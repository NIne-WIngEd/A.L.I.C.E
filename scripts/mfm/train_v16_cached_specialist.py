"""Train the unchanged MFM specialist from verified CPU BF16 source states.

This separately versioned route uses FP32 specialist arithmetic on P100/T4 or
CPU. It does not quantize, shard or update Gemma. Teacher-fit artifacts remain
unqualified; diagnostic development is never used for gradients or FINAL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from time import perf_counter

from cognitive_kernel.canonical import CognitiveKernelContractError
from . import train_v16_formation_specialist as training
from . import formation_feature_bank as features

RUN_SCHEMA = "mfm-v16-cached-specialist-run-v1"
ARTIFACT_SCHEMA = "mfm-v16-cached-specialist-artifact-v1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-training-manifest", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--owner-authorization-ref", required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--feature-bank-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--specialist-width", type=int, default=768)
    parser.add_argument("--logit-chunk-tokens", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--session-seconds", type=int,
                        help="checkpoint at an optimizer boundary before this session ends")
    return parser.parse_args()


def _payload(root, example, export_digest, tokenizer, config, device):
    import torch
    _, tensors = features.read_case(root, example, export_digest,
                                    width=config.base_hidden_size)
    inputs, labels = training._target_ids(tokenizer, example, config.max_target_tokens)
    return {"base_states": tensors["states"].to(device),
            "source_mask": tensors["attention_mask"].to(device),
            "input_ids": torch.tensor([inputs], device=device),
            "labels": torch.tensor([labels], device=device)}


def optimizer_step(specialist, optimizer, examples, *, root, export_digest,
                   tokenizer, device) -> dict:
    import torch
    if not examples or any(e.split != "train" for e in examples):
        raise CognitiveKernelContractError("only admitted train examples enter gradients")
    specialist.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for example in examples:
        payload = _payload(root, example, export_digest, tokenizer, specialist.config, device)
        _, loss = specialist(**payload)
        if not torch.isfinite(loss):
            raise CognitiveKernelContractError("cached training loss is nonfinite")
        (loss / len(examples)).backward()
        losses.append(float(loss.detach().cpu()))
    norm = torch.nn.utils.clip_grad_norm_(specialist.parameters(), 1.0,
                                         error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise CognitiveKernelContractError("cached specialist has no finite nonzero gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not torch.isfinite(p).all() for p in specialist.parameters()):
        raise CognitiveKernelContractError("cached optimizer produced nonfinite parameters")
    return {"mean_loss": sum(losses) / len(losses), "gradient_norm": norm.item(),
            "train_examples": len(examples)}


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_tree(v) for v in value]
    return value


def compare_tree(actual, expected, *, atol=1e-6, rtol=1e-5) -> float:
    import torch
    if isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype or \
                not torch.allclose(actual.cpu(), expected, atol=atol, rtol=rtol):
            raise CognitiveKernelContractError("resumed optimizer continuation differs")
        return float((actual.cpu() - expected).abs().max()) if actual.numel() else 0.0
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise CognitiveKernelContractError("resumed optimizer tree differs")
        return max((compare_tree(actual[k], expected[k], atol=atol, rtol=rtol)
                    for k in expected), default=0.0)
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise CognitiveKernelContractError("resumed optimizer list differs")
        return max((compare_tree(a, b, atol=atol, rtol=rtol)
                    for a, b in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise CognitiveKernelContractError("resumed optimizer scalar differs")
    return 0.0


def probe(specialist, optimizer, examples, *, args, run_digest, step_kwargs) -> dict:
    import torch
    before = _cpu_tree(specialist.state_dict())
    results = []
    for _ in range(2):
        results.append(optimizer_step(specialist, optimizer, examples, **step_kwargs))
    training._checkpoint(args.output_dir, specialist=specialist, optimizer=optimizer,
                         run_digest=run_digest, epoch=0, next_case=len(examples), step=2)
    expected_result = optimizer_step(specialist, optimizer, examples, **step_kwargs)
    expected_model = _cpu_tree(specialist.state_dict())
    expected_optimizer = _cpu_tree(optimizer.state_dict())
    training._resume(args.output_dir / "checkpoint-00000002", args.output_dir,
                     specialist, optimizer, run_digest)
    actual_result = optimizer_step(specialist, optimizer, examples, **step_kwargs)
    if abs(actual_result["mean_loss"] - expected_result["mean_loss"]) > 1e-6:
        raise CognitiveKernelContractError("resumed next loss differs")
    model_difference = compare_tree(specialist.state_dict(), expected_model)
    optimizer_difference = compare_tree(optimizer.state_dict(), expected_optimizer)
    if all(torch.equal(p.detach().cpu(), before[name])
           for name, p in specialist.state_dict().items()):
        raise CognitiveKernelContractError("cached probe did not change weights")
    from safetensors.torch import save_file
    path = args.output_dir / "probe-specialist.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in specialist.state_dict().items()}, str(path))
    return {"first_and_later_steps": results, "resumed_next_step": actual_result,
            "model_max_abs_difference": model_difference,
            "optimizer_max_abs_difference": optimizer_difference,
            "atol": 1e-6, "rtol": 1e-5, "checkpoint_reload_executed": True,
            "optimizer_steps": 3, "backward_and_restart_passed": True,
            "probe_weights_sha256": training.shared._digest(path)}


def development_loss(specialist, examples, **kwargs) -> float:
    import torch
    if not examples or any(e.split != "development" for e in examples):
        raise CognitiveKernelContractError("diagnostic development split differs")
    specialist.eval()
    total = count = 0
    with torch.no_grad():
        for example in examples:
            payload = _payload(kwargs["root"], example, kwargs["export_digest"],
                               kwargs["tokenizer"], specialist.config, kwargs["device"])
            _, loss = specialist(**payload)
            tokens = payload["labels"].numel()
            if not torch.isfinite(loss):
                raise CognitiveKernelContractError("diagnostic development loss is nonfinite")
            total += float(loss.cpu()) * tokens
            count += tokens
    return total / count


def run(args) -> dict:
    training.shared._require_private_network_isolation()
    args.mode = "train"
    args.teacher_fit = True
    args.full_fit = False
    args.admitted_manifest = args.public_synthetic_curriculum = None
    train, development, status = training._examples(args)
    preflight = training.shared._read_sealed(args.preflight_receipt, training.PREFLIGHT_SCHEMA)
    if preflight.get("corpus_sha256") != args.input_sha256 or \
            preflight.get("corpus_status") != status or preflight.get("teacher_fit") is not True or \
            preflight.get("full_fit") is not False or \
            preflight.get("trainer_sha256") != training.shared._digest(Path(training.__file__)) or \
            preflight.get("codec_sha256") != training._codec_sha256() or \
            preflight.get("semantics_sha256") != training._semantics_sha256() or \
            preflight.get("decoder_implementation_sha256") != training._decoder_sha256() or \
            preflight.get("owner_authorization_ref") != args.owner_authorization_ref or \
            preflight.get("train_cases") != len(train) or \
            preflight.get("development_cases") != len(development):
        raise CognitiveKernelContractError("cached preflight binding differs")
    bank, export_record = features.verify_bank(
        args.feature_bank, expected_sha256=args.feature_bank_sha256,
        examples=(*train, *development), preflight=preflight)
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import save_file, load_file
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
    device = torch.device(args.device)
    if device.type == "cuda" and (device.index is None or not torch.cuda.is_available() or
                                  device.index >= torch.cuda.device_count()):
        raise CognitiveKernelContractError("cached CUDA device is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise CognitiveKernelContractError("cached trainer requires explicit CPU or CUDA")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.feature_bank / "processor"),
                                              trust_remote_code=False, local_files_only=True)
    config_record = json.loads((args.feature_bank / "processor" / "config.json").read_bytes())
    config = SpecialistConfig(
        base_hidden_size=config_record["text_config"]["hidden_size"],
        vocabulary_size=len(tokenizer), width=args.specialist_width,
        layers=preflight["specialist_layers"], heads=preflight["specialist_heads"],
        max_target_tokens=preflight["max_target_tokens"],
        pad_token_id=tokenizer.pad_token_id, start_token_id=tokenizer.bos_token_id,
        end_token_id=tokenizer.eos_token_id, logit_chunk_tokens=args.logit_chunk_tokens)
    specialist = FormationSpecialist(config).to(device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(specialist.parameters(), lr=args.learning_rate)
    run_record = {"schema": RUN_SCHEMA, "objective": training.OBJECTIVE_VERSION_V16,
                  "feature_bank_sha256": bank["record_sha256"],
                  "preflight_sha256": preflight["record_sha256"],
                  "corpus_sha256": args.input_sha256, "corpus_status": status,
                  "prepared_base_receipt_sha256": export_record["prepared_base_receipt_sha256"],
                  "specialist_config": config.record(), "epochs": args.epochs,
                  "gradient_accumulation": args.gradient_accumulation,
                  "learning_rate": args.learning_rate, "seed": args.seed,
                  "specialist_dtype": "float32", "device": str(device),
                  "torch_version": torch.__version__,
                  "trainer_sha256": training.shared._digest(Path(__file__)),
                  "probe_only": args.probe_only, "teacher_fit": True,
                  "full_fit": False, "qualified_for_product": False}
    digest = training.shared._record_hash(run_record)
    if args.resume_checkpoint:
        if args.probe_only or not args.output_dir.is_dir() or (args.output_dir / "component.json").exists():
            raise CognitiveKernelContractError("cached resume needs an incomplete matching run")
        old = training.shared._read_sealed(args.output_dir / "run.json", RUN_SCHEMA)
        if old["record_sha256"] != digest:
            raise CognitiveKernelContractError("cached resume run differs")
        training._verify_seed_control(args.output_dir, digest, config)
        epoch_cursor, case_cursor, steps = training._resume(
            args.resume_checkpoint, args.output_dir, specialist, optimizer, digest)
    else:
        args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        training.shared._write_new(args.output_dir / "run.json", run_record)
        training._seed_control(args.output_dir, specialist, digest, config)
        epoch_cursor = case_cursor = steps = 0
    step_kwargs = dict(root=args.feature_bank, export_digest=export_record["record_sha256"],
                       tokenizer=tokenizer, device=device)
    started = perf_counter()
    if args.probe_only:
        stress = preflight["stress_train_case_ids"]
        if not stress or not set(stress).issubset({e.case_id for e in train}):
            raise CognitiveKernelContractError("probe stress cases are not admitted train")
        selected = sorted(train, key=lambda e: stress.index(e.case_id) if e.case_id in stress else len(stress))
        if args.gradient_accumulation < len(stress) or len(train) < args.gradient_accumulation:
            raise CognitiveKernelContractError("probe omits complete stress accumulation")
        result = probe(specialist, optimizer, selected[:args.gradient_accumulation],
                       args=args, run_digest=digest, step_kwargs=step_kwargs)
    else:
        for epoch in range(epoch_cursor, args.epochs):
            order = list(train)
            random.Random(args.seed + epoch).shuffle(order)
            cursor = case_cursor if epoch == epoch_cursor else 0
            if not 0 <= cursor <= len(order) or (cursor != len(order) and cursor % args.gradient_accumulation):
                raise CognitiveKernelContractError("cached resume cursor is not an optimizer boundary")
            while cursor < len(order):
                group = order[cursor:cursor + args.gradient_accumulation]
                step = optimizer_step(specialist, optimizer, group, **step_kwargs)
                cursor += len(group)
                steps += 1
                pause = args.session_seconds is not None and perf_counter() - started >= args.session_seconds
                if steps % args.save_every_steps == 0 or pause or cursor == len(order):
                    training._checkpoint(args.output_dir, specialist=specialist, optimizer=optimizer,
                                         run_digest=digest, epoch=epoch, next_case=cursor, step=steps)
                print(json.dumps({"optimizer_step": steps, "epoch": epoch,
                                  "train_cases_in_epoch": cursor, **step}), flush=True)
                if pause and not (epoch == args.epochs - 1 and cursor == len(order)):
                    return {"state": "CHECKPOINTED_INCOMPLETE", "optimizer_steps": steps,
                            "resume_checkpoint": str(args.output_dir / f"checkpoint-{steps:08d}"),
                            "qualified_for_product": False}
        trained_path = args.output_dir / "formation-specialist.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in specialist.state_dict().items()}, str(trained_path))
        trained_loss = development_loss(specialist, development, **step_kwargs)
        specialist.load_state_dict(load_file(str(args.output_dir / "seed-control.safetensors")))
        seed_loss = development_loss(specialist, development, **step_kwargs)
        specialist.load_state_dict(load_file(str(trained_path)))
        seed = training._verify_seed_control(args.output_dir, digest, config)
        if training.shared._digest(trained_path) == seed["specialist_sha256"]:
            raise CognitiveKernelContractError("cached full fit did not change weights")
        result = {"optimizer_steps": steps, "all_train_epochs_completed": True,
                  "trained_weights_sha256": training.shared._digest(trained_path),
                  "diagnostic_development_teacher_forced_loss": trained_loss,
                  "seeded_development_teacher_forced_loss": seed_loss,
                  "development_has_no_gradients": True, "independent_qualification": False}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["hardware"] = {"device_name": torch.cuda.get_device_name(device),
                              "total_bytes": torch.cuda.get_device_properties(device).total_memory,
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    component = training.shared._write_new(args.output_dir / "component.json", {
        "schema": ARTIFACT_SCHEMA, "run_sha256": digest,
        "feature_bank_sha256": bank["record_sha256"], "specialist_config": config.record(),
        "probe_only": args.probe_only, "teacher_fit": True, "full_fit": False,
        "qualified_for_product": False, "result": result,
        "scope": "cache-only teacher learning; diagnostic CE is not generated formation behavior"})
    return {"state": "PROBE_COMPLETED" if args.probe_only else "TEACHER_FIT_COMPLETED_UNQUALIFIED",
            "component_sha256": component["record_sha256"], **result,
            "seconds": perf_counter() - started, "qualified_for_product": False}


def main():
    args = arguments()
    if any(v < 1 for v in (args.specialist_width, args.logit_chunk_tokens, args.epochs,
                          args.gradient_accumulation, args.save_every_steps)) or \
            args.learning_rate <= 0 or (args.session_seconds is not None and args.session_seconds < 1):
        raise CognitiveKernelContractError("invalid cached training recipe")
    if args.probe_only and args.resume_checkpoint:
        raise CognitiveKernelContractError("probe and resume are distinct runs")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true"})
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
