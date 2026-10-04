"""Matched generated diagnostics for the separately versioned FP32 teacher fit.

Only admitted diagnostic development is decoded. Targets never enter decoding;
raw invalid output and missing EOS are retained. This is not independent gold,
FINAL, memory admission or product qualification. The live CUDA BF16 inference
runner cannot be substituted: this route must retain FP32 on P100.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from cognitive_kernel.formation_semantics_v16 import (
    bundle_v16_from_output, validate_formation_grounding_v16,
)
from . import train_v16_cached_specialist as cached
from . import formation_feature_bank as features

SCHEMA = "mfm-v16-cached-generated-diagnostic-v1"
RESUME_SCHEMA = "mfm-v16-cached-diagnostic-recovery-plan-v1"
LEGACY_RUNNER_SHA256 = "2874fe7b9e1fea19456810bd3728135610a0da7d931aa83ef9d6b29a3cb943bf"


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CognitiveKernelContractError("duplicate generated JSON key")
        result[key] = value
    return result


def verify_component(root, *, expected_sha256, bank_digest, preflight, corpus_digest):
    shared = cached.training.shared
    component = shared._read_sealed(root / "component.json", cached.ARTIFACT_SCHEMA)
    run = shared._read_sealed(root / "run.json", cached.RUN_SCHEMA)
    if component["record_sha256"] != require_sha256(expected_sha256, "component_sha256") or \
            component.get("run_sha256") != run["record_sha256"] or \
            component.get("specialist_config") != run.get("specialist_config") or \
            run.get("trainer_sha256") != shared._digest(Path(cached.__file__)) or \
            run.get("objective") != cached.training.OBJECTIVE_VERSION_V16 or \
            run.get("corpus_sha256") != corpus_digest or \
            run.get("preflight_sha256") != preflight["record_sha256"] or \
            run.get("specialist_dtype") != "float32" or \
            any(r.get("feature_bank_sha256") != bank_digest or r.get("teacher_fit") is not True or
                r.get("probe_only") is not False or r.get("full_fit") is not False or
                r.get("qualified_for_product") is not False for r in (run, component)):
        raise CognitiveKernelContractError("cached diagnostic component lineage differs")
    result = component.get("result", {})
    if result.get("all_train_epochs_completed") is not True or \
            type(result.get("optimizer_steps")) is not int or result["optimizer_steps"] < 1:
        raise CognitiveKernelContractError("cached diagnostic needs a completed teacher fit")
    from cognitive_kernel.formation_v1_specialist import SpecialistConfig
    config = SpecialistConfig(**run["specialist_config"])
    if config.layers != preflight["specialist_layers"] or config.heads != preflight["specialist_heads"] or \
            config.max_target_tokens != preflight["max_target_tokens"]:
        raise CognitiveKernelContractError("cached diagnostic decoder budgets differ")
    seed = cached.training._verify_seed_control(root, run["record_sha256"], config)
    weights = root / "formation-specialist.safetensors"
    if shared._digest(weights) != result.get("trained_weights_sha256") or \
            result["trained_weights_sha256"] == seed["specialist_sha256"]:
        raise CognitiveKernelContractError("cached trained weights differ or are the seeded control")
    return component, run, config, {
        "trained": (weights, result["trained_weights_sha256"]),
        "seeded-untrained": (root / "seed-control.safetensors", seed["specialist_sha256"])}


def greedy_fp32(specialist, *, states, mask, max_new_tokens):
    import torch
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= specialist.config.max_target_tokens:
        raise CognitiveKernelContractError("cached diagnostic token budget differs")
    if any(p.dtype != torch.float32 for p in specialist.parameters()):
        raise CognitiveKernelContractError("cached diagnostic requires FP32 specialist arithmetic")
    device = next(specialist.parameters()).device
    states, mask = states.to(device), mask.to(device)
    prefix = [specialist.config.start_token_id]
    generated = []
    specialist.eval()
    # No BF16 autocast: P100 has no native BF16 arithmetic. This is prefix
    # recomputation, matching the existing decoder's reference algorithm.
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        for _ in range(max_new_tokens):
            inputs = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = specialist.next_token_logits(base_states=states, source_mask=mask, input_ids=inputs)
            if not torch.isfinite(logits).all():
                raise CognitiveKernelContractError("cached diagnostic logits are nonfinite")
            token = int(logits[0].argmax())
            if token == specialist.config.end_token_id:
                return generated, True
            prefix.append(token)
            generated.append(token)
    return generated, False


def validate_generated(raw, ended, example, weight_digest, run_id):
    record = {"raw_output_text": raw, "eos_observed": ended,
              "validation_status": "invalid", "validation_error": None}
    if not ended:
        record["validation_error"] = "no EOS inside declared diagnostic budget"
        return record
    try:
        body = json.loads(raw, object_pairs_hook=_unique_json)
        bundle = bundle_v16_from_output(example.context, body,
                                       artifact_sha256=weight_digest, inference_run_id=run_id)
        validate_formation_grounding_v16(example.context, bundle, example.opened_sources)
    except (ValueError, TypeError, KeyError, CognitiveKernelContractError) as exc:
        record["validation_error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["validation_status"] = "grounded_teacher_proposal_only"
    return record


def load_resume_rows(path, *, expected_sha256, control, weight_digest, run_id,
                     examples, tokenizer, config, max_new_tokens):
    """Accept only complete, externally pinned prefix rows; preserve failures."""
    if cached.training.shared._digest(path) != require_sha256(expected_sha256, "partial_output_sha256"):
        raise CognitiveKernelContractError("partial diagnostic output digest differs")
    rows = [json.loads(line, object_pairs_hook=_unique_json) for line in path.read_text().splitlines()]
    if len(rows) > len(examples):
        raise CognitiveKernelContractError("partial diagnostic has extra cases")
    for row, example in zip(rows, examples):
        bindings = {"case_id": example.case_id, "split": "development", "control": control,
                    "weights_sha256": weight_digest, "inference_run_id": run_id,
                    "context_sha256": example.context.content_digest(),
                    "model_input_sha256": features.case_key(example)}
        if any(row.get(key) != value for key, value in bindings.items()):
            raise CognitiveKernelContractError("partial diagnostic row binding differs")
        tokens, ended = row.get("generated_token_ids"), row.get("eos_observed")
        if type(tokens) is not list or type(ended) is not bool or \
                any(type(t) is not int or not 0 <= t < config.vocabulary_size or
                    t == config.end_token_id for t in tokens) or \
                len(tokens) > max_new_tokens or (ended and len(tokens) >= max_new_tokens) or \
                (not ended and len(tokens) != max_new_tokens):
            raise CognitiveKernelContractError("partial diagnostic tokens or stopping differ")
        validated = validate_generated(tokenizer.decode(tokens, skip_special_tokens=False),
                                       ended, example, weight_digest, run_id)
        expected = {**bindings, "generated_token_ids": tokens, **validated}
        if row != expected:
            raise CognitiveKernelContractError("partial diagnostic validation differs")
    return rows


def prepare_diagnostic(args):
    cached.training.shared._require_private_network_isolation()
    args.mode, args.teacher_fit, args.full_fit = "train", True, False
    args.admitted_manifest = args.public_synthetic_curriculum = None
    train, development, status = cached.training._examples(args)
    preflight = cached.training.shared._read_sealed(args.preflight_receipt, cached.training.PREFLIGHT_SCHEMA)
    if preflight.get("corpus_sha256") != args.input_sha256 or preflight.get("corpus_status") != status or \
            preflight.get("owner_authorization_ref") != args.owner_authorization_ref or \
            preflight.get("trainer_sha256") != cached.training.shared._digest(Path(cached.training.__file__)) or \
            preflight.get("codec_sha256") != cached.training._codec_sha256() or \
            preflight.get("semantics_sha256") != cached.training._semantics_sha256() or \
            preflight.get("decoder_implementation_sha256") != cached.training._decoder_sha256():
        raise CognitiveKernelContractError("cached diagnostic preflight differs")
    bank, export = features.verify_bank(args.feature_bank, expected_sha256=args.feature_bank_sha256,
                                        examples=(*train, *development), preflight=preflight)
    component, run_record, config, controls = verify_component(args.component_dir,
        expected_sha256=args.component_sha256, bank_digest=bank["record_sha256"],
        preflight=preflight, corpus_digest=args.input_sha256)
    if not preflight["longest_target_tokens"] <= args.max_new_tokens <= config.max_target_tokens:
        raise CognitiveKernelContractError("diagnostic budget omits observed complete target length")
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist
    device = torch.device(args.device)
    if device.type == "cuda" and (device.index is None or not torch.cuda.is_available() or
                                  device.index >= torch.cuda.device_count()):
        raise CognitiveKernelContractError("cached diagnostic CUDA ordinal unavailable")
    if device.type not in {"cuda", "cpu"}:
        raise CognitiveKernelContractError("cached diagnostic device unavailable")
    tokenizer = AutoTokenizer.from_pretrained(str(args.feature_bank / "processor"),
                                              local_files_only=True, trust_remote_code=False)
    from types import SimpleNamespace
    return SimpleNamespace(train=train, development=development, preflight=preflight,
        bank=bank, export=export, component=component, run_record=run_record,
        config=config, controls=controls, tokenizer=tokenizer, device=device)


def run(args):
    prepared = prepare_diagnostic(args)
    development, bank, export = prepared.development, prepared.bank, prepared.export
    component, run_record, config = prepared.component, prepared.run_record, prepared.config
    controls, tokenizer, device = prepared.controls, prepared.tokenizer, prepared.device
    import torch
    from safetensors.torch import load_file
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist
    resume = None
    resume_plan_path = getattr(args, "resume_plan", None)
    resume_pin = getattr(args, "resume_plan_sha256", None)
    if bool(resume_plan_path) != bool(resume_pin):
        raise CognitiveKernelContractError("diagnostic recovery needs an external plan pin")
    if resume_plan_path:
        resume = cached.training.shared._read_sealed(resume_plan_path, RESUME_SCHEMA)
        if resume["record_sha256"] != require_sha256(resume_pin, "resume_plan_sha256") or \
                any(resume.get(key) != value for key, value in {
                    "component_sha256": component["record_sha256"],
                    "training_run_sha256": run_record["record_sha256"],
                    "feature_bank_sha256": bank["record_sha256"], "corpus_sha256": args.input_sha256,
                    "decode": {"device": str(device), "dtype": "float32", "max_new_tokens": args.max_new_tokens,
                               "method": "greedy-prefix-reference", "target_visible": False},
                }.items()) or resume.get("runner_sha256") not in {
                    LEGACY_RUNNER_SHA256, cached.training.shared._digest(Path(__file__))} or \
                set(resume.get("outputs", {})) != set(controls):
            raise CognitiveKernelContractError("diagnostic recovery plan lineage or settings differ")
    retained = {}
    for control, (_, digest) in controls.items():
        rows = []
        if resume:
            entry = resume["outputs"][control]
            rows = load_resume_rows(Path(resume["output_dir"]) / (control + ".jsonl"),
                expected_sha256=entry["sha256"], control=control, weight_digest=digest,
                run_id="cached-diagnostic-" + component["record_sha256"][:16] + "-" + control,
                examples=development, tokenizer=tokenizer, config=config, max_new_tokens=args.max_new_tokens)
            if len(rows) != entry["completed_cases"]:
                raise CognitiveKernelContractError("diagnostic recovery case count differs")
        retained[control] = rows
    model = FormationSpecialist(config).to(device, dtype=torch.float32)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    summary = {}
    artifacts = {}
    for control, (path, digest) in controls.items():
        weights = load_file(str(path))
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in weights.values()):
            raise CognitiveKernelContractError("cached diagnostic weights changed dtype or are nonfinite")
        model.load_state_dict(weights)
        del weights
        run_id = "cached-diagnostic-" + component["record_sha256"][:16] + "-" + control
        output_path = args.output_dir / (control + ".jsonl")
        valid = sum(r["validation_status"] != "invalid" for r in retained[control])
        with output_path.open("x", encoding="utf-8") as output:
            for row in retained[control]:
                output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            for index in range(len(retained[control]), len(development)):
                example = development[index]
                _, tensors = features.read_case(args.feature_bank, example, export["record_sha256"],
                                                 width=config.base_hidden_size)
                tokens, ended = greedy_fp32(model, states=tensors["states"],
                    mask=tensors["attention_mask"], max_new_tokens=args.max_new_tokens)
                raw = tokenizer.decode(tokens, skip_special_tokens=False)
                record = validate_generated(raw, ended, example, digest, run_id)
                valid += record["validation_status"] != "invalid"
                output.write(json.dumps({"case_id": example.case_id, "split": "development",
                    "control": control, "weights_sha256": digest, "inference_run_id": run_id,
                    "context_sha256": example.context.content_digest(),
                    "model_input_sha256": features.case_key(example),
                    "generated_token_ids": tokens, **record}, sort_keys=True) + "\n")
                output.flush()
                print(json.dumps({"control": control, "completed": index + 1,
                                  "total": len(development), "grounded_so_far": valid}), flush=True)
        summary[control] = {"cases": len(development), "grounded_proposals": valid,
                            "invalid_outputs": len(development) - valid}
        artifacts[control] = cached.training.shared._digest(output_path)
    return cached.training.shared._write_new(args.output_dir / "diagnostic.json", {
        "schema": SCHEMA, "component_sha256": component["record_sha256"],
        "training_run_sha256": run_record["record_sha256"],
        "feature_bank_sha256": bank["record_sha256"], "corpus_sha256": args.input_sha256,
        "runner_sha256": cached.training.shared._digest(Path(__file__)),
        "decode": {"method": "greedy-prefix-reference", "dtype": "float32",
                   "device": str(device), "max_new_tokens": args.max_new_tokens,
                   "matched_between_controls": True, "target_visible": False},
        "outputs_sha256": artifacts, "summary": summary,
        "recovery": {"plan_sha256": resume["record_sha256"] if resume else None,
                     "reused_cases": {name: len(rows) for name, rows in retained.items()}},
        "independent_gold": False, "memory_gate_executed": False,
        "final_payloads_opened": False, "qualified_for_product": False,
        "scope": "generated and source-grounded teacher development diagnostic; no semantic accuracy or downstream usefulness claim"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-training-manifest", "preflight-receipt", "feature-bank",
                 "component-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("input-sha256", "owner-authorization-ref", "feature-bank-sha256", "component-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--resume-plan", type=Path)
    parser.add_argument("--resume-plan-sha256")
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true"})
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
