"""Measure the full unchanged decoder on maximum-length synthetic tensors.

No corpus payloads or backbone are opened. This early hardware test does not
replace the later admitted-corpus stress/restart probe or qualify learning.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from . import train_v16_cached_specialist as cached


def run(args):
    shared = cached.training.shared
    # Purely public random tensors. There is deliberately no corpus, model,
    # processor or preflight-file argument. Private training keeps its separate
    # mandatory namespace guard; this probe cannot be used to bypass it.
    reference = require_sha256(args.shape_reference_sha256, "shape_reference_sha256")
    import torch
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CognitiveKernelContractError("capacity probe requires one actual CUDA device")
    source_length, target_length = args.source_tokens, args.target_tokens
    if not 0 < source_length <= 32768 or not 1 < target_length <= 8192:
        raise CognitiveKernelContractError("capacity stress lengths invalid")
    torch.manual_seed(73129)
    torch.cuda.manual_seed_all(73129)
    config = SpecialistConfig(3840, 262144, 768, 6, 12, 8192, 0, 2, 1, 64)
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    record = shared._write_new(args.output_dir / "run.json", {
        "schema": "mfm-v16-cached-synthetic-capacity-run-v1",
        "shape_reference_sha256": reference, "shape_reference_file_opened": False,
        "specialist_config": config.record(),
        "probe_sha256": shared._digest(Path(__file__)),
        "decoder_sha256": cached.training._decoder_sha256(),
        "source_length": source_length, "target_length": target_length,
        "gradient_accumulation": 16, "specialist_dtype": "float32", "source_dtype": "bfloat16",
        "torch_version": torch.__version__, "teacher_corpus_opened": False,
        "qualified_for_product": False,
    })
    digest = record["record_sha256"]
    started = perf_counter()
    model = FormationSpecialist(config).to("cuda:0", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    states = torch.randn(1, source_length, config.base_hidden_size, device="cuda:0").bfloat16()
    mask = torch.ones(1, source_length, dtype=torch.long, device="cuda:0")
    labels = torch.randint(3, config.vocabulary_size, (1, target_length), device="cuda:0")
    labels[:, -1] = config.end_token_id
    tokens = torch.cat((torch.full((1, 1), config.start_token_id, device="cuda:0"), labels[:, :-1]), dim=1)
    before = cached._cpu_tree(model.state_dict())

    def step(index):
        start = perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for accumulation in range(16):
            _, loss = model(base_states=states, source_mask=mask, input_ids=tokens, labels=labels)
            if not torch.isfinite(loss):
                raise CognitiveKernelContractError("capacity loss nonfinite")
            (loss / 16).backward()
            losses.append(float(loss.detach().cpu()))
            if accumulation == 0:
                print(json.dumps({"event": "capacity_first_backward", "step": index,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()}), flush=True)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not torch.isfinite(norm) or norm.item() <= 0:
            raise CognitiveKernelContractError("capacity gradient missing or invalid")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise CognitiveKernelContractError("capacity parameters nonfinite")
        torch.cuda.synchronize()
        result = {"step": index, "mean_loss": sum(losses) / len(losses), "gradient_norm": norm.item(),
            "seconds": perf_counter() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        print(json.dumps(result), flush=True)
        return result

    print(json.dumps({"event": "full_capacity_model_ready", "parameters": sum(p.numel() for p in model.parameters()),
        "source_length": source_length, "target_length": target_length,
        "device": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}), flush=True)
    first = step(1)
    second = step(2)
    shared._checkpoint(args.output_dir, specialist=model, optimizer=optimizer,
        run_digest=digest, epoch=0, next_case=16, step=2)
    expected_loss = step(3)
    expected_model = cached._cpu_tree(model.state_dict())
    expected_optimizer = cached._cpu_tree(optimizer.state_dict())
    shared._resume(args.output_dir / "checkpoint-00000002", args.output_dir, model, optimizer, digest)
    resumed = step(3)
    if abs(resumed["mean_loss"] - expected_loss["mean_loss"]) > 1e-6:
        raise CognitiveKernelContractError("capacity resumed next loss differs")
    model_difference = cached.compare_tree(model.state_dict(), expected_model)
    optimizer_difference = cached.compare_tree(optimizer.state_dict(), expected_optimizer)
    if all(torch.equal(p.detach().cpu(), before[name]) for name, p in model.state_dict().items()):
        raise CognitiveKernelContractError("capacity optimizer did not change weights")
    return shared._write_new(args.output_dir / "capacity.json", {
        "schema": "mfm-v16-cached-synthetic-capacity-v1", "run_sha256": digest,
        "synthetic_tensor_capacity_passed": True, "empirical_corpus_stress_passed": False,
        "checkpoint_reload_executed": True, "model_max_abs_difference": model_difference,
        "optimizer_max_abs_difference": optimizer_difference, "first_step": first,
        "second_step": second, "resumed_third_step": resumed,
        "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "seconds": perf_counter() - started, "teacher_corpus_opened": False,
        "qualified_for_product": False,
        "scope": "full configured decoder and maximal-length synthetic tensor resource/restart measurement; no teacher learning or admitted-corpus stress pass",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-reference-sha256", required=True)
    parser.add_argument("--source-tokens", type=int, default=3035)
    parser.add_argument("--target-tokens", type=int, default=1607)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true"})
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
