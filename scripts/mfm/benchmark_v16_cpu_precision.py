"""Measure higher-precision CPU execution against committed BF16 source states.

This does not train, write a feature bank, change source weights or qualify
behavior. BF16 publisher values are exactly representable in FP32; arithmetic
and resulting states can differ and are reported, never called equivalent.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from . import train_v16_formation_specialist as training
from . import formation_feature_bank as features


def run(args):
    training.shared._require_private_network_isolation()
    args.mode, args.teacher_fit, args.full_fit = "train", True, False
    args.admitted_manifest = args.public_synthetic_curriculum = None
    args.trust_roster_sha256 = args.signed_review_receipt_sha256 = None
    train, development, status = training._examples(args)
    sealed = training.shared._read_sealed(args.preflight_receipt, training.PREFLIGHT_SCHEMA)
    for field in ("max_source_tokens", "max_target_tokens", "specialist_heads", "specialist_layers"):
        setattr(args, field, sealed[field])
    prepared = training.shared._prepared_base(args)
    processor, preflight = training._processor_preflight(args, train, development, status, prepared)
    reference = training.shared._read_sealed(args.reference_bank / "export.json", features.EXPORT_SCHEMA)
    if reference["record_sha256"] != require_sha256(args.reference_export_sha256, "reference_export_sha256") or \
            reference.get("preflight") != preflight or reference.get("backbone_dtype") != "bfloat16" or \
            reference.get("backbone_device") != "cpu" or reference.get("base_frozen") is not True:
        raise CognitiveKernelContractError("precision reference lineage differs")
    import torch
    import resource
    from transformers import AutoModelForMultimodalLM
    started = perf_counter()
    base = AutoModelForMultimodalLM.from_pretrained(prepared["snapshot_path"],
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, device_map={"": "cpu"}, low_cpu_mem_usage=True)
    base.requires_grad_(False).eval()
    if any(p.dtype != torch.float32 or p.device.type != "cpu" or p.requires_grad for p in base.parameters()):
        raise CognitiveKernelContractError("precision benchmark base differs")
    load_seconds = perf_counter() - started
    print(json.dumps({"event": "fp32_base_loaded", "seconds": load_seconds,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    rows = []
    for example in train[:2]:
        _, original = features.read_case(args.reference_bank, example, reference["record_sha256"],
                                         width=base.config.text_config.hidden_size)
        encoded = training.source_batch_v16(processor, example.context, example.opened_sources,
                                             args.max_source_tokens, case_id=example.case_id)
        if any(not torch.equal(encoded[k], original[k]) for k in ("attention_mask", "input_ids")):
            raise CognitiveKernelContractError("precision benchmark processor/source differs")
        start = perf_counter()
        with torch.no_grad():
            states = base.model(**encoded, use_cache=False, return_dict=True).last_hidden_state
        seconds = perf_counter() - start
        if states.dtype != torch.float32 or states.shape != original["states"].shape or not torch.isfinite(states).all():
            raise CognitiveKernelContractError("precision benchmark states invalid")
        baseline = original["states"].float()
        delta = states - baseline
        row = {"model_input_sha256": features.case_key(example), "source_tokens": states.shape[1],
               "forward_seconds": seconds, "max_abs_difference": delta.abs().max().item(),
               "root_mean_squared_difference": delta.square().mean().sqrt().item(),
               "cosine_similarity": torch.nn.functional.cosine_similarity(states.flatten(), baseline.flatten(), dim=0).item(),
               "bit_identical": torch.equal(states, baseline),
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del states, baseline, delta, original, encoded
    return training.shared._write_new(args.output_receipt, {
        "schema": "mfm-v16-cpu-fp32-precision-benchmark-v1", "corpus_sha256": args.input_sha256,
        "preflight_sha256": preflight["record_sha256"], "reference_export_sha256": reference["record_sha256"],
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "benchmark_sha256": training.shared._digest(Path(__file__)), "torch_version": torch.__version__,
        "backbone_device": "cpu", "backbone_dtype": "float32", "base_load_seconds": load_seconds,
        "rows": rows, "final_payloads_opened": False, "targets_visible_to_backbone": False,
        "qualified_for_product": False, "scope": "two admitted source forwards only; numerical difference and capacity measurement, no semantic or training qualification"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-training-manifest", "prepared-base-dir", "prepared-base-receipt",
                 "preflight-receipt", "reference-bank", "output-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("input-sha256", "owner-authorization-ref", "reference-export-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true"})
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
