"""Export the complete admitted teacher corpus through the exact frozen base.

CPU BF16 preserves the established representation dtype. Serialization is
lossless. Each case is committed atomically; a later identical invocation can
resume verified cases. Only a completed bank is usable for specialist training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

from cognitive_kernel.canonical import CognitiveKernelContractError
from . import train_v16_formation_specialist as training
from . import formation_feature_bank as features


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-training-manifest", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--owner-authorization-ref", required=True)
    parser.add_argument("--prepared-base-dir", type=Path, required=True)
    parser.add_argument("--prepared-base-receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-source-tokens", type=int, default=32768)
    parser.add_argument("--max-target-tokens", type=int, default=8192)
    parser.add_argument("--specialist-heads", type=int, default=12)
    parser.add_argument("--specialist-layers", type=int, default=6)
    return parser.parse_args()


def export(args) -> dict:
    training.shared._require_private_network_isolation()
    args.mode = "train"
    args.teacher_fit = True
    args.full_fit = False
    args.admitted_manifest = args.public_synthetic_curriculum = None
    args.trust_roster_sha256 = args.signed_review_receipt_sha256 = None
    train, development, status = training._examples(args)
    prepared = training.shared._prepared_base(args)
    processor, preflight = training._processor_preflight(
        args, train, development, status, prepared)
    import torch
    from transformers import AutoModelForMultimodalLM
    binding = features.feature_binding(preflight, prepared, torch.__version__)
    binding["schema"] = features.EXPORT_SCHEMA
    root = args.output_dir
    if args.resume:
        if not root.is_dir() or root.is_symlink():
            raise CognitiveKernelContractError("feature resume needs existing custody directory")
        export_record = training.shared._read_sealed(root / "export.json", features.EXPORT_SCHEMA)
        if export_record != training.shared._seal(binding) or (root / "bank.json").exists():
            raise CognitiveKernelContractError("feature resume differs or bank is already complete")
    else:
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        export_record = training.shared._write_new(root / "export.json", binding)
        (root / "export.json").chmod(0o600)
    rows = features.training_prepared_files(preflight)
    processor_root = features._within(root, "processor")
    processor_root.mkdir(mode=0o700, exist_ok=True)
    for row in rows:
        source = Path(prepared["snapshot_path"]) / row["path"]
        target = features._within(root, "processor/" + row["path"])
        if not target.exists():
            with source.open("rb") as src, target.open("xb") as out:
                shutil.copyfileobj(src, out)
            target.chmod(0o600)
        if training.shared._digest(target) != row["sha256"]:
            raise CognitiveKernelContractError("staged processor differs from publisher")
    start = perf_counter()
    base = AutoModelForMultimodalLM.from_pretrained(
        prepared["snapshot_path"], trust_remote_code=False, local_files_only=True,
        use_safetensors=True, dtype=torch.bfloat16,
        device_map={"": "cpu"}, low_cpu_mem_usage=True)
    if getattr(base.config, "model_type", None) != "gemma4_unified" or not hasattr(base, "model"):
        raise CognitiveKernelContractError("feature export lacks Gemma representation path")
    base.requires_grad_(False).eval()
    if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 or p.requires_grad
           for p in base.parameters()):
        raise CognitiveKernelContractError("feature export changed base dtype, placement or gradients")
    print(json.dumps({"event": "base_loaded", "seconds": perf_counter() - start,
                      "dtype": "bfloat16", "device": "cpu", "language_head_used": False}), flush=True)
    examples = (*train, *development)
    for index, example in enumerate(examples):
        case_start = perf_counter()
        key = features.case_key(example)
        if (root / "cases" / key).exists():
            features.read_case(root, example, export_record["record_sha256"],
                               width=base.config.text_config.hidden_size)
            event = "verified_resume_case"
        else:
            encoded = training.source_batch_v16(
                processor, example.context, example.opened_sources,
                args.max_source_tokens, case_id=example.case_id)
            with torch.no_grad():
                states = base.model(**encoded, use_cache=False, return_dict=True).last_hidden_state
            features.write_case(root, example, encoded, states, export_record["record_sha256"])
            del states, encoded
            event = "exported_case"
        print(json.dumps({"event": event, "completed": index + 1, "total": len(examples),
                          "split": example.split, "seconds": perf_counter() - case_start}), flush=True)
    bank = features.finalize(root, export_record, examples, rows)
    return {"feature_bank_sha256": bank["record_sha256"], "train_cases": len(train),
            "development_cases": len(development), "final_payloads_opened": False,
            "qualified_for_product": False, "seconds": perf_counter() - start}


def main():
    args = arguments()
    if any(v < 1 for v in (args.max_source_tokens, args.max_target_tokens,
                          args.specialist_heads, args.specialist_layers)):
        raise CognitiveKernelContractError("invalid feature export budgets")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true"})
    print(json.dumps(export(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
