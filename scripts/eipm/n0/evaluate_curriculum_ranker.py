#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum import (
    sha256_file,
    validate_curriculum_manifest,
    validate_curriculum_rows,
)
from alice_personality.n0.curriculum_data import (
    CurriculumCollator,
    CurriculumDataset,
    load_tokenizer,
    score_group,
)
from alice_personality.n0.ranker import build_ranker_from_mlm_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an N0 curriculum ranker and emit failure-driven repair inputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--mlm-checkpoint", required=True)
    parser.add_argument("--ranker", required=True, help="ranker.safetensors produced by training")
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before curriculum evaluation") from exc

    manifest = validate_curriculum_manifest(args.curriculum, args.curriculum_manifest)
    curriculum_summary = validate_curriculum_rows(args.curriculum)
    config = load_n0_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    if len(tokenizer) != config.vocab_size:
        raise SystemExit(f"tokenizer size {len(tokenizer)} != config vocab {config.vocab_size}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        device = torch.device(args.device)

    model = build_ranker_from_mlm_checkpoint(
        args.mlm_checkpoint,
        hidden_size=config.hidden_size,
    )
    state = load_file(args.ranker, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"ranker state mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    model.to(device)
    model.eval()

    dataset = CurriculumDataset(args.curriculum, args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CurriculumCollator(tokenizer, args.max_length),
    )

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    competency_totals: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for batch in dataloader:
            scores = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            offset = 0
            for row_id, competency, size, preferred in zip(
                batch["ids"],
                batch["competencies"],
                batch["group_sizes"],
                batch["preferred_masks"],
            ):
                group_scores = scores[offset : offset + size]
                offset += size
                result = score_group(group_scores, preferred)
                preferred_indices = [
                    index for index, is_preferred in enumerate(preferred.tolist())
                    if is_preferred
                ]
                record = {
                    "id": row_id,
                    "competency": competency,
                    "preferred_indices": preferred_indices,
                    "scores": [float(value) for value in group_scores.detach().cpu().tolist()],
                    **result,
                }
                records.append(record)

                bucket = competency_totals.setdefault(
                    competency,
                    {"correct": 0.0, "separated": 0.0, "margin": 0.0, "count": 0.0},
                )
                bucket["correct"] += float(result["top_supported"])
                bucket["separated"] += float(result["supported_set_separated"])
                bucket["margin"] += float(result["separation_margin"])
                bucket["count"] += 1.0

                if not result["top_supported"] or not result["supported_set_separated"]:
                    failures.append(record)

    total = max(len(records), 1)
    summary = {
        "split": args.split,
        "examples": len(records),
        "top1_accuracy": sum(float(row["top_supported"]) for row in records) / total,
        "supported_set_separation_rate": (
            sum(float(row["supported_set_separated"]) for row in records) / total
        ),
        "mean_separation_margin": (
            sum(float(row["separation_margin"]) for row in records) / total
        ),
        "failures": len(failures),
        "per_competency": {},
    }
    summary["per_competency"] = {
        competency: {
            "examples": int(stats["count"]),
            "top1_accuracy": stats["correct"] / stats["count"],
            "supported_set_separation_rate": stats["separated"] / stats["count"],
            "mean_separation_margin": stats["margin"] / stats["count"],
        }
        for competency, stats in sorted(competency_totals.items())
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.split}_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / f"{args.split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (output / f"{args.split}_failures.jsonl").open("w", encoding="utf-8") as handle:
        for record in failures:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    receipt = {
        "stage": "N0_curriculum_failure_driven_evaluation",
        "model_id": config.model_id,
        "split": args.split,
        "config_sha256": sha256_file(Path(args.config)),
        "curriculum_sha256": sha256_file(Path(args.curriculum)),
        "curriculum_manifest_sha256": sha256_file(Path(args.curriculum_manifest)),
        "curriculum_origin_type": manifest["origin_type"],
        "curriculum_summary": curriculum_summary,
        "mlm_checkpoint": args.mlm_checkpoint,
        "ranker_sha256": sha256_file(Path(args.ranker)),
        "metrics": summary,
        "private_identity_gradient": False,
    }
    (output / f"{args.split}_evaluation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
