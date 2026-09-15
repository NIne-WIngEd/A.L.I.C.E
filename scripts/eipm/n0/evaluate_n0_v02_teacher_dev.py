#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import (
    CurriculumCollator,
    CurriculumDataset,
    load_tokenizer,
    score_group,
)
from alice_personality.n0.ranker import build_ranker_from_mlm_checkpoint
from alice_personality.n0.v02_training import verify_teacher_registry


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


class RowList(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def rotated_dev_rows(
    rows: list[dict[str, Any]],
    order_variants: int,
) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for row in rows:
        candidates = [str(value) for value in row["candidates"]]
        preferred = [int(value) for value in row["preferred_indices"]]
        variants = min(max(order_variants, 1), len(candidates))
        for shift in range(variants):
            order = list(range(len(candidates)))[shift:] + list(range(len(candidates)))[:shift]
            inverse = {old: new for new, old in enumerate(order)}
            compiled.append(
                {
                    **row,
                    "id": f"{row['id']}.order{shift}",
                    "base_id": str(row["id"]),
                    "candidate_order_variant": shift,
                    "candidates": [candidates[index] for index in order],
                    "preferred_indices": sorted(inverse[index] for index in preferred),
                }
            )
    return compiled


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "examples": 0,
            "top1_accuracy": 0.0,
            "separation_rate": 0.0,
            "mean_margin": 0.0,
            "median_margin": 0.0,
            "margin_p10": 0.0,
            "margin_p90": 0.0,
            "mean_abs_score": 0.0,
            "max_abs_score": 0.0,
        }
    margins = [float(row["separation_margin"]) for row in rows]
    abs_scores = [abs(float(score)) for row in rows for score in row["scores"]]
    return {
        "examples": len(rows),
        "top1_accuracy": sum(float(row["top_supported"]) for row in rows) / len(rows),
        "separation_rate": sum(float(row["supported_set_separated"]) for row in rows) / len(rows),
        "mean_margin": statistics.fmean(margins),
        "median_margin": statistics.median(margins),
        "margin_p10": percentile(margins, 0.10),
        "margin_p90": percentile(margins, 0.90),
        "mean_abs_score": statistics.fmean(abs_scores) if abs_scores else 0.0,
        "max_abs_score": max(abs_scores) if abs_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an N0 v0.2 ranker on the held-out 255-row governed teacher-dev bank "
            "with deterministic candidate-order rotations."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--mlm-checkpoint", required=True)
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--teacher-registry", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--order-variants", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before teacher-dev evaluation") from exc

    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    ranker_path = Path(args.ranker).resolve()
    output_dir = Path(args.output_dir).resolve()
    registry_path = Path(args.teacher_registry).resolve()
    audit_path = Path(args.teacher_audit).resolve()

    curriculum_paths, teacher_report = verify_teacher_registry(
        repo_root,
        registry_path,
        audit_path,
    )
    dev = CurriculumDataset(curriculum_paths, "dev")
    counts = Counter(str(row["competency"]) for row in dev.rows)
    if len(dev) != 255:
        raise SystemExit(f"expected 255 held-out teacher dev rows, observed {len(dev)}")
    if len(counts) != 51 or any(count != 5 for count in counts.values()):
        raise SystemExit(
            "teacher dev bank must contain exactly five rows for each of 51 competencies; "
            f"observed={dict(sorted(counts.items()))}"
        )

    compiled_rows = rotated_dev_rows(dev.rows, args.order_variants)
    dataset = RowList(compiled_rows)
    tokenizer = load_tokenizer(tokenizer_dir)
    config = load_n0_config(config_path)
    if len(tokenizer) != config.vocab_size:
        raise SystemExit("tokenizer/config vocabulary mismatch")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_ranker_from_mlm_checkpoint(
        args.mlm_checkpoint,
        hidden_size=config.hidden_size,
    )
    state = load_file(str(ranker_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"ranker state mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CurriculumCollator(tokenizer, args.max_length),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    row_by_id = {str(row["id"]): row for row in compiled_rows}

    records: list[dict[str, Any]] = []
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_competency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
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
                source = row_by_id[str(row_id)]
                record = {
                    "id": str(row_id),
                    "base_id": str(source["base_id"]),
                    "competency": str(competency),
                    "candidate_order_variant": int(source["candidate_order_variant"]),
                    "scores": [float(value) for value in group_scores.detach().cpu().tolist()],
                    **result,
                }
                records.append(record)
                by_base[record["base_id"]].append(record)
                by_competency[record["competency"]].append(record)
                by_order[record["candidate_order_variant"]].append(record)

    base_pass = {
        base_id: all(
            bool(row["top_supported"]) and bool(row["supported_set_separated"])
            for row in bucket
        )
        for base_id, bucket in by_base.items()
    }
    competency_base_pass: dict[str, list[bool]] = defaultdict(list)
    base_competency = {str(row["id"]): str(row["competency"]) for row in dev.rows}
    for base_id, passed in base_pass.items():
        competency_base_pass[base_competency[base_id]].append(passed)

    metrics = {
        **summarize(records),
        "base_rows": len(dev),
        "compiled_rows": len(records),
        "competency_count": len(counts),
        "full_order_invariance_pass_rate": sum(base_pass.values()) / len(base_pass),
        "full_order_invariance_failures": sorted(
            base_id for base_id, passed in base_pass.items() if not passed
        ),
        "per_competency": {
            name: {
                **summarize(bucket),
                "base_rows": len(competency_base_pass[name]),
                "full_order_invariance_pass_rate": (
                    sum(competency_base_pass[name]) / len(competency_base_pass[name])
                ),
            }
            for name, bucket in sorted(by_competency.items())
        },
        "per_candidate_order_variant": {
            str(order): summarize(bucket) for order, bucket in sorted(by_order.items())
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teacher_dev_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "teacher_dev_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    receipt = {
        "schema": "alice.eipm.n0.v02-teacher-dev-evaluation.v0.1",
        "status": "PASS",
        "model_id": config.model_id,
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "ranker_sha256": sha256_file(ranker_path),
        "mlm_checkpoint": str(Path(args.mlm_checkpoint).resolve()),
        "teacher_registry_sha256": sha256_file(registry_path),
        "teacher_audit_sha256": sha256_file(audit_path),
        "teacher_registered_rows": int(teacher_report["registered_rows"]),
        "teacher_dev_rows": len(dev),
        "teacher_dev_competencies": len(counts),
        "candidate_order_variants": args.order_variants,
        "eval_only": True,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "metrics": metrics,
    }
    (output_dir / "teacher_dev_evaluation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
