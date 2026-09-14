#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import CurriculumCollator, load_tokenizer, score_group
from alice_personality.n0.ranker import build_ranker_from_mlm_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FixedRows(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError("compiled fixed benchmark is empty")
        for row in self.rows:
            if row.get("eval_only") is not True or row.get("training_authorized") is not False:
                raise ValueError(f"benchmark row is not eval-only: {row.get('id')}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen N0 ranker on the v0.2 fixed suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--mlm-checkpoint", required=True)
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before fixed-suite evaluation") from exc

    benchmark = Path(args.benchmark)
    benchmark_receipt = json.loads(Path(args.benchmark_receipt).read_text(encoding="utf-8"))
    if benchmark_receipt.get("training_authorized") is not False:
        raise SystemExit("fixed benchmark receipt must explicitly forbid training")
    if benchmark_receipt.get("compiled_sha256") != sha256_file(benchmark):
        raise SystemExit("compiled fixed benchmark hash does not match receipt")

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

    model = build_ranker_from_mlm_checkpoint(args.mlm_checkpoint, hidden_size=config.hidden_size)
    state = load_file(args.ranker, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"ranker state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    model.to(device)
    model.eval()

    dataset = FixedRows(benchmark)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CurriculumCollator(tokenizer, args.max_length),
    )
    row_by_id = {str(row["id"]): row for row in dataset.rows}

    records: list[dict[str, object]] = []
    base_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    competency_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    order_buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
    paraphrase_buckets: dict[int, list[dict[str, object]]] = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            scores = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            offset = 0
            for row_id, competency, size, preferred in zip(
                batch["ids"], batch["competencies"], batch["group_sizes"], batch["preferred_masks"]
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
                    "paraphrase_variant": int(source["paraphrase_variant"]),
                    "scores": [float(x) for x in group_scores.detach().cpu().tolist()],
                    **result,
                }
                records.append(record)
                base_buckets[record["base_id"]].append(record)
                competency_buckets[record["competency"]].append(record)
                order_buckets[record["candidate_order_variant"]].append(record)
                paraphrase_buckets[record["paraphrase_variant"]].append(record)

    def summarize(bucket: list[dict[str, object]]) -> dict[str, float | int]:
        n = max(len(bucket), 1)
        return {
            "examples": len(bucket),
            "top1_accuracy": sum(float(row["top_supported"]) for row in bucket) / n,
            "separation_rate": sum(float(row["supported_set_separated"]) for row in bucket) / n,
            "mean_margin": sum(float(row["separation_margin"]) for row in bucket) / n,
        }

    base_pass = {
        base_id: all(bool(row["top_supported"]) and bool(row["supported_set_separated"]) for row in bucket)
        for base_id, bucket in base_buckets.items()
    }
    summary = {
        **summarize(records),
        "base_cases": len(base_buckets),
        "full_invariance_pass_rate": sum(base_pass.values()) / max(len(base_pass), 1),
        "full_invariance_failures": sorted(base_id for base_id, passed in base_pass.items() if not passed),
        "per_competency": {name: summarize(bucket) for name, bucket in sorted(competency_buckets.items())},
        "per_candidate_order_variant": {
            str(name): summarize(bucket) for name, bucket in sorted(order_buckets.items())
        },
        "per_paraphrase_variant": {
            str(name): summarize(bucket) for name, bucket in sorted(paraphrase_buckets.items())
        },
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fixed_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "fixed_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    receipt = {
        "schema": "alice.eipm.n0.fixed-readiness-evaluation.v0.2",
        "model_id": config.model_id,
        "config_sha256": sha256_file(Path(args.config)),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_receipt_sha256": sha256_file(Path(args.benchmark_receipt)),
        "ranker_sha256": sha256_file(Path(args.ranker)),
        "mlm_checkpoint": args.mlm_checkpoint,
        "metrics": summary,
        "eval_only": True,
        "private_identity_gradient": False,
    }
    (output / "fixed_evaluation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
