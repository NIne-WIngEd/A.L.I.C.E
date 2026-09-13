#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum import sha256_file, validate_curriculum_manifest
from alice_personality.n0.ranker import build_ranker_from_mlm_checkpoint, listwise_preference_loss


class CurriculumDataset(Dataset):
    def __init__(self, path: str | Path, split: str) -> None:
        self.rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split", "train") == split:
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"no curriculum rows found for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class CurriculumCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        prompts: list[str] = []
        candidates: list[str] = []
        group_sizes: list[int] = []
        preferred_masks: list[torch.Tensor] = []
        ids: list[str] = []

        for row in rows:
            row_candidates = list(row["candidates"])
            preferred = set(int(x) for x in row["preferred_indices"])
            if not row_candidates or not preferred:
                raise ValueError(f"invalid curriculum row {row.get('id')}")
            if max(preferred) >= len(row_candidates):
                raise ValueError(f"preferred index out of range in {row.get('id')}")

            ids.append(str(row["id"]))
            group_sizes.append(len(row_candidates))
            preferred_masks.append(
                torch.tensor([index in preferred for index in range(len(row_candidates))], dtype=torch.bool)
            )
            prompts.extend([str(row["prompt"])] * len(row_candidates))
            candidates.extend(str(candidate) for candidate in row_candidates)

        encoded = self.tokenizer(
            prompts,
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "group_sizes": group_sizes,
            "preferred_masks": preferred_masks,
            "ids": ids,
        }


def evaluate(model, dataloader, accelerator) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    tie_exact = 0
    with torch.no_grad():
        for batch in dataloader:
            scores = model(batch["input_ids"], batch["attention_mask"])
            offset = 0
            for size, preferred in zip(batch["group_sizes"], batch["preferred_masks"]):
                group_scores = scores[offset : offset + size]
                offset += size
                preferred = preferred.to(group_scores.device)
                top = torch.argmax(group_scores).item()
                correct += int(bool(preferred[top]))
                total += 1
                max_score = group_scores.max()
                predicted_tie = torch.isclose(group_scores, max_score, rtol=0.0, atol=1e-6)
                tie_exact += int(torch.equal(predicted_tie, preferred))
    stats = torch.tensor([correct, total, tie_exact], device=accelerator.device, dtype=torch.float64)
    stats = accelerator.reduce(stats, reduction="sum")
    return {
        "top1_accuracy": float(stats[0] / stats[1].clamp_min(1)),
        "tie_exact_rate": float(stats[2] / stats[1].clamp_min(1)),
        "examples": int(stats[1].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the N0 universal semantic candidate ranker.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--mlm-checkpoint", required=True, help="Directory containing the N0 MLM model")
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    try:
        from accelerate import Accelerator
        from safetensors.torch import save_file
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before curriculum training") from exc

    curriculum_manifest = validate_curriculum_manifest(args.curriculum, args.curriculum_manifest)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_n0_config(args.config)
    accelerator = Accelerator(
        mixed_precision=("bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16")
        if torch.cuda.is_available()
        else "no"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    model = build_ranker_from_mlm_checkpoint(args.mlm_checkpoint, hidden_size=config.hidden_size)

    train_set = CurriculumDataset(args.curriculum, "train")
    dev_set = CurriculumDataset(args.curriculum, "dev")
    collator = CurriculumCollator(tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bias" in name.lower() or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.98),
    )

    model, optimizer, train_loader, dev_loader = accelerator.prepare(
        model, optimizer, train_loader, dev_loader
    )

    best_accuracy = -1.0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch["input_ids"], batch["attention_mask"])
            loss = listwise_preference_loss(scores, batch["group_sizes"], batch["preferred_masks"])
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach())
            batches += 1

        metrics = evaluate(model, dev_loader, accelerator)
        if accelerator.is_main_process:
            print(json.dumps({"epoch": epoch, "train_loss": running_loss / max(batches, 1), **metrics}))

        if metrics["top1_accuracy"] > best_accuracy:
            best_accuracy = metrics["top1_accuracy"]
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                state = {key: value.detach().cpu().contiguous() for key, value in unwrapped.state_dict().items()}
                save_file(state, str(output / "ranker.safetensors"))
                receipt = {
                    "model_id": config.model_id,
                    "stage": "N0_targeted_curriculum_ranker",
                    "best_dev_top1_accuracy": best_accuracy,
                    "curriculum_sha256": sha256_file(Path(args.curriculum)),
                    "curriculum_manifest_sha256": sha256_file(Path(args.curriculum_manifest)),
                    "curriculum_origin_type": curriculum_manifest["origin_type"],
                    "config_sha256": sha256_file(Path(args.config)),
                    "seed": args.seed,
                    "epochs_completed": epoch,
                    "mlm_checkpoint": args.mlm_checkpoint,
                    "private_identity_gradient": False
                }
                (output / "receipt.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
