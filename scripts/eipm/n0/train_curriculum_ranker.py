#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
from alice_personality.n0.ranker import build_ranker_from_mlm_checkpoint, listwise_preference_loss


def evaluate(model, dataloader, accelerator, competencies: list[str]) -> dict[str, object]:
    model.eval()
    competency_index = {name: index for index, name in enumerate(competencies)}
    overall = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
    by_competency = torch.zeros(
        (len(competencies), 4), device=accelerator.device, dtype=torch.float64
    )

    with torch.no_grad():
        for batch in dataloader:
            scores = model(batch["input_ids"], batch["attention_mask"])
            offset = 0
            for size, preferred, competency in zip(
                batch["group_sizes"],
                batch["preferred_masks"],
                batch["competencies"],
            ):
                group_scores = scores[offset : offset + size]
                offset += size
                result = score_group(group_scores, preferred)
                values = torch.tensor(
                    [
                        float(result["top_supported"]),
                        1.0,
                        float(result["supported_set_separated"]),
                        float(result["separation_margin"]),
                    ],
                    device=accelerator.device,
                    dtype=torch.float64,
                )
                overall += values
                by_competency[competency_index[competency]] += values

    overall = accelerator.reduce(overall, reduction="sum")
    by_competency = accelerator.reduce(by_competency, reduction="sum")

    examples = max(int(overall[1].item()), 1)
    result: dict[str, object] = {
        "top1_accuracy": float(overall[0].item() / examples),
        "supported_set_separation_rate": float(overall[2].item() / examples),
        "mean_separation_margin": float(overall[3].item() / examples),
        "examples": int(overall[1].item()),
        "per_competency": {},
    }

    per_competency: dict[str, dict[str, float | int]] = {}
    for name, index in competency_index.items():
        stats = by_competency[index]
        count = int(stats[1].item())
        if count == 0:
            continue
        per_competency[name] = {
            "top1_accuracy": float(stats[0].item() / count),
            "supported_set_separation_rate": float(stats[2].item() / count),
            "mean_separation_margin": float(stats[3].item() / count),
            "examples": count,
        }
    result["per_competency"] = per_competency
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the N0 universal semantic candidate ranker.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--mlm-checkpoint", required=True, help="Directory containing the N0 MLM model")
    parser.add_argument("--curriculum", action="append", required=True)
    parser.add_argument("--curriculum-manifest", action="append", required=True)
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
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before curriculum training") from exc

    if len(args.curriculum) != len(args.curriculum_manifest):
        raise SystemExit("repeat --curriculum and --curriculum-manifest the same number of times")

    curriculum_entries: list[dict[str, object]] = []
    for curriculum_path, manifest_path in zip(args.curriculum, args.curriculum_manifest):
        manifest = validate_curriculum_manifest(curriculum_path, manifest_path)
        summary = validate_curriculum_rows(curriculum_path)
        curriculum_entries.append(
            {
                "path": curriculum_path,
                "sha256": sha256_file(Path(curriculum_path)),
                "manifest_path": manifest_path,
                "manifest_sha256": sha256_file(Path(manifest_path)),
                "origin_type": manifest["origin_type"],
                "actor": manifest.get("actor"),
                "summary": summary,
            }
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_n0_config(args.config)
    accelerator = Accelerator(
        mixed_precision=(
            "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
        )
        if torch.cuda.is_available()
        else "no"
    )

    tokenizer = load_tokenizer(args.tokenizer_dir)
    if len(tokenizer) != config.vocab_size:
        raise SystemExit(f"tokenizer size {len(tokenizer)} != config vocab {config.vocab_size}")

    model = build_ranker_from_mlm_checkpoint(args.mlm_checkpoint, hidden_size=config.hidden_size)

    train_set = CurriculumDataset(args.curriculum, "train")
    dev_set = CurriculumDataset(args.curriculum, "dev")
    competencies = sorted({str(row["competency"]) for row in dev_set.rows})
    collator = CurriculumCollator(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

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
    best_separation = -1.0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch["input_ids"], batch["attention_mask"])
            loss = listwise_preference_loss(
                scores,
                batch["group_sizes"],
                batch["preferred_masks"],
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

        metrics = evaluate(model, dev_loader, accelerator, competencies)
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": running_loss / max(batches, 1),
                        **metrics,
                    },
                    sort_keys=True,
                )
            )

        accuracy = float(metrics["top1_accuracy"])
        separation = float(metrics["supported_set_separation_rate"])
        improved = accuracy > best_accuracy or (
            accuracy == best_accuracy and separation > best_separation
        )
        if improved:
            best_accuracy = accuracy
            best_separation = separation
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                state = {
                    key: value.detach().cpu().contiguous()
                    for key, value in unwrapped.state_dict().items()
                }
                save_file(state, str(output / "ranker.safetensors"))
                (output / "best_dev_metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                receipt = {
                    "model_id": config.model_id,
                    "stage": "N0_targeted_curriculum_ranker",
                    "best_dev_top1_accuracy": best_accuracy,
                    "best_dev_supported_set_separation_rate": best_separation,
                    "best_dev_metrics_file": "best_dev_metrics.json",
                    "curricula": curriculum_entries,
                    "config_sha256": sha256_file(Path(args.config)),
                    "seed": args.seed,
                    "epochs_completed": epoch,
                    "mlm_checkpoint": args.mlm_checkpoint,
                    "private_identity_gradient": False,
                }
                (output / "receipt.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
