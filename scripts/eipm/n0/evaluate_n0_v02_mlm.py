#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
from alice_personality.n0.v02_training import (
    sha256_file,
    verify_public_corpus_v021,
    verify_tokenizer_v021,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate N0 v0.2 MLM on the untouched public dev split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-receipt", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--mask-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        from transformers import ModernBertForMaskedLM
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before N0 v0.2 MLM evaluation") from exc

    config_path = Path(args.config).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_receipt_path = Path(args.model_receipt).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    source_config = Path(args.source_config).resolve()
    output = Path(args.output).resolve()

    config = load_n0_config(config_path)
    if args.sequence_length not in config.train_sequence_lengths:
        raise SystemExit("evaluation sequence length is not allowed by N0 config")
    if args.batch_size < 1 or args.max_batches < 1 or args.mask_repeats < 1:
        raise SystemExit("batch size, max batches, and mask repeats must be positive")

    tokenizer_receipt = verify_tokenizer_v021(tokenizer_dir)
    corpus_receipt, shard_paths = verify_public_corpus_v021(corpus_dir, source_config)
    model_receipt = json.loads(model_receipt_path.read_text(encoding="utf-8"))
    if model_receipt.get("model_id") != config.model_id:
        raise SystemExit("model receipt does not match active v0.2 model id")
    if model_receipt.get("config_sha256") != sha256_file(config_path):
        raise SystemExit("model receipt config hash does not match active config")
    if model_receipt.get("private_identity_gradient") is not False:
        raise SystemExit("N0 v0.2 evaluation refuses a private-gradient checkpoint")
    if model_receipt.get("corpus_receipt_sha256") not in (
        None,
        sha256_file(corpus_dir / "corpus_receipt.json"),
    ):
        raise SystemExit("checkpoint is bound to a different public corpus")

    tokenizer = load_tokenizer(tokenizer_dir)
    if len(tokenizer) != config.vocab_size:
        raise SystemExit("tokenizer/config vocabulary mismatch")
    if tokenizer_receipt.get("tokenizer_sha256") != sha256_file(
        tokenizer_dir / "tokenizer.json"
    ):
        raise SystemExit("tokenizer receipt binding failed")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModernBertForMaskedLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    weighted_loss_sum = 0.0
    masked_tokens_total = 0
    reference_visible_tokens: int | None = None
    per_repeat: list[dict[str, int | float]] = []

    with torch.inference_mode():
        for repeat in range(args.mask_repeats):
            dataset = PackedJSONLIterableDataset(
                paths=shard_paths,
                tokenizer=tokenizer,
                sequence_length=args.sequence_length,
                split="dev",
            )
            collator = SpanMLMCollator(
                tokenizer=tokenizer,
                mlm_probability=config.mlm_probability,
                mean_span=config.mean_mask_span,
                max_span=config.max_mask_span,
                seed=args.seed + repeat * 1_000_003,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
            repeat_loss = 0.0
            repeat_masked = 0
            repeat_visible = 0
            repeat_batches = 0
            for batch in loader:
                if repeat_batches >= args.max_batches:
                    break
                batch_masked = int((batch["labels"] != -100).sum().item())
                if batch_masked == 0:
                    continue
                repeat_visible += int(batch["attention_mask"].sum().item())
                batch = {key: value.to(device) for key, value in batch.items()}
                result = model(**batch)
                repeat_loss += float(result.loss.detach().cpu()) * batch_masked
                repeat_masked += batch_masked
                repeat_batches += 1

            if repeat_batches == 0 or repeat_masked == 0:
                raise RuntimeError("public dev split produced no evaluable masked tokens")
            if reference_visible_tokens is None:
                reference_visible_tokens = repeat_visible
            elif reference_visible_tokens != repeat_visible:
                raise RuntimeError("dev document exposure changed across mask repeats")

            nll = repeat_loss / repeat_masked
            per_repeat.append(
                {
                    "repeat": repeat,
                    "batches": repeat_batches,
                    "visible_tokens": repeat_visible,
                    "masked_tokens": repeat_masked,
                    "mean_masked_token_nll": nll,
                    "masked_token_perplexity": math.exp(nll) if nll < 80 else float("inf"),
                }
            )
            weighted_loss_sum += repeat_loss
            masked_tokens_total += repeat_masked

    mean_nll = weighted_loss_sum / masked_tokens_total
    repeat_nlls = [float(row["mean_masked_token_nll"]) for row in per_repeat]
    repeat_mean = sum(repeat_nlls) / len(repeat_nlls)
    variance = sum((value - repeat_mean) ** 2 for value in repeat_nlls) / len(repeat_nlls)
    result = {
        "schema": "alice.eipm.n0.v02-mlm-evaluation.v0.1",
        "status": "PASS",
        "model_id": config.model_id,
        "model_receipt_sha256": sha256_file(model_receipt_path),
        "corpus_receipt_sha256": sha256_file(corpus_dir / "corpus_receipt.json"),
        "corpus_source_count": len(corpus_receipt.get("sources", [])),
        "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "split": "dev",
        "sequence_length": args.sequence_length,
        "mask_repeats": args.mask_repeats,
        "reference_visible_tokens": reference_visible_tokens,
        "masked_tokens_total": masked_tokens_total,
        "mean_masked_token_nll": mean_nll,
        "masked_token_perplexity": math.exp(mean_nll) if mean_nll < 80 else float("inf"),
        "repeat_mean_nll": repeat_mean,
        "repeat_nll_stddev": math.sqrt(variance),
        "per_repeat": per_repeat,
        "device": str(device),
        "eval_only": True,
        "private_identity_data": False,
        "private_identity_gradient": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
