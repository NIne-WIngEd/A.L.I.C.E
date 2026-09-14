#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.corpus_receipt import verify_corpus_receipt
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an N0 MLM checkpoint on the deterministic held-out dev split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-receipt", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=256)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument(
        "--mask-repeats",
        type=int,
        default=8,
        help=(
            "Repeat evaluation over the exact same held-out documents with deterministic "
            "independent masking seeds. This reduces masking variance without pretending "
            "that training documents are newly held out."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        from transformers import ModernBertForMaskedLM, PreTrainedTokenizerFast
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before N0 evaluation") from exc

    config_path = Path(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    model_dir = checkpoint_dir / "model"
    checkpoint_receipt_path = checkpoint_dir / "receipt.json"
    tokenizer_path = Path(args.tokenizer)
    tokenizer_receipt_path = Path(args.tokenizer_receipt)
    corpus_dir = Path(args.corpus_dir)
    source_config = Path(args.source_config)

    if not model_dir.is_dir() or not checkpoint_receipt_path.is_file():
        raise SystemExit("checkpoint-dir must contain model/ and receipt.json")
    if not tokenizer_path.is_file() or not tokenizer_receipt_path.is_file():
        raise SystemExit("tokenizer and tokenizer receipt are required")

    config = load_n0_config(config_path)
    if args.sequence_length not in config.train_sequence_lengths:
        raise SystemExit(f"sequence length must be one of {config.train_sequence_lengths}")
    if args.batch_size < 1 or args.max_batches < 1 or args.mask_repeats < 1:
        raise SystemExit("batch-size, max-batches, and mask-repeats must be positive")

    corpus_verification = verify_corpus_receipt(corpus_dir, source_config)
    checkpoint_receipt = json.loads(checkpoint_receipt_path.read_text(encoding="utf-8"))
    tokenizer_receipt = json.loads(tokenizer_receipt_path.read_text(encoding="utf-8"))

    if checkpoint_receipt.get("model_id") != config.model_id:
        raise SystemExit("checkpoint model_id does not match active N0 config")
    if checkpoint_receipt.get("config_sha256") != sha256_file(config_path):
        raise SystemExit("checkpoint config hash does not match active N0 config")
    if checkpoint_receipt.get("corpus_receipt_sha256") != corpus_verification["receipt_sha256"]:
        raise SystemExit("checkpoint corpus receipt hash does not match active corpus")
    if checkpoint_receipt.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise SystemExit("checkpoint tokenizer hash does not match active tokenizer")
    if tokenizer_receipt.get("private_identity_data") is not False:
        raise SystemExit("N0 tokenizer receipt must declare private_identity_data=false")
    if tokenizer_receipt.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise SystemExit("tokenizer receipt hash binding failed")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    if len(tokenizer) != config.vocab_size:
        raise SystemExit(f"tokenizer size {len(tokenizer)} != config vocab {config.vocab_size}")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModernBertForMaskedLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    shard_paths = [Path(path) for path in corpus_verification["shard_paths"]]

    weighted_loss_sum = 0.0
    masked_tokens_total = 0
    visible_token_exposures = 0
    batches_total = 0
    reference_visible_tokens: int | None = None
    per_repeat: list[dict[str, int | float]] = []

    with torch.inference_mode():
        for repeat_index in range(args.mask_repeats):
            repeat_seed = args.seed + repeat_index * 1_000_003
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
                seed=repeat_seed,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            repeat_weighted_loss = 0.0
            repeat_masked_tokens = 0
            repeat_visible_tokens = 0
            repeat_batches = 0

            for batch in dataloader:
                if repeat_batches >= args.max_batches:
                    break
                labels = batch["labels"]
                batch_masked = int((labels != -100).sum().item())
                if batch_masked == 0:
                    continue
                repeat_visible_tokens += int(batch["attention_mask"].sum().item())
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                repeat_weighted_loss += float(outputs.loss.detach().cpu()) * batch_masked
                repeat_masked_tokens += batch_masked
                repeat_batches += 1

            if repeat_batches == 0 or repeat_masked_tokens == 0:
                raise RuntimeError(
                    f"held-out dev split produced no evaluable masked tokens for repeat {repeat_index}"
                )

            if reference_visible_tokens is None:
                reference_visible_tokens = repeat_visible_tokens
            elif repeat_visible_tokens != reference_visible_tokens:
                raise RuntimeError(
                    "held-out document exposure changed across mask repeats; deterministic dev set violated"
                )

            repeat_nll = repeat_weighted_loss / repeat_masked_tokens
            per_repeat.append(
                {
                    "repeat": repeat_index,
                    "mask_seed": repeat_seed,
                    "batches": repeat_batches,
                    "visible_tokens": repeat_visible_tokens,
                    "masked_tokens": repeat_masked_tokens,
                    "mean_masked_token_nll": repeat_nll,
                    "masked_token_perplexity": (
                        math.exp(repeat_nll) if repeat_nll < 80 else float("inf")
                    ),
                }
            )
            weighted_loss_sum += repeat_weighted_loss
            masked_tokens_total += repeat_masked_tokens
            visible_token_exposures += repeat_visible_tokens
            batches_total += repeat_batches

    mean_nll = weighted_loss_sum / masked_tokens_total
    repeat_nlls = [float(item["mean_masked_token_nll"]) for item in per_repeat]
    repeat_mean = sum(repeat_nlls) / len(repeat_nlls)
    repeat_variance = sum((value - repeat_mean) ** 2 for value in repeat_nlls) / len(repeat_nlls)

    result = {
        "schema": "alice.eipm.n0.mlm-evaluation.v0.2",
        "status": "PASS",
        "model_id": config.model_id,
        "checkpoint_step": int(checkpoint_receipt["step"]),
        "checkpoint_receipt_sha256": sha256_file(checkpoint_receipt_path),
        "corpus_receipt_sha256": corpus_verification["receipt_sha256"],
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "split": "dev",
        "sequence_length": args.sequence_length,
        "base_mask_seed": args.seed,
        "mask_repeats": args.mask_repeats,
        "heldout_documents_reused_across_mask_repeats": True,
        "reference_visible_tokens": reference_visible_tokens,
        "visible_token_exposures": visible_token_exposures,
        "batches_total": batches_total,
        "masked_tokens_total": masked_tokens_total,
        "mean_masked_token_nll": mean_nll,
        "masked_token_perplexity": math.exp(mean_nll) if mean_nll < 80 else float("inf"),
        "repeat_mean_nll": repeat_mean,
        "repeat_nll_stddev": math.sqrt(repeat_variance),
        "per_repeat": per_repeat,
        "device": str(device),
        "private_identity_data": False,
        "private_identity_gradient": False,
    }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
