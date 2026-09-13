#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
from alice_personality.n0.model import build_masked_lm, count_parameters


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train alice-n0-semantic-v0.1 from random initialization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--train", action="append", required=True, help="Normalized JSONL; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    try:
        from accelerate import Accelerator
        from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before N0 training") from exc

    config_path = Path(args.config)
    tokenizer_path = Path(args.tokenizer)
    config = load_n0_config(config_path)
    if args.sequence_length not in config.train_sequence_lengths:
        raise SystemExit(f"sequence length must be one of {config.train_sequence_lengths}")

    seed_everything(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")

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

    model = build_masked_lm(config)
    total, trainable = count_parameters(model)
    if accelerator.is_main_process:
        print(json.dumps({"parameters_total": total, "parameters_trainable": trainable}))

    dataset = PackedJSONLIterableDataset(
        paths=args.train,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        split="train",
    )
    collator = SpanMLMCollator(
        tokenizer=tokenizer,
        mlm_probability=config.mlm_probability,
        mean_span=config.mean_mask_span,
        max_span=config.max_mask_span,
        seed=args.seed + accelerator.process_index,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    running_loss = 0.0

    while step < args.max_steps:
        saw_batch = False
        for batch in dataloader:
            saw_batch = True
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach())
            if accelerator.sync_gradients:
                step += 1
                if accelerator.is_main_process and step % 10 == 0:
                    print(json.dumps({"step": step, "loss": running_loss / 10.0, "lr": scheduler.get_last_lr()[0]}))
                    running_loss = 0.0

                if step % args.save_every == 0 or step == args.max_steps:
                    accelerator.wait_for_everyone()
                    checkpoint_dir = output_dir / f"step-{step:08d}"
                    accelerator.save_state(str(checkpoint_dir / "accelerator_state"))
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(checkpoint_dir / "model", safe_serialization=True)
                        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
                        receipt = {
                            "model_id": config.model_id,
                            "step": step,
                            "sequence_length": args.sequence_length,
                            "seed": args.seed,
                            "config_sha256": sha256_file(config_path),
                            "tokenizer_sha256": sha256_file(tokenizer_path),
                            "train_inputs": [str(Path(path)) for path in args.train],
                            "parameters_total": total,
                            "parameters_trainable": trainable,
                            "torch_version": torch.__version__,
                            "cuda_available": torch.cuda.is_available(),
                            "world_size": accelerator.num_processes
                        }
                        (checkpoint_dir / "receipt.json").write_text(
                            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                        )
                    accelerator.wait_for_everyone()

                if step >= args.max_steps:
                    break
        if not saw_batch:
            raise RuntimeError("training dataset produced zero batches")


if __name__ == "__main__":
    main()
