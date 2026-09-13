#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.corpus_receipt import verify_corpus_receipt
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
from alice_personality.n0.model import build_masked_lm, count_parameters


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_tokenizer_receipt(tokenizer_path: Path, receipt_path: Path) -> dict:
    if not receipt_path.is_file():
        raise SystemExit(f"tokenizer receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("private_identity_data") is not False:
        raise SystemExit("tokenizer receipt must declare private_identity_data=false")
    actual = sha256_file(tokenizer_path)
    if receipt.get("tokenizer_sha256") != actual:
        raise SystemExit("tokenizer SHA256 does not match tokenizer receipt")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train alice-n0-semantic-v0.1 from random initialization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--tokenizer-receipt", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
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
    parser.add_argument("--mixed-precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Existing step directory containing accelerator_state/ and receipt.json",
    )
    args = parser.parse_args()

    try:
        from accelerate import Accelerator
        from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before N0 training") from exc

    config_path = Path(args.config)
    tokenizer_path = Path(args.tokenizer)
    tokenizer_receipt_path = Path(args.tokenizer_receipt)
    corpus_dir = Path(args.corpus_dir)
    source_config_path = Path(args.source_config)
    config = load_n0_config(config_path)
    if args.sequence_length not in config.train_sequence_lengths:
        raise SystemExit(f"sequence length must be one of {config.train_sequence_lengths}")
    if args.max_steps < 1 or args.save_every < 1 or args.grad_accum < 1:
        raise SystemExit("max-steps, save-every, and grad-accum must be positive")

    corpus_verification = verify_corpus_receipt(corpus_dir, source_config_path)
    train_paths = [Path(path).resolve() for path in args.train]
    manifested_paths = {Path(path).resolve() for path in corpus_verification["shard_paths"]}
    if set(train_paths) != manifested_paths:
        missing = sorted(str(path) for path in manifested_paths - set(train_paths))
        extra = sorted(str(path) for path in set(train_paths) - manifested_paths)
        raise SystemExit(
            f"training shard set does not match verified corpus receipt: missing={missing[:5]} extra={extra[:5]}"
        )

    tokenizer_receipt = verify_tokenizer_receipt(tokenizer_path, tokenizer_receipt_path)

    seed_everything(args.seed)
    if args.mixed_precision == "auto":
        if not torch.cuda.is_available():
            mixed_precision = "no"
        elif torch.cuda.is_bf16_supported():
            mixed_precision = "bf16"
        else:
            mixed_precision = "fp16"
    else:
        mixed_precision = args.mixed_precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=mixed_precision,
    )

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
    if not args.no_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    total, trainable = count_parameters(model)
    if accelerator.is_main_process:
        print(json.dumps({"parameters_total": total, "parameters_trainable": trainable}))

    dataset = PackedJSONLIterableDataset(
        paths=train_paths,
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
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    previous_global_tokens = 0
    resume_parent = None
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        receipt_path = resume_dir / "receipt.json"
        state_dir = resume_dir / "accelerator_state"
        if not receipt_path.is_file() or not state_dir.is_dir():
            raise SystemExit("resume-from must contain receipt.json and accelerator_state/")
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("model_id") != config.model_id:
            raise SystemExit("resume checkpoint model_id does not match active config")
        if previous.get("config_sha256") != sha256_file(config_path):
            raise SystemExit("resume checkpoint config hash does not match active config")
        if previous.get("tokenizer_sha256") != sha256_file(tokenizer_path):
            raise SystemExit("resume checkpoint tokenizer hash does not match active tokenizer")
        step = int(previous["step"])
        previous_global_tokens = int(
            previous.get("tokens_seen_total", previous.get("tokens_seen_since_launch", 0))
        )
        resume_parent = str(resume_dir)
        accelerator.load_state(str(state_dir))
        if accelerator.is_main_process:
            print(json.dumps({"resume_from": resume_parent, "resume_step": step}))

    if step >= args.max_steps:
        if accelerator.is_main_process:
            print(json.dumps({"status": "already_complete", "step": step, "max_steps": args.max_steps}))
        return

    model.train()
    loss_sum = 0.0
    loss_microbatches = 0
    local_tokens_seen = 0
    code_revision = git_revision()

    while step < args.max_steps:
        saw_batch = False
        for batch in dataloader:
            saw_batch = True
            local_tokens_seen += int(batch["attention_mask"].sum().item())
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += float(loss.detach().cpu())
            loss_microbatches += 1
            if accelerator.sync_gradients:
                step += 1
                if accelerator.is_main_process and step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "loss": loss_sum / max(loss_microbatches, 1),
                                "lr": scheduler.get_last_lr()[0],
                            }
                        )
                    )
                    loss_sum = 0.0
                    loss_microbatches = 0

                if step % args.save_every == 0 or step == args.max_steps:
                    accelerator.wait_for_everyone()
                    checkpoint_dir = output_dir / f"step-{step:08d}"
                    accelerator.save_state(str(checkpoint_dir / "accelerator_state"))
                    token_tensor = torch.tensor(
                        local_tokens_seen, device=accelerator.device, dtype=torch.long
                    )
                    global_tokens_this_launch = int(
                        accelerator.reduce(token_tensor, reduction="sum").item()
                    )
                    global_tokens_total = previous_global_tokens + global_tokens_this_launch
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        model_dir = checkpoint_dir / "model"
                        tokenizer_dir = checkpoint_dir / "tokenizer"
                        unwrapped.save_pretrained(model_dir, safe_serialization=True)
                        tokenizer.save_pretrained(tokenizer_dir)
                        receipt = {
                            "schema": "alice.eipm.n0.mlm-checkpoint-receipt.v0.2",
                            "model_id": config.model_id,
                            "step": step,
                            "sequence_length": args.sequence_length,
                            "tokens_seen_this_launch": global_tokens_this_launch,
                            "tokens_seen_total": global_tokens_total,
                            "seed": args.seed,
                            "config_sha256": sha256_file(config_path),
                            "tokenizer_sha256": sha256_file(tokenizer_path),
                            "tokenizer_receipt_sha256": sha256_file(tokenizer_receipt_path),
                            "tokenizer_origin_git_revision": tokenizer_receipt.get("git_revision"),
                            "corpus_receipt_sha256": corpus_verification["receipt_sha256"],
                            "source_config_sha256": corpus_verification["source_config_sha256"],
                            "corpus_verified_shards": corpus_verification["verified_shards"],
                            "corpus_verified_bytes": corpus_verification["verified_bytes"],
                            "train_inputs": [str(path) for path in train_paths],
                            "parameters_total": total,
                            "parameters_trainable": trainable,
                            "torch_version": torch.__version__,
                            "cuda_available": torch.cuda.is_available(),
                            "cuda_devices": [
                                torch.cuda.get_device_name(index)
                                for index in range(torch.cuda.device_count())
                            ],
                            "world_size": accelerator.num_processes,
                            "mixed_precision": mixed_precision,
                            "gradient_checkpointing": not args.no_gradient_checkpointing,
                            "git_revision": code_revision,
                            "resume_parent": resume_parent,
                            "model_artifact_sha256": directory_hashes(model_dir),
                            "tokenizer_artifact_sha256": directory_hashes(tokenizer_dir),
                            "private_identity_gradient": False,
                        }
                        (checkpoint_dir / "receipt.json").write_text(
                            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    accelerator.wait_for_everyone()

                if step >= args.max_steps:
                    break
        if not saw_batch:
            raise RuntimeError("training dataset produced zero batches")


if __name__ == "__main__":
    main()
