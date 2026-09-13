#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.model import build_masked_lm, count_parameters


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast N0 model/runtime preflight before spending GPU time.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer.json to verify vocabulary size")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backward", action="store_true", help="Also verify one backward pass")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_n0_config(config_path)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    tokenizer_size = None
    if args.tokenizer:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit("tokenizers is required to verify --tokenizer") from exc
        tokenizer = Tokenizer.from_file(args.tokenizer)
        tokenizer_size = tokenizer.get_vocab_size()
        if tokenizer_size != config.vocab_size:
            raise SystemExit(f"tokenizer vocab {tokenizer_size} != config vocab {config.vocab_size}")

    model = build_masked_lm(config).to(device)
    total, trainable = count_parameters(model)
    if not 300_000_000 <= total <= 400_000_000:
        raise SystemExit(f"unexpected parameter count: {total}")
    if trainable != total:
        raise SystemExit(f"unexpected frozen parameters: trainable={trainable} total={total}")

    input_ids = torch.randint(5, config.vocab_size, (1, 16), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, 3:6] = input_ids[0, 3:6]
    input_ids[0, 3:6] = config.mask_token_id

    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = float(outputs.loss.detach().cpu())
    if not math.isfinite(loss):
        raise SystemExit(f"non-finite preflight loss: {loss}")
    if args.backward:
        outputs.loss.backward()
        if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
            raise SystemExit("backward pass produced no gradients")

    receipt = {
        "status": "PASS",
        "model_id": config.model_id,
        "config_sha256": sha256_file(config_path),
        "tokenizer_verified": bool(args.tokenizer),
        "tokenizer_vocab_size": tokenizer_size,
        "parameters_total": total,
        "parameters_trainable": trainable,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "forward_loss": loss,
        "backward_verified": args.backward,
        "private_identity_gradient": False,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
