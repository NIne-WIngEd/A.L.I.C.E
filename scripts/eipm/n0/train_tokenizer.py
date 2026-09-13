#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


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


def text_iterator(paths: list[str]):
    for raw_path in paths:
        with Path(raw_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Alice N0 48k byte-complete BPE tokenizer.")
    parser.add_argument("--input", action="append", required=True, help="Normalized JSONL; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=48_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.normalizers import NFC
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before training the tokenizer") from exc

    inputs = [Path(path) for path in args.input]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"tokenizer input is missing: {path}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_iterator([str(path) for path in inputs]), trainer=trainer)

    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            raise RuntimeError(f"special token {token} expected id {expected_id}, got {actual}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    special_tokens_path = output / "special_tokens_map.json"
    special_tokens_path.write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema": "alice.eipm.n0.tokenizer-receipt.v0.1",
        "tokenizer_type": "byte_fallback_bpe",
        "vocab_size_requested": args.vocab_size,
        "vocab_size_observed": tokenizer.get_vocab_size(),
        "min_frequency": args.min_frequency,
        "special_tokens": SPECIAL_TOKENS,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "special_tokens_map_sha256": sha256_file(special_tokens_path),
        "input_shards": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inputs
        ],
        "git_revision": git_revision(),
        "private_identity_data": False,
    }
    receipt_path = output / "tokenizer_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "vocab_size": tokenizer.get_vocab_size(),
                "output": str(output),
                "tokenizer_sha256": receipt["tokenizer_sha256"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
