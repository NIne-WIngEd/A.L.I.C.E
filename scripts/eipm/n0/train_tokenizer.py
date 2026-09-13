#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


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
    tokenizer.train_from_iterator(text_iterator(args.input), trainer=trainer)

    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            raise RuntimeError(f"special token {token} expected id {expected_id}, got {actual}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / "tokenizer.json"), pretty=True)
    (output / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]"
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"vocab_size": tokenizer.get_vocab_size(), "output": str(output)}))


if __name__ == "__main__":
    main()
