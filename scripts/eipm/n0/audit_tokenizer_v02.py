#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPECIAL_TOKEN_IDS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the governed Alice N0 v0.2 tokenizer.")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-rows-per-source", type=int, default=64)
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before auditing the tokenizer") from exc

    corpus_dir = Path(args.corpus_dir).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_receipt_path = tokenizer_dir / "tokenizer_receipt.json"
    corpus_receipt_path = corpus_dir / "corpus_receipt.json"
    for path in (tokenizer_path, tokenizer_receipt_path, corpus_receipt_path):
        if not path.is_file():
            raise SystemExit(f"required audit artifact missing: {path}")

    receipt = json.loads(tokenizer_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "alice.eipm.n0.tokenizer-receipt.v0.2":
        raise SystemExit("unexpected tokenizer receipt schema")
    if receipt.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise SystemExit("tokenizer hash does not match tokenizer receipt")
    if receipt.get("corpus_receipt_sha256") != sha256_file(corpus_receipt_path):
        raise SystemExit("tokenizer receipt is not bound to this tokenizer corpus")
    for key in ("private_identity_data", "private_identity_gradient", "model_training_performed"):
        if receipt.get(key) is not False:
            raise SystemExit(f"tokenizer receipt must declare {key}=false")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 48_000:
        raise SystemExit(f"expected 48000-token vocabulary, got {tokenizer.get_vocab_size()}")
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            raise SystemExit(f"special token {token} expected id {expected_id}, got {actual}")

    corpus_receipt = json.loads(corpus_receipt_path.read_text(encoding="utf-8"))
    shard_paths: list[Path] = []
    for source in corpus_receipt.get("sources", []):
        for shard in source.get("shards", []):
            path = corpus_dir / str(shard["path"])
            if not path.is_file() or sha256_file(path) != str(shard["sha256"]):
                raise SystemExit(f"corpus shard failed audit hash verification: {path}")
            shard_paths.append(path)

    train_kept: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_chars: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    source_chars: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    unknown_tokens = 0
    roundtrip_failures: list[dict[str, Any]] = []

    for path in sorted(shard_paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_id = str(row.get("source_id") or "unknown")
                split = str(row.get("split") or "unknown")
                if split == "train":
                    if train_kept[source_id] >= args.train_rows_per_source:
                        continue
                    train_kept[source_id] += 1
                text = unicodedata.normalize("NFC", str(row.get("text") or ""))
                if not text:
                    continue
                encoded = tokenizer.encode(text, add_special_tokens=False)
                token_count = len(encoded.ids)
                if token_count == 0:
                    raise SystemExit(f"tokenizer produced zero tokens for non-empty row from {source_id}")
                unknown_tokens += sum(token_id == 1 for token_id in encoded.ids)
                decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
                if decoded != text and len(roundtrip_failures) < 10:
                    roundtrip_failures.append(
                        {"source_id": source_id, "split": split, "text_sha256": row.get("text_sha256")}
                    )
                split_rows[split] += 1
                split_chars[split] += len(text)
                split_tokens[split] += token_count
                source_chars[source_id] += len(text)
                source_tokens[source_id] += token_count

    if unknown_tokens:
        raise SystemExit(f"tokenizer emitted {unknown_tokens} [UNK] tokens during audit")
    if roundtrip_failures:
        raise SystemExit(f"tokenizer round-trip failed on sampled corpus rows: {roundtrip_failures[:3]}")
    if split_rows["dev"] < 1 or split_rows["test"] < 1:
        raise SystemExit(
            f"audit requires held-out coverage; observed dev={split_rows['dev']} test={split_rows['test']}"
        )

    def ratio(chars: int, tokens: int) -> float | None:
        return chars / tokens if tokens else None

    result = {
        "schema": "alice.eipm.n0.tokenizer-audit.v0.2",
        "status": "PASS",
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "tokenizer_receipt_sha256": sha256_file(tokenizer_receipt_path),
        "corpus_receipt_sha256": sha256_file(corpus_receipt_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "unknown_token_count": 0,
        "roundtrip_failures": 0,
        "sampled_rows_by_split": dict(sorted(split_rows.items())),
        "sampled_chars_by_split": dict(sorted(split_chars.items())),
        "sampled_tokens_by_split": dict(sorted(split_tokens.items())),
        "chars_per_token_by_split": {
            split: ratio(split_chars[split], split_tokens[split]) for split in sorted(split_rows)
        },
        "chars_per_token_by_source": {
            source: ratio(source_chars[source], source_tokens[source]) for source in sorted(source_chars)
        },
        "private_identity_data": False,
        "private_identity_gradient": False,
        "model_training_performed": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
