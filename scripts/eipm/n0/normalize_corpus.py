#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for_hash(digest: str) -> str:
    bucket = int(digest[:8], 16) % 10_000
    if bucket < 10:
        return "test"
    if bucket < 20:
        return "dev"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize/deduplicate N0 public text JSONL.")
    parser.add_argument("--input", action="append", required=True, help="JSONL path; repeatable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--license-expression", required=True)
    parser.add_argument("--min-chars", type=int, default=80)
    args = parser.parse_args()

    seen: set[str] = set()
    counters = {"read": 0, "kept": 0, "empty_or_short": 0, "duplicate": 0}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as sink:
        for raw_path in args.input:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8") as source:
                for line_no, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    counters["read"] += 1
                    row = json.loads(line)
                    text = normalize_text(str(row.get("text", "")))
                    if len(text) < args.min_chars:
                        counters["empty_or_short"] += 1
                        continue
                    digest = sha256_text(text)
                    if digest in seen:
                        counters["duplicate"] += 1
                        continue
                    seen.add(digest)
                    sink.write(
                        json.dumps(
                            {
                                "text": text,
                                "text_sha256": digest,
                                "split": split_for_hash(digest),
                                "source_id": args.source_id,
                                "license_expression": args.license_expression,
                                "source_file": path.name,
                                "source_line": line_no,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counters["kept"] += 1

    print(json.dumps(counters, sort_keys=True))


if __name__ == "__main__":
    main()
