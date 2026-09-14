#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
EXPECTED_CORPUS_SCHEMA = "alice.eipm.n0.derived-tokenizer-corpus-receipt.v0.2.1"
EXPECTED_SOURCE_SCHEMA = "alice.eipm.n0.public-corpus-sources.v0.2.1"


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


def verify_corpus_lineage(
    corpus_dir: Path,
    source_config_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    receipt_path = corpus_dir / "corpus_receipt.json"
    if not receipt_path.is_file():
        raise SystemExit(f"missing tokenizer-corpus receipt: {receipt_path}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != EXPECTED_CORPUS_SCHEMA:
        raise SystemExit(f"unexpected tokenizer-corpus receipt schema: {receipt.get('schema')}")
    if receipt.get("status") != "PASS":
        raise SystemExit("tokenizer corpus must have status=PASS")
    for key in ("private_identity_data", "private_identity_gradient", "model_training_performed"):
        if receipt.get(key) is not False:
            raise SystemExit(f"tokenizer corpus must declare {key}=false")
    if receipt.get("network_access_required") is not False:
        raise SystemExit("v0.2.1 tokenizer corpus must record network_access_required=false")

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if source_config.get("schema") != EXPECTED_SOURCE_SCHEMA:
        raise SystemExit("tokenizer training requires the activated v0.2.1 source manifest")
    if source_config.get("status") != "activated_public_n0_v02":
        raise SystemExit("v0.2.1 source manifest is not activated")
    if receipt.get("source_config_sha256") != sha256_file(source_config_path):
        raise SystemExit("tokenizer corpus source_config_sha256 does not match active manifest")

    active_sources = source_config.get("sources")
    receipt_sources = receipt.get("sources")
    if not isinstance(active_sources, list) or len(active_sources) != 21:
        raise SystemExit("active v0.2.1 manifest must contain exactly 21 sources")
    if not isinstance(receipt_sources, list) or len(receipt_sources) != 21:
        raise SystemExit("tokenizer corpus receipt must contain exactly 21 sources")

    active_by_id = {str(row["source_id"]): row for row in active_sources}
    if len(active_by_id) != len(active_sources):
        raise SystemExit("active source manifest contains duplicate source IDs")

    shard_paths: list[Path] = []
    seen_shards: set[Path] = set()
    for source in receipt_sources:
        source_id = str(source.get("source_id", ""))
        active = active_by_id.get(source_id)
        if active is None:
            raise SystemExit(f"receipt contains source absent from active manifest: {source_id}")
        if str(source.get("revision")) != str(active.get("revision")):
            raise SystemExit(f"source revision mismatch for {source_id}")
        if abs(float(source.get("target_share", 0.0)) - float(active.get("target_share", 0.0))) > 1e-12:
            raise SystemExit(f"source target-share mismatch for {source_id}")
        if float(source.get("fill_ratio", 0.0)) < 0.999:
            raise SystemExit(f"source fill ratio is below the accepted tokenizer threshold: {source_id}")

        shards = source.get("shards")
        if not isinstance(shards, list) or not shards:
            raise SystemExit(f"source has no materialized shards: {source_id}")
        for shard in shards:
            rel = Path(str(shard["path"]))
            if rel.is_absolute() or ".." in rel.parts:
                raise SystemExit(f"unsafe tokenizer-corpus shard path: {rel}")
            path = (corpus_dir / rel).resolve()
            if corpus_dir.resolve() not in path.parents:
                raise SystemExit(f"tokenizer-corpus shard escapes corpus root: {rel}")
            if path in seen_shards:
                raise SystemExit(f"duplicate tokenizer-corpus shard in receipt: {rel}")
            seen_shards.add(path)
            if not path.is_file():
                raise SystemExit(f"missing tokenizer-corpus shard: {rel}")
            if path.stat().st_size != int(shard["bytes"]):
                raise SystemExit(f"tokenizer-corpus shard byte mismatch: {rel}")
            if sha256_file(path) != str(shard["sha256"]):
                raise SystemExit(f"tokenizer-corpus shard SHA256 mismatch: {rel}")
            shard_paths.append(path)

    if int(receipt.get("accepted_chars", 0)) < 99_000_000:
        raise SystemExit("tokenizer corpus is unexpectedly small (<99M accepted characters)")
    return receipt, sorted(shard_paths)


class TrainTextStream:
    """Yield only train-split public text while retaining auditable counters."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.rows_seen = 0
        self.rows_used = 0
        self.chars_used = 0
        self.skipped_by_split: Counter[str] = Counter()
        self.rows_by_source: Counter[str] = Counter()
        self.chars_by_source: Counter[str] = Counter()

    def __iter__(self) -> Iterator[str]:
        for path in self.paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.rows_seen += 1
                    split = str(row.get("split") or "")
                    if split != "train":
                        self.skipped_by_split[split or "missing"] += 1
                        continue
                    text = str(row.get("text") or "").strip()
                    if not text:
                        continue
                    source_id = str(row.get("source_id") or "unknown")
                    self.rows_used += 1
                    self.chars_used += len(text)
                    self.rows_by_source[source_id] += 1
                    self.chars_by_source[source_id] += len(text)
                    yield text


def validate_tokenizer(tokenizer: Any, vocab_size: int) -> dict[str, Any]:
    observed_vocab = int(tokenizer.get_vocab_size())
    if observed_vocab != vocab_size:
        raise RuntimeError(f"tokenizer vocab size mismatch: requested={vocab_size} observed={observed_vocab}")

    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            raise RuntimeError(f"special token {token} expected id {expected_id}, got {actual}")

    probes = [
        "Plain ASCII with numbers 0123456789 and punctuation!?",
        "Café naïve résumé — symbols: α β γ ∑ ≠ ≤ ≥",
        "বাংলা লেখা এবং English mixed together.",
        "Emoji stay representable: 🙂🚀🧠✨",
        "def f(x):\n    return x ** 2  # code and indentation",
        "Uncertainty: evidence supports A or B; do not invent C.",
    ]
    unk_id = tokenizer.token_to_id("[UNK]")
    roundtrip_failures: list[str] = []
    unknown_token_count = 0
    probe_token_count = 0
    for raw in probes:
        normalized = unicodedata.normalize("NFC", raw)
        encoded = tokenizer.encode(normalized, add_special_tokens=False)
        probe_token_count += len(encoded.ids)
        if unk_id is not None:
            unknown_token_count += sum(token_id == unk_id for token_id in encoded.ids)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        if decoded != normalized:
            roundtrip_failures.append(raw)

    if unknown_token_count:
        raise RuntimeError(f"byte-complete tokenizer emitted {unknown_token_count} [UNK] probe tokens")
    if roundtrip_failures:
        raise RuntimeError(f"tokenizer round-trip failed for {len(roundtrip_failures)} probes")

    return {
        "probe_count": len(probes),
        "probe_token_count": probe_token_count,
        "unknown_token_count": unknown_token_count,
        "roundtrip_failures": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the governed Alice N0 v0.2 48k byte-complete BPE tokenizer."
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
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

    corpus_dir = Path(args.corpus_dir).resolve()
    source_config_path = Path(args.source_config).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty tokenizer directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    corpus_receipt, shard_paths = verify_corpus_lineage(corpus_dir, source_config_path)
    stream = TrainTextStream(shard_paths)

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
    tokenizer.train_from_iterator(stream, trainer=trainer)

    if stream.rows_used < 1 or stream.chars_used < 1:
        raise RuntimeError("tokenizer trainer consumed no train-split text")
    validation = validate_tokenizer(tokenizer, args.vocab_size)

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
    tokenizer_config_path = output / "tokenizer_config.json"
    tokenizer_config_path.write_text(
        json.dumps(
            {
                "model_max_length": 8192,
                "padding_side": "right",
                "truncation_side": "right",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema": "alice.eipm.n0.tokenizer-receipt.v0.2",
        "model_id": "alice-n0-semantic-v0.2",
        "tokenizer_type": "byte_complete_bpe",
        "vocab_size_requested": args.vocab_size,
        "vocab_size_observed": tokenizer.get_vocab_size(),
        "min_frequency": args.min_frequency,
        "special_tokens": SPECIAL_TOKENS,
        "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "special_tokens_map_sha256": sha256_file(special_tokens_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "corpus_receipt_sha256": sha256_file(corpus_dir / "corpus_receipt.json"),
        "source_config_sha256": sha256_file(source_config_path),
        "parent_corpus_receipt_sha256": corpus_receipt.get("parent_corpus_receipt_sha256"),
        "corpus_accepted_chars": int(corpus_receipt["accepted_chars"]),
        "corpus_accepted_rows": int(corpus_receipt["accepted_rows"]),
        "corpus_source_count": int(corpus_receipt["source_count"]),
        "train_rows_used": stream.rows_used,
        "train_chars_used": stream.chars_used,
        "rows_skipped_by_split": dict(sorted(stream.skipped_by_split.items())),
        "train_rows_by_source": dict(sorted(stream.rows_by_source.items())),
        "train_chars_by_source": dict(sorted(stream.chars_by_source.items())),
        "training_split_policy": "train_only_dev_and_test_excluded_from_tokenizer_fit",
        "validation": validation,
        "input_shards": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in shard_paths
        ],
        "git_revision": git_revision(),
        "private_identity_data": False,
        "private_identity_gradient": False,
        "model_training_performed": False,
        "third_party_model_weights_used": False,
    }
    receipt_path = output / "tokenizer_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "vocab_size": tokenizer.get_vocab_size(),
                "train_rows_used": stream.rows_used,
                "train_chars_used": stream.chars_used,
                "rows_skipped_by_split": dict(sorted(stream.skipped_by_split.items())),
                "unknown_token_count": validation["unknown_token_count"],
                "roundtrip_failures": validation["roundtrip_failures"],
                "tokenizer_sha256": receipt["tokenizer_sha256"],
                "receipt_sha256": sha256_file(receipt_path),
                "output": str(output),
                "private_identity_data": False,
                "private_identity_gradient": False,
                "model_training_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
