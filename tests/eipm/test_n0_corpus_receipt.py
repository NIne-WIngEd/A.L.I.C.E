from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_personality.n0.corpus_receipt import sha256_file, verify_corpus_receipt


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    shards = corpus / "shards"
    shards.mkdir(parents=True)

    source_config = tmp_path / "sources.json"
    source_config.write_text(
        json.dumps(
            {
                "sources": [
                    {"source_id": "a"},
                    {"source_id": "b"},
                ]
            }
        ),
        encoding="utf-8",
    )

    a = shards / "a-00000.jsonl"
    b = shards / "b-00000.jsonl"
    a.write_text('{"text":"alpha"}\n', encoding="utf-8")
    b.write_text('{"text":"beta"}\n', encoding="utf-8")

    receipt = {
        "source_config_sha256": sha256_file(source_config),
        "private_identity_data": False,
        "sources": [
            {
                "source_id": "a",
                "resolved_revision": "sha-a",
                "counters": {"accepted": 1, "accepted_chars": 5},
                "shards": [
                    {
                        "path": "shards/a-00000.jsonl",
                        "sha256": sha256_file(a),
                        "bytes": a.stat().st_size,
                    }
                ],
            },
            {
                "source_id": "b",
                "resolved_revision": "sha-b",
                "counters": {"accepted": 1, "accepted_chars": 4},
                "shards": [
                    {
                        "path": "shards/b-00000.jsonl",
                        "sha256": sha256_file(b),
                        "bytes": b.stat().st_size,
                    }
                ],
            },
        ],
    }
    (corpus / "corpus_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return corpus, source_config


def test_verify_corpus_receipt_binds_every_shard(tmp_path: Path) -> None:
    corpus, source_config = make_fixture(tmp_path)
    result = verify_corpus_receipt(corpus, source_config)
    assert result["status"] == "PASS"
    assert result["source_count"] == 2
    assert result["verified_shards"] == 2
    assert result["accepted_rows"] == 2
    assert result["accepted_chars"] == 9


def test_verify_corpus_receipt_rejects_mutated_shard(tmp_path: Path) -> None:
    corpus, source_config = make_fixture(tmp_path)
    (corpus / "shards/a-00000.jsonl").write_text(
        '{"text":"mutated"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        verify_corpus_receipt(corpus, source_config)


def test_verify_corpus_receipt_rejects_source_config_drift(tmp_path: Path) -> None:
    corpus, source_config = make_fixture(tmp_path)
    source_config.write_text(
        json.dumps({"sources": [{"source_id": "a"}, {"source_id": "c"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_config_sha256"):
        verify_corpus_receipt(corpus, source_config)
