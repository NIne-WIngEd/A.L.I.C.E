from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.retrieval import build_index
from alice_vault.semantic_retrieval import (
    build_semantic_index,
    hybrid_search,
    prepare_embedding_model,
    semantic_search,
    verify_semantic_index,
    _segment_chunk_for_embeddings,
    load_semantic_policy,
)
from alice_vault.semantic_evaluation import (
    validate_semantic_benchmark,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


class FakeModel:
    max_seq_length = 512
    tokenizer = FakeTokenizer()

    def get_sentence_embedding_dimension(self):
        return 8

    def save_pretrained(self, path, safe_serialization=True):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"safe")
        (target / "config.json").write_text(
            '{"fake": true}',
            encoding="utf-8",
        )

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            lowered = text.casefold()
            if any(word in lowered for word in ("afm", "surface", "microscopy")):
                vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif any(word in lowered for word in ("vanderbilt", "university", "transfer")):
                vector = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            norm = math.sqrt(sum(value * value for value in vector))
            rows.append([value / norm for value in vector])
        return rows


def fake_loader(*args, **kwargs):
    return FakeModel()


class SemanticRetrievalTests(unittest.TestCase):
    def make_vault(self, base: Path) -> Path:
        vault = base / "vault"
        chunk_root = (
            vault
            / "derived"
            / "pilot-v1"
            / "chunks"
            / "chunk-set-test"
        )
        text_root = chunk_root / "text"
        text_root.mkdir(parents=True)
        (vault / "manifests" / "exports").mkdir(parents=True)
        (vault / "temporary").mkdir()
        texts = [
            (
                "c1",
                "source-a",
                0,
                "A U-Net analyzed nanoscale images and generated masks.",
                "research_project",
            ),
            (
                "c2",
                "source-b",
                0,
                "A university admission and transfer plan changed.",
                "education",
            ),
            (
                "c3",
                "source-c",
                0,
                "A mechanical design project created an engine rotor.",
                "work",
            ),
        ]
        records = []
        for chunk_id, source, index, text, family in texts:
            path = text_root / f"{chunk_id}.txt"
            path.write_text(text, encoding="utf-8")
            records.append(
                {
                    "chunk_id": chunk_id,
                    "source_content_sha256": source,
                    "source_text_sha256": "t" * 64,
                    "normalized_source_text_sha256": "n" * 64,
                    "chunk_index": index,
                    "start_char": 0,
                    "end_char": len(text),
                    "char_count": len(text),
                    "chunk_text_sha256": digest(path),
                    "family": family,
                    "parser_id": "test",
                    "extraction_registry_digest": "r" * 64,
                    "source_extraction_truncated": False,
                    "source_extraction_warnings": [],
                    "provenance_path_count": 1,
                    "provenance": [
                        {
                            "file_id": f"file-{chunk_id}",
                            "original_relative_path": f"{chunk_id}.txt",
                            "filename": f"{chunk_id}.txt",
                            "role": "primary",
                            "family": family,
                            "source_bucket": "test",
                            "year_hint": "2026",
                            "duplicate_control_group": "",
                            "known_contradiction_group": "",
                        }
                    ],
                }
            )
        records_path = chunk_root / "chunk-records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        (chunk_root / "chunk-manifest.json").write_text(
            json.dumps(
                {
                    "chunk_set_id": "chunk-set-test",
                    "manifest_fingerprint": "m" * 64,
                    "chunk_records_sha256": digest(records_path),
                    "chunk_count": len(records),
                }
            ),
            encoding="utf-8",
        )
        return vault

    def policy_path(self, base: Path) -> Path:
        data = json.loads(
            (
                ROOT / "policies" / "semantic_retrieval_policy.json"
            ).read_text(encoding="utf-8")
        )
        data["model"]["embedding_dimension"] = 8
        data["model"]["maximum_sequence_tokens"] = 512
        path = base / "semantic-policy.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_prepare_build_verify_search_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            vault = self.make_vault(base)
            policy = self.policy_path(base)

            prepared = prepare_embedding_model(
                vault_root=vault,
                policy_path=policy,
                model_loader=fake_loader,
            )
            self.assertTrue(prepared["prepared_now"])

            lexical_policy = (
                ROOT / "policies" / "retrieval_policy.json"
            )
            build_index(
                vault_root=vault,
                policy_path=lexical_policy,
            )

            built = build_semantic_index(
                vault_root=vault,
                policy_path=policy,
                model_loader=fake_loader,
            )
            self.assertEqual(built["chunk_count"], 3)

            verification = verify_semantic_index(
                vault_root=vault,
                policy_path=policy,
            )
            self.assertTrue(
                verification["ready_for_semantic_search"]
            )

            semantic = semantic_search(
                vault_root=vault,
                policy_path=policy,
                query="surface microscopy segmentation",
                model_loader=fake_loader,
            )
            self.assertEqual(
                semantic["results"][0][
                    "source_content_sha256"
                ],
                "source-a",
            )

            hybrid = hybrid_search(
                vault_root=vault,
                semantic_policy_path=policy,
                lexical_policy_path=lexical_policy,
                query="surface microscopy segmentation",
                model_loader=fake_loader,
            )
            self.assertEqual(
                hybrid["results"][0][
                    "source_content_sha256"
                ],
                "source-a",
            )

            resumed = build_semantic_index(
                vault_root=vault,
                policy_path=policy,
                model_loader=fake_loader,
            )
            self.assertTrue(
                resumed["resumed_existing_index"]
            )

    def test_token_aware_segmentation_prevents_model_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            policy_path = self.policy_path(base)
            policy = load_semantic_policy(policy_path)
            text = " ".join(
                f"researchtoken{index}" for index in range(1400)
            )
            segments = _segment_chunk_for_embeddings(
                model=FakeModel(),
                text=text,
                parent_chunk_id="parent-chunk",
                policy=policy,
            )
            self.assertGreater(len(segments), 1)
            self.assertEqual(segments[0]["segment_start_char"], 0)
            self.assertEqual(
                segments[-1]["segment_end_char"],
                len(text),
            )
            self.assertTrue(
                all(
                    segment["segment_token_count"] <= 480
                    for segment in segments
                )
            )
            self.assertEqual(
                len(
                    {
                        segment["semantic_segment_id"]
                        for segment in segments
                    }
                ),
                len(segments),
            )

    def test_benchmark_requires_approved_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            vault = self.make_vault(base)
            policy = self.policy_path(base)
            benchmark = base / "benchmark.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "benchmark_id": "b1",
                        "cases": [
                            {
                                "query_id": f"q{index}",
                                "question": "question",
                                "status": "approved",
                                "expected_source_sha256": (
                                    ["source-a"] if index < 9 else []
                                ),
                            }
                            for index in range(10)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = validate_semantic_benchmark(
                vault_root=vault,
                benchmark_path=benchmark,
                semantic_policy_path=policy,
            )
            self.assertFalse(result["ready_for_evaluation"])
            self.assertGreater(result["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
