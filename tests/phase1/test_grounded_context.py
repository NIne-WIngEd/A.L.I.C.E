from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_context import (
    build_grounded_context,
    verify_grounded_context_package,
)
from alice_vault.grounded_context_evaluation import evaluate_grounded_context


def fake_search(**kwargs):
    return {
        "results": [
            {
                "rank": 1,
                "source_content_sha256": "source-a",
                "rrf_score": 0.03,
                "lexical_rank": 1,
                "semantic_rank": 2,
                "chunk_id": "chunk-a",
                "chunk_index": 0,
                "family": "research_project",
                "source_extraction_truncated": False,
                "snippet": "The AFM project used a U-Net to create segmentation masks.",
                "provenance": [
                    {
                        "file_id": "a1",
                        "original_relative_path": "research/afm.pdf",
                        "filename": "afm.pdf",
                        "role": "primary",
                        "source_bucket": "research",
                        "year_hint": "2026",
                        "duplicate_control_group": "dup-1",
                        "known_contradiction_group": "project-status",
                    },
                    {
                        "file_id": "a2",
                        "original_relative_path": "backup/afm.pdf",
                        "filename": "afm.pdf",
                        "role": "duplicate_control",
                        "source_bucket": "backup",
                        "year_hint": "2026",
                        "duplicate_control_group": "dup-1",
                        "known_contradiction_group": "project-status",
                    },
                ],
            },
            {
                "rank": 2,
                "source_content_sha256": "source-b",
                "rrf_score": 0.02,
                "lexical_rank": None,
                "semantic_rank": 1,
                "chunk_id": "chunk-b",
                "chunk_index": 1,
                "family": "education",
                "source_extraction_truncated": True,
                "snippet": "A later record contains a different project status.",
                "provenance": [
                    {
                        "file_id": "b1",
                        "original_relative_path": "notes/status.txt",
                        "filename": "status.txt",
                        "role": "primary",
                        "source_bucket": "notes",
                        "year_hint": "2026",
                        "duplicate_control_group": "",
                        "known_contradiction_group": "project-status",
                    }
                ],
            },
        ]
    }


class GroundedContextTests(unittest.TestCase):
    def test_context_is_cited_read_only_and_conflict_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            vault.mkdir()
            result = build_grounded_context(
                vault_root=vault,
                query="Which project used segmentation?",
                policy_path=ROOT / "policies" / "grounded_context_policy.json",
                search_fn=fake_search,
            )
            package = result["package"]
            self.assertEqual([x["citation_id"] for x in package["evidence"]], ["S1", "S2"])
            self.assertEqual(len(package["evidence"][0]["provenance"]), 2)
            self.assertTrue(package["contradiction_groups"][0]["unresolved"])
            self.assertFalse(package["guardrails"]["memory_write_allowed"])

            verification = verify_grounded_context_package(
                package_path=Path(result["summary"]["package_path"]),
                policy_path=ROOT / "policies" / "grounded_context_policy.json",
            )
            self.assertTrue(verification["ready_for_llm_context"])

    def test_benchmark_coverage_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            vault = base / "vault"
            vault.mkdir()
            benchmark = base / "benchmark.json"
            benchmark.write_text(json.dumps({
                "benchmark_id": "b1",
                "cases": [
                    {
                        "query_id": "q1",
                        "question": "Which project used segmentation?",
                        "status": "approved",
                        "expected_source_sha256": ["source-a"],
                    },
                    {
                        "query_id": "q2",
                        "question": "Which later record exists?",
                        "status": "approved",
                        "expected_source_sha256": ["source-b"],
                    },
                ],
            }), encoding="utf-8")

            result = evaluate_grounded_context(
                vault_root=vault,
                benchmark_path=benchmark,
                policy_path=ROOT / "policies" / "grounded_context_policy.json",
                search_fn=fake_search,
            )
            self.assertEqual(result["expected_source_coverage"], 1.0)
            self.assertEqual(result["citation_integrity_rate"], 1.0)
            self.assertEqual(result["contradiction_safety_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
