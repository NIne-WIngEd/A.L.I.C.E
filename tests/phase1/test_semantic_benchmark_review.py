from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.semantic_benchmark_review import review_semantic_benchmark


class BenchmarkReviewerTests(unittest.TestCase):
    def test_shows_preview_and_writes_selected_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            vault = base / "vault"
            vault.mkdir()
            benchmark = base / "benchmark.json"
            benchmark.write_text(
                json.dumps({
                    "benchmark_id": "b1",
                    "cases": [{
                        "query_id": "q1",
                        "question": "Which project used masks?",
                        "status": "pending",
                        "expected_source_sha256": [],
                    }],
                }),
                encoding="utf-8",
            )

            def fake_search(**kwargs):
                return {
                    "results": [{
                        "rank": 1,
                        "source_content_sha256": "correct-sha",
                        "chunk_id": "c1",
                        "family": "research_project",
                        "snippet": "The AFM project used a U-Net for masks.",
                        "rrf_score": 0.03,
                        "lexical_rank": 2,
                        "semantic_rank": 1,
                        "source_extraction_truncated": False,
                        "provenance": [{
                            "filename": "AFM.pdf",
                            "original_relative_path": "research/AFM.pdf",
                        }],
                    }]
                }

            output = []
            result = review_semantic_benchmark(
                vault_root=vault,
                benchmark_path=benchmark,
                search_fn=fake_search,
                model_loader=lambda *args, **kwargs: object(),
                input_fn=lambda prompt: "1",
                output_fn=output.append,
            )
            updated = json.loads(benchmark.read_text(encoding="utf-8"))
            self.assertEqual(
                updated["cases"][0]["expected_source_sha256"],
                ["correct-sha"],
            )
            self.assertIn("U-Net for masks", "\n".join(output))
            self.assertEqual(
                result["final_status_counts"]["approved"], 1
            )


if __name__ == "__main__":
    unittest.main()
