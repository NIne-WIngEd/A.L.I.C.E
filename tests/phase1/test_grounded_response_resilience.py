from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import (
    RESPONSE_JSON_SCHEMA,
    load_grounded_response_policy,
    ollama_generate,
)
from alice_vault.grounded_response_evaluation import (
    evaluate_grounded_responses,
)


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class GroundedResponseResilienceTests(unittest.TestCase):
    def test_ollama_timeout_is_retried(self):
        policy = load_grounded_response_policy(
            ROOT / "policies" / "grounded_response_policy.json"
        )
        payload = {
            "response": json.dumps({
                "answer_type": "insufficient_evidence",
                "answer": "Insufficient evidence.",
                "claims": [],
                "uncertainty_notes": ["missing evidence"],
                "contradiction_notes": [],
            }),
            "done": True,
        }
        with patch(
            "alice_vault.grounded_response.urllib.request.urlopen",
            side_effect=[socket.timeout("slow"), FakeHttpResponse(payload)],
        ), patch("alice_vault.grounded_response.time.sleep"):
            result = ollama_generate(
                policy,
                RESPONSE_JSON_SCHEMA,
                "system",
                "user",
            )
        self.assertEqual(result["ollama"]["attempt_count"], 2)

    def test_evaluation_checkpoint_resumes_completed_case(self):
        def context_search(**kwargs):
            return {"results": [{
                "rank": 1,
                "source_content_sha256": "source-a",
                "rrf_score": 0.03,
                "lexical_rank": 1,
                "semantic_rank": 1,
                "chunk_id": "chunk-a",
                "chunk_index": 0,
                "family": "research_project",
                "source_extraction_truncated": False,
                "snippet": "AFM used segmentation.",
                "provenance": [{
                    "file_id": "a",
                    "original_relative_path": "afm.pdf",
                    "filename": "afm.pdf",
                    "role": "primary",
                    "source_bucket": "research",
                    "year_hint": "2026",
                    "duplicate_control_group": "",
                    "known_contradiction_group": "",
                }],
            }]}

        calls = {"count": 0}
        def flaky_client(policy, schema, system_prompt, user_prompt):
            calls["count"] += 1
            if calls["count"] == 2:
                raise TimeoutError("simulated")
            return {
                "structured": {
                    "answer_type": "grounded",
                    "answer": "AFM used segmentation. [S1]",
                    "claims": [{
                        "text": "AFM used segmentation.",
                        "claim_type": "fact",
                        "citations": ["[S1]"],
                    }],
                    "uncertainty_notes": [],
                    "contradiction_notes": [],
                },
                "ollama": {},
            }

        def good_client(policy, schema, system_prompt, user_prompt):
            return {
                "structured": {
                    "answer_type": "grounded",
                    "answer": "AFM used segmentation. [S1]",
                    "claims": [{
                        "text": "AFM used segmentation.",
                        "claim_type": "fact",
                        "citations": ["[S1]"],
                    }],
                    "uncertainty_notes": [],
                    "contradiction_notes": [],
                },
                "ollama": {},
            }

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            vault = base / "vault"
            vault.mkdir()
            (vault / "temporary").mkdir()
            benchmark = base / "benchmark.json"
            benchmark.write_text(json.dumps({
                "benchmark_id": "bench-resume",
                "cases": [
                    {"query_id": "q1", "question": "Q1", "status": "approved", "expected_source_sha256": ["source-a"]},
                    {"query_id": "q2", "question": "Q2", "status": "approved", "expected_source_sha256": ["source-a"]},
                ],
            }), encoding="utf-8")

            with self.assertRaises(RuntimeError):
                evaluate_grounded_responses(
                    vault_root=vault,
                    benchmark_path=benchmark,
                    response_policy_path=ROOT / "policies" / "grounded_response_policy.json",
                    context_policy_path=ROOT / "policies" / "grounded_context_policy.json",
                    context_search_fn=context_search,
                    response_model_client=flaky_client,
                )

            result = evaluate_grounded_responses(
                vault_root=vault,
                benchmark_path=benchmark,
                response_policy_path=ROOT / "policies" / "grounded_response_policy.json",
                context_policy_path=ROOT / "policies" / "grounded_context_policy.json",
                context_search_fn=context_search,
                response_model_client=good_client,
            )
            self.assertEqual(result["resumed_case_count"], 1)
            self.assertEqual(result["new_case_count"], 1)
            self.assertTrue(result["passes_all_thresholds"])


if __name__ == "__main__":
    unittest.main()
