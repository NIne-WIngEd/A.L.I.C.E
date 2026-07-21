from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.response_reranker import (
    load_response_reranker_policy,
    rerank_candidates,
)


class FakeReranker:
    def predict(
        self,
        pairs,
        batch_size=16,
        show_progress_bar=False,
    ):
        return [
            9.0 if "direct answer" in passage else 2.0
            for _, passage in pairs
        ]


class ResponseRerankerTests(unittest.TestCase):
    def test_ms_marco_reranker_is_enabled_for_p1_11_passage_selection(self):
        policy = load_response_reranker_policy(
            ROOT
            / "policies"
            / "response_reranker_policy.json"
        )
        self.assertTrue(policy.enabled)
        self.assertEqual(
            policy.model_id,
            "cross-encoder/ms-marco-MiniLM-L6-v2",
        )

    def test_cross_encoder_reranking_logic_still_works_when_enabled(self):
        candidates = [
            {
                "semantic_segment_id": "a",
                "selection_score": 0.95,
                "text": "related topic",
            },
            {
                "semantic_segment_id": "b",
                "selection_score": 0.70,
                "text": "direct answer",
            },
        ]
        result = rerank_candidates(
            query="q",
            candidates=candidates,
            reranker=FakeReranker(),
            batch_size=4,
        )
        self.assertEqual(
            result[0]["semantic_segment_id"],
            "b",
        )


if __name__ == "__main__":
    unittest.main()
