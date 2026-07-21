from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.response_context_enrichment import (
    _interval_overlap_ratio,
    _lexical_overlap,
    _query_terms,
    _select_nonredundant,
)
from alice_vault.grounded_response import (
    load_grounded_response_policy,
)


class ResponseEvidenceExpansionTests(unittest.TestCase):
    def test_policy_enables_source_preserving_expansion(self):
        policy = load_grounded_response_policy(
            ROOT
            / "policies"
            / "grounded_response_policy.json"
        )
        self.assertTrue(
            policy.evidence_expansion["enabled"]
        )
        self.assertEqual(
            policy.evidence_expansion[
                "passages_per_source"
            ],
            3,
        )

    def test_lexical_overlap_rewards_question_terms(self):
        terms = _query_terms(
            "What research used AFM image segmentation?"
        )
        relevant = _lexical_overlap(
            terms,
            "AFM research used image segmentation and a U-Net.",
        )
        irrelevant = _lexical_overlap(
            terms,
            "A mechanical rotor was modeled in CAD.",
        )
        self.assertGreater(relevant, irrelevant)

    def test_redundant_overlapping_segments_are_collapsed(self):
        candidates = [
            {
                "chunk_id": "c1",
                "segment_start_char": 0,
                "segment_end_char": 500,
                "selection_score": 1.0,
            },
            {
                "chunk_id": "c1",
                "segment_start_char": 40,
                "segment_end_char": 520,
                "selection_score": 0.9,
            },
            {
                "chunk_id": "c1",
                "segment_start_char": 600,
                "segment_end_char": 1000,
                "selection_score": 0.8,
            },
        ]
        selected = _select_nonredundant(
            candidates,
            limit=3,
        )
        self.assertEqual(len(selected), 2)
        self.assertGreater(
            _interval_overlap_ratio(
                0, 500, 40, 520
            ),
            0.72,
        )


if __name__ == "__main__":
    unittest.main()
