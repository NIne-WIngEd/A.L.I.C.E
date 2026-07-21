from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.claim_entailment_gate import (
    _build_sentence_windows,
    cited_passages_for_claim,
    load_claim_entailment_policy,
)


class ClaimFocusedNLIWindowTests(unittest.TestCase):
    def test_policy_keeps_strict_thresholds_and_enables_compact_windows(self):
        policy = load_claim_entailment_policy(
            ROOT
            / "policies"
            / "claim_entailment_policy.json"
        )
        self.assertEqual(
            policy.entailment_threshold,
            0.70,
        )
        self.assertEqual(
            policy.contradiction_threshold,
            0.80,
        )
        self.assertLessEqual(
            policy.maximum_window_characters,
            900,
        )
        self.assertGreaterEqual(
            policy.sentence_window_size,
            1,
        )

    def test_sentence_windows_are_compact_and_overlapping(self):
        text = (
            "First sentence about unrelated work. "
            "Second sentence mentions AFM images. "
            "Third sentence says the project used U-Net. "
            "Fourth sentence is unrelated again."
        )
        windows = _build_sentence_windows(
            passage=text,
            window_size=2,
            stride=1,
            maximum_characters=900,
        )
        self.assertGreaterEqual(len(windows), 3)
        self.assertTrue(
            any(
                "AFM images" in window
                and "U-Net" in window
                for window in windows
            )
        )

    def test_claim_focused_windows_rank_relevant_span_first(self):
        claim = {
            "text": "The AFM project used a U-Net.",
            "citations": ["[S1]"],
        }
        context = {
            "evidence": [
                {
                    "citation": "[S1]",
                    "owner_relation": "owner_self_record",
                    "context_text": (
                        "Unrelated mechanical design work. "
                        "Another unrelated sentence. "
                        "The AFM project used a U-Net for segmentation. "
                        "More unrelated information."
                    ),
                }
            ]
        }
        windows = cited_passages_for_claim(
            claim=claim,
            context_package=context,
            limit=12,
            sentence_window_size=2,
            sentence_window_stride=1,
            maximum_window_characters=900,
            maximum_windows=12,
        )
        self.assertTrue(windows)
        self.assertIn(
            "AFM project used a U-Net",
            windows[0]["premise"],
        )


if __name__ == "__main__":
    unittest.main()
