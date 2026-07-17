from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.claim_entailment_gate import (
    filter_model_output_by_entailment,
    load_claim_entailment_policy,
)


class FakeNLI:
    """Fake logits use FEVER-NLI order:
    entailment, neutral, contradiction.
    """

    def predict(
        self,
        pairs,
        batch_size=8,
        show_progress_bar=False,
    ):
        return [
            [4.0, 0.2, 0.1]
            for _ in pairs
        ]


def renderer(claims):
    return "\n".join(
        claim["text"]
        + " "
        + " ".join(claim["citations"])
        for claim in claims
    )


class ClaimEntailmentReportingTests(unittest.TestCase):
    def test_gate_summary_reports_enabled_when_it_runs(self):
        policy = load_claim_entailment_policy(
            ROOT
            / "policies"
            / "claim_entailment_policy.json"
        )
        output = {
            "answer_type": "grounded",
            "answer": "old",
            "claims": [
                {
                    "text": "The AFM project used a U-Net.",
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        }
        context = {
            "evidence": [
                {
                    "citation": "[S1]",
                    "context_text": (
                        "The AFM project used a U-Net "
                        "for segmentation."
                    ),
                    "owner_relation": "owner_self_record",
                }
            ]
        }

        _, summary = filter_model_output_by_entailment(
            model_output=output,
            context_package=context,
            model=FakeNLI(),
            policy=policy,
            answer_renderer=renderer,
        )

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["input_claim_count"], 1)
        self.assertEqual(summary["kept_claim_count"], 1)
        self.assertEqual(summary["dropped_claim_count"], 0)


if __name__ == "__main__":
    unittest.main()
