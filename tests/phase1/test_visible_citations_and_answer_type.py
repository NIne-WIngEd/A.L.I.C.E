from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import (
    _normalize_model_output,
    load_grounded_response_policy,
    verify_grounded_response_data,
)


def context():
    return {
        "evidence": [
            {
                "citation": "[S1]",
                "source_content_sha256": "source-a",
            },
            {
                "citation": "[S2]",
                "source_content_sha256": "source-b",
            },
        ],
        "contradiction_groups": [],
        "guardrails": {
            "memory_write_allowed": False,
            "external_action_allowed": False,
        },
    }


class VisibleCitationAndAnswerTypeTests(unittest.TestCase):
    def policy(self):
        return load_grounded_response_policy(
            ROOT
            / "policies"
            / "grounded_response_policy.json"
        )

    def test_uncited_free_form_answer_is_rendered_from_cited_claims(self):
        output = {
            "answer_type": "grounded",
            "answer": "The project involved AFM analysis.",
            "claims": [
                {
                    "text": "The project involved AFM analysis.",
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                },
                {
                    "text": "The work included segmentation.",
                    "claim_type": "fact",
                    "citations": ["[S2]"],
                },
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        }

        normalized = _normalize_model_output(
            output,
            context(),
        )

        self.assertIn("[S1]", normalized["answer"])
        self.assertIn("[S2]", normalized["answer"])

        verification = verify_grounded_response_data(
            context_package=context(),
            model_output=normalized,
            policy=self.policy(),
        )
        self.assertTrue(verification["verified"])
        self.assertEqual(
            verification["inline_answer_citation_count"],
            2,
        )

    def test_false_contradictory_answer_type_becomes_grounded(self):
        output = {
            "answer_type": "contradictory_evidence",
            "answer": "The project involved AFM analysis.",
            "claims": [
                {
                    "text": "The project involved AFM analysis.",
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [
                {
                    "label": "invented-conflict",
                    "citations": ["[S1]"],
                    "note": "There is a conflict.",
                }
            ],
        }

        normalized = _normalize_model_output(
            output,
            context(),
        )
        self.assertEqual(
            normalized["answer_type"],
            "grounded",
        )
        self.assertEqual(
            normalized["contradiction_notes"],
            [],
        )

        verification = verify_grounded_response_data(
            context_package=context(),
            model_output=normalized,
            policy=self.policy(),
        )
        self.assertTrue(verification["verified"])

    def test_verifier_rejects_raw_false_contradictory_answer_type(self):
        raw = {
            "answer_type": "contradictory_evidence",
            "answer": "The project involved AFM analysis. [S1]",
            "claims": [
                {
                    "text": "The project involved AFM analysis.",
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        }

        verification = verify_grounded_response_data(
            context_package=context(),
            model_output=raw,
            policy=self.policy(),
        )
        self.assertFalse(verification["verified"])
        self.assertTrue(
            any(
                "without an actual context contradiction group"
                in error
                for error in verification["errors"]
            )
        )


if __name__ == "__main__":
    unittest.main()
