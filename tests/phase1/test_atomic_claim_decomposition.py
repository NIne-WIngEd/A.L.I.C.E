from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.atomic_claim_decomposition import (
    decompose_model_output_claims,
    load_atomic_claim_decomposition_policy,
)


class AtomicClaimDecompositionTests(unittest.TestCase):
    def policy(self):
        return load_atomic_claim_decomposition_policy(
            ROOT
            / "policies"
            / "atomic_claim_decomposition_policy.json"
        )

    def test_compound_claim_is_split_and_citations_are_inherited(self):
        model_output = {
            "answer_type": "grounded",
            "answer": "old",
            "claims": [
                {
                    "text": (
                        "The user built an AFM platform "
                        "and used U-Net segmentation."
                    ),
                    "claim_type": "fact",
                    "citations": ["[S4]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        }

        def fake_client(
            *,
            policy,
            claims,
        ):
            return {
                "structured": {
                    "atomic_claims": [
                        {
                            "parent_claim_index": 1,
                            "text": (
                                "The user built an AFM platform."
                            ),
                        },
                        {
                            "parent_claim_index": 1,
                            "text": (
                                "The user used U-Net segmentation."
                            ),
                        },
                    ]
                },
                "runtime": {},
            }

        output, summary = decompose_model_output_claims(
            model_output=model_output,
            policy=self.policy(),
            client=fake_client,
        )

        self.assertEqual(
            len(output["claims"]),
            2,
        )
        self.assertEqual(
            output["claims"][0]["citations"],
            ["[S4]"],
        )
        self.assertEqual(
            output["claims"][1]["citations"],
            ["[S4]"],
        )
        self.assertEqual(
            summary["input_claim_count"],
            1,
        )
        self.assertEqual(
            summary["output_atomic_claim_count"],
            2,
        )

    def test_missing_parent_decomposition_falls_back_to_original_claim(self):
        model_output = {
            "answer_type": "grounded",
            "answer": "old",
            "claims": [
                {
                    "text": (
                        "The user worked with AFM images."
                    ),
                    "claim_type": "fact",
                    "citations": ["[S4]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        }

        def fake_client(
            *,
            policy,
            claims,
        ):
            return {
                "structured": {
                    "atomic_claims": []
                },
                "runtime": {},
            }

        output, _ = decompose_model_output_claims(
            model_output=model_output,
            policy=self.policy(),
            client=fake_client,
        )

        self.assertEqual(
            len(output["claims"]),
            1,
        )
        self.assertEqual(
            output["claims"][0]["text"],
            "The user worked with AFM images.",
        )


if __name__ == "__main__":
    unittest.main()
