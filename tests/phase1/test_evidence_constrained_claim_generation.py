from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.evidence_claim_generation import (
    generate_evidence_constrained_claims,
    load_evidence_claim_generation_policy,
)


class EvidenceConstrainedClaimGenerationTests(unittest.TestCase):
    def policy(self):
        return load_evidence_claim_generation_policy(
            ROOT
            / "policies"
            / "evidence_claim_generation_policy.json"
        )

    def context(self):
        return {
            "query": (
                "What research experience do I have "
                "with AFM images?"
            ),
            "evidence": [
                {
                    "citation": "[S4]",
                    "owner_relation": (
                        "owner_self_record"
                    ),
                    "owner_relation_confidence": (
                        "high"
                    ),
                    "context_text": (
                        "The portfolio states that the "
                        "owner developed an AI-powered "
                        "platform integrating AFM analysis "
                        "pipelines and U-Net segmentation."
                    ),
                },
                {
                    "citation": "[S5]",
                    "owner_relation": "unknown",
                    "owner_relation_confidence": "none",
                    "context_text": (
                        "Unrelated third-party content."
                    ),
                },
            ],
        }

    def test_claims_are_directly_generated_with_valid_citations(self):
        def fake_client(
            *,
            policy,
            context_package,
        ):
            return {
                "structured": {
                    "claims": [
                        {
                            "text": (
                                "The user developed a "
                                "platform integrating AFM "
                                "analysis pipelines."
                            ),
                            "citations": ["[S4]"],
                        },
                        {
                            "text": (
                                "The user used U-Net "
                                "segmentation."
                            ),
                            "citations": ["[S4]"],
                        },
                    ]
                },
                "runtime": {},
            }

        claims, summary = (
            generate_evidence_constrained_claims(
                context_package=self.context(),
                policy=self.policy(),
                client=fake_client,
            )
        )

        self.assertEqual(
            len(claims),
            2,
        )
        self.assertTrue(
            all(
                claim["claim_type"] == "fact"
                for claim in claims
            )
        )
        self.assertTrue(
            all(
                claim["citations"] == ["[S4]"]
                for claim in claims
            )
        )
        self.assertEqual(
            summary["generated_claim_count"],
            2,
        )

    def test_invalid_citations_are_removed_and_citationless_claims_are_dropped(self):
        def fake_client(
            *,
            policy,
            context_package,
        ):
            return {
                "structured": {
                    "claims": [
                        {
                            "text": "Valid claim.",
                            "citations": [
                                "[S4]",
                                "[S999]",
                            ],
                        },
                        {
                            "text": "Invalid claim.",
                            "citations": [
                                "[S999]"
                            ],
                        },
                    ]
                },
                "runtime": {},
            }

        claims, _ = (
            generate_evidence_constrained_claims(
                context_package=self.context(),
                policy=self.policy(),
                client=fake_client,
            )
        )

        self.assertEqual(
            len(claims),
            1,
        )
        self.assertEqual(
            claims[0]["citations"],
            ["[S4]"],
        )


if __name__ == "__main__":
    unittest.main()
