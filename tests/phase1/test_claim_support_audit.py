from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.claim_support_audit import (
    audit_claims,
    build_claim_audit_items,
    load_claim_support_audit_policy,
)


def context():
    return {
        "evidence": [
            {
                "citation": "[S1]",
                "source_content_sha256": "a",
                "context_text": (
                    "The owner's AFM project used a U-Net "
                    "for segmentation."
                ),
                "owner_relation": "owner_self_record",
                "owner_relation_confidence": "high",
            },
            {
                "citation": "[S2]",
                "source_content_sha256": "b",
                "context_text": (
                    "A separate unrelated CAD project."
                ),
                "owner_relation": "owner_self_record",
                "owner_relation_confidence": "high",
            },
        ]
    }


class ClaimSupportAuditTests(unittest.TestCase):
    def policy(self):
        return load_claim_support_audit_policy(
            ROOT
            / "policies"
            / "claim_support_audit_policy.json"
        )

    def test_only_cited_evidence_is_exposed_to_auditor(self):
        claims = [
            {
                "text": (
                    "The AFM project used a U-Net."
                ),
                "claim_type": "fact",
                "citations": ["[S1]"],
            }
        ]
        normalized, evidence = (
            build_claim_audit_items(
                claims=claims,
                context_package=context(),
            )
        )
        self.assertEqual(
            normalized[0]["citations"],
            ["[S1]"],
        )
        self.assertEqual(
            set(evidence),
            {"[S1]"},
        )

    def test_supported_claim_passes_strict_audit(self):
        def fake_client(
            policy,
            schema,
            system_prompt,
            user_prompt,
        ):
            return {
                "structured": {
                    "assessments": [
                        {
                            "claim_index": 1,
                            "verdict": "supported",
                            "confidence": 0.97,
                            "supporting_citations": [
                                "[S1]"
                            ],
                            "unsupported_aspects": [],
                            "rationale": (
                                "The cited evidence directly "
                                "states the U-Net use."
                            ),
                        }
                    ]
                },
                "runtime": {},
            }

        result = audit_claims(
            question=(
                "What AFM research did I do?"
            ),
            claims=[
                {
                    "text": (
                        "The AFM project used a U-Net."
                    ),
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            context_package=context(),
            policy=self.policy(),
            model_client=fake_client,
        )
        self.assertEqual(
            result["citation_support_rate"],
            1.0,
        )
        self.assertEqual(
            result[
                "high_confidence_support_rate"
            ],
            1.0,
        )
        self.assertEqual(
            result[
                "manual_review_required_claim_count"
            ],
            0,
        )

    def test_partial_claim_requires_manual_review(self):
        def fake_client(
            policy,
            schema,
            system_prompt,
            user_prompt,
        ):
            return {
                "structured": {
                    "assessments": [
                        {
                            "claim_index": 1,
                            "verdict": (
                                "partially_supported"
                            ),
                            "confidence": 0.92,
                            "supporting_citations": [
                                "[S1]"
                            ],
                            "unsupported_aspects": [
                                "No evidence for publication."
                            ],
                            "rationale": (
                                "U-Net use is supported, but "
                                "publication is not."
                            ),
                        }
                    ]
                },
                "runtime": {},
            }

        result = audit_claims(
            question=(
                "What AFM research did I do?"
            ),
            claims=[
                {
                    "text": (
                        "The AFM project used a U-Net "
                        "and was published."
                    ),
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            context_package=context(),
            policy=self.policy(),
            model_client=fake_client,
        )
        self.assertEqual(
            result[
                "partially_supported_claim_count"
            ],
            1,
        )
        self.assertEqual(
            result[
                "manual_review_required_claim_count"
            ],
            1,
        )
        self.assertEqual(
            result["citation_support_rate"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
