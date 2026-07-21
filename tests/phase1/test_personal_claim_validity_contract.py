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


class PersonalClaimValidityContractTests(unittest.TestCase):
    def policy(self):
        return load_evidence_claim_generation_policy(
            ROOT / "policies" / "evidence_claim_generation_policy.json"
        )

    def generate(self, *, query, evidence, claims):
        context = {"query": query, "evidence": evidence}

        def fake_client(*, policy, context_package):
            return {"structured": {"claims": claims}, "runtime": {}}

        result, _ = generate_evidence_constrained_claims(
            context_package=context,
            policy=self.policy(),
            client=fake_client,
        )
        return result

    def test_personal_query_rejects_unrelated_author_claim(self):
        claims = self.generate(
            query="What do my records say about my technical background?",
            evidence=[{
                "citation": "[S1]",
                "owner_relation": "unknown",
                "context_text": "The author believes textbooks should explain mysteries.",
            }],
            claims=[{
                "text": "The author believes that textbooks should explain mysteries and make the profound obvious.",
                "citations": ["[S1]"],
            }],
        )
        self.assertEqual(claims, [])

    def test_personal_query_allows_subject_neutral_claim_for_independent_validation(self):
        claims = self.generate(
            query="What research experience do I have with AFM images?",
            evidence=[{
                "citation": "[S1]",
                "owner_relation": "owner_self_record",
                "context_text": "The portfolio states that the owner developed an AFM analysis platform.",
            }],
            claims=[{
                "text": "Valid claim.",
                "citations": ["[S1]"],
            }],
        )
        self.assertEqual(len(claims), 1)

    def test_relative_time_claim_is_rejected(self):
        claims = self.generate(
            query="What happened in my work history?",
            evidence=[{
                "citation": "[S1]",
                "owner_relation": "owner_self_record",
                "context_text": "08-13-2025: Starbucks training finished today.",
            }],
            claims=[{
                "text": "The user's Starbucks training finished today.",
                "citations": ["[S1]"],
            }],
        )
        self.assertEqual(claims, [])

    def test_promotional_self_record_impact_is_rejected_but_concrete_action_kept(self):
        claims = self.generate(
            query="What mechanical design work have I done?",
            evidence=[{
                "citation": "[S1]",
                "owner_relation": "owner_self_record",
                "context_text": (
                    "Designed and refined precision parametric assemblies using SolidWorks xDesign. "
                    "Leverage computational modeling and intelligent design tools to accelerate R&D cycles."
                ),
            }],
            claims=[
                {
                    "text": "The user leveraged computational modeling and intelligent design tools to accelerate R&D cycles.",
                    "citations": ["[S1]"],
                },
                {
                    "text": "The user designed and refined precision parametric assemblies using SolidWorks xDesign.",
                    "citations": ["[S1]"],
                },
            ],
        )
        self.assertEqual(len(claims), 1)
        self.assertIn("SolidWorks xDesign", claims[0]["text"])


if __name__ == "__main__":
    unittest.main()
