from __future__ import annotations

import unittest

from alice_vault.evidence_claim_generation import (
    _select_claim_generation_context,
)


class ClaimGenerationSourcePrefilterTests(unittest.TestCase):
    def evidence(
        self,
        citation,
        relation,
    ):
        return {
            "citation": citation,
            "owner_relation": relation,
            "context_text": "example",
        }

    def test_personal_query_prefers_owner_self_records(self):
        context = {
            "query": "What research experience do I have?",
            "evidence": [
                self.evidence(
                    "[S1]",
                    "unknown",
                ),
                self.evidence(
                    "[S2]",
                    "owner_self_record",
                ),
                self.evidence(
                    "[S3]",
                    "owner_self_record",
                ),
            ],
        }

        selected, summary = (
            _select_claim_generation_context(
                context
            )
        )

        self.assertEqual(
            [
                item["citation"]
                for item in selected["evidence"]
            ],
            [
                "[S2]",
                "[S3]",
            ],
        )
        self.assertTrue(
            summary[
                "source_prefilter_applied"
            ]
        )

    def test_non_personal_query_keeps_all_sources(self):
        context = {
            "query": "What does this document describe?",
            "evidence": [
                self.evidence(
                    "[S1]",
                    "unknown",
                ),
                self.evidence(
                    "[S2]",
                    "owner_self_record",
                ),
            ],
        }

        selected, summary = (
            _select_claim_generation_context(
                context
            )
        )

        self.assertIs(
            selected,
            context,
        )
        self.assertFalse(
            summary[
                "source_prefilter_applied"
            ]
        )

    def test_personal_query_falls_back_when_no_self_record_exists(self):
        context = {
            "query": "What happened in my records?",
            "evidence": [
                self.evidence(
                    "[S1]",
                    "unknown",
                ),
            ],
        }

        selected, summary = (
            _select_claim_generation_context(
                context
            )
        )

        self.assertIs(
            selected,
            context,
        )
        self.assertEqual(
            summary["strategy"],
            "fallback_all_sources_no_owner_self_record",
        )


if __name__ == "__main__":
    unittest.main()
