from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import (
    _normalize_model_output,
    _response_schema_for_context,
    load_grounded_response_policy,
    verify_grounded_response_data,
)


def context(with_contradiction: bool = False):
    package = {
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
    if with_contradiction:
        package["contradiction_groups"] = [
            {
                "label": "status-conflict",
                "citations": ["S1", "[S2]"],
                "unresolved": True,
                "resolution": None,
            }
        ]
    return package


class CitationContractTests(unittest.TestCase):
    def policy(self):
        return load_grounded_response_policy(
            ROOT / "policies" / "grounded_response_policy.json"
        )

    def test_bare_citation_ids_are_canonicalized(self):
        output = {
            "answer_type": "grounded",
            "answer": "The project used segmentation. S1",
            "claims": [
                {
                    "text": "The project used segmentation.",
                    "claim_type": "fact",
                    "citations": ["S1"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        }
        normalized = _normalize_model_output(
            output,
            context(),
        )
        self.assertEqual(
            normalized["claims"][0]["citations"],
            ["[S1]"],
        )
        self.assertIn("[S1]", normalized["answer"])
        result = verify_grounded_response_data(
            context_package=context(),
            model_output=normalized,
            policy=self.policy(),
        )
        self.assertTrue(result["verified"])
        self.assertEqual(
            result["cited_source_sha256"],
            ["source-a"],
        )

    def test_schema_enumerates_only_context_citations(self):
        schema = _response_schema_for_context(context())
        enum = schema["properties"]["claims"]["items"][
            "properties"
        ]["citations"]["items"]["enum"]
        self.assertEqual(enum, ["[S1]", "[S2]"])

    def test_unrelated_context_contradiction_does_not_force_failure(self):
        output = {
            "answer_type": "grounded",
            "answer": "The project used segmentation. [S1]",
            "claims": [
                {
                    "text": "The project used segmentation.",
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        }
        normalized = _normalize_model_output(
            output,
            context(with_contradiction=True),
        )
        self.assertEqual(
            normalized["contradiction_notes"][0]["label"],
            "status-conflict",
        )
        result = verify_grounded_response_data(
            context_package=context(with_contradiction=True),
            model_output=normalized,
            policy=self.policy(),
        )
        self.assertTrue(result["verified"])


if __name__ == "__main__":
    unittest.main()
