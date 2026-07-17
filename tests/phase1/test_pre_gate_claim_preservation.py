from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import (
    generate_grounded_response,
)


def context():
    return {
        "context_package_schema_version": 1,
        "package_id": "context-pre-gate",
        "pilot_name": "pilot-v1",
        "query": "What AFM work did I do?",
        "query_sha256": "q" * 64,
        "package_fingerprint": "p" * 64,
        "evidence": [
            {
                "citation_id": "S1",
                "citation": "[S1]",
                "source_content_sha256": "source-a",
                "chunk_id": "chunk-a",
                "chunk_index": 0,
                "family": "research_project",
                "retrieval_agreement": (
                    "lexical_and_semantic"
                ),
                "source_extraction_truncated": False,
                "context_text": (
                    "The AFM project used a U-Net "
                    "for segmentation."
                ),
                "provenance": [],
                "contradiction_labels": [],
            }
        ],
        "contradiction_groups": [],
        "guardrails": {
            "memory_write_allowed": False,
            "answer_generation_allowed": False,
            "external_action_allowed": False,
            "contradictions_auto_resolved": False,
            "source_text_is_untrusted_data": True,
            "private_output_only": True,
        },
    }


def client(
    policy,
    schema,
    system_prompt,
    user_prompt,
):
    return {
        "structured": {
            "answer_type": "grounded",
            "answer": (
                "The AFM project used a U-Net. [S1]"
            ),
            "claims": [
                {
                    "text": (
                        "The AFM project used a U-Net."
                    ),
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        },
        "ollama": {},
    }


class PreGateClaimPreservationTests(unittest.TestCase):
    def test_pre_gate_output_is_preserved_privately(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            vault.mkdir()
            context_path = vault / "context.json"
            context_path.write_text(
                json.dumps(context()),
                encoding="utf-8",
            )

            result = generate_grounded_response(
                vault_root=vault,
                context_package_path=context_path,
                policy_path=(
                    ROOT
                    / "policies"
                    / "grounded_response_policy.json"
                ),
                model_client=client,
                save=False,
            )

            package = result[
                "response_package"
            ]
            self.assertIn(
                "pre_gate_model_output",
                package,
            )
            self.assertEqual(
                len(
                    package[
                        "pre_gate_model_output"
                    ]["claims"]
                ),
                1,
            )
            self.assertEqual(
                result["summary"][
                    "pre_gate_claim_count"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
