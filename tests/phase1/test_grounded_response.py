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
    verify_grounded_response_package,
)
from alice_vault.grounded_response_evaluation import (
    evaluate_grounded_responses,
)


def make_context() -> dict:
    return {
        "context_package_schema_version": 1,
        "package_id": "context-1",
        "pilot_name": "pilot-v1",
        "query": "Which project used image segmentation?",
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
                "retrieval_agreement": "lexical_and_semantic",
                "source_extraction_truncated": False,
                "context_text": (
                    "The AFM project used a U-Net to create "
                    "segmentation masks."
                ),
                "provenance": [
                    {
                        "filename": "afm.pdf",
                        "file_id": "a",
                        "original_relative_path": "research/afm.pdf",
                    }
                ],
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


def good_model_client(policy, schema, system_prompt, user_prompt):
    return {
        "structured": {
            "answer_type": "grounded",
            "answer": (
                "The AFM project used a U-Net for image "
                "segmentation. [S1]"
            ),
            "claims": [
                {
                    "text": (
                        "The AFM project used a U-Net "
                        "for image segmentation."
                    ),
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                }
            ],
            "uncertainty_notes": [],
            "contradiction_notes": [],
        },
        "ollama": {"model": "fake"},
    }


class GroundedResponseTests(unittest.TestCase):
    def test_generates_and_verifies_cited_response(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            vault.mkdir()
            context_path = vault / "context.json"
            context_path.write_text(
                json.dumps(make_context()),
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
                model_client=good_model_client,
            )
            self.assertTrue(
                result["summary"]["verified"]
            )
            self.assertEqual(
                result["summary"][
                    "claim_citation_coverage"
                ],
                1.0,
            )

            verification = verify_grounded_response_package(
                response_path=Path(
                    result["summary"][
                        "response_path"
                    ]
                ),
                context_package_path=context_path,
                policy_path=(
                    ROOT
                    / "policies"
                    / "grounded_response_policy.json"
                ),
            )
            self.assertTrue(
                verification[
                    "ready_for_conversation"
                ]
            )

    def test_invalid_citation_is_rejected(self):
        def bad_client(
            policy,
            schema,
            system_prompt,
            user_prompt,
        ):
            return {
                "structured": {
                    "answer_type": "grounded",
                    "answer": "It was project X. [S99]",
                    "claims": [
                        {
                            "text": "It was project X.",
                            "claim_type": "fact",
                            "citations": ["[S99]"],
                        }
                    ],
                    "uncertainty_notes": [],
                    "contradiction_notes": [],
                },
                "ollama": {},
            }

        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            vault.mkdir()
            context_path = vault / "context.json"
            context_path.write_text(
                json.dumps(make_context()),
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
                model_client=bad_client,
                save=False,
            )
            self.assertFalse(
                result["summary"]["verified"]
            )

    def test_evaluation_checks_expected_source_citation(self):
        def fake_context_search(**kwargs):
            return {
                "results": [
                    {
                        "rank": 1,
                        "source_content_sha256": "source-a",
                        "rrf_score": 0.03,
                        "lexical_rank": 1,
                        "semantic_rank": 1,
                        "chunk_id": "chunk-a",
                        "chunk_index": 0,
                        "family": "research_project",
                        "source_extraction_truncated": False,
                        "snippet": (
                            "The AFM project used a U-Net "
                            "for segmentation."
                        ),
                        "provenance": [
                            {
                                "file_id": "a",
                                "original_relative_path": "afm.pdf",
                                "filename": "afm.pdf",
                                "role": "primary",
                                "source_bucket": "research",
                                "year_hint": "2026",
                                "duplicate_control_group": "",
                                "known_contradiction_group": "",
                            }
                        ],
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            vault = base / "vault"
            vault.mkdir()
            (vault / "temporary").mkdir()
            benchmark = base / "benchmark.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "benchmark_id": "b1",
                        "cases": [
                            {
                                "query_id": "q1",
                                "question": (
                                    "Which project used segmentation?"
                                ),
                                "status": "approved",
                                "expected_source_sha256": [
                                    "source-a"
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_grounded_responses(
                vault_root=vault,
                benchmark_path=benchmark,
                response_policy_path=(
                    ROOT
                    / "policies"
                    / "grounded_response_policy.json"
                ),
                context_policy_path=(
                    ROOT
                    / "policies"
                    / "grounded_context_policy.json"
                ),
                context_search_fn=fake_context_search,
                response_model_client=good_model_client,
            )
            self.assertEqual(
                result["verified_response_rate"],
                1.0,
            )
            self.assertEqual(
                result[
                    "expected_source_citation_rate"
                ],
                1.0,
            )
            self.assertTrue(
                result["passes_all_thresholds"]
            )


if __name__ == "__main__":
    unittest.main()
