from __future__ import annotations

import unittest

from alice_vault.query_claim_relevance_gate import (
    filter_model_output_by_query_claim_relevance,
    load_query_claim_relevance_gate_policy,
)


class _FakeModel:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, show_progress_bar=False):
        return self.scores[: len(pairs)]


def _render(claims):
    return "\n".join(claim["text"] for claim in claims)


class QueryClaimRelevanceGateTests(unittest.TestCase):
    def test_policy_uses_frozen_holdout_validated_threshold(self):
        policy = load_query_claim_relevance_gate_policy()
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.frozen_threshold, -9.76161)
        self.assertEqual(policy.objective, "non_irrelevance_filter")
        self.assertFalse(policy.memory_write_allowed)
        self.assertFalse(policy.external_action_allowed)
        self.assertFalse(policy.tool_calling_allowed)
        self.assertFalse(policy.web_access_allowed)

    def test_irrelevant_claim_is_dropped_and_relevant_claim_is_kept(self):
        policy = load_query_claim_relevance_gate_policy()
        model_output = {
            "answer_type": "grounded",
            "answer": "old",
            "claims": [
                {"text": "Relevant claim", "claim_type": "fact", "citations": ["[S1]"]},
                {"text": "Irrelevant claim", "claim_type": "fact", "citations": ["[S2]"]},
            ],
        }
        context = {
            "query": "What research recognition did I receive?",
            "contradiction_groups": [],
        }
        output, summary = filter_model_output_by_query_claim_relevance(
            model_output=model_output,
            context_package=context,
            model=_FakeModel([-3.0, -11.0]),
            policy=policy,
            answer_renderer=_render,
        )
        self.assertEqual(len(output["claims"]), 1)
        self.assertEqual(output["claims"][0]["text"], "Relevant claim")
        self.assertEqual(summary["kept_claim_count"], 1)
        self.assertEqual(summary["dropped_claim_count"], 1)

    def test_all_irrelevant_claims_fall_back_to_insufficient_evidence(self):
        policy = load_query_claim_relevance_gate_policy()
        model_output = {
            "answer_type": "grounded",
            "answer": "old",
            "claims": [
                {
                    "text": "Unrelated diary claim",
                    "claim_type": "fact",
                    "citations": ["[S1]"],
                },
            ],
        }
        context = {
            "query": "What award did I receive?",
            "contradiction_groups": [],
        }
        output, summary = filter_model_output_by_query_claim_relevance(
            model_output=model_output,
            context_package=context,
            model=_FakeModel([-11.2]),
            policy=policy,
            answer_renderer=_render,
        )
        self.assertEqual(output["answer_type"], "insufficient_evidence")
        self.assertEqual(output["claims"], [])
        self.assertEqual(summary["dropped_claim_count"], 1)


if __name__ == "__main__":
    unittest.main()
