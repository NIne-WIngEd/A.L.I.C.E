from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_candidate import evaluate_candidate, training_examples
from cognitive_kernel.formation_contracts import MemoryProposalBundle
from cognitive_kernel.formation_gold import load_frozen_formation_gold


MANIFEST = (Path(__file__).resolve().parents[1] /
            "fixtures/mfm/fictional_formation_gold_v1.manifest.json")


class OracleForHarnessOnly:
    """Test double for evaluation wiring; no inference or learned weights."""

    artifact_sha256 = "a" * 64

    def __init__(self, cases):
        self.by_context = {case.gold.context.content_digest(): case.gold.expected
                           for case in cases}

    def infer(self, *, context, opened_sources):
        if {ref for ref, _ in opened_sources} != {ref.ref_id for ref in context.evidence}:
            raise ValueError("test harness failed to deliver context evidence")
        return MemoryProposalBundle(
            scope=context.scope, authority_namespace_id=context.authority_namespace_id,
            bundle_id=f"bundle-{context.experience_refs[0]}",
            experience_refs=context.experience_refs,
            context_digest=context.content_digest(),
            model_artifact_digest=self.artifact_sha256,
            inference_run_id="harness-test-only",
            proposals=self.by_context[context.content_digest()],
        )


class CandidateHarnessTests(unittest.TestCase):
    def test_train_isolation_and_challenge_evaluation(self):
        cases = load_frozen_formation_gold(MANIFEST)
        self.assertEqual({c.host_family for c in training_examples(cases)},
                         {"fictional-aria"})
        result = evaluate_candidate(OracleForHarnessOnly(cases), cases, split="challenge")
        self.assertEqual(len(result.cases), 5)
        self.assertEqual(result.totals["false_negatives"], 0)
        self.assertEqual(result.critical_failures, ())

    def test_wrong_artifact_and_train_scoring_blocked(self):
        cases = load_frozen_formation_gold(MANIFEST)
        candidate = OracleForHarnessOnly(cases)
        candidate.artifact_sha256 = "b" * 64
        original = candidate.infer
        def wrong_model(*, context, opened_sources):
            return replace(original(context=context, opened_sources=opened_sources),
                           model_artifact_digest="a" * 64)
        candidate.infer = wrong_model
        with self.assertRaisesRegex(CognitiveKernelContractError, "different model"):
            evaluate_candidate(candidate, cases, split="challenge")
        with self.assertRaisesRegex(CognitiveKernelContractError, "development or challenge"):
            evaluate_candidate(candidate, cases, split="train")


if __name__ == "__main__":
    unittest.main()
