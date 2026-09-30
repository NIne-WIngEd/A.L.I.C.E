from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_contracts import FormationEvidenceAnchor, FormationProposal, MemoryProposalBundle
from cognitive_kernel.formation_evaluation import assess_formation
from cognitive_kernel.formation_gold import (
    compile_formation_case, load_formation_gold, load_frozen_formation_gold,
)


CORPUS = Path(__file__).resolve().parents[1] / "fixtures/mfm/fictional_formation_gold_v1.json"
MANIFEST = CORPUS.with_suffix(".manifest.json")


class FormationGoldTests(unittest.TestCase):
    def test_scoped_defer_preserves_narrower_proposal(self):
        rows = json.loads(CORPUS.read_text())
        plan = dict(rows[2], case_id="aria-plan-scoped")
        plan["expected"] = [dict(plan["expected"][0],
                                 disposition_scope_ref="scheduled_plan")]
        plan["dispositions"] = [
            {"scope_ref": "outcome_claim", "action": "defer",
             "evidence_refs": ["aria-calendar"]},
            {"scope_ref": "scheduled_plan", "action": "propose",
             "evidence_refs": ["aria-calendar"]},
        ]
        compiled = compile_formation_case(plan)
        self.assertEqual(len(compiled.gold.expected), 1)
        self.assertEqual(compiled.gold.expected[0].kind, "goal")
        self.assertEqual(compiled.gold.expected_dispositions[0].action, "defer")
        self.assertEqual(compiled.gold.expected_dispositions[0].scope_ref, "outcome_claim")
        self.assertEqual(compiled.gold.expected[0].disposition_scope_ref, "scheduled_plan")
        output = MemoryProposalBundle(
            scope=compiled.gold.context.scope,
            authority_namespace_id=compiled.gold.context.authority_namespace_id,
            bundle_id="narrower-plan", experience_refs=compiled.gold.context.experience_refs,
            context_digest=compiled.gold.context.content_digest(),
            model_artifact_digest="f" * 64, inference_run_id="scoped-run",
            proposals=compiled.gold.expected,
            dispositions=compiled.gold.expected_dispositions,
        )
        assessed = assess_formation(compiled.gold, output)
        self.assertEqual((assessed.true_positives, assessed.disposition_true_positives), (1, 2))
        legacy = dict(rows[2], case_id="aria-plan-legacy", decision_action="propose",
                      decision_scope="scheduled_plan")
        self.assertEqual(compile_formation_case(legacy).gold.expected[0].disposition_scope_ref,
                         "scheduled_plan")

    def test_explicit_multiple_actions_and_deletion_target_refs(self):
        rows = json.loads(CORPUS.read_text())
        deletion = dict(rows[5], case_id="ben-delete-scoped",
                        deletion_target_refs=["old-travel-log"])
        deletion["dispositions"] = [
            {"scope_ref": "deletion_request", "action": "propose",
             "evidence_refs": ["ben-delete-event"], "target_refs": ["old-travel-log"]},
            {"scope_ref": "historical_reconstruction", "action": "abstain",
             "evidence_refs": ["ben-delete-event"]},
        ]
        compiled = compile_formation_case(deletion)
        self.assertEqual(compiled.gold.expected[0].target_refs, ("old-travel-log",))
        self.assertEqual(len(compiled.gold.expected_dispositions), 2)
        output = MemoryProposalBundle(
            scope=compiled.gold.context.scope,
            authority_namespace_id=compiled.gold.context.authority_namespace_id,
            bundle_id="deletion-scoped", experience_refs=compiled.gold.context.experience_refs,
            context_digest=compiled.gold.context.content_digest(),
            model_artifact_digest="f" * 64, inference_run_id="scoped-run",
            proposals=compiled.gold.expected, dispositions=compiled.gold.expected_dispositions,
        )
        self.assertEqual(assess_formation(compiled.gold, output).disposition_true_positives, 2)
        wrong_target = replace(compiled.gold.expected[0], target_refs=("other-log",))
        assessed = assess_formation(compiled.gold, replace(output, proposals=(wrong_target,)))
        self.assertEqual((assessed.true_positives, assessed.false_negatives), (0, 1))
        mixed = dict(deletion, decision_action="propose", decision_scope="formation_proposal")
        with self.assertRaisesRegex(CognitiveKernelContractError, "cannot mix"):
            compile_formation_case(mixed)

    def test_matching_words_with_wrong_source_span_do_not_score_as_gold(self):
        case = next(c for c in load_formation_gold(CORPUS)
                    if c.gold.case_id == "aria-change")
        original = case.gold.expected[0]
        changed = replace(original, anchors=(FormationEvidenceAnchor(
            original.evidence_refs[0], 0, 1),))
        bundle = MemoryProposalBundle(
            scope=case.gold.context.scope,
            authority_namespace_id=case.gold.context.authority_namespace_id,
            bundle_id="wrong-span", experience_refs=case.gold.context.experience_refs,
            context_digest=case.gold.context.content_digest(),
            model_artifact_digest="f" * 64, inference_run_id="span-run",
            proposals=(changed,),
        )
        result = assess_formation(case.gold, bundle)
        self.assertEqual((result.true_positives, result.false_positives,
                          result.false_negatives), (0, 1, 1))

    def test_independent_host_splits_and_provenance_validity(self):
        cases = load_frozen_formation_gold(MANIFEST)
        self.assertEqual(len(cases), 11)
        by_split = {}
        for case in cases:
            by_split.setdefault(case.split, set()).add(case.host_family)
            self.assertTrue(case.reason)
            self.assertTrue(case.gold.critical_forbidden)
            self.assertTrue(case.texts)
        self.assertEqual(set(by_split), {"train", "development", "challenge"})
        self.assertFalse(by_split["train"] & by_split["challenge"])
        self.assertFalse(by_split["development"] & by_split["challenge"])

    def test_critical_false_memory_is_visible_despite_other_success(self):
        case = next(c for c in load_formation_gold(CORPUS)
                    if c.gold.case_id == "ben-reconstruction")
        kind, domain, subject, value, status = case.gold.critical_forbidden[0]
        forbidden = FormationProposal(
            proposal_id="invented-history", kind=kind, domain=domain,
            subject_ref=subject, value_ref="forbidden-value", value_text=value,
            evidence_refs=(case.gold.context.evidence[0].ref_id,),
            anchors=(FormationEvidenceAnchor(
                case.gold.context.evidence[0].ref_id, 0,
                len(case.texts[0][1].encode("utf-8"))),),
            epistemic_status=status,
        )
        # The role gate must reject this outright. A generated rehearsal cannot
        # be cited as owner-attested source history, even if other rows are good.
        bundle = MemoryProposalBundle(
            scope=case.gold.context.scope,
            authority_namespace_id=case.gold.context.authority_namespace_id,
            bundle_id="bad-bundle", experience_refs=case.gold.context.experience_refs,
            context_digest=case.gold.context.content_digest(),
            model_artifact_digest="f" * 64, inference_run_id="bad-run",
            proposals=case.gold.expected + (forbidden,),
        )
        with self.assertRaisesRegex(CognitiveKernelContractError, "matching evidence"):
            assess_formation(case.gold, bundle)

    def test_split_leak_and_bad_label_rejected(self):
        rows = json.loads(CORPUS.read_text())
        leaked = dict(rows[0], case_id="leaked-quote", split="challenge")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leak.json"
            path.write_text(json.dumps(rows + [leaked]))
            with self.assertRaisesRegex(CognitiveKernelContractError, "leaks across splits"):
                load_formation_gold(path)
        bad = json.loads(CORPUS.read_text())[0]
        bad["expected"][0]["epistemic_status"] = "owner_statement"
        with self.assertRaisesRegex(CognitiveKernelContractError, "matching evidence"):
            compile_formation_case(bad)

    def test_changed_gold_bytes_cannot_pass_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / CORPUS.name).write_bytes(CORPUS.read_bytes() + b" ")
            (folder / MANIFEST.name).write_bytes(MANIFEST.read_bytes())
            with self.assertRaisesRegex(CognitiveKernelContractError, "frozen digest"):
                load_frozen_formation_gold(folder / MANIFEST.name)


if __name__ == "__main__":
    unittest.main()
