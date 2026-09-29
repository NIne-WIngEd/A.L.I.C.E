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
