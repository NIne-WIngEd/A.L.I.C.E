from __future__ import annotations

from dataclasses import replace
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FormationContextPacket,
    FormationEvidenceRef,
    FormationProposal,
    MemoryProposalBundle,
    validate_formation_binding,
)
from cognitive_kernel.formation_evaluation import FormationGoldCase, assess_formation


def _bundle(product: str = "alice") -> MemoryProposalBundle:
    return MemoryProposalBundle(
        scope=ProductHostScope.create(
            product_id=product,
            host_instance_id="owner-instance",
            schema_version="1.0.0",
            encryption_domain="private-owner",
        ),
        authority_namespace_id="owner-namespace",
        bundle_id="bundle-1",
        experience_refs=("experience-1",),
        context_digest="a" * 64,
        model_artifact_digest="b" * 64,
        inference_run_id="run-1",
        proposals=(FormationProposal(
            proposal_id="proposal-1",
            kind="correction_request",
            domain="host",
            subject_ref="entity-1",
            value_ref="private-value-1",
            evidence_refs=("experience-1",),
            valid_from="2026-09-23T00:00:00Z",
        ),),
    )


class FormationContractsTests(unittest.TestCase):
    def _context(self, bundle: MemoryProposalBundle) -> FormationContextPacket:
        return FormationContextPacket(
            scope=bundle.scope,
            authority_namespace_id=bundle.authority_namespace_id,
            experience_refs=bundle.experience_refs,
            evidence=(FormationEvidenceRef(
                ref_id="experience-1", scope=bundle.scope,
                authority_namespace_id=bundle.authority_namespace_id,
                content_digest="c" * 64,
                role="authenticated_owner_statement", modality="audio",
                subject_ref="entity-1",
            ),),
        )

    def test_bundle_is_host_scoped_and_backend_neutral(self) -> None:
        alice = _bundle()
        friday = _bundle("friday")
        alice_record = alice.metadata_record()
        friday_record = friday.metadata_record()
        self.assertNotEqual(alice.content_digest(), friday.content_digest())
        self.assertEqual(alice_record["scope"]["product_id"], "alice")
        self.assertEqual(friday_record["scope"]["product_id"], "friday")
        self.assertNotIn("backend", alice_record)
        self.assertNotIn("adjudication", alice_record)
        self.assertEqual(alice_record["proposals"][0]["kind"], "correction_request")

    def test_proposals_require_evidence_and_reject_authority(self) -> None:
        bundle = _bundle()
        with self.assertRaisesRegex(CognitiveKernelContractError, "evidence"):
            replace(bundle, proposals=(replace(bundle.proposals[0], evidence_refs=()),)).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "kind"):
            replace(bundle, proposals=(replace(bundle.proposals[0], kind="authoritative_write"),)).validate()

    def test_duplicate_proposals_and_backward_time_fail(self) -> None:
        bundle = _bundle()
        with self.assertRaisesRegex(CognitiveKernelContractError, "duplicate"):
            replace(bundle, proposals=bundle.proposals * 2).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "valid_to"):
            replace(bundle, proposals=(replace(
                bundle.proposals[0], valid_to="2026-09-22T00:00:00Z"
            ),)).validate()

    def test_binding_rejects_unseen_evidence_and_reclassified_external_text(self) -> None:
        original = _bundle()
        context = self._context(original)
        bundle = replace(original, context_digest=context.content_digest())
        validate_formation_binding(context, bundle)
        forged = replace(bundle, proposals=(replace(
            bundle.proposals[0], evidence_refs=("made-up-owner-statement",),
        ),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "absent"):
            validate_formation_binding(context, forged)
        external = replace(context, evidence=(replace(
            context.evidence[0], role="outside_source"
        ),))
        falsely_owner = replace(bundle, context_digest=external.content_digest(), proposals=(replace(
            bundle.proposals[0], epistemic_status="owner_statement"
        ),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "matching evidence"):
            validate_formation_binding(external, falsely_owner)
        mixed = replace(context, evidence=context.evidence + (replace(
            context.evidence[0], ref_id="outside-quote", role="outside_source"
        ),))
        mixed_claim = replace(bundle, context_digest=mixed.content_digest(),
                              proposals=(replace(bundle.proposals[0],
                                  epistemic_status="owner_statement",
                                  evidence_refs=("experience-1", "outside-quote")),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "matching evidence"):
            validate_formation_binding(mixed, mixed_claim)

    def test_binding_rejects_cross_host_and_changed_context(self) -> None:
        original = _bundle()
        context = self._context(original)
        bundle = replace(original, context_digest=context.content_digest())
        foreign_scope = _bundle("friday").scope
        injected = replace(context, evidence=(replace(
            context.evidence[0], scope=foreign_scope
        ),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "foreign-scope"):
            injected.validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "digest"):
            validate_formation_binding(context, replace(bundle, context_digest="d" * 64))

    def test_outcomes_and_relationship_development_remain_distinct(self) -> None:
        original = _bundle()
        context = self._context(original)
        outcome = replace(original.proposals[0], kind="outcome", domain="mission",
                          epistemic_status="observation")
        relationship = replace(original.proposals[0], proposal_id="norm-2",
                               kind="relationship_norm", domain="relationship")
        bundle = replace(original, context_digest=context.content_digest(),
                         proposals=(outcome, relationship))
        validate_formation_binding(context, bundle)
        self.assertNotEqual(bundle.proposals[0].domain, bundle.proposals[1].domain)

    def test_gold_assessment_exposes_critical_errors_and_wrong_lineage(self) -> None:
        original = _bundle()
        context = self._context(original)
        correct = replace(original.proposals[0], epistemic_status="owner_statement")
        bad = replace(correct, proposal_id="p-2", domain="source_person",
                      epistemic_status="inference")
        gold = FormationGoldCase(
            case_id="source-host-boundary",
            context=context,
            expected=(correct,),
            critical_forbidden=((bad.kind, bad.domain, bad.subject_ref,
                                 bad.value_ref, bad.epistemic_status),),
        )
        output = replace(original, context_digest=context.content_digest(),
                         proposals=(correct, bad))
        report = assess_formation(gold, output)
        self.assertEqual((report.true_positives, report.false_positives), (1, 1))
        self.assertFalse(report.passes_critical_gate)

        other_ref = replace(context.evidence[0], ref_id="historical-2",
                            role="historical_experience")
        extended = replace(context, evidence=context.evidence + (other_ref,))
        wrong_lineage = replace(correct, evidence_refs=("historical-2",),
                                epistemic_status="inference")
        inference_gold = replace(gold, context=extended,
                                 expected=(replace(correct, epistemic_status="inference"),),
                                 critical_forbidden=())
        wrong_output = replace(original, context_digest=extended.content_digest(),
                               proposals=(wrong_lineage,))
        lineage_report = assess_formation(inference_gold, wrong_output)
        self.assertEqual((lineage_report.true_positives, lineage_report.false_negatives), (0, 1))
        stale = replace(original, context_digest=context.content_digest(),
                        proposals=(replace(correct, valid_from="2025-09-23T00:00:00Z"),))
        temporal_report = assess_formation(gold, stale)
        self.assertEqual((temporal_report.true_positives, temporal_report.false_negatives), (0, 1))

    def test_abstention_can_be_the_gold_behavior(self) -> None:
        original = _bundle()
        context = self._context(original)
        empty = replace(original, context_digest=context.content_digest(), proposals=())
        gold = FormationGoldCase(case_id="no-supported-personal-claim",
                                 context=context, expected=())
        report = assess_formation(gold, empty)
        self.assertEqual((report.true_positives, report.false_positives,
                          report.false_negatives), (0, 0, 0))
        self.assertTrue(report.passes_critical_gate)


if __name__ == "__main__":
    unittest.main()
