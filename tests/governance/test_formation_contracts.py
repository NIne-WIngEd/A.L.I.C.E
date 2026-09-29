from __future__ import annotations

from dataclasses import replace
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FormationContextPacket,
    FormationDisposition,
    FormationEvidenceAnchor,
    FormationEvidenceRef,
    FormationProposal,
    MemoryProposalBundle,
    validate_formation_binding,
    validate_formation_grounding,
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
            value_text="The owner requested a correction.",
            anchors=(FormationEvidenceAnchor("experience-1", locator="entire-source"),),
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
        self.assertEqual(alice_record["proposals"][0]["value_text"],
                         "The owner requested a correction.")

    def test_proposals_require_evidence_and_reject_authority(self) -> None:
        bundle = _bundle()
        with self.assertRaisesRegex(CognitiveKernelContractError, "evidence"):
            replace(bundle, proposals=(replace(bundle.proposals[0], evidence_refs=()),)).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "kind"):
            replace(bundle, proposals=(replace(bundle.proposals[0], kind="authoritative_write"),)).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "semantic value"):
            replace(bundle, proposals=(replace(bundle.proposals[0], value_text=""),)).validate()

    def test_duplicate_proposals_and_backward_time_fail(self) -> None:
        bundle = _bundle()
        with self.assertRaisesRegex(CognitiveKernelContractError, "duplicate"):
            replace(bundle, proposals=bundle.proposals * 2).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "valid_to"):
            replace(bundle, proposals=(replace(
                bundle.proposals[0], valid_to="2026-09-22T00:00:00Z"
            ),)).validate()

    def test_day_granularity_is_an_inclusive_date_marker_not_an_instant(self) -> None:
        bundle = _bundle()
        source = replace(self._context(bundle).evidence[0],
                         observed_at="2026-01-03T00:00:00.000000Z",
                         temporal_granularity="day")
        source.validate()
        self.assertEqual(source.metadata_record()["temporal_granularity"], "day")
        proposal = replace(bundle.proposals[0],
                           valid_from="2026-01-03T00:00:00.000000Z",
                           valid_to="2026-01-03T00:00:00.000000Z",
                           temporal_granularity="day")
        proposal.validate()
        self.assertEqual(proposal.record()["temporal_granularity"], "day")
        with self.assertRaisesRegex(CognitiveKernelContractError, "UTC date marker"):
            replace(source, observed_at="2026-01-03T12:00:00.000000Z").validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "UTC date marker"):
            replace(proposal, valid_to="2026-01-03T12:00:00.000000Z").validate()

        context = replace(self._context(bundle), evidence=(source,))
        gold = FormationGoldCase("day-marker", context, (proposal,))
        output = replace(bundle, context_digest=context.content_digest(),
                         proposals=(replace(proposal, temporal_granularity="instant"),))
        assessment = assess_formation(gold, output)
        self.assertEqual(assessment.true_positives, 0)
        self.assertEqual(assessment.false_positives, 1)
        self.assertEqual(assessment.false_negatives, 1)

    def test_binding_rejects_unseen_evidence_and_reclassified_external_text(self) -> None:
        original = _bundle()
        context = self._context(original)
        bundle = replace(original, context_digest=context.content_digest())
        validate_formation_binding(context, bundle)
        forged = replace(bundle, proposals=(replace(
            bundle.proposals[0], evidence_refs=("made-up-owner-statement",),
            anchors=(FormationEvidenceAnchor("made-up-owner-statement", locator="entire-source"),),
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
                                  evidence_refs=("experience-1", "outside-quote"),
                                  anchors=(FormationEvidenceAnchor("experience-1", locator="entire-source"),
                                           FormationEvidenceAnchor("outside-quote", locator="entire-source"))),))
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
                                 bad.value_text, bad.epistemic_status),),
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
                                anchors=(FormationEvidenceAnchor("historical-2", locator="entire-source"),),
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

    def test_scoped_defer_coexists_with_valid_narrower_proposal(self) -> None:
        original = _bundle()
        context = self._context(original)
        plan = replace(original.proposals[0], kind="goal", epistemic_status="observation",
                       value_text="A plan was scheduled, with no completed outcome established.",
                       disposition_scope_ref="scheduled_plan")
        defer = FormationDisposition("outcome_claim", "defer", ("experience-1",))
        propose_plan = FormationDisposition("scheduled_plan", "propose", ("experience-1",))
        retain = FormationDisposition("self_skill_promotion", "retain_raw", ("experience-1",))
        gold = FormationGoldCase("scope-specific-action", context, (plan,),
                                 expected_dispositions=(defer, propose_plan, retain))
        output = replace(original, context_digest=context.content_digest(), proposals=(plan,),
                         dispositions=(defer, propose_plan, retain))
        result = assess_formation(gold, output)
        self.assertEqual((result.true_positives, result.false_positives,
                          result.disposition_true_positives), (1, 0, 3))
        self.assertTrue(result.passes_critical_gate)
        premature = replace(output, dispositions=(replace(defer, action="propose"),
                                                propose_plan, retain))
        blocked = assess_formation(gold, premature)
        self.assertIn("premature_propose:outcome_claim", blocked.critical_failures)
        self.assertEqual(blocked.true_positives, 1)

    def test_target_references_and_disposition_evidence_are_bound(self) -> None:
        original = _bundle()
        context = self._context(original)
        correction = replace(original.proposals[0], target_refs=("older-claim",),
                             disposition_scope_ref="correction_scope")
        decision = FormationDisposition("correction_scope", "propose", ("experience-1",),
                                        ("older-claim",))
        output = replace(original, context_digest=context.content_digest(),
                         proposals=(correction,), dispositions=(decision,))
        validate_formation_binding(context, output)
        self.assertEqual(output.metadata_record()["proposals"][0]["target_refs"], ["older-claim"])
        self.assertEqual(output.metadata_record()["dispositions"][0]["target_refs"], ["older-claim"])
        wrong_source = replace(output, dispositions=(replace(
            decision, evidence_refs=("unopened-claim",)),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "disposition cites evidence absent"):
            validate_formation_binding(context, wrong_source)
        with self.assertRaisesRegex(CognitiveKernelContractError, "duplicate formation disposition scope"):
            replace(output, dispositions=(decision, decision)).validate()
        contradictory = replace(output, dispositions=(replace(decision, action="defer"),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "matching propose disposition"):
            validate_formation_binding(context, contradictory)
        unscoped = replace(output, proposals=(replace(correction, disposition_scope_ref=None),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "matching propose disposition"):
            validate_formation_binding(context, unscoped)

    def test_grounding_checks_exact_source_bytes_and_anchor_bounds(self) -> None:
        original = _bundle()
        text = b"Owner asked for a correction"
        from hashlib import sha256
        context = replace(self._context(original), evidence=(replace(
            self._context(original).evidence[0], modality="text",
            content_digest=sha256(text).hexdigest()),))
        proposal = replace(original.proposals[0], anchors=(
            FormationEvidenceAnchor("experience-1", 0, len(text)),))
        bundle = replace(original, context_digest=context.content_digest(),
                         proposals=(proposal,))
        validate_formation_grounding(context, bundle, (("experience-1", text),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "span exceeds"):
            validate_formation_grounding(context, replace(bundle, proposals=(replace(
                proposal, anchors=(FormationEvidenceAnchor("experience-1", 0, len(text) + 1),)),)),
                (("experience-1", text),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "digest mismatch"):
            validate_formation_grounding(context, bundle, (("experience-1", b"tampered"),))


if __name__ == "__main__":
    unittest.main()
