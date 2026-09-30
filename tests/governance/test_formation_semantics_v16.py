from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FORMATION_SCHEMA_VERSION, FormationContextPacket, FormationEvidenceAnchor,
    FormationEvidenceRef, FormationProposal, MemoryProposalBundle,
)
from cognitive_kernel.formation_learning import output_record
from cognitive_kernel.formation_evaluation import FormationGoldCase, assess_formation
from cognitive_kernel.formation_semantics_v16 import (
    ADJUDICATION_DIMENSIONS, EpisodeSemantics, FormationContextV16,
    FormationGoldCaseV16, FormationProposalV16, MemoryProposalBundleV16,
    RegisteredFormationTarget, RegisteredSourceSensitivity, assess_formation_v16,
    bundle_v16_from_output, context_v16_from_record, require_full_role_structural_coverage_v16,
    validate_formation_grounding_v16,
)


SOURCE = b"Owner and Alice planned a new project together."
TIME = "2026-09-30T00:00:00.000000Z"


def fixture():
    scope = ProductHostScope.create(product_id="alice", host_instance_id="owner-one",
                                    schema_version="1.0.0", encryption_domain="owner-private")
    evidence = FormationEvidenceRef(
        ref_id="source-1", scope=scope, authority_namespace_id="owner-namespace",
        content_digest=sha256(SOURCE).hexdigest(), role="authenticated_owner_statement",
        modality="text", speaker_ref="owner", subject_ref="owner", observed_at=TIME,
    )
    base_context = FormationContextPacket(scope, "owner-namespace", ("source-1",),
                                          (evidence,))
    targets = tuple(RegisteredFormationTarget(
        ref_id=ref, kind=kind, scope=scope,
        authority_namespace_id="owner-namespace", supporting_evidence_refs=("source-1",))
        for ref, kind in (("owner", "entity"), ("scene-home", "scene"),
                          ("mission-7", "mission"), ("workspace-9", "workspace")))
    context = FormationContextV16(
        base_context, (RegisteredSourceSensitivity("source-1", "private"),), targets)
    base_proposal = FormationProposal(
        proposal_id="episode-1", kind="episode", domain="host", subject_ref="owner",
        value_ref="private-value", value_text="Owner and Alice planned a project.",
        evidence_refs=("source-1",),
        anchors=(FormationEvidenceAnchor("source-1", 0, len(SOURCE)),),
        epistemic_status="owner_statement", valid_from=TIME,
    )
    proposal = FormationProposalV16(
        base_proposal, "private", EpisodeSemantics(TIME, None, ("owner",),
                                                    ("scene-home",)),
        mission_target_refs=("mission-7",), workspace_target_refs=("workspace-9",))
    base_bundle = MemoryProposalBundle(
        scope, "owner-namespace", "bundle-1", ("source-1",),
        base_context.content_digest(), "a" * 64, "run-1", (base_proposal,))
    bundle = MemoryProposalBundleV16(base_bundle, context.content_digest(), (proposal,))
    return context, bundle


class FormationSemanticsV16Tests(unittest.TestCase):
    def test_exact_v16_roundtrip_preserves_v15_bytes_and_hashes(self):
        context, bundle = fixture()
        old_context_digest = context.base.content_digest()
        old_output = output_record(bundle.base)
        self.assertEqual(FORMATION_SCHEMA_VERSION, "1.5.0")
        parsed_context = context_v16_from_record(context.record())
        parsed_bundle = bundle_v16_from_output(
            parsed_context, bundle.output_record(), artifact_sha256="a" * 64,
            inference_run_id="run-1")
        self.assertEqual(parsed_bundle.output_record(), bundle.output_record())
        self.assertEqual(parsed_context.content_digest(), context.content_digest())
        self.assertEqual(context.base.content_digest(), old_context_digest)
        self.assertEqual(output_record(bundle.base), old_output)
        validate_formation_grounding_v16(parsed_context, parsed_bundle,
                                         (("source-1", SOURCE),))

    def test_model_hint_cannot_lower_registered_source_policy(self):
        context, bundle = fixture()
        low = replace(bundle.proposals[0], sensitivity_hint="internal")
        with self.assertRaisesRegex(CognitiveKernelContractError, "lowers source policy"):
            validate_formation_grounding_v16(
                context, replace(bundle, base=replace(bundle.base, proposals=(low.base,)),
                                 proposals=(low,)), (("source-1", SOURCE),))
        no_hint = replace(bundle.proposals[0], sensitivity_hint=None)
        validate_formation_grounding_v16(
            context, replace(bundle, proposals=(no_hint,)), (("source-1", SOURCE),))
        self.assertEqual(context.source_sensitivities[0].minimum, "private")

    def test_registered_targets_fail_cross_scope_and_ungrounded_links(self):
        context, bundle = fixture()
        foreign = ProductHostScope.create(product_id="alice", host_instance_id="other-owner",
                                          schema_version="1.0.0", encryption_domain="other-private")
        with self.assertRaisesRegex(CognitiveKernelContractError, "foreign-scope"):
            replace(context, targets=(replace(context.targets[0], scope=foreign),
                                      *context.targets[1:])).validate()
        unknown_scene = replace(bundle.proposals[0],
                                episode=replace(bundle.proposals[0].episode,
                                                scene_refs=("invented-scene",)))
        with self.assertRaisesRegex(CognitiveKernelContractError, "unregistered"):
            validate_formation_grounding_v16(
                context, replace(bundle, proposals=(unknown_scene,)),
                (("source-1", SOURCE),))
        other = b"A distinct outside source"
        second = replace(context.base.evidence[0], ref_id="source-2",
                         content_digest=sha256(other).hexdigest(), role="outside_source")
        extended = replace(context,
                           base=replace(context.base, evidence=(*context.base.evidence, second)),
                           source_sensitivities=(*context.source_sensitivities,
                                                 RegisteredSourceSensitivity("source-2", "public")),
                           targets=tuple(replace(t, supporting_evidence_refs=("source-2",))
                                         if t.ref_id == "scene-home" else t for t in context.targets))
        extended_bundle = replace(
            bundle, context_v16_digest=extended.content_digest(),
            base=replace(bundle.base, context_digest=extended.base.content_digest()))
        with self.assertRaisesRegex(CognitiveKernelContractError, "no cited supporting evidence"):
            validate_formation_grounding_v16(
                extended, extended_bundle, (("source-1", SOURCE), ("source-2", other)))

    def test_event_relationship_and_mission_semantics_are_explicit(self):
        context, bundle = fixture()
        proposal = bundle.proposals[0]
        with self.assertRaisesRegex(CognitiveKernelContractError, "requires event semantics"):
            replace(proposal, episode=None).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "episode end precedes start"):
            replace(proposal.episode, event_end="2026-09-29T00:00:00.000000Z").validate()
        relation_base = replace(proposal.base, proposal_id="relationship-1", kind="relationship",
                                domain="relationship", subject_ref="alice-self")
        relation = FormationProposalV16(relation_base, "private",
                                        relationship_counterpart_ref="owner")
        extended = replace(bundle, base=replace(bundle.base, proposals=(relation_base,)),
                           proposals=(relation,))
        validate_formation_grounding_v16(context, extended, (("source-1", SOURCE),))
        with self.assertRaisesRegex(CognitiveKernelContractError, "needs counterpart"):
            replace(relation, relationship_counterpart_ref=None).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "relationship domain"):
            replace(relation, base=replace(relation_base, domain="host"),
                    relationship_counterpart_ref=None).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "wrong kind"):
            bad = replace(proposal, mission_target_refs=("scene-home",))
            validate_formation_grounding_v16(
                context, replace(bundle, proposals=(bad,)), (("source-1", SOURCE),))

    def test_v16_evaluator_scores_link_change_and_requires_adjudication(self):
        context, bundle = fixture()
        gold = FormationGoldCaseV16(
            "case-1", context, bundle.proposals, (), (), ADJUDICATION_DIMENSIONS,
            ("reviewer-a", "reviewer-b"), (("source-1", SOURCE),))
        self.assertEqual(assess_formation_v16(gold, bundle).true_positives, 1)
        equivalent = replace(bundle.proposals[0], base=replace(
            bundle.proposals[0].base, proposal_id="generated-id", value_ref="new-local-ref"))
        equivalent_output = replace(bundle, base=replace(bundle.base,
            proposals=(equivalent.base,)), proposals=(equivalent,))
        self.assertEqual(assess_formation_v16(gold, equivalent_output).true_positives, 1)
        alt = replace(bundle.proposals[0], mission_target_refs=())
        report = assess_formation_v16(gold, replace(bundle, proposals=(alt,)))
        self.assertEqual((report.true_positives, report.false_positives,
                          report.false_negatives), (0, 1, 1))
        with self.assertRaisesRegex(CognitiveKernelContractError, "unadjudicated"):
            assess_formation_v16(replace(gold, adjudicated_dimensions=frozenset()), bundle)
        with self.assertRaisesRegex(CognitiveKernelContractError, "reviewer"):
            replace(gold, reviewer_refs=("reviewer-a",)).validate()

    def test_old_corpus_is_not_admitted_as_full_role_fit(self):
        with self.assertRaisesRegex(CognitiveKernelContractError, "historical v1.5"):
            require_full_role_structural_coverage_v16(
                corpus_schema="mfm-training-mixture-v1",
                split_roster={"train": 49819}, observed_label_counts={},
                sealed_final_sha256=None, independent_adjudication_receipt_sha256=None,
                authenticated_rights_receipt_sha256=None)
        with self.assertRaisesRegex(CognitiveKernelContractError, "observed full-role"):
            require_full_role_structural_coverage_v16(
                corpus_schema="mfm-full-role-corpus-v1.6",
                split_roster={"train": 1, "development": 1, "final": 1},
                observed_label_counts={s: {"sensitivity": 1} for s in
                                       ("train", "development", "final")},
                sealed_final_sha256="a" * 64,
                independent_adjudication_receipt_sha256="b" * 64,
                authenticated_rights_receipt_sha256="c" * 64)

    def test_parser_rejects_unknown_fields_and_legacy_output(self):
        context, bundle = fixture()
        with self.assertRaisesRegex(CognitiveKernelContractError, "v1.5 evaluator only"):
            assess_formation(FormationGoldCase("old-gold", context.base, ()), bundle)
        with self.assertRaisesRegex(CognitiveKernelContractError, "v1.5 output serializer only"):
            output_record(bundle)
        with self.assertRaisesRegex(CognitiveKernelContractError, "unsupported v1.6 output"):
            bundle_v16_from_output(context, output_record(bundle.base),
                                   artifact_sha256="a" * 64, inference_run_id="run-1")
        row = bundle.output_record()
        row["proposals"][0]["unexpected"] = "foreign"
        with self.assertRaisesRegex(CognitiveKernelContractError, "unsupported fields"):
            bundle_v16_from_output(context, row, artifact_sha256="a" * 64,
                                   inference_run_id="run-1")


if __name__ == "__main__":
    unittest.main()
