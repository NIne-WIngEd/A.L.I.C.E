"""Synthetic causal wiring check from v1.6 formation to the real P2 gate.

This is an integration regression, not a trained MFM or native judgment test.
The P2 candidate authority is the currently callable promotion boundary. Its
Rayan-specific schema is not a general Fable host authority implementation.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from alice_memory.candidate_assessment import (
    MemoryCandidateAssessmentAuthorization,
    assess_memory_candidate,
)
from alice_memory.deletion import (
    ORDINARY_MEMORY_DELETION_SCOPE,
    MemoryDeletionAuthorization,
    MemoryDeletionRequestAuthorization,
    delete_memory,
    request_memory_deletion,
)
from alice_memory.formation import (
    MemoryCandidateCreateRequest,
    MemoryCandidateWriteAuthorization,
    propose_memory_candidate,
)
from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.promotion import (
    MemoryCandidatePromotionAuthorization,
    MemoryCandidatePromotionAuthorizationError,
    promote_memory_candidate,
)
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import MemoryRetrievalAuthorization, MemorySearchRequest
from alice_memory.service import MemoryContentAccessAuthorization, load_memory_content
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.transition_promotion import (
    MemoryCandidateTransitionAuthorization,
    MemoryCandidateTransitionAuthorizationError,
    promote_memory_candidate_with_transition,
)
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FormationContextPacket,
    FormationEvidenceAnchor,
    FormationEvidenceRef,
    FormationProposal,
    MemoryProposalBundle,
)
from cognitive_kernel.formation_semantics_v16 import (
    FormationContextV16,
    FormationProposalV16,
    MemoryProposalBundleV16,
    RegisteredSourceSensitivity,
    validate_formation_grounding_v16,
)

T0 = "2026-09-30T15:00:00.000000Z"
T1 = "2026-09-30T16:00:00.000000Z"
T2 = "2026-09-30T17:00:00.000000Z"
QUERY = "desktop theme"
MEMORY_KEY = "host.desktop.theme"


def _formation(theme: str, time: str, *, version: int):
    # Invented owner statements; none assert a fact about the actual Rayan.
    source = (f"For this fictional case, I prefer {theme} theme on my desktop."
              if version == 1 else
              f"Correction: I now prefer {theme} theme on my desktop.").encode()
    scope = ProductHostScope.create(product_id="alice", host_instance_id="fictional-rayan",
                                    schema_version="1.0.0", encryption_domain="synthetic-private")
    ref = f"synthetic-source-{version}"
    evidence = FormationEvidenceRef(
        ref_id=ref, scope=scope, authority_namespace_id="fictional-owner",
        content_digest=sha256(source).hexdigest(), role="authenticated_owner_statement",
        modality="text", subject_ref="fictional-rayan", speaker_ref="fictional-rayan",
        observed_at=time,
    )
    base_context = FormationContextPacket(scope, "fictional-owner", (ref,), (evidence,))
    context = FormationContextV16(
        base_context, (RegisteredSourceSensitivity(ref, "private"),), ())
    base_proposal = FormationProposal(
        proposal_id=f"synthetic-preference-{version}", kind="preference", domain="host",
        subject_ref="fictional-rayan", value_ref=f"theme-{theme}-{version}",
        value_text=f"The fictional owner prefers {theme} theme on desktop.",
        evidence_refs=(ref,), anchors=(FormationEvidenceAnchor(ref, 0, len(source)),),
        epistemic_status="owner_statement", valid_from=time, confidence=0.95,
        target_refs=(("synthetic-preference-1",) if version == 2 else ()),
    )
    proposal = FormationProposalV16(base_proposal, sensitivity_hint="private")
    base_bundle = MemoryProposalBundle(
        scope, "fictional-owner", f"synthetic-bundle-{version}", (ref,),
        base_context.content_digest(), "a" * 64, f"synthetic-gold-injection-{version}",
        (base_proposal,),
    )
    bundle = MemoryProposalBundleV16(base_bundle, context.content_digest(), (proposal,))
    validate_formation_grounding_v16(context, bundle, ((ref, source),))
    return context, bundle, source


def _stage(connection, context, bundle, source, *, version: int):
    validate_formation_grounding_v16(
        context, bundle, ((context.base.experience_refs[0], source),))
    proposal = bundle.proposals[0].base
    if (proposal.domain != "host" or proposal.kind != "preference" or
            proposal.epistemic_status != "owner_statement" or
            proposal.subject_ref != context.base.scope.host_instance_id or
            context.base.evidence[0].speaker_ref != proposal.subject_ref or
            context.source_sensitivities[0].minimum != "private"):
        raise ValueError("synthetic P2 bridge only admits owner-stated host preferences")
    candidate_id = f"synthetic-candidate-{version}"
    candidate = propose_memory_candidate(
        connection,
        request=MemoryCandidateCreateRequest(
            candidate_id=candidate_id, content=proposal.value_text,
            memory_key=MEMORY_KEY, category="profile", knowledge_status="rayan_statement",
            confidence=proposal.confidence, data_classification="PRIVATE",
            recorded_at=proposal.valid_from,
            valid_from=proposal.valid_from,
            sources=(MemorySourceSpec(
                source_type="rayan_direct_statement",
                source_ref=f"synthetic:v16:{context.base.evidence[0].ref_id}",
                source_content_sha256=sha256(source).hexdigest(),
                support_relation="supports",
            ),),
            origin="model_proposed", rayan_confirmed=False,
            policy_version="synthetic-v16-intervention",
            model="untrained-gold-injection", model_version="synthetic-1",
            prompt_version="none", run_id=bundle.base.inference_run_id,
        ),
        authorization=MemoryCandidateWriteAuthorization(
            actor="synthetic-formation-injector", allowed=True),
        proposed_at=proposal.valid_from,
    )
    assessment = assess_memory_candidate(
        connection, candidate_id=candidate.candidate_id,
        authorization=MemoryCandidateAssessmentAuthorization(
            actor="separate-synthetic-assessor", allowed=True),
        assessed_at=proposal.valid_from,
    )
    assert assessment.outcome == "review_required"
    assert "model_proposals_require_user_review" in assessment.reason_codes
    return candidate_id, assessment


def _fixed_decision_probe(connection, vault: Path, repository: Path) -> tuple[str, tuple[str, ...]]:
    """A constant question and fixed rule; never pass this as native judgment."""
    result = search_memories(
        connection, vault,
        request=MemorySearchRequest(query=QUERY, memory_key=MEMORY_KEY),
        authorization=MemoryRetrievalAuthorization(
            actor="synthetic-judge-probe", allowed=True,
            purpose="evaluate authoritative retrieval effect", max_classification="PRIVATE"),
        repository_root=repository,
    )
    contents = tuple(load_memory_content(
        connection, memory_id=row.memory_id,
        authorization=MemoryContentAccessAuthorization(
            actor="synthetic-judge-probe", allowed=True,
            reason="evaluate authoritative retrieval effect"),
    ) for row in result.results)
    if len(contents) == 1 and contents[0].endswith("dark theme on desktop."):
        return "choose-dark", tuple(row.memory_id for row in result.results)
    if len(contents) == 1 and contents[0].endswith("light theme on desktop."):
        return "choose-light", tuple(row.memory_id for row in result.results)
    return "ask-owner", tuple(row.memory_id for row in result.results)


def test_versioned_formation_gate_retrieval_and_fixed_decision_intervention(tmp_path: Path):
    repository, vault = tmp_path / "repository-sentinel", tmp_path / "private-vault"
    repository.mkdir()
    vault.mkdir()
    context1, bundle1, source1 = _formation("dark", T0, version=1)
    context2, bundle2, source2 = _formation("light", T1, version=2)
    with open_memory_store(vault, repository_root=repository) as connection:
        def rebuild():
            build_memory_lexical_index(
                connection, vault, repository_root=repository, built_at=T2)

        rebuild()
        before = _fixed_decision_probe(connection, vault, repository)
        candidate1, assessment1 = _stage(connection, context1, bundle1, source1, version=1)
        assert _fixed_decision_probe(connection, vault, repository) == before
        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            promote_memory_candidate(
                connection, candidate_id=candidate1,
                authorization=MemoryCandidatePromotionAuthorization(
                    actor="independent-owner-reviewer", allowed=True,
                    candidate_id=candidate1, authorization_id="auth-1",
                    user_confirmed=False), promoted_at=T0)
        assert _fixed_decision_probe(connection, vault, repository) == before
        approved1 = promote_memory_candidate(
            connection, candidate_id=candidate1,
            authorization=MemoryCandidatePromotionAuthorization(
                actor="independent-owner-reviewer", allowed=True,
                candidate_id=candidate1, authorization_id="auth-1-confirmed",
                user_confirmed=True), promoted_at=T0)
        rebuild()
        after_dark = _fixed_decision_probe(connection, vault, repository)
        assert after_dark == ("choose-dark", (approved1.memory.memory_id,))

        candidate2, assessment2 = _stage(connection, context2, bundle2, source2, version=2)
        assert "current_memory_exists_for_key" in assessment2.reason_codes
        assert _fixed_decision_probe(connection, vault, repository) == after_dark
        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            promote_memory_candidate_with_transition(
                connection, candidate_id=candidate2,
                authorization=MemoryCandidateTransitionAuthorization(
                    actor="independent-owner-reviewer", allowed=True,
                    candidate_id=candidate2, target_memory_id=approved1.memory.memory_id,
                    transition_type="correction", authorization_id="auth-2-denied",
                    user_confirmed=False), promoted_at=T1)
        approved2 = promote_memory_candidate_with_transition(
            connection, candidate_id=candidate2,
            authorization=MemoryCandidateTransitionAuthorization(
                actor="independent-owner-reviewer", allowed=True,
                candidate_id=candidate2, target_memory_id=approved1.memory.memory_id,
                transition_type="correction", authorization_id="auth-2-confirmed",
                user_confirmed=True), promoted_at=T1)
        assert approved2.relation is not None
        assert approved2.relation.relation_type == "corrects"
        rebuild()
        after_light = _fixed_decision_probe(connection, vault, repository)
        assert after_light == ("choose-light", (approved2.memory.memory_id,))

        request_memory_deletion(
            connection, memory_id=approved2.memory.memory_id,
            authorization=MemoryDeletionRequestAuthorization(
                actor="independent-owner-reviewer", allowed=True,
                memory_id=approved2.memory.memory_id,
                deletion_scope=ORDINARY_MEMORY_DELETION_SCOPE,
                authorization_id="synthetic-delete-request"), requested_at=T2)
        rebuild()
        after_deletion_request = _fixed_decision_probe(connection, vault, repository)
        assert after_deletion_request == ("ask-owner", ())
        delete_memory(
            connection, memory_id=approved2.memory.memory_id,
            authorization=MemoryDeletionAuthorization(
                actor="independent-owner-reviewer", allowed=True,
                memory_id=approved2.memory.memory_id,
                deletion_scope=ORDINARY_MEMORY_DELETION_SCOPE,
                authorization_id="synthetic-delete-final", strongly_confirmed=True,
                issued_at="2026-09-30T17:01:00Z",
                expires_at="2026-09-30T17:03:00Z"),
            deleted_at="2026-09-30T17:02:00Z")
        rebuild()
        assert _fixed_decision_probe(connection, vault, repository) == ("ask-owner", ())
        assert [step for step, _ in (before, after_dark, after_light, after_deletion_request)] == [
            "ask-owner", "choose-dark", "choose-light", "ask-owner"]
        assert assessment1.outcome == assessment2.outcome == "review_required"


def test_v16_cross_host_source_and_proposal_cannot_enter_intervention():
    context, bundle, source = _formation("dark", T0, version=1)
    foreign = ProductHostScope.create(
        product_id="alice", host_instance_id="another-fictional-owner",
        schema_version="1.0.0", encryption_domain="another-private")
    poisoned_source = replace(context.base.evidence[0], scope=foreign)
    with pytest.raises(CognitiveKernelContractError, match="foreign-scope"):
        validate_formation_grounding_v16(
            replace(context, base=replace(context.base, evidence=(poisoned_source,))),
            bundle, (("synthetic-source-1", source),))
    foreign_proposal = replace(bundle.proposals[0].base,
                               subject_ref="another-fictional-owner")
    with pytest.raises(ValueError, match="owner-stated host preferences"):
        # The legacy P2 runtime does not provide this multi-host binding itself.
        _stage(None, context, replace(bundle,
            base=replace(bundle.base, proposals=(foreign_proposal,)),
            proposals=(replace(bundle.proposals[0], base=foreign_proposal),)),
            source, version=1)


def test_uncertain_formation_does_not_silently_become_authoritative(tmp_path: Path):
    context, bundle, source = _formation("dark", T0, version=1)
    uncertain_base = replace(bundle.proposals[0].base,
                             epistemic_status="uncertain", confidence=0.2)
    uncertain = replace(bundle, base=replace(bundle.base, proposals=(uncertain_base,)),
                        proposals=(replace(bundle.proposals[0], base=uncertain_base),))
    validate_formation_grounding_v16(context, uncertain,
                                     (("synthetic-source-1", source),))
    repository, vault = tmp_path / "repository-sentinel", tmp_path / "private-vault"
    repository.mkdir()
    vault.mkdir()
    with open_memory_store(vault, repository_root=repository) as connection:
        with pytest.raises(ValueError, match="owner-stated host preferences"):
            _stage(connection, context, uncertain, source, version=1)
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM memory_candidates").fetchone()[0] == 0
