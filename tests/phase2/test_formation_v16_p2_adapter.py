"""A narrow MFM 1.6 → P2 gate exercise using synthetic, untrained output."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from alice_memory.candidate_assessment import (
    MemoryCandidateAssessmentAuthorization, assess_memory_candidate,
)
from alice_memory.formation import MemoryCandidateWriteAuthorization
from alice_memory.formation_v16_p2_adapter import (
    CurrentFormationSource, FormationP2AdapterError, P2FormationGateBinding,
    prepare_p2_candidate_v16, stage_p2_candidate_v16,
)
from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.promotion import (
    MemoryCandidatePromotionAuthorization, MemoryCandidatePromotionAuthorizationError,
    promote_memory_candidate,
)
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import MemoryRetrievalAuthorization, MemorySearchRequest
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FormationContextPacket, FormationDisposition, FormationEvidenceAnchor,
    FormationEvidenceRef, FormationProposal, MemoryProposalBundle,
)
from cognitive_kernel.formation_semantics_v16 import (
    EpisodeSemantics, FormationContextV16, FormationProposalV16,
    MemoryProposalBundleV16, RegisteredFormationTarget, RegisteredSourceSensitivity,
)


SOURCE = b"Owner prefers jasmine tea each morning."
TIME = "2026-09-30T00:00:00.000000Z"
ARTIFACT = "a" * 64


def _fixture():
    scope = ProductHostScope.create(
        product_id="alice", host_instance_id="owner-one",
        schema_version="1.0.0", encryption_domain="owner-private",
    )
    evidence = FormationEvidenceRef(
        ref_id="owner-statement-1", scope=scope,
        authority_namespace_id="owner-namespace", content_digest=sha256(SOURCE).hexdigest(),
        role="authenticated_owner_statement", modality="text",
        subject_ref="owner", speaker_ref="owner", observed_at=TIME,
    )
    base_context = FormationContextPacket(
        scope, "owner-namespace", (evidence.ref_id,), (evidence,))
    context = FormationContextV16(
        base_context, (RegisteredSourceSensitivity(evidence.ref_id, "private"),))
    claim = FormationProposal(
        proposal_id="claim-1", kind="claim", domain="host", subject_ref="owner",
        value_ref="preference-1", value_text="Owner prefers jasmine tea.",
        evidence_refs=(evidence.ref_id,),
        anchors=(FormationEvidenceAnchor(evidence.ref_id, 0, len(SOURCE)),),
        epistemic_status="owner_statement", valid_from=TIME, confidence=0.93,
        disposition_scope_ref="owner_profile",
    )
    proposal = FormationProposalV16(claim, sensitivity_hint="private")
    disposition = FormationDisposition("owner_profile", "propose", (evidence.ref_id,))
    base_bundle = MemoryProposalBundle(
        scope, "owner-namespace", "formation-run-1", (evidence.ref_id,),
        base_context.content_digest(), ARTIFACT, "run-1", (claim,), (disposition,))
    bundle = MemoryProposalBundleV16(
        base_bundle, context.content_digest(), (proposal,))
    binding = P2FormationGateBinding(scope, "owner-namespace", "owner", ARTIFACT)
    registration = CurrentFormationSource(
        evidence, "private", MemorySourceSpec(
            source_type="rayan_direct_statement", source_ref=evidence.ref_id,
            support_relation="derived_from",
            source_content_sha256=evidence.content_digest,
            source_text_sha256=evidence.content_digest, source_date=TIME,
        ), active=True, candidate_write_allowed=True,
    )
    return context, bundle, binding, registration


def _prepare(context, bundle, binding, registration, **overrides):
    arguments = dict(
        context=context, output=bundle.output_record(),
        opened_sources=(("owner-statement-1", SOURCE),), binding=binding,
        resolve_source=lambda ref: registration if ref == "owner-statement-1" else None,
        artifact_sha256=ARTIFACT, inference_run_id="run-1", proposed_at=TIME,
    )
    arguments.update(overrides)
    return prepare_p2_candidate_v16(**arguments)


def _store(tmp_path: Path):
    repo, vault = tmp_path / "repo", tmp_path / "vault"
    repo.mkdir()
    vault.mkdir()
    return repo, vault


def test_synthetic_v16_owner_claim_reaches_p2_gate_and_authorized_retrieval(tmp_path: Path):
    context, bundle, binding, registration = _fixture()
    prepared = _prepare(context, bundle, binding, registration)
    assert _prepare(context, bundle, binding, registration).receipt == prepared.receipt
    assert prepared.request.origin == "model_proposed"
    assert prepared.request.knowledge_status == "alice_inference"
    assert prepared.request.rayan_confirmed is False
    assert prepared.request.data_classification == "PRIVATE"
    assert prepared.receipt.bundle_sha256 == bundle.content_digest()
    assert prepared.receipt.source_sha256 == (sha256(SOURCE).hexdigest(),)

    repo, vault = _store(tmp_path)
    with open_memory_store(vault, repository_root=repo) as connection:
        staged = stage_p2_candidate_v16(
            connection, authorization=MemoryCandidateWriteAuthorization(
                "model-runner", True, "synthetic bridge test"),
            context=context, output=bundle.output_record(),
            opened_sources=(("owner-statement-1", SOURCE),), binding=binding,
            resolve_source=lambda _: registration, artifact_sha256=ARTIFACT,
            inference_run_id="run-1", proposed_at=TIME,
        )
        assert staged.receipt == prepared.receipt
        assert staged.candidate.candidate_id == prepared.request.candidate_id
        assert staged.candidate.content_sha256 == sha256(
            prepared.request.content.encode("utf-8")).hexdigest()
        source_row = connection.execute(
            "SELECT source_ref, source_content_sha256, source_text_sha256, "
            "support_relation FROM memory_candidate_sources WHERE candidate_id = ?",
            (staged.candidate.candidate_id,),
        ).fetchone()
        assert tuple(source_row) == (
            "owner-statement-1", sha256(SOURCE).hexdigest(),
            sha256(SOURCE).hexdigest(), "derived_from")
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        build_memory_lexical_index(
            connection, vault, repository_root=repo,
            built_at="2026-09-30T00:00:30Z")
        retrieval_auth = MemoryRetrievalAuthorization(
            "owner-reviewer", True, "synthetic retrieval", "PRIVATE")
        assert search_memories(
            connection, vault, request=MemorySearchRequest(query="jasmine tea"),
            authorization=retrieval_auth, repository_root=repo,
        ).results == ()
        assessment = assess_memory_candidate(
            connection, candidate_id=staged.candidate.candidate_id,
            authorization=MemoryCandidateAssessmentAuthorization("gate", True),
            assessed_at="2026-09-30T00:01:00Z",
        )
        assert assessment.outcome == "review_required"
        assert "model_proposals_require_user_review" in assessment.reason_codes
        denied = MemoryCandidatePromotionAuthorization(
            actor="model-runner", allowed=True,
            candidate_id=staged.candidate.candidate_id,
            authorization_id="review-1", user_confirmed=True,
        )
        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            promote_memory_candidate(
                connection, candidate_id=staged.candidate.candidate_id,
                authorization=denied, promoted_at="2026-09-30T00:02:00Z")
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        promotion = promote_memory_candidate(
            connection, candidate_id=staged.candidate.candidate_id,
            authorization=replace(denied, actor="owner-reviewer"),
            promoted_at="2026-09-30T00:02:00Z",
        )
        assert promotion.memory.content_sha256 == staged.candidate.content_sha256
        build_memory_lexical_index(
            connection, vault, repository_root=repo,
            built_at="2026-09-30T00:03:00Z")
        result = search_memories(
            connection, vault, request=MemorySearchRequest(query="jasmine tea"),
            authorization=retrieval_auth,
            repository_root=repo,
        )
        assert [item.memory_id for item in result.results] == [promotion.memory.memory_id]
        assert result.results[0].knowledge_status == "alice_inference"


@pytest.mark.parametrize("change", [
    "episode", "relationship", "mission", "workspace", "contradiction",
    "target", "uncertainty", "day", "source_person", "second_proposal", "abstain",
])
def test_unrepresented_v16_semantics_fail_closed(change: str):
    context, bundle, binding, registration = _fixture()
    proposal = bundle.proposals[0]
    base = proposal.base
    if change == "episode":
        base = replace(base, kind="episode")
        proposal = replace(proposal, base=base,
                           episode=EpisodeSemantics(TIME, None, (), ()))
    elif change == "relationship":
        base = replace(base, kind="relationship", domain="relationship",
                       subject_ref="alice-self")
        proposal = replace(proposal, base=base, relationship_counterpart_ref="owner")
    elif change == "mission":
        proposal = replace(proposal, mission_target_refs=("mission-1",))
    elif change == "workspace":
        proposal = replace(proposal, workspace_target_refs=("workspace-1",))
    elif change == "contradiction":
        base = replace(base, contradicts=("claim-old",))
    elif change == "target":
        base = replace(base, target_refs=("memory-old",))
    elif change == "uncertainty":
        base = replace(base, uncertainty_ref="uncertain-source")
    elif change == "day":
        base = replace(base, temporal_granularity="day")
    elif change == "source_person":
        base = replace(base, kind="source_person_evidence", domain="source_person")
    elif change == "second_proposal":
        second = replace(proposal, base=replace(base, proposal_id="claim-2"))
        bundle = replace(bundle, base=replace(bundle.base, proposals=(base, second.base)),
                         proposals=(proposal, second))
    elif change == "abstain":
        bundle = replace(bundle, base=replace(bundle.base, dispositions=(
            replace(bundle.base.dispositions[0], action="abstain"),)))
    if change not in {"second_proposal", "abstain"}:
        proposal = replace(proposal, base=base)
        bundle = replace(bundle, base=replace(bundle.base, proposals=(base,)),
                         proposals=(proposal,))
    registered = {
        "relationship": ("owner", "entity"),
        "mission": ("mission-1", "mission"),
        "workspace": ("workspace-1", "workspace"),
    }.get(change)
    if registered is not None:
        context = replace(context, targets=(RegisteredFormationTarget(
            registered[0], registered[1], binding.scope,
            binding.authority_namespace_id, ("owner-statement-1",)),))
        bundle = replace(bundle, context_v16_digest=context.content_digest())
    with pytest.raises(FormationP2AdapterError if change != "abstain" else ValueError):
        _prepare(context, bundle, binding, registration)


def test_live_registration_and_exact_bytes_are_required():
    context, bundle, binding, registration = _fixture()
    foreign = ProductHostScope.create(
        product_id="alice", host_instance_id="another-host",
        schema_version="1.0.0", encryption_domain="owner-private")
    cases = (
        {"binding": replace(binding, scope=foreign)},
        {"binding": replace(binding, approved_artifact_sha256="b" * 64)},
        {"registration": replace(registration, active=False)},
        {"registration": replace(registration, candidate_write_allowed=False)},
        {"registration": replace(registration, evidence=replace(
            registration.evidence, content_digest="b" * 64))},
        {"registration": replace(registration, source=replace(
            registration.source, source_content_sha256="b" * 64))},
        {"registration": replace(registration, sensitivity="highly_sensitive")},
        {"opened_sources": (("owner-statement-1", b"changed"),)},
        {"resolve_source": lambda _: None},
    )
    for changes in cases:
        changes = changes.copy()
        current = changes.pop("registration", registration)
        bound = changes.pop("binding", binding)
        with pytest.raises((FormationP2AdapterError, ValueError)):
            _prepare(context, bundle, bound, current, **changes)


def test_current_policy_escalation_changes_candidate_identity_and_classification():
    context, bundle, binding, registration = _fixture()
    original = _prepare(context, bundle, binding, registration)
    # The stale model context and hint say PRIVATE; the current registry is stricter.
    escalated_context = replace(context, source_sensitivities=(
        RegisteredSourceSensitivity("owner-statement-1", "public"),))
    public_proposal = replace(bundle.proposals[0], sensitivity_hint="public")
    public_bundle = replace(bundle, base=replace(bundle.base,
        context_digest=escalated_context.base.content_digest()),
        context_v16_digest=escalated_context.content_digest(),
        proposals=(public_proposal,))
    still_private = _prepare(escalated_context, public_bundle, binding, registration)
    assert still_private.request.data_classification == "PRIVATE"
    assert still_private.receipt.candidate_id != original.receipt.candidate_id
    assert still_private.receipt.binding_sha256 != original.receipt.binding_sha256
    with pytest.raises(ValueError):
        _prepare(context, bundle, binding, registration,
                 output={**bundle.output_record(), "unknown": None})
