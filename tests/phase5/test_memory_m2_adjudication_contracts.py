"""Memory M2.2 adjudication-contract regression tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel.adjudication_contracts import (
    ADJUDICATION_OUTCOMES,
    CLAIM_CANDIDATE_ACTIONS,
    CLAIM_CONFLICT_TYPES,
    AdjudicationRecord,
    ClaimCandidate,
    ClaimConflictRecord,
    ClaimEvidenceRelation,
)
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.claim_contracts import (
    CanonicalTaggedValue,
    ClaimIdentity,
    ClaimQualifier,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope

REFERENCE_TIME = "2026-08-06T04:00:00Z"
PROVENANCE_DIGEST = "a" * 64
CONTENT_DIGEST = "b" * 64
REQUEST_DIGEST = "c" * 64


def scope(
    *,
    product_id: str = "alice",
    host_instance_id: str = "owner-primary",
    encryption_domain: str = "owner-private",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host_instance_id,
        schema_version="1.0.0",
        encryption_domain=encryption_domain,
    )


def envelope(
    *,
    record_id: str,
    record_type: str,
    authority_role: str,
    source_records: tuple[str, ...] = (),
    product_scope: ProductHostScope | None = None,
    idempotency_key: str | None = None,
) -> MemoryUnitEnvelope:
    selected_scope = product_scope or scope()
    namespace = (
        "owner-primary"
        if selected_scope.product_id == "alice"
        else "synthetic-host-a"
    )
    return MemoryUnitEnvelope.create(
        scope=selected_scope,
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id=namespace,
        host_or_cluster_id=selected_scope.host_instance_id,
        authority_role=authority_role,
        deployment_profile="single_workstation",
        created_at=REFERENCE_TIME,
        valid_from=REFERENCE_TIME,
        valid_to=None,
        transaction_time=REFERENCE_TIME,
        logical_clock=1,
        causal_parents=(),
        source_records=source_records,
        generation=0,
        state="committed",
        data_classification="highly_sensitive",
        retention_class="authoritative_source",
        deletion_state="active",
        provenance_digest=PROVENANCE_DIGEST,
        content_digest=CONTENT_DIGEST,
        writer="shadow_adjudicator",
        workflow_or_request_id="request-1",
        idempotency_namespace="shadow_adjudication",
        idempotency_key=idempotency_key or record_id,
    )


def text(value: str) -> CanonicalTaggedValue:
    return CanonicalTaggedValue.create(type_tag="text", value=value)


def qualifier(key: str, value: str) -> ClaimQualifier:
    return ClaimQualifier.create(key=key, value=text(value))


def identity(
    *,
    product_scope: ProductHostScope | None = None,
) -> ClaimIdentity:
    selected_scope = product_scope or scope()
    return ClaimIdentity.create(
        envelope=envelope(
            record_id="claim-1",
            record_type="claim_identity",
            authority_role="claim_authority",
            product_scope=selected_scope,
        ),
        claim_id="claim-1",
        canonical_subject=CanonicalTaggedValue.create(
            type_tag="identifier",
            value="owner-primary",
        ),
        canonical_predicate="prefers_interface",
        canonical_value=text("dark mode"),
        qualifiers=(),
        semantic_scope=("user_interface", "owner_preference"),
        canonicalization_version="1.0.0",
    )


def relation(
    *,
    relation_id: str = "relation-1",
    evidence_id: str = "evidence-1",
    relation_type: str = "support",
    candidate_id: str = "candidate-1",
    product_scope: ProductHostScope | None = None,
) -> ClaimEvidenceRelation:
    selected_scope = product_scope or scope()
    return ClaimEvidenceRelation.create(
        envelope=envelope(
            record_id=relation_id,
            record_type="claim_evidence_relation",
            authority_role="candidate",
            source_records=(evidence_id, candidate_id),
            product_scope=selected_scope,
        ),
        relation_id=relation_id,
        evidence_record_id=evidence_id,
        target_record_id=candidate_id,
        target_record_type="claim_candidate",
        relation_type=relation_type,
        source_class="experience_ledger_event",
        source_authority_class="owner_attested",
        extractor_component_id="candidate-extractor",
        extractor_version="1.0.0",
        confidence=0.9,
    )


def candidate(
    *,
    relations: tuple[ClaimEvidenceRelation, ...] | None = None,
    confidence: float | None = 0.9,
    proposed_action: str = "add",
    product_scope: ProductHostScope | None = None,
) -> ClaimCandidate:
    selected_scope = product_scope or scope()
    selected_relations = relations or (
        relation(product_scope=selected_scope),
    )
    relation_ids = tuple(item.relation_id for item in selected_relations)
    return ClaimCandidate.create(
        envelope=envelope(
            record_id="candidate-1",
            record_type="claim_candidate",
            authority_role="candidate",
            source_records=("claim-1", *relation_ids),
            product_scope=selected_scope,
        ),
        candidate_id="candidate-1",
        identity=identity(product_scope=selected_scope),
        value=text("dark mode"),
        qualifiers=(qualifier("context", "desktop"),),
        evidence_relation_ids=relation_ids,
        proposed_action=proposed_action,
        candidate_state="submitted",
        extractor_component_id="candidate-extractor",
        extractor_version="1.0.0",
        model_or_rule_id="rule-owner-preference-v1",
        confidence=confidence,
        request_digest=REQUEST_DIGEST,
    )


def conflict() -> ClaimConflictRecord:
    return ClaimConflictRecord.create(
        envelope=envelope(
            record_id="conflict-1",
            record_type="claim_conflict_record",
            authority_role="evaluation_artifact",
            source_records=(
                "evidence-1",
                "evidence-2",
                "relation-1",
                "relation-2",
            ),
        ),
        conflict_id="conflict-1",
        claim_id="claim-1",
        member_record_ids=("evidence-1", "evidence-2"),
        evidence_relation_ids=("relation-1", "relation-2"),
        conflict_type="support_contradiction",
        resolution_state="quarantined",
        detected_by="shadow-adjudication-prototype",
        detection_rule_id="support-contradiction-v1",
        resolution_adjudication_id=None,
        rollback_reference="delete-shadow-prototype-database",
    )


def adjudication(
    *,
    canonical_effect: bool = False,
    execution_mode: str = "shadow",
    conflict_id: str | None = None,
) -> AdjudicationRecord:
    sources = ["candidate-1", "claim-1", "relation-1"]
    if conflict_id is not None:
        sources.append(conflict_id)
    return AdjudicationRecord.create(
        envelope=envelope(
            record_id="adjudication-1",
            record_type="adjudication_record",
            authority_role="evaluation_artifact",
            source_records=tuple(sources),
        ),
        adjudication_id="adjudication-1",
        candidate_id="candidate-1",
        claim_id="claim-1",
        authority_class="algorithmic",
        authority_actor_id="shadow-adjudication-prototype",
        policy_profile="memory.m2.shadow_adjudication",
        rule_id="deterministic-shadow-adjudication",
        rule_version="1.0.0",
        evidence_relation_ids=("relation-1",),
        alternatives=("add", "quarantine", "reject"),
        confidence=0.9,
        outcome="add",
        execution_mode=execution_mode,
        canonical_effect=canonical_effect,
        conflict_record_id=conflict_id,
        rationale_codes=("shadow_eligibility_satisfied",),
        rollback_reference="delete-shadow-prototype-database",
    )


def test_contract_vocabularies_include_ratified_actions() -> None:
    assert {"add", "revise", "reject", "quarantine"} <= ADJUDICATION_OUTCOMES
    assert {"add", "retain_as_evidence", "request_owner_input"} <= CLAIM_CANDIDATE_ACTIONS
    assert "support_contradiction" in CLAIM_CONFLICT_TYPES


def test_evidence_relation_binds_evidence_and_candidate() -> None:
    value = relation()
    assert value.relation_type == "support"
    assert set(value.envelope.source_records) == {
        "candidate-1",
        "evidence-1",
    }
    value.validate()


def test_evidence_relation_rejects_unknown_relation_type() -> None:
    with pytest.raises(CognitiveKernelContractError):
        relation(relation_type="proof_by_confidence")


def test_candidate_persists_full_content_but_is_not_truth() -> None:
    value = candidate()
    record = value.metadata_record()
    assert record["value"]["value"] == "dark mode"
    assert record["candidate_state"] == "submitted"
    assert value.identity.claim_id == "claim-1"
    assert "store_sequence" not in record
    value.validate()


def test_candidate_requires_every_relation_in_envelope_lineage() -> None:
    support = relation()
    with pytest.raises(CognitiveKernelContractError):
        ClaimCandidate.create(
            envelope=envelope(
                record_id="candidate-1",
                record_type="claim_candidate",
                authority_role="candidate",
                source_records=("claim-1",),
            ),
            candidate_id="candidate-1",
            identity=identity(),
            value=text("dark mode"),
            evidence_relation_ids=(support.relation_id,),
            proposed_action="add",
            extractor_component_id="candidate-extractor",
            extractor_version="1.0.0",
            model_or_rule_id="rule-owner-preference-v1",
            confidence=0.9,
            request_digest=REQUEST_DIGEST,
        )


def test_candidate_rejects_cross_product_identity() -> None:
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="synthetic-private",
    )
    support = relation()
    with pytest.raises(CognitiveKernelContractError):
        ClaimCandidate.create(
            envelope=envelope(
                record_id="candidate-1",
                record_type="claim_candidate",
                authority_role="candidate",
                source_records=("claim-1", support.relation_id),
            ),
            candidate_id="candidate-1",
            identity=identity(product_scope=friday_scope),
            value=text("dark mode"),
            evidence_relation_ids=(support.relation_id,),
            proposed_action="add",
            extractor_component_id="candidate-extractor",
            extractor_version="1.0.0",
            model_or_rule_id="rule-owner-preference-v1",
            confidence=0.9,
            request_digest=REQUEST_DIGEST,
        )


def test_conflict_record_binds_members_and_relations() -> None:
    value = conflict()
    assert value.resolution_state == "quarantined"
    assert value.member_record_ids == ("evidence-1", "evidence-2")
    value.validate()


def test_conflict_requires_two_members() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ClaimConflictRecord.create(
            envelope=envelope(
                record_id="conflict-1",
                record_type="claim_conflict_record",
                authority_role="evaluation_artifact",
                source_records=("evidence-1",),
            ),
            conflict_id="conflict-1",
            claim_id="claim-1",
            member_record_ids=("evidence-1",),
            conflict_type="duplicate",
            resolution_state="open",
            detected_by="shadow-adjudication-prototype",
            detection_rule_id="duplicate-v1",
            resolution_adjudication_id=None,
            rollback_reference="delete-shadow-prototype-database",
        )


def test_resolved_conflict_requires_resolution_adjudication() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            conflict(),
            resolution_state="resolved",
        ).validate()


def test_shadow_adjudication_has_no_canonical_effect() -> None:
    value = adjudication()
    assert value.execution_mode == "shadow"
    assert value.canonical_effect is False
    value.assert_adjudicates(candidate())


def test_shadow_adjudication_rejects_canonical_effect() -> None:
    with pytest.raises(CognitiveKernelContractError):
        adjudication(canonical_effect=True)


def test_adjudication_requires_all_lineage_sources() -> None:
    with pytest.raises(CognitiveKernelContractError):
        AdjudicationRecord.create(
            envelope=envelope(
                record_id="adjudication-1",
                record_type="adjudication_record",
                authority_role="evaluation_artifact",
                source_records=("candidate-1",),
            ),
            adjudication_id="adjudication-1",
            candidate_id="candidate-1",
            claim_id="claim-1",
            authority_class="algorithmic",
            authority_actor_id="shadow-adjudication-prototype",
            policy_profile="memory.m2.shadow_adjudication",
            rule_id="deterministic-shadow-adjudication",
            rule_version="1.0.0",
            evidence_relation_ids=("relation-1",),
            alternatives=("add", "reject"),
            confidence=0.9,
            outcome="add",
            execution_mode="shadow",
            canonical_effect=False,
            conflict_record_id=None,
            rationale_codes=("shadow_eligibility_satisfied",),
            rollback_reference="delete-shadow-prototype-database",
        )


def test_contract_digests_detect_tampering() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(candidate(), candidate_sha256="0" * 64).validate()
    with pytest.raises(CognitiveKernelContractError):
        replace(conflict(), conflict_sha256="0" * 64).validate()
    with pytest.raises(CognitiveKernelContractError):
        replace(adjudication(), adjudication_sha256="0" * 64).validate()
