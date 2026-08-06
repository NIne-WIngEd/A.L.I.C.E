from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel import (
    AUTHORITY_ROLES,
    CAPABILITY_STATES,
    EVIDENCE_RELATION_TYPES,
    CognitiveKernelContractError,
    EvidenceBinding,
    MemoryUnitEnvelope,
    ProductHostScope,
    StoreRegistration,
)

REFERENCE_TIME = "2026-08-06T04:00:00Z"
PROVENANCE_DIGEST = "a" * 64
CONTENT_DIGEST = "b" * 64


def scope(
    *,
    product_id: str = "alice",
    host_instance_id: str = "owner-primary",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host_instance_id,
        schema_version="1.0.0",
        encryption_domain=(
            "owner-private"
            if product_id == "alice"
            else "synthetic-private"
        ),
    )


def registration() -> StoreRegistration:
    return StoreRegistration.create(
        scope=scope(),
        registration_id="registration-1",
        component_id="claim-store-embedded",
        authority_namespace_id="owner-primary",
        authority_role="claim_authority",
        capability_ids=(
            "claim.append",
            "claim.current_projection",
        ),
        backend_type="sqlite",
        backend_version="3.0.0",
        deployment_profile="single_workstation",
        capability_state="compatibility_only",
        consistency_model="serializable",
        availability_profile="single_process",
        encryption_profile="host_keyed",
        region_or_device_scope="owner_workstation",
        health_state="healthy",
        performance_profile="baseline",
        cost_profile="local",
        deletion_endpoint="kernel://claim-store/delete",
        rollback_endpoint="kernel://claim-store/rollback",
        backup_profile="encrypted_snapshot",
        created_at=REFERENCE_TIME,
    )


def envelope() -> MemoryUnitEnvelope:
    return MemoryUnitEnvelope.create(
        scope=scope(),
        record_id="record-1",
        record_type="evidence_event",
        authority_namespace_id="owner-primary",
        host_or_cluster_id="owner-primary",
        authority_role="evidence_authority",
        deployment_profile="single_workstation",
        created_at=REFERENCE_TIME,
        valid_from=REFERENCE_TIME,
        valid_to=None,
        transaction_time=REFERENCE_TIME,
        logical_clock=1,
        causal_parents=(),
        source_records=("source-1",),
        generation=0,
        state="committed",
        data_classification="highly_sensitive",
        retention_class="authoritative_source",
        deletion_state="active",
        provenance_digest=PROVENANCE_DIGEST,
        content_digest=CONTENT_DIGEST,
        writer="experience-ledger",
        workflow_or_request_id="request-1",
        idempotency_namespace="experience-ledger",
        idempotency_key="event-1",
    )


def binding() -> EvidenceBinding:
    return EvidenceBinding.create(
        scope=scope(),
        binding_id="binding-1",
        authority_namespace_id="owner-primary",
        evidence_record_id="evidence-1",
        target_record_id="claim-1",
        target_record_type="claim_version",
        relation_type="support",
        responsible_component="claim-adjudicator",
        workflow_or_request_id="request-1",
        created_at=REFERENCE_TIME,
        confidence=0.8,
    )


def test_store_registration_is_metadata_only_and_digest_bound() -> None:
    value = registration()
    record = value.metadata_record()

    assert value.authority_role in AUTHORITY_ROLES
    assert value.capability_state in CAPABILITY_STATES
    assert len(value.registration_sha256) == 64
    assert "payload" not in record
    assert "ciphertext" not in record

    value.validate()


def test_store_registration_rejects_unknown_authority_role() -> None:
    original = registration()
    with pytest.raises(CognitiveKernelContractError):
        replace(
            original,
            authority_role="unregistered_authority",
        ).validate()


def test_store_registration_detects_digest_tampering() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            registration(),
            registration_sha256="0" * 64,
        ).validate()


def test_memory_unit_envelope_carries_ratified_common_fields() -> None:
    value = envelope()
    record = value.metadata_record()

    assert record["scope"]["product_id"] == "alice"
    assert record["authority_namespace_id"] == "owner-primary"
    assert record["authority_role"] == "evidence_authority"
    assert record["logical_clock"] == 1
    assert record["generation"] == 0
    assert record["provenance_digest"] == PROVENANCE_DIGEST
    assert record["content_digest"] == CONTENT_DIGEST
    assert "payload" not in record
    assert "ciphertext" not in record

    value.validate()


def test_memory_unit_envelope_rejects_reversed_validity() -> None:
    with pytest.raises(CognitiveKernelContractError):
        MemoryUnitEnvelope.create(
            scope=scope(),
            record_id="record-1",
            record_type="evidence_event",
            authority_namespace_id="owner-primary",
            host_or_cluster_id="owner-primary",
            authority_role="evidence_authority",
            deployment_profile="single_workstation",
            created_at=REFERENCE_TIME,
            valid_from="2026-08-07T00:00:00Z",
            valid_to="2026-08-06T00:00:00Z",
            transaction_time=REFERENCE_TIME,
            logical_clock=1,
            source_records=("source-1",),
            generation=0,
            state="committed",
            data_classification="highly_sensitive",
            retention_class="authoritative_source",
            deletion_state="active",
            provenance_digest=PROVENANCE_DIGEST,
            content_digest=CONTENT_DIGEST,
            writer="experience-ledger",
            workflow_or_request_id="request-1",
            idempotency_namespace="experience-ledger",
            idempotency_key="event-1",
        )


def test_memory_unit_envelope_rejects_unknown_retention_class() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            envelope(),
            retention_class="forever",
        ).validate()


def test_memory_unit_envelope_detects_material_tampering() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            envelope(),
            content_digest="c" * 64,
        ).validate()


def test_evidence_binding_uses_ratified_relation_vocabulary() -> None:
    value = binding()

    assert value.relation_type in EVIDENCE_RELATION_TYPES
    assert value.metadata_record()["target_record_type"] == (
        "claim_version"
    )
    assert "evidence_payload" not in value.metadata_record()

    value.validate()


def test_evidence_binding_rejects_unknown_relation() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            binding(),
            relation_type="authority",
        ).validate()


def test_evidence_binding_detects_digest_tampering() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            binding(),
            binding_sha256="0" * 64,
        ).validate()


def test_contracts_preserve_product_and_host_isolation() -> None:
    alice = envelope()
    friday = MemoryUnitEnvelope.create(
        scope=scope(
            product_id="friday",
            host_instance_id="synthetic-host-a",
        ),
        record_id="record-1",
        record_type="evidence_event",
        authority_namespace_id="synthetic-host-a",
        host_or_cluster_id="synthetic-host-a",
        authority_role="evidence_authority",
        deployment_profile="single_workstation",
        created_at=REFERENCE_TIME,
        valid_from=REFERENCE_TIME,
        valid_to=None,
        transaction_time=REFERENCE_TIME,
        logical_clock=1,
        generation=0,
        state="committed",
        data_classification="private",
        retention_class="ordinary_experience",
        deletion_state="active",
        provenance_digest=PROVENANCE_DIGEST,
        content_digest=CONTENT_DIGEST,
        writer="experience-ledger",
        workflow_or_request_id="request-1",
        idempotency_namespace="experience-ledger",
        idempotency_key="event-1",
    )

    assert (
        alice.scope.storage_scope()
        != friday.scope.storage_scope()
    )
    alice.scope.assert_isolated_from(friday.scope)
