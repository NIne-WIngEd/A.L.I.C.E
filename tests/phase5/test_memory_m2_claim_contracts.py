"""Memory M2.1 claim-contract regression tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel import (
    CLAIM_ADJUDICATION_STATES,
    CLAIM_CONFLICT_STATES,
    CLAIM_DELETION_STATES,
    CLAIM_VALIDITY_STATES,
    CanonicalTaggedValue,
    ClaimIdentity,
    ClaimQualifier,
    ClaimVersion,
    CognitiveKernelContractError,
    CurrentClaimProjection,
    MemoryUnitEnvelope,
    ProductHostScope,
)

REFERENCE_TIME = "2026-08-06T04:00:00Z"
LATER_TIME = "2026-08-06T05:00:00Z"
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
    supersedes: tuple[str, ...] = (),
    deletion_state: str = "active",
    product_scope: ProductHostScope | None = None,
) -> MemoryUnitEnvelope:
    selected_scope = product_scope or scope()
    return MemoryUnitEnvelope.create(
        scope=selected_scope,
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id=(
            "owner-primary"
            if selected_scope.product_id == "alice"
            else "synthetic-host-a"
        ),
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
        deletion_state=deletion_state,
        provenance_digest=PROVENANCE_DIGEST,
        content_digest=CONTENT_DIGEST,
        writer="claim_authority",
        workflow_or_request_id="request-1",
        idempotency_namespace="claim_authority",
        idempotency_key=record_id,
        supersedes=supersedes,
    )


def text(value: str) -> CanonicalTaggedValue:
    return CanonicalTaggedValue.create(type_tag="text", value=value)


def qualifier(key: str, value: str) -> ClaimQualifier:
    return ClaimQualifier.create(key=key, value=text(value))


def identity(
    *,
    claim_id: str = "claim-1",
    product_scope: ProductHostScope | None = None,
    qualifiers: tuple[ClaimQualifier, ...] = (),
) -> ClaimIdentity:
    return ClaimIdentity.create(
        envelope=envelope(
            record_id=claim_id,
            record_type="claim_identity",
            authority_role="claim_authority",
            product_scope=product_scope,
        ),
        claim_id=claim_id,
        canonical_subject=CanonicalTaggedValue.create(
            type_tag="identifier",
            value="owner-primary",
        ),
        canonical_predicate="prefers_interface",
        canonical_value=text("dark mode"),
        qualifiers=qualifiers,
        semantic_scope=("user_interface", "owner_preference"),
        canonicalization_version="1.0.0",
    )


def version(
    *,
    claim_id: str = "claim-1",
    claim_version_id: str = "claim_version-1",
    adjudication_state: str = "accepted",
    conflict_set_id: str | None = None,
    product_scope: ProductHostScope | None = None,
) -> ClaimVersion:
    return ClaimVersion.create(
        envelope=envelope(
            record_id=claim_version_id,
            record_type="claim_version",
            authority_role="claim_authority",
            source_records=(claim_id, "evidence-1"),
            product_scope=product_scope,
        ),
        claim_version_id=claim_version_id,
        claim_id=claim_id,
        version_sequence=1,
        store_sequence=17,
        event_stream_position=42,
        value=text("dark mode"),
        qualifiers=(qualifier("context", "desktop"),),
        authority_class="owner_attested",
        confidence=0.99,
        adjudication_state=adjudication_state,
        evidence_relation_ids=("binding-1",),
        conflict_set_id=conflict_set_id,
        request_digest=REQUEST_DIGEST,
    )


def projection(
    *,
    claim_id: str = "claim-1",
    claim_version_id: str = "claim_version-1",
    product_scope: ProductHostScope | None = None,
) -> CurrentClaimProjection:
    return CurrentClaimProjection.create(
        envelope=envelope(
            record_id="projection-1",
            record_type="current_claim_projection",
            authority_role="registered_projection",
            source_records=(claim_id, claim_version_id),
            product_scope=product_scope,
        ),
        projection_id="projection-1",
        claim_id=claim_id,
        current_claim_version_id=claim_version_id,
        authority_generation=2,
        projection_generation=3,
        adjudication_state="accepted",
        validity_state="current",
        conflict_state="none",
        deletion_state="active",
        source_position=17,
    )


def test_type_tagged_values_are_canonical_and_extensible() -> None:
    standard = CanonicalTaggedValue.create(
        type_tag="map",
        value={"longitude": -73.9, "latitude": 40.7},
    )
    extension = CanonicalTaggedValue.create(
        type_tag="geospatial.point.v1",
        value={"latitude": 40.7, "longitude": -73.9},
    )

    assert standard.value() == extension.value()
    assert standard.value_sha256 != extension.value_sha256
    standard.validate()
    extension.validate()


def test_type_tag_validation_rejects_mismatched_material() -> None:
    with pytest.raises(CognitiveKernelContractError):
        CanonicalTaggedValue.create(type_tag="integer", value=True)


def test_qualifiers_are_sorted_and_unique() -> None:
    first = qualifier("location", "home")
    second = qualifier("modality", "text")
    value = identity(qualifiers=(second, first))

    assert [item.key for item in value.qualifiers] == [
        "location",
        "modality",
    ]
    with pytest.raises(CognitiveKernelContractError):
        ClaimIdentity.create(
            envelope=envelope(
                record_id="claim-duplicate",
                record_type="claim_identity",
                authority_role="claim_authority",
            ),
            claim_id="claim-duplicate",
            canonical_subject=text("subject"),
            canonical_predicate="has_value",
            canonical_value=text("value"),
            qualifiers=(first, first),
            semantic_scope=("test",),
        )


def test_semantic_digest_excludes_claim_identifier_and_operational_time() -> None:
    first = identity(claim_id="claim-1")
    second = identity(claim_id="claim-2")

    assert first.semantic_digest == second.semantic_digest
    assert first.identity_sha256 != second.identity_sha256
    assert first.semantically_equals(second)


def test_semantic_digest_includes_authority_namespace_and_product() -> None:
    alice = identity()
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="synthetic-private",
    )
    friday = identity(
        claim_id="claim-1",
        product_scope=friday_scope,
    )

    assert alice.semantic_digest != friday.semantic_digest
    assert not alice.semantically_equals(friday)


def test_claim_identity_requires_claim_authority_envelope() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ClaimIdentity.create(
            envelope=envelope(
                record_id="claim-1",
                record_type="claim_identity",
                authority_role="candidate",
            ),
            claim_id="claim-1",
            canonical_subject=text("subject"),
            canonical_predicate="has_value",
            canonical_value=text("value"),
        )


def test_claim_retirement_is_append_only_metadata_with_paired_reason() -> None:
    original = identity()
    retired = ClaimIdentity.create(
        envelope=original.envelope,
        claim_id=original.claim_id,
        canonical_subject=original.canonical_subject,
        canonical_predicate=original.canonical_predicate,
        canonical_value=original.canonical_value,
        qualifiers=original.qualifiers,
        semantic_scope=original.semantic_scope,
        canonicalization_version=original.canonicalization_version,
        retired_at=LATER_TIME,
        retirement_reason="owner_retirement",
    )

    assert retired.semantic_digest == original.semantic_digest
    assert retired.retired_at.endswith("Z")
    with pytest.raises(CognitiveKernelContractError):
        replace(retired, retirement_reason=None).validate()


def test_claim_version_binds_claim_identity_and_commit_order() -> None:
    value = version()

    assert value.claim_id in value.envelope.source_records
    assert value.store_sequence == 17
    assert value.idempotency_record() == {
        "idempotency_namespace": "claim_authority",
        "idempotency_key": "claim_version-1",
        "request_digest": REQUEST_DIGEST,
    }
    value.validate()


def test_claim_version_rejects_nonpositive_authoritative_sequence() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(version(), store_sequence=0).validate()


def test_disputed_version_requires_explicit_conflict_set() -> None:
    with pytest.raises(CognitiveKernelContractError):
        version(adjudication_state="disputed")

    disputed = version(
        adjudication_state="disputed",
        conflict_set_id="conflict-1",
    )
    assert disputed.conflict_set_id == "conflict-1"


def test_correction_targets_must_be_superseded_by_the_envelope() -> None:
    corrected_envelope = envelope(
        record_id="claim_version-2",
        record_type="claim_version",
        authority_role="claim_authority",
        source_records=("claim-1", "evidence-2"),
        supersedes=("claim_version-1",),
    )
    corrected = ClaimVersion.create(
        envelope=corrected_envelope,
        claim_version_id="claim_version-2",
        claim_id="claim-1",
        version_sequence=2,
        store_sequence=18,
        event_stream_position=43,
        value=text("light mode"),
        authority_class="owner_correction",
        confidence=1.0,
        adjudication_state="revised",
        evidence_relation_ids=("binding-2",),
        correction_of=("claim_version-1",),
        request_digest="d" * 64,
    )

    corrected.validate()
    with pytest.raises(CognitiveKernelContractError):
        replace(corrected, correction_of=("claim_version-x",)).validate()


def test_current_projection_is_rebuildable_metadata_only_state() -> None:
    value = projection()
    record = value.metadata_record()

    assert value.adjudication_state in CLAIM_ADJUDICATION_STATES
    assert value.validity_state in CLAIM_VALIDITY_STATES
    assert value.conflict_state in CLAIM_CONFLICT_STATES
    assert value.deletion_state in CLAIM_DELETION_STATES
    assert "payload" not in record
    assert "ciphertext" not in record
    value.validate()


def test_current_projection_validates_identity_version_and_source_position() -> None:
    current_identity = identity()
    current_version = version()
    current_projection = projection()

    current_projection.assert_projects(current_identity, current_version)

    with pytest.raises(CognitiveKernelContractError):
        replace(current_projection, source_position=16).assert_projects(
            current_identity,
            current_version,
        )


def test_projection_rejects_cross_product_claim_state() -> None:
    alice_projection = projection()
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="synthetic-private",
    )
    friday_identity = identity(product_scope=friday_scope)
    friday_version = version(product_scope=friday_scope)

    with pytest.raises(CognitiveKernelContractError):
        alice_projection.assert_projects(friday_identity, friday_version)


def test_claim_contracts_detect_digest_tampering() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(identity(), semantic_digest="0" * 64).validate()
    with pytest.raises(CognitiveKernelContractError):
        replace(version(), version_sha256="0" * 64).validate()
    with pytest.raises(CognitiveKernelContractError):
        replace(projection(), projection_sha256="0" * 64).validate()
