from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    OpaquePrivateCompanionReference,
    ProductHostScope,
    ProvenanceReference,
)


def alice_scope(host: str = "rayan-primary") -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id=host,
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def test_scope_binds_product_host_schema_and_encryption() -> None:
    scope = alice_scope()
    assert scope.storage_scope() == "alice/rayan-primary/owner-private"
    assert len(scope.scope_sha256()) == 64


def test_equal_scope_is_not_isolated_but_distinct_hosts_are() -> None:
    first = alice_scope()
    with pytest.raises(CognitiveKernelContractError):
        first.assert_isolated_from(alice_scope())
    first.assert_isolated_from(alice_scope("rayan-secondary"))


def test_generated_reconstruction_requires_sources_and_derivation() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ProvenanceReference.create(
            provenance_type="generated_reconstruction",
            responsible_component="clone-reconstructor",
        )

    provenance = ProvenanceReference.create(
        provenance_type="generated_reconstruction",
        source_reference_ids=("source-ref-1",),
        derivation_activity_id="derivation-1",
        responsible_component="clone-reconstructor",
        model_id="model-qwen3-8b",
        confidence=0.72,
    )
    assert provenance.metadata_record()["provenance_type"] == (
        "generated_reconstruction"
    )


def test_owner_correction_requires_supersession() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ProvenanceReference.create(
            provenance_type="owner_correction",
            responsible_component="owner-review",
        )


def test_private_reference_is_alice_only_and_metadata_only() -> None:
    provenance = ProvenanceReference.create(
        provenance_type="owner_attested_canonical",
        source_reference_ids=("private-source-ref-1",),
        responsible_component="private-loader",
    )
    reference = OpaquePrivateCompanionReference.create(
        scope=alice_scope(),
        reference_id="companion-record-1",
        directive_code="PX-CLONE-01",
        identity_layer="source_history",
        provenance=provenance,
    )
    record = reference.metadata_record()
    assert record["private_payload_included"] is False
    assert "payload" not in record
    assert "ciphertext" not in record

    friday = ProductHostScope.create(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        schema_version="1.0.0",
        encryption_domain="host-local",
    )
    with pytest.raises(CognitiveKernelContractError):
        OpaquePrivateCompanionReference.create(
            scope=friday,
            reference_id="companion-record-1",
            directive_code="PX-CLONE-01",
            identity_layer="source_history",
            provenance=provenance,
        )


def test_private_reference_detects_digest_tampering() -> None:
    provenance = ProvenanceReference.create(
        provenance_type="owner_attested_canonical",
        source_reference_ids=("private-source-ref-1",),
        responsible_component="private-loader",
    )
    reference = OpaquePrivateCompanionReference.create(
        scope=alice_scope(),
        reference_id="companion-record-1",
        directive_code="PX-CANON-01",
        identity_layer="source_history",
        provenance=provenance,
    )
    with pytest.raises(CognitiveKernelContractError):
        replace(reference, reference_sha256="0" * 64).validate()
