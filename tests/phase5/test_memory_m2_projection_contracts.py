from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel.canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope
from cognitive_kernel.projection_contracts import (
    EPISODE_KINDS,
    PROJECTION_MODALITIES,
    PROJECTION_SUBJECT_TYPES,
    EpisodeRecord,
    ProjectionVersion,
)

TIME = "2026-08-06T04:00:00Z"
LATER = "2026-08-06T05:00:00Z"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


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
    product_scope: ProductHostScope | None = None,
    generation: int = 0,
    source_records: tuple[str, ...] = ("source-1",),
) -> MemoryUnitEnvelope:
    selected_scope = product_scope or scope()
    namespace = (
        "owner-primary"
        if selected_scope.product_id == "alice"
        else selected_scope.host_instance_id
    )
    return MemoryUnitEnvelope.create(
        scope=selected_scope,
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id=namespace,
        host_or_cluster_id=selected_scope.host_instance_id,
        authority_role="registered_projection",
        deployment_profile="single_workstation",
        created_at=TIME,
        valid_from=TIME,
        valid_to=None,
        transaction_time=TIME,
        logical_clock=1,
        causal_parents=(),
        source_records=source_records,
        generation=generation,
        state="committed",
        data_classification="highly_sensitive",
        retention_class="high_value_experience",
        deletion_state="active",
        provenance_digest=DIGEST_A,
        content_digest=DIGEST_B,
        writer="projection_builder",
        workflow_or_request_id="request-1",
        idempotency_namespace="projection_fabric",
        idempotency_key=record_id,
    )


def episode(
    *,
    generation: int = 1,
    supersedes: str | None = None,
) -> EpisodeRecord:
    full = {"events": ["event-1"], "meaning": "owner preference"}
    summary = {"summary": "preference event"}
    return EpisodeRecord.create(
        envelope=envelope(
            record_id="episode-1",
            record_type="episode",
        ),
        episode_id="episode-1",
        episode_kind="interaction",
        episode_state="accepted",
        member_evidence_ids=("evidence-1",),
        member_claim_version_ids=("claim-version-1",),
        participant_ids=("owner-primary",),
        valid_from=TIME,
        valid_to=None,
        formed_at=LATER,
        formation_component_id="episode-builder",
        formation_version="v1",
        summary_content_digest=canonical_sha256(summary),
        full_content_digest=canonical_sha256(full),
        confidence=0.9,
        supersedes_episode_id=supersedes,
        generation=generation,
    )


def projection(
    *,
    generation: int = 1,
    supersedes: str | None = None,
    modalities: tuple[str, ...] = ("graph", "vector"),
    projection_type: str = "owner_model",
    subject_type: str = "owner",
    content: dict[str, object] | None = None,
) -> ProjectionVersion:
    selected_content = content or {
        "traits": {"prefers_interface": "dark mode"}
    }
    version_id = f"projection-version-{generation}"
    return ProjectionVersion.create(
        envelope=envelope(
            record_id=version_id,
            record_type="projection_version",
            generation=generation,
        ),
        projection_id="owner-model-1",
        version_id=version_id,
        projection_type=projection_type,
        subject_type=subject_type,
        subject_id="owner-primary",
        modalities=modalities,
        generation=generation,
        source_episode_ids=("episode-1",),
        source_claim_version_ids=("claim-version-1",),
        valid_from=TIME,
        valid_to=None,
        produced_at=LATER,
        projection_state="shadow",
        responsible_component="projection-builder",
        model_id="challenger-model",
        model_version="v1",
        content_digest=canonical_sha256(selected_content),
        vector_space_id=(
            "owner-model-space"
            if "vector" in modalities
            else None
        ),
        graph_namespace_id=(
            "owner-model-graph"
            if "graph" in modalities
            else None
        ),
        supersedes_version_id=supersedes,
        confidence=0.8,
    )


def test_episode_contract_preserves_lineage_and_content_digests() -> None:
    value = episode()
    assert value.episode_kind in EPISODE_KINDS
    assert value.member_evidence_ids == ("evidence-1",)
    assert len(value.episode_sha256) == 64
    assert "full_content" not in value.metadata_record()
    value.validate()


def test_episode_requires_source_lineage() -> None:
    with pytest.raises(CognitiveKernelContractError):
        EpisodeRecord.create(
            envelope=envelope(
                record_id="episode-1",
                record_type="episode",
                source_records=(),
            ),
            episode_id="episode-1",
            episode_kind="interaction",
            episode_state="accepted",
            valid_from=TIME,
            valid_to=None,
            formed_at=LATER,
            formation_component_id="episode-builder",
            formation_version="v1",
            summary_content_digest=DIGEST_A,
            full_content_digest=DIGEST_B,
            confidence=0.8,
        )


def test_episode_generation_requires_predecessor() -> None:
    with pytest.raises(CognitiveKernelContractError):
        episode(generation=2)


def test_episode_digest_tampering_is_detected() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(episode(), episode_sha256="0" * 64).validate()


def test_projection_contract_supports_graph_vector_and_temporal_lineage() -> None:
    value = projection()
    assert set(value.modalities) == {"graph", "vector"}
    assert set(value.modalities).issubset(PROJECTION_MODALITIES)
    assert value.subject_type in PROJECTION_SUBJECT_TYPES
    assert value.graph_namespace_id == "owner-model-graph"
    assert value.vector_space_id == "owner-model-space"
    value.validate()


def test_projection_requires_source_lineage() -> None:
    content = {"traits": {}}
    with pytest.raises(CognitiveKernelContractError):
        ProjectionVersion.create(
            envelope=envelope(
                record_id="projection-version-1",
                record_type="projection_version",
                source_records=(),
            ),
            projection_id="owner-model-1",
            version_id="projection-version-1",
            projection_type="owner_model",
            subject_type="owner",
            subject_id="owner-primary",
            modalities=("symbolic",),
            generation=1,
            valid_from=TIME,
            valid_to=None,
            produced_at=LATER,
            projection_state="shadow",
            responsible_component="projection-builder",
            model_id=None,
            model_version=None,
            content_digest=canonical_sha256(content),
        )


def test_vector_modality_requires_vector_space() -> None:
    content = {"traits": {}}
    with pytest.raises(CognitiveKernelContractError):
        ProjectionVersion.create(
            envelope=envelope(
                record_id="projection-version-1",
                record_type="projection_version",
            ),
            projection_id="owner-model-1",
            version_id="projection-version-1",
            projection_type="owner_model",
            subject_type="owner",
            subject_id="owner-primary",
            modalities=("vector",),
            generation=1,
            source_episode_ids=("episode-1",),
            valid_from=TIME,
            valid_to=None,
            produced_at=LATER,
            projection_state="shadow",
            responsible_component="projection-builder",
            model_id=None,
            model_version=None,
            content_digest=canonical_sha256(content),
        )


def test_graph_modality_requires_graph_namespace() -> None:
    content = {"traits": {}}
    with pytest.raises(CognitiveKernelContractError):
        ProjectionVersion.create(
            envelope=envelope(
                record_id="projection-version-1",
                record_type="projection_version",
            ),
            projection_id="owner-model-1",
            version_id="projection-version-1",
            projection_type="owner_model",
            subject_type="owner",
            subject_id="owner-primary",
            modalities=("graph",),
            generation=1,
            source_episode_ids=("episode-1",),
            valid_from=TIME,
            valid_to=None,
            produced_at=LATER,
            projection_state="shadow",
            responsible_component="projection-builder",
            model_id=None,
            model_version=None,
            content_digest=canonical_sha256(content),
        )


def test_model_identity_and_version_must_appear_together() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(projection(), model_version=None).validate()


def test_projection_generation_requires_current_predecessor() -> None:
    with pytest.raises(CognitiveKernelContractError):
        projection(generation=2)


def test_projection_digest_tampering_is_detected() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            projection(),
            projection_sha256="0" * 64,
        ).validate()


@pytest.mark.parametrize(
    ("projection_type", "subject_type"),
    [
        ("owner_model", "owner"),
        ("source_person_model", "source_person"),
        ("self_model", "alice_self"),
        ("temporal", "episode"),
    ],
)
def test_projection_subject_families_are_explicit(
    projection_type: str,
    subject_type: str,
) -> None:
    value = projection(
        modalities=("symbolic", "temporal"),
        projection_type=projection_type,
        subject_type=subject_type,
    )
    assert value.projection_type == projection_type
    assert value.subject_type == subject_type
