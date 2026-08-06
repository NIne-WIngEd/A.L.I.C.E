"""M2.3 reversible projection-prototype tests.

The false production-influence state below is scoped to the named M2.3
prototype profile. Successor production profiles remain governed by the
registered activation, evaluation, deletion, rollback, and owner-approval
conditions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from cognitive_kernel.canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope
from cognitive_kernel.projection_contracts import (
    EpisodeRecord,
    ProjectionVersion,
)
from cognitive_kernel.projection_prototype import (
    PROJECTION_PROTOTYPE_STATE,
    ProjectionGraphEdge,
    ProjectionPrototypeConflictError,
    ProjectionPrototypeIntegrityError,
    ProjectionPrototypeIsolationError,
    ProjectionPrototypeProfile,
    UnsafeProjectionPrototypePathError,
    open_projection_prototype,
)

TIME = "2026-08-06T04:00:00Z"
LATER = "2026-08-06T05:00:00Z"
LATEST = "2026-08-06T06:00:00Z"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
REQUEST_A = "c" * 64
REQUEST_B = "d" * 64

EPISODE_CONTENT = {
    "events": [{"id": "evidence-1", "text": "owner chose dark mode"}],
    "interpretation": "stable interface preference candidate",
}
EPISODE_SUMMARY = {
    "summary": "Owner selected dark mode during an interface session."
}
PROJECTION_CONTENT_V1 = {
    "subject": "owner-primary",
    "preferences": {"interface_theme": "dark"},
    "confidence_notes": ["supported by episode-1"],
}
PROJECTION_CONTENT_V2 = {
    "subject": "owner-primary",
    "preferences": {
        "interface_theme": "dark",
        "response_style": "direct",
    },
    "confidence_notes": ["supported by episode-1", "reviewed"],
}


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
    generation: int,
    product_scope: ProductHostScope | None = None,
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
        logical_clock=generation,
        causal_parents=(),
        source_records=("source-1",),
        generation=generation,
        state="committed",
        data_classification="highly_sensitive",
        retention_class="high_value_experience",
        deletion_state="active",
        provenance_digest=DIGEST_A,
        content_digest=DIGEST_B,
        writer="projection-builder",
        workflow_or_request_id="request-1",
        idempotency_namespace="projection-fabric",
        idempotency_key=record_id,
    )


def episode(
    *,
    product_scope: ProductHostScope | None = None,
) -> EpisodeRecord:
    return EpisodeRecord.create(
        envelope=envelope(
            record_id="episode-1",
            record_type="episode",
            generation=1,
            product_scope=product_scope,
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
        summary_content_digest=canonical_sha256(EPISODE_SUMMARY),
        full_content_digest=canonical_sha256(EPISODE_CONTENT),
        confidence=0.9,
        generation=1,
    )


def projection(
    *,
    generation: int = 1,
    content: dict[str, object] | None = None,
    product_scope: ProductHostScope | None = None,
) -> ProjectionVersion:
    selected_content = (
        PROJECTION_CONTENT_V1
        if content is None
        else content
    )
    version_id = f"owner-projection-v{generation}"
    return ProjectionVersion.create(
        envelope=envelope(
            record_id=version_id,
            record_type="projection_version",
            generation=generation,
            product_scope=product_scope,
        ),
        projection_id="owner-model-1",
        version_id=version_id,
        projection_type="owner_model",
        subject_type="owner",
        subject_id="owner-primary",
        modalities=("graph", "symbolic", "temporal", "vector"),
        generation=generation,
        source_episode_ids=("episode-1",),
        source_claim_version_ids=("claim-version-1",),
        valid_from=TIME,
        valid_to=None,
        produced_at=LATER if generation == 1 else LATEST,
        projection_state="shadow",
        responsible_component="projection-builder",
        model_id="challenger-model",
        model_version="v1",
        content_digest=canonical_sha256(selected_content),
        vector_space_id="owner-model-space",
        graph_namespace_id="owner-model-graph",
        supersedes_version_id=(
            None
            if generation == 1
            else f"owner-projection-v{generation - 1}"
        ),
        confidence=0.8,
    )


def edge(
    *,
    edge_id: str = "edge-1",
    relation_type: str = "prefers",
    valid_to: str | None = None,
) -> ProjectionGraphEdge:
    return ProjectionGraphEdge.create(
        edge_id=edge_id,
        graph_namespace_id="owner-model-graph",
        source_node_id="owner-primary",
        relation_type=relation_type,
        target_node_id="interface-dark-mode",
        valid_from=TIME,
        valid_to=valid_to,
        weight=0.9,
        source_record_ids=("claim-version-1", "episode-1"),
    )


def profile(
    *,
    product_scope: ProductHostScope | None = None,
) -> ProjectionPrototypeProfile:
    selected_scope = product_scope or scope()
    namespace = (
        "owner-primary"
        if selected_scope.product_id == "alice"
        else selected_scope.host_instance_id
    )
    return ProjectionPrototypeProfile.create(
        scope=selected_scope,
        authority_namespace_id=namespace,
        store_id="projection-prototype-1",
    )


def open_store(
    tmp_path: Path,
    *,
    product_scope: ProductHostScope | None = None,
):
    return open_projection_prototype(
        tmp_path / "projection.sqlite3",
        profile=profile(product_scope=product_scope),
        repository_root=tmp_path / "repository",
    )


def append_episode(store) -> None:
    store.append_episode(
        episode(),
        full_content=EPISODE_CONTENT,
        summary_content=EPISODE_SUMMARY,
        idempotency_namespace="episode-write",
        idempotency_key="episode-1",
        request_digest=REQUEST_A,
    )


def append_projection_v1(store):
    return store.append_projection(
        projection(),
        full_content=PROJECTION_CONTENT_V1,
        idempotency_namespace="projection-write",
        idempotency_key="owner-model-v1",
        request_digest=REQUEST_A,
        expected_current_generation=None,
        vector=(1.0, 0.0, 0.5),
        graph_edges=(edge(),),
    )


def test_profile_is_full_content_and_nonproduction() -> None:
    value = profile()
    assert value.state == PROJECTION_PROTOTYPE_STATE
    assert value.full_content_persistence is True
    assert value.graph_projection is True
    assert value.vector_projection is True
    assert value.production_influence is False
    assert value.canonical_authority is False
    value.validate()


def test_repository_path_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(UnsafeProjectionPrototypePathError):
        open_projection_prototype(
            repository / "projection.sqlite3",
            profile=profile(),
            repository_root=repository,
        )


def test_episode_content_is_persisted_and_retrievable(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        loaded = store.get_episode("episode-1")
        assert loaded is not None
        assert loaded["full_content"] == EPISODE_CONTENT
        assert loaded["summary_content"] == EPISODE_SUMMARY


def test_episode_append_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        first = store.append_episode(
            episode(),
            full_content=EPISODE_CONTENT,
            summary_content=EPISODE_SUMMARY,
            idempotency_namespace="episode-write",
            idempotency_key="episode-1",
            request_digest=REQUEST_A,
        )
        second = store.append_episode(
            episode(),
            full_content=EPISODE_CONTENT,
            summary_content=EPISODE_SUMMARY,
            idempotency_namespace="episode-write",
            idempotency_key="episode-1",
            request_digest=REQUEST_A,
        )
        assert first == second


def test_idempotency_key_reuse_with_changed_request_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        with pytest.raises(ProjectionPrototypeConflictError):
            store.append_episode(
                episode(),
                full_content=EPISODE_CONTENT,
                summary_content=EPISODE_SUMMARY,
                idempotency_namespace="episode-write",
                idempotency_key="episode-1",
                request_digest=REQUEST_B,
            )


def test_graph_vector_projection_is_persisted(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        append_projection_v1(store)
        loaded = store.get_projection_version(
            "owner-projection-v1"
        )
        assert loaded is not None
        assert loaded["full_content"] == PROJECTION_CONTENT_V1
        assert loaded["vector"] == [1.0, 0.0, 0.5]
        assert len(loaded["graph_edges"]) == 1


def test_current_projection_and_history_advance(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        append_projection_v1(store)
        store.append_projection(
            projection(
                generation=2,
                content=PROJECTION_CONTENT_V2,
            ),
            full_content=PROJECTION_CONTENT_V2,
            idempotency_namespace="projection-write",
            idempotency_key="owner-model-v2",
            request_digest=REQUEST_B,
            expected_current_generation=1,
            vector=(1.0, 0.2, 0.5),
            graph_edges=(
                edge(edge_id="edge-2"),
            ),
        )
        current = store.get_current_projection("owner-model-1")
        assert current is not None
        assert current["projection"]["generation"] == 2
        assert len(store.projection_history("owner-model-1")) == 2


def test_expected_current_generation_is_enforced(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        append_projection_v1(store)
        with pytest.raises(ProjectionPrototypeConflictError):
            store.append_projection(
                projection(
                    generation=2,
                    content=PROJECTION_CONTENT_V2,
                ),
                full_content=PROJECTION_CONTENT_V2,
                idempotency_namespace="projection-write",
                idempotency_key="owner-model-v2",
                request_digest=REQUEST_B,
                expected_current_generation=0,
                vector=(1.0, 0.2, 0.5),
                graph_edges=(edge(edge_id="edge-2"),),
            )


def test_similarity_search_returns_best_match(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        append_projection_v1(store)
        results = store.similarity_search(
            (1.0, 0.0, 0.5),
            vector_space_id="owner-model-space",
        )
        assert results[0]["projection_version_id"] == (
            "owner-projection-v1"
        )
        assert results[0]["similarity"] == pytest.approx(1.0)


def test_temporal_graph_neighbor_query(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        append_projection_v1(store)
        neighbors = store.neighbors(
            graph_namespace_id="owner-model-graph",
            node_id="owner-primary",
            relation_type="prefers",
            valid_at=LATER,
        )
        assert len(neighbors) == 1
        assert neighbors[0]["target_node_id"] == (
            "interface-dark-mode"
        )


def test_projection_as_of_returns_prior_generation(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    with open_store(tmp_path) as store:
        append_episode(store)
        append_projection_v1(store)
        store.append_projection(
            projection(
                generation=2,
                content=PROJECTION_CONTENT_V2,
            ),
            full_content=PROJECTION_CONTENT_V2,
            idempotency_namespace="projection-write",
            idempotency_key="owner-model-v2",
            request_digest=REQUEST_B,
            expected_current_generation=1,
            vector=(1.0, 0.2, 0.5),
            graph_edges=(edge(edge_id="edge-2"),),
        )
        prior = store.projection_as_of(
            "owner-model-1",
            produced_at=LATER,
        )
        assert prior is not None
        assert prior["projection"]["generation"] == 1


def test_projection_persists_across_reopen(tmp_path: Path) -> None:
    (tmp_path / "repository").mkdir()
    database = tmp_path / "projection.sqlite3"
    first_profile = profile()
    with open_projection_prototype(
        database,
        profile=first_profile,
        repository_root=tmp_path / "repository",
    ) as store:
        append_episode(store)
        append_projection_v1(store)
    with open_projection_prototype(
        database,
        profile=first_profile,
        repository_root=tmp_path / "repository",
    ) as reopened:
        assert reopened.get_current_projection(
            "owner-model-1"
        ) is not None


def test_product_host_isolation_is_enforced(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    database = tmp_path / "projection.sqlite3"
    alice_profile = profile()
    with open_projection_prototype(
        database,
        profile=alice_profile,
        repository_root=tmp_path / "repository",
    ):
        pass
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="host-local",
    )
    with pytest.raises(ProjectionPrototypeIsolationError):
        open_projection_prototype(
            database,
            profile=profile(product_scope=friday_scope),
            repository_root=tmp_path / "repository",
        )


def test_integrity_verification_detects_content_tampering(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    database = tmp_path / "projection.sqlite3"
    with open_store(tmp_path) as store:
        append_episode(store)
        append_projection_v1(store)
        assert store.verify_integrity().valid is True
    connection = sqlite3.connect(database)
    connection.execute(
        """
        UPDATE projection_versions
        SET content_json = ?
        WHERE version_id = ?
        """,
        ('{"tampered":true}', "owner-projection-v1"),
    )
    connection.commit()
    connection.close()
    with open_store(tmp_path) as store:
        report = store.verify_integrity()
        assert report.valid is False
        assert any(
            "projection content mismatch" in error
            for error in report.errors
        )


def test_database_can_be_reversibly_removed(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    database = tmp_path / "projection.sqlite3"
    store = open_store(tmp_path)
    append_episode(store)
    store.delete_database()
    assert not database.exists()


def test_projection_scope_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository").mkdir()
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="host-local",
    )
    with open_store(tmp_path) as store:
        with pytest.raises(ProjectionPrototypeIsolationError):
            store.append_episode(
                episode(product_scope=friday_scope),
                full_content=EPISODE_CONTENT,
                summary_content=EPISODE_SUMMARY,
                idempotency_namespace="episode-write",
                idempotency_key="episode-1",
                request_digest=REQUEST_A,
            )
