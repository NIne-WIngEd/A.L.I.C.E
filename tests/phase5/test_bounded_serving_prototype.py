"""M2.4 reversible bounded-serving prototype tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_kernel.bounded_serving_prototype import (
    BoundedServingIntegrityError,
    BoundedServingIsolationError,
    BoundedServingProfile,
    BoundedServingTransactionError,
    UnsafeBoundedServingPathError,
    open_bounded_serving_prototype,
    validate_bounded_serving_path,
)
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope

NOW = "2026-08-06T22:00:00Z"


def scope(
    product_id: str = "alice",
    host_instance_id: str = "owner-primary",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host_instance_id,
        schema_version="1.0.0",
        encryption_domain=(
            "owner-private" if product_id == "alice" else "host-local"
        ),
    )


def profile(
    *,
    selected_scope: ProductHostScope | None = None,
    item_budget: int = 3,
    byte_budget: int = 4096,
    expansion_depth: int = 2,
) -> BoundedServingProfile:
    return BoundedServingProfile.create(
        scope=selected_scope or scope(),
        authority_namespace_id=(
            "owner-primary"
            if (selected_scope or scope()).product_id == "alice"
            else "synthetic-host"
        ),
        profile_id="memory-m2-serving",
        item_budget=item_budget,
        byte_budget=byte_budget,
        expansion_depth=expansion_depth,
        lexical_weight=1.0,
        vector_weight=1.0,
        graph_weight=1.0,
        freshness_weight=0.25,
        fallback_mode="bounded_scan",
        production_influence=False,
    )


def add_documents(store) -> None:
    store.upsert_document(
        record_id="claim-alpha",
        record_version_id="claim-alpha-v1",
        source_kind="claim",
        authority_namespace_id="owner-primary",
        searchable_text="graph memory architecture owner preference",
        full_content={
            "subject": "memory architecture",
            "value": "graph capable",
        },
        embedding=(1.0, 0.0, 0.0),
        graph_neighbors=("episode-beta",),
        mission_node_ids=("mission-memory",),
        valid_from=NOW,
        generation=1,
        updated_at=NOW,
    )
    store.upsert_document(
        record_id="episode-beta",
        record_version_id="episode-beta-v1",
        source_kind="episode",
        authority_namespace_id="owner-primary",
        searchable_text="owner discussed graph and vector projections",
        full_content={
            "episode": "projection design",
            "details": ["graph", "vector"],
        },
        embedding=(0.9, 0.1, 0.0),
        graph_neighbors=("claim-alpha", "projection-gamma"),
        mission_node_ids=("mission-memory",),
        valid_from=NOW,
        generation=2,
        updated_at=NOW,
    )
    store.upsert_document(
        record_id="projection-gamma",
        record_version_id="projection-gamma-v1",
        source_kind="projection",
        authority_namespace_id="owner-primary",
        searchable_text="self model temporal projection",
        full_content={
            "projection": "self model",
            "state": "research",
        },
        embedding=(0.0, 1.0, 0.0),
        graph_neighbors=("episode-beta",),
        mission_node_ids=("mission-self",),
        valid_from=NOW,
        generation=3,
        updated_at=NOW,
    )
    store.upsert_document(
        record_id="claim-delta",
        record_version_id="claim-delta-v1",
        source_kind="claim",
        authority_namespace_id="owner-primary",
        searchable_text="unrelated cooking note",
        full_content={"note": "cooking"},
        embedding=(0.0, 0.0, 1.0),
        graph_neighbors=(),
        valid_from=NOW,
        generation=1,
        updated_at=NOW,
    )


def ready_indexes(store) -> None:
    for kind in ("lexical", "vector", "graph", "temporal"):
        store.set_index_state(
            index_kind=kind,
            generation=3,
            state="ready",
            updated_at=NOW,
        )


def serve(store, *, key: str = "request-key"):
    return store.serve(
        request_id="request-1",
        query_text="graph memory projection",
        query_embedding=(1.0, 0.0, 0.0),
        seed_record_ids=("claim-alpha",),
        mission_node_ids=("mission-memory",),
        idempotency_namespace="test-serving",
        idempotency_key=key,
        now=NOW,
    )


def test_profile_is_explicitly_nonproduction() -> None:
    value = profile()
    value.validate()
    assert value.production_influence is False


def test_profile_rejects_production_influence() -> None:
    with pytest.raises(CognitiveKernelContractError):
        BoundedServingProfile.create(
            scope=scope(),
            authority_namespace_id="owner-primary",
            profile_id="memory-m2-serving",
            item_budget=3,
            byte_budget=4096,
            expansion_depth=2,
            production_influence=True,
        )


def test_profile_budgets_are_selected_not_universal() -> None:
    small = profile(item_budget=1, byte_budget=256)
    large = profile(item_budget=9, byte_budget=16384)
    assert small.item_budget != large.item_budget
    assert small.byte_budget != large.byte_budget


def test_path_must_remain_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(UnsafeBoundedServingPathError):
        validate_bounded_serving_path(
            repository / "serving.sqlite3",
            repository_root=repository,
        )


def test_document_persistence_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "serving.sqlite3"
    first = open_bounded_serving_prototype(
        path=database,
        profile=profile(),
    )
    add_documents(first)
    first.close()
    second = open_bounded_serving_prototype(
        path=database,
        profile=profile(),
    )
    assert second.verify_integrity().checked_documents == 4
    second.close()


def test_scope_mismatch_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "serving.sqlite3"
    original = open_bounded_serving_prototype(
        path=database,
        profile=profile(),
    )
    original.close()
    friday_scope = scope("friday", "synthetic-host")
    with pytest.raises(BoundedServingIsolationError):
        open_bounded_serving_prototype(
            path=database,
            profile=profile(selected_scope=friday_scope),
        )


def test_changed_document_requires_advancing_generation(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    store.upsert_document(
        record_id="claim-alpha",
        record_version_id="claim-alpha-v1",
        source_kind="claim",
        authority_namespace_id="owner-primary",
        searchable_text="first",
        full_content={"value": "first"},
        embedding=(1.0, 0.0),
        valid_from=NOW,
        generation=1,
        updated_at=NOW,
    )
    with pytest.raises(BoundedServingTransactionError):
        store.upsert_document(
            record_id="claim-alpha",
            record_version_id="claim-alpha-v2",
            source_kind="claim",
            authority_namespace_id="owner-primary",
            searchable_text="changed",
            full_content={"value": "changed"},
            embedding=(1.0, 0.0),
            valid_from=NOW,
            generation=1,
            updated_at=NOW,
        )
    store.close()


def test_ready_indexes_produce_non_degraded_packet(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    assert receipt.packet.degraded is False
    assert receipt.packet.fallback_used is False
    assert receipt.trace.stale_index_observed is False
    store.close()


def test_stale_index_uses_bounded_scan_fallback(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    store.set_index_state(
        index_kind="vector",
        generation=1,
        state="stale",
        updated_at=NOW,
    )
    receipt = serve(store)
    assert receipt.fallback_used is True
    assert receipt.stale_index_observed is True
    assert receipt.packet.packet_state == "stale_fallback"
    assert any(
        step.fallback_used
        for step in receipt.trace.steps
    )
    store.close()


def test_unavailable_indexes_use_bounded_scan_fallback(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    receipt = serve(store)
    assert receipt.fallback_used is True
    assert receipt.packet.hydrated_item_count > 0
    store.close()


def test_lexical_vector_and_graph_sources_are_fused(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    selected = [item.record_id for item in receipt.packet.selections]
    assert selected[0] == "claim-alpha"
    assert "episode-beta" in selected
    assert any(
        "graph_expansion" in item.reason_codes
        for item in receipt.packet.selections
    )
    store.close()



def test_mission_filtering_is_recorded(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    assert any(
        "mission_match" in item.reason_codes
        for item in receipt.packet.selections
    )
    assert any(
        step.stage_kind == "mission_filtering"
        for step in receipt.trace.steps
    )
    store.close()

def test_graph_expansion_reaches_second_hop(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(item_budget=4, expansion_depth=2),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    assert "projection-gamma" in {
        item.record_id for item in receipt.packet.selections
    }
    store.close()


def test_item_budget_bounds_hydration(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(item_budget=1),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    assert receipt.packet.hydrated_item_count == 1
    assert len(receipt.full_packet_content) == 1
    store.close()


def test_byte_budget_bounds_hydration(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(byte_budget=80),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    assert receipt.packet.hydrated_byte_count <= 80
    store.close()


def test_packet_stores_full_content_outside_public_git(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    assert receipt.full_packet_content
    assert "full_content" in receipt.full_packet_content[0]
    store.close()


def test_trace_explains_fusion_and_packet_assembly(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    receipt = serve(store)
    kinds = {step.stage_kind for step in receipt.trace.steps}
    assert "fusion" in kinds
    assert "packet_assembly" in kinds
    store.close()


def test_idempotent_retry_returns_same_packet(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    first = serve(store)
    second = serve(store)
    assert first.packet.packet_sha256 == second.packet.packet_sha256
    assert first.trace.trace_sha256 == second.trace.trace_sha256
    assert store.list_packet_ids() == (first.packet.packet_id,)
    store.close()


def test_idempotency_key_rejects_changed_request(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    serve(store)
    with pytest.raises(BoundedServingTransactionError):
        store.serve(
            request_id="request-2",
            query_text="different query",
            query_embedding=(0.0, 1.0, 0.0),
            seed_record_ids=(),
            mission_node_ids=(),
            idempotency_namespace="test-serving",
            idempotency_key="request-key",
            now=NOW,
        )
    store.close()


def test_packet_and_trace_survive_restart(tmp_path: Path) -> None:
    database = tmp_path / "serving.sqlite3"
    first = open_bounded_serving_prototype(
        path=database,
        profile=profile(),
    )
    add_documents(first)
    ready_indexes(first)
    receipt = serve(first)
    first.close()
    second = open_bounded_serving_prototype(
        path=database,
        profile=profile(),
    )
    loaded = second.load_receipt(
        packet_id=receipt.packet.packet_id,
        trace_id=receipt.trace.trace_id,
    )
    assert loaded.packet.packet_sha256 == receipt.packet.packet_sha256
    assert loaded.full_packet_content == receipt.full_packet_content
    second.close()


def test_integrity_report_is_healthy(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    ready_indexes(store)
    serve(store)
    report = store.require_integrity()
    assert report.healthy
    assert report.checked_documents == 4
    assert report.checked_packets == 1
    assert report.checked_traces == 1
    store.close()


def test_document_tampering_is_detected(tmp_path: Path) -> None:
    store = open_bounded_serving_prototype(
        path=tmp_path / "serving.sqlite3",
        profile=profile(),
    )
    add_documents(store)
    store._connection.execute(
        "UPDATE serving_documents SET full_content_json = ? "
        "WHERE record_id = ?",
        (json.dumps({"tampered": True}), "claim-alpha"),
    )
    store._connection.commit()
    with pytest.raises(BoundedServingIntegrityError):
        store.require_integrity()
    store.close()


def test_reversible_removal_deletes_database(tmp_path: Path) -> None:
    database = tmp_path / "serving.sqlite3"
    store = open_bounded_serving_prototype(
        path=database,
        profile=profile(),
    )
    add_documents(store)
    store.remove_database()
    assert not database.exists()
