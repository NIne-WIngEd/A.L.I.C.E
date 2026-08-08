from __future__ import annotations

from pathlib import Path
import json

import pytest

from cognitive_kernel.canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.shadow_migration_stage_f_g import (
    CanonicalChangeEnvelope,
    ProjectionBuildReceipt,
    build_synthetic_stage_f_manifest,
    build_synthetic_stage_g_manifest,
)
from cognitive_kernel.shadow_migration_stage_f_g_persistent import (
    PersistentBackendCandidate,
    PersistentIntegrationManifest,
    SQLitePersistentStageFGReferenceAdapter,
    build_persistent_backend_candidate_registry,
    build_synthetic_persistent_integration_manifest,
    persistent_backend_candidate_registry_sha256,
)
from cognitive_kernel.shadow_migration_stage_f_g_persistent_evaluation import (
    build_persistent_stage_f_g_report,
)

TS = "2026-08-07T15:30:00Z"


def _scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="persistent-stage-f-g-test-host",
        schema_version="1.0.0",
        encryption_domain="persistent-stage-f-g-test-domain",
    )


def _change(
    manifest_id: str,
    sequence: int,
    operation: str = "upsert",
) -> CanonicalChangeEnvelope:
    return CanonicalChangeEnvelope.create(
        manifest_id=manifest_id,
        change_id=f"change.{sequence}",
        authority_namespace="alice.owner.memory",
        outbox_sequence=sequence,
        operation=operation,
        canonical_record_sha256=canonical_sha256(
            {"sequence": sequence, "operation": operation}
        ),
        mapping_version="1.0.0",
        evidence_lineage_ids=(f"evidence.{sequence}",),
        deletion_lineage_ids=(
            (f"deletion.{sequence}",)
            if operation == "delete"
            else ()
        ),
    )


def test_candidate_registry_is_polyglot_and_non_ceiling() -> None:
    candidates = build_persistent_backend_candidate_registry()
    ids = {item.candidate_id for item in candidates}
    assert {
        "alice.reference.sqlite",
        "alice.candidate.kurrentdb",
        "alice.candidate.neo4j",
        "alice.candidate.qdrant",
        "alice.candidate.temporal",
    }.issubset(ids)
    assert all(
        item.architecture_state == "candidate_not_destination_ceiling"
        for item in candidates
    )
    assert len(persistent_backend_candidate_registry_sha256()) == 64


def test_candidate_requires_roles_and_profiles() -> None:
    with pytest.raises(CognitiveKernelContractError):
        PersistentBackendCandidate.create(
            candidate_id="candidate.empty",
            engine_family="example",
            roles=(),
            deployment_profiles=("single_workstation",),
            client_package=None,
            candidate_state="research_candidate",
        )


def test_owner_authorized_persistent_manifest_requires_reference() -> None:
    with pytest.raises(CognitiveKernelContractError):
        PersistentIntegrationManifest.create(
            scope=_scope(),
            manifest_id="persistent.owner",
            reference_adapter_id="alice.reference.sqlite",
            candidate_registry_sha256=(
                persistent_backend_candidate_registry_sha256()
            ),
            workload_class="owner_authorized",
            created_at=TS,
        )


def test_synthetic_manifest_keeps_phase2_writer_current() -> None:
    manifest = build_synthetic_persistent_integration_manifest(
        scope=_scope(),
        created_at=TS,
    )
    assert (
        manifest.canonical_writer_state
        == "phase2_released_writer_remains_current"
    )
    assert (
        manifest.profile_state
        == "nonproduction_persistent_integration_reference"
    )


def test_sqlite_reference_persists_and_replays_across_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "reference.sqlite3"
    stage_f = build_synthetic_stage_f_manifest(
        scope=_scope(),
        created_at=TS,
    )
    change = _change(stage_f.manifest_id, 1)

    with SQLitePersistentStageFGReferenceAdapter(database) as adapter:
        first = adapter.write(change)
        first_digest = adapter.state_sha256()
        assert first.outcome == "applied"
        assert adapter.mirror_record_count == 1

    with SQLitePersistentStageFGReferenceAdapter(database) as adapter:
        replay = adapter.write(change)
        assert replay.outcome == "duplicate"
        assert adapter.mirror_record_count == 1
        assert adapter.state_sha256() == first_digest
        integrity = adapter.integrity_receipt()
        assert integrity.integrity_state == "ok"
        assert integrity.journal_mode == "wal"


def test_outbox_sequence_conflict_is_quarantined(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sequence.sqlite3"
    stage_f = build_synthetic_stage_f_manifest(
        scope=_scope(),
        created_at=TS,
    )
    first = _change(stage_f.manifest_id, 1)
    conflicting = CanonicalChangeEnvelope.create(
        manifest_id=stage_f.manifest_id,
        change_id="different.change",
        authority_namespace="alice.owner.memory",
        outbox_sequence=1,
        operation="correction",
        canonical_record_sha256=canonical_sha256(
            {"different": True}
        ),
        mapping_version="1.0.0",
        evidence_lineage_ids=("evidence.other",),
    )

    with SQLitePersistentStageFGReferenceAdapter(database) as adapter:
        assert adapter.write(first).outcome == "applied"
        assert adapter.write(conflicting).outcome == "quarantined"
        assert adapter.mirror_record_count == 1


def test_projection_receipt_persists_and_replays(
    tmp_path: Path,
) -> None:
    database = tmp_path / "projection.sqlite3"
    manifest = build_synthetic_stage_g_manifest(
        scope=_scope(),
        created_at=TS,
    )
    receipt = ProjectionBuildReceipt.create(
        manifest=manifest,
        source_record_count=3,
        graph_node_count=3,
        graph_edge_count=2,
        vector_record_count=2,
        workflow_activity_count=1,
        deletion_exclusion_count=1,
        repair_action_count=0,
        graph_generation_sha256=canonical_sha256({"graph": 1}),
        vector_generation_sha256=canonical_sha256({"vector": 1}),
        workflow_generation_sha256=canonical_sha256(
            {"workflow": 1}
        ),
        completed_at=TS,
    )

    with SQLitePersistentStageFGReferenceAdapter(database) as adapter:
        first = adapter.persist_projection(manifest, receipt)
        assert first.outcome == "applied"
        assert adapter.projection_generation_count == 1

    with SQLitePersistentStageFGReferenceAdapter(database) as adapter:
        replay = adapter.persist_projection(manifest, receipt)
        assert replay.outcome == "duplicate"
        assert adapter.projection_generation_count == 1


def test_synthetic_persistent_evaluation_is_deterministic(
    tmp_path: Path,
) -> None:
    report_a = build_persistent_stage_f_g_report(
        database_path=tmp_path / "a.sqlite3"
    )
    report_b = build_persistent_stage_f_g_report(
        database_path=tmp_path / "b.sqlite3"
    )
    assert report_a["report_sha256"] == report_b["report_sha256"]
    assert (
        report_a["authority"]["canonical_writer"]
        == "phase2_released_writer_remains_current"
    )
    assert (
        report_a["external_candidate_connections"]
        == "not_performed_by_this_evaluation"
    )
    assert report_a["destination_selection"] == "not_performed"


def test_persistent_policy_and_docs_preserve_successor_paths() -> None:
    repo = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (
            repo
            / "policies"
            / "memory_shadow_migration_stage_f_g_persistent_policy.json"
        ).read_text(encoding="utf-8")
    )
    assert policy["capability_ceiling"] is False
    assert (
        policy["authority_state"]["canonical_writer"]
        == "phase2_released_writer_remains_current"
    )
    assert (
        policy["reference_adapter"]["role"]
        == "compatibility_reference_not_destination_selection"
    )
    later = set(policy["later_activation_states"])
    assert "candidate_specific_live_backend_integrations" in later
    assert "owner_authorized_historical_private_backfill_execution" in later
    assert "stage_h_bounded_canary_review" in later
    assert "canonical_authority_transfer" in later
    assert "production_serving_influence" in later

    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "persistent Stage F+G integration" in readme
    assert "Phase 2 remains" in readme

    implementation = (
        repo
        / "docs"
        / "MEMORY_SHADOW_MIGRATION_STAGE_F_G_PERSISTENT_IMPLEMENTATION.md"
    ).read_text(encoding="utf-8")
    assert "SQLite is a compatibility/reference durability oracle" in implementation
    assert "KurrentDB" in implementation
    assert "Neo4j" in implementation
    assert "Qdrant" in implementation
    assert "Temporal" in implementation
