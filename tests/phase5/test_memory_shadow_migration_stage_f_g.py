from __future__ import annotations

from pathlib import Path
import json

import pytest

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_sha256
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.shadow_migration_stage_f_g import (
    CanonicalChangeEnvelope,
    ControlledMirrorManifest,
    InMemoryIdempotentMirrorSink,
    ProjectionBuildManifest,
    ProjectionBuildReceipt,
    build_synthetic_stage_f_manifest,
    build_synthetic_stage_g_manifest,
    run_controlled_mirror_batch,
)
from cognitive_kernel.shadow_migration_stage_f_g_evaluation import (
    build_synthetic_stage_f_g_report,
)

TS = "2026-08-07T15:00:00Z"


def _scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="stage-f-g-test-host",
        schema_version="1.0.0",
        encryption_domain="stage-f-g-test-domain",
    )


def _change(manifest_id: str, sequence: int, operation: str = "upsert") -> CanonicalChangeEnvelope:
    return CanonicalChangeEnvelope.create(
        manifest_id=manifest_id,
        change_id=f"change.{sequence}",
        authority_namespace="alice.owner.memory",
        outbox_sequence=sequence,
        operation=operation,
        canonical_record_sha256=canonical_sha256({"sequence": sequence}),
        mapping_version="1.0.0",
        evidence_lineage_ids=(f"evidence.{sequence}",),
        deletion_lineage_ids=((f"deletion.{sequence}",) if operation == "delete" else ()),
    )


def test_stage_f_requires_owner_authorization_reference() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ControlledMirrorManifest.create(
            scope=_scope(),
            manifest_id="owner.private.mirror",
            canonical_writer_id="phase2.writer",
            canonical_authority_generation="phase2.generation",
            destination_candidate_id="candidate.one",
            outbox_stream_id="outbox.one",
            mapping_version="1.0.0",
            workload_class="owner_authorized",
            created_at=TS,
        )


def test_stage_f_synthetic_manifest_preserves_nonproduction_profile() -> None:
    manifest = build_synthetic_stage_f_manifest(scope=_scope(), created_at=TS)
    assert manifest.profile_state == "nonproduction_controlled_mirroring"
    assert manifest.authority_transition_state == "unchanged"
    assert manifest.serving_state == "shadow_only"


def test_stage_f_mirror_is_idempotent_and_keeps_canonical_writer_state() -> None:
    manifest = build_synthetic_stage_f_manifest(scope=_scope(), created_at=TS)
    changes = (_change(manifest.manifest_id, 1), _change(manifest.manifest_id, 2, "delete"))
    sink = InMemoryIdempotentMirrorSink()
    first = run_controlled_mirror_batch(
        manifest=manifest,
        changes=changes,
        write_change=sink.write,
        completed_at=TS,
    )
    replay = run_controlled_mirror_batch(
        manifest=manifest,
        changes=changes,
        write_change=sink.write,
        completed_at=TS,
    )
    assert first.applied_count == 2
    assert replay.duplicate_count == 2
    assert first.canonical_writer_state == "unchanged"
    assert first.authority_transition_state == "unchanged"
    assert first.deletion_lineage_sha256 == replay.deletion_lineage_sha256


def test_stage_f_requires_ordered_unique_outbox_sequence() -> None:
    manifest = build_synthetic_stage_f_manifest(scope=_scope(), created_at=TS)
    sink = InMemoryIdempotentMirrorSink()
    changes = (_change(manifest.manifest_id, 2), _change(manifest.manifest_id, 1))
    with pytest.raises(CognitiveKernelContractError):
        run_controlled_mirror_batch(
            manifest=manifest,
            changes=changes,
            write_change=sink.write,
            completed_at=TS,
        )


def test_stage_g_requires_graph_vector_workflow_planes() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ProjectionBuildManifest.create(
            scope=_scope(),
            build_id="build.one",
            destination_candidate_id="candidate.one",
            source_generation_id="source.g1",
            source_snapshot_sha256=canonical_sha256({"source": 1}),
            graph_generation_id="graph.g1",
            vector_generation_id="vector.g1",
            workflow_generation_id="workflow.g1",
            deletion_watermark="deletion.1",
            projection_planes=("graph", "vector"),
            created_at=TS,
        )


def test_stage_g_receipt_binds_generations_and_deletion_watermark() -> None:
    manifest = build_synthetic_stage_g_manifest(scope=_scope(), created_at=TS)
    receipt = ProjectionBuildReceipt.create(
        manifest=manifest,
        source_record_count=4,
        graph_node_count=4,
        graph_edge_count=3,
        vector_record_count=3,
        workflow_activity_count=2,
        deletion_exclusion_count=1,
        repair_action_count=0,
        graph_generation_sha256=canonical_sha256({"graph": 1}),
        vector_generation_sha256=canonical_sha256({"vector": 1}),
        workflow_generation_sha256=canonical_sha256({"workflow": 1}),
        completed_at=TS,
    )
    assert receipt.deletion_watermark == manifest.deletion_watermark
    assert receipt.deletion_exclusion_count == 1
    assert len(receipt.receipt_sha256) == 64


def test_stage_f_g_synthetic_report_is_deterministic() -> None:
    first = build_synthetic_stage_f_g_report()
    second = build_synthetic_stage_f_g_report()
    assert first == second
    assert first["stage_f_first_batch"]["applied_count"] == 3
    assert first["stage_f_replay_batch"]["duplicate_count"] == 3
    assert first["stage_g_receipt"]["deletion_exclusion_count"] == 1
    assert first["material_state"]["authority_transition"] == "unchanged"


def test_stage_f_g_policy_keeps_successor_capabilities_open() -> None:
    policy_path = (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "memory_shadow_migration_stage_f_g_policy.json"
    )
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    assert payload["capability_ceiling"] is False
    assert payload["research_status"] == "allowed"
    assert "owner_authorized_historical_private_backfill_execution" in payload["later_activation_states"]
    assert "canary_authority" in payload["later_activation_states"]
    assert "canonical_authority_transfer" in payload["later_activation_states"]
    assert "production_serving_influence" in payload["later_activation_states"]
    assert "p5_1e_storage_admission" in payload["later_activation_states"]


def test_stage_f_g_docs_do_not_claim_production_or_private_execution() -> None:
    repo = Path(__file__).resolve().parents[2]
    doc = (repo / "docs/MEMORY_SHADOW_MIGRATION_STAGE_F_G_IMPLEMENTATION.md").read_text(encoding="utf-8")
    assert "nonproduction" in doc.lower()
    assert "real private" in doc.lower()
    assert "Phase 2 remains" in doc
