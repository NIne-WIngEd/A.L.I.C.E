"""Deterministic synthetic evaluation for shadow migration Stages F+G."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import canonical_sha256
from .contracts import ProductHostScope
from .shadow_migration_stage_f_g import (
    CanonicalChangeEnvelope,
    InMemoryIdempotentMirrorSink,
    ProjectionBuildReceipt,
    build_synthetic_stage_f_manifest,
    build_synthetic_stage_g_manifest,
    run_controlled_mirror_batch,
)

TS = "2026-08-07T15:00:00Z"


def _scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="stage-f-g-synthetic-host",
        schema_version="1.0.0",
        encryption_domain="stage-f-g-synthetic-domain",
    )


def _change(
    *,
    manifest_id: str,
    change_id: str,
    sequence: int,
    operation: str,
    deletion: tuple[str, ...] = (),
) -> CanonicalChangeEnvelope:
    return CanonicalChangeEnvelope.create(
        manifest_id=manifest_id,
        change_id=change_id,
        authority_namespace="alice.owner.memory",
        outbox_sequence=sequence,
        operation=operation,
        canonical_record_sha256=canonical_sha256(
            {"change_id": change_id, "operation": operation}
        ),
        mapping_version="1.0.0",
        evidence_lineage_ids=(f"evidence.{sequence}",),
        deletion_lineage_ids=deletion,
    )


def build_synthetic_stage_f_g_report() -> dict[str, object]:
    scope = _scope()
    mirror_manifest = build_synthetic_stage_f_manifest(scope=scope, created_at=TS)
    sink = InMemoryIdempotentMirrorSink()
    changes = (
        _change(
            manifest_id=mirror_manifest.manifest_id,
            change_id="change.1",
            sequence=1,
            operation="upsert",
        ),
        _change(
            manifest_id=mirror_manifest.manifest_id,
            change_id="change.2",
            sequence=2,
            operation="correction",
        ),
        _change(
            manifest_id=mirror_manifest.manifest_id,
            change_id="change.3",
            sequence=3,
            operation="delete",
            deletion=("deletion.3",),
        ),
    )
    first = run_controlled_mirror_batch(
        manifest=mirror_manifest,
        changes=changes,
        write_change=sink.write,
        completed_at=TS,
    )
    replay = run_controlled_mirror_batch(
        manifest=mirror_manifest,
        changes=changes,
        write_change=sink.write,
        completed_at=TS,
    )

    projection_manifest = build_synthetic_stage_g_manifest(
        scope=scope, created_at=TS
    )
    projection = ProjectionBuildReceipt.create(
        manifest=projection_manifest,
        source_record_count=3,
        graph_node_count=3,
        graph_edge_count=2,
        vector_record_count=2,
        workflow_activity_count=3,
        deletion_exclusion_count=1,
        repair_action_count=0,
        graph_generation_sha256=canonical_sha256(
            {"generation": projection_manifest.graph_generation_id, "nodes": 3}
        ),
        vector_generation_sha256=canonical_sha256(
            {"generation": projection_manifest.vector_generation_id, "records": 2}
        ),
        workflow_generation_sha256=canonical_sha256(
            {
                "generation": projection_manifest.workflow_generation_id,
                "activities": 3,
            }
        ),
        completed_at=TS,
    )

    report: dict[str, object] = {
        "evaluation_id": "phase2-shadow-migration-stage-f-g-synthetic-1",
        "schema_version": "1.0.0",
        "material_state": {
            "stage_f_controlled_mirroring_prototype": "operational_nonproduction",
            "stage_g_projection_generation_prototype": "operational_nonproduction",
            "canonical_writer": "phase2_released_writer_remains_current",
            "authority_transition": "unchanged",
            "production_serving": "not_activated_by_this_profile",
            "historical_private_payload_execution": "not_performed",
        },
        "stage_f_manifest": mirror_manifest.metadata_record(),
        "stage_f_first_batch": first.metadata_record(),
        "stage_f_replay_batch": replay.metadata_record(),
        "stage_f_synthetic_sink_record_count": sink.record_count,
        "stage_g_manifest": projection_manifest.metadata_record(),
        "stage_g_receipt": projection.metadata_record(),
        "next_gate": (
            "expanded_stage_f_g_backend_integration_or_owner_authorized_"
            "stage_d_execution_or_stage_h_canary_admission"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _assert_report_outside_repo(report: Path, repo_root: Path) -> None:
    report_resolved = report.resolve()
    repo_resolved = repo_root.resolve()
    if report_resolved == repo_resolved or repo_resolved in report_resolved.parents:
        raise ValueError("Stage F+G evaluation report must remain outside public Git")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report_path = Path(args.report).resolve()
    _assert_report_outside_repo(report_path, repo_root)

    report = build_synthetic_stage_f_g_report()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"stage_f_g_evaluation_id={report['evaluation_id']}")
    print(f"stage_f_g_report_sha256={report['report_sha256']}")
    print("stage_f_state=operational_nonproduction")
    print("stage_g_state=operational_nonproduction")
    print("canonical_writer=phase2_released_writer_remains_current")
    print("production_serving=not_activated_by_this_profile")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
