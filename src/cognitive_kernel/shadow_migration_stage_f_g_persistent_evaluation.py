"""Deterministic persistent-reference evaluation for Stage F+G integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import canonical_sha256
from .contracts import ProductHostScope
from .shadow_migration_stage_f_g import (
    CanonicalChangeEnvelope,
    ProjectionBuildReceipt,
    build_synthetic_stage_f_manifest,
    build_synthetic_stage_g_manifest,
    run_controlled_mirror_batch,
)
from .shadow_migration_stage_f_g_persistent import (
    SQLitePersistentStageFGReferenceAdapter,
    build_persistent_backend_candidate_registry,
    build_synthetic_persistent_integration_manifest,
    persistent_backend_candidate_registry_sha256,
)

TS = "2026-08-07T15:30:00Z"


def _scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="stage-f-g-persistent-synthetic-host",
        schema_version="1.0.0",
        encryption_domain="stage-f-g-persistent-synthetic-domain",
    )


def _change(
    *,
    manifest_id: str,
    sequence: int,
    operation: str,
) -> CanonicalChangeEnvelope:
    return CanonicalChangeEnvelope.create(
        manifest_id=manifest_id,
        change_id=f"persistent.change.{sequence}",
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


def build_persistent_stage_f_g_report(
    *, database_path: Path
) -> dict[str, object]:
    scope = _scope()
    integration_manifest = (
        build_synthetic_persistent_integration_manifest(
            scope=scope,
            created_at=TS,
        )
    )
    stage_f_manifest = build_synthetic_stage_f_manifest(
        scope=scope,
        created_at=TS,
    )
    changes = (
        _change(
            manifest_id=stage_f_manifest.manifest_id,
            sequence=1,
            operation="upsert",
        ),
        _change(
            manifest_id=stage_f_manifest.manifest_id,
            sequence=2,
            operation="correction",
        ),
        _change(
            manifest_id=stage_f_manifest.manifest_id,
            sequence=3,
            operation="delete",
        ),
    )

    first_results = []

    with SQLitePersistentStageFGReferenceAdapter(
        database_path
    ) as adapter:
        def write_first(change):
            result = adapter.write(change)
            first_results.append(result)
            return result

        first_batch = run_controlled_mirror_batch(
            manifest=stage_f_manifest,
            changes=changes,
            write_change=write_first,
            completed_at=TS,
        )
        first_state = adapter.state_sha256()
        first_integrity = adapter.integrity_receipt()

    replay_results = []

    with SQLitePersistentStageFGReferenceAdapter(
        database_path
    ) as adapter:
        def write_replay(change):
            result = adapter.write(change)
            replay_results.append(result)
            return result

        replay_batch = run_controlled_mirror_batch(
            manifest=stage_f_manifest,
            changes=changes,
            write_change=write_replay,
            completed_at=TS,
        )
        replay_state = adapter.state_sha256()

        stage_g_manifest = build_synthetic_stage_g_manifest(
            scope=scope,
            created_at=TS,
        )
        projection_receipt = ProjectionBuildReceipt.create(
            manifest=stage_g_manifest,
            source_record_count=3,
            graph_node_count=3,
            graph_edge_count=2,
            vector_record_count=2,
            workflow_activity_count=1,
            deletion_exclusion_count=1,
            repair_action_count=0,
            graph_generation_sha256=canonical_sha256(
                {"plane": "graph", "generation": "g1"}
            ),
            vector_generation_sha256=canonical_sha256(
                {"plane": "vector", "generation": "g1"}
            ),
            workflow_generation_sha256=canonical_sha256(
                {"plane": "workflow", "generation": "g1"}
            ),
            completed_at=TS,
        )
        projection_first = adapter.persist_projection(
            stage_g_manifest,
            projection_receipt,
        )
        projection_replay = adapter.persist_projection(
            stage_g_manifest,
            projection_receipt,
        )
        final_integrity = adapter.integrity_receipt()
        checkpoint = adapter.checkpoint_wal()

    candidates = build_persistent_backend_candidate_registry()

    payload: dict[str, object] = {
        "evaluation_id": (
            "phase2-shadow-migration-stage-f-g-persistent-synthetic-1"
        ),
        "integration_manifest": integration_manifest.metadata_record(),
        "candidate_registry_sha256": (
            persistent_backend_candidate_registry_sha256()
        ),
        "candidate_registry": [
            item.metadata_record() for item in candidates
        ],
        "reference_adapter": {
            "adapter_id": (
                SQLitePersistentStageFGReferenceAdapter.adapter_id
            ),
            "role": (
                "compatibility_reference_durability_oracle_"
                "not_destination_selection"
            ),
            "artifact_kind": "sqlite_reference_database",
        },
        "mirror": {
            "first_batch_sha256": first_batch.receipt_sha256,
            "replay_batch_sha256": replay_batch.receipt_sha256,
            "first_state_sha256": first_state,
            "replay_state_sha256": replay_state,
            "first_outcomes": [
                result.outcome for result in first_results
            ],
            "replay_outcomes": [
                result.outcome for result in replay_results
            ],
        },
        "projection": {
            "first_outcome": projection_first.outcome,
            "replay_outcome": projection_replay.outcome,
            "persistent_projection_receipt_sha256": (
                projection_first.receipt_sha256
            ),
        },
        "integrity": {
            "first": first_integrity.metadata_record(),
            "final": final_integrity.metadata_record(),
            "wal_checkpoint_state": (
                "completed" if int(checkpoint[0]) == 0 else "busy"
            ),
        },
        "authority": {
            "canonical_writer": (
                "phase2_released_writer_remains_current"
            ),
            "authority_transition": "unchanged",
            "stage_h_state": "review_not_activated",
        },
        "private_execution": (
            "synthetic_reference_evaluation_only"
        ),
        "external_candidate_connections": (
            "not_performed_by_this_evaluation"
        ),
        "destination_selection": "not_performed",
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--database", required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    report_path = Path(args.report).resolve()
    database_path = Path(args.database).resolve()

    if report_path.is_relative_to(repo):
        raise SystemExit(
            "persistent integration report must remain outside public Git"
        )
    if database_path.is_relative_to(repo):
        raise SystemExit(
            "persistent reference database must remain outside public Git"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_persistent_stage_f_g_report(
        database_path=database_path
    )
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "stage_f_g_persistent_evaluation_id="
        + str(payload["evaluation_id"])
    )
    print(
        "stage_f_g_persistent_report_sha256="
        + str(payload["report_sha256"])
    )
    print("persistent_reference_state=operational_nonproduction")
    print("candidate_registry_state=active_research")
    print("external_candidate_connections=not_performed")
    print("phase2_canonical_writer_remains_current=True")
    print("stage_h_state=review_not_activated")
    print("report=" + str(report_path))
    print("database=" + str(database_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
