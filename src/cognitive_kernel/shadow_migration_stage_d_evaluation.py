"""Deterministic synthetic evaluation for Phase 2 shadow migration Stage D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import canonical_sha256
from .contracts import ProductHostScope
from .shadow_migration_stage_d import (
    HistoricalBackfillCheckpoint,
    HistoricalBackfillRecord,
    InMemoryIdempotentBackfillSink,
    build_synthetic_stage_d_manifest,
    run_historical_backfill_batch,
)

TS = "2026-08-07T06:15:00Z"


def _scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="stage-d-synthetic-host",
        schema_version="1.0.0",
        encryption_domain="stage-d-synthetic-domain",
    )


def _record(
    *,
    manifest_id: str,
    record_id: str,
    checkpoint: str,
    disposition: str = "accepted",
    provenance_state: str = "complete",
    reason: str | None = None,
    evidence: tuple[str, ...] = (),
    deletion: tuple[str, ...] = (),
) -> HistoricalBackfillRecord:
    return HistoricalBackfillRecord.create(
        manifest_id=manifest_id,
        source_record_id=record_id,
        source_checkpoint=checkpoint,
        source_record_sha256=canonical_sha256(
            {"source_record_id": record_id, "checkpoint": checkpoint}
        ),
        mapped_record_sha256=canonical_sha256(
            {"mapped_record_id": record_id, "mapping": "1.0.0"}
        ),
        mapping_version="1.0.0",
        provenance_state=provenance_state,
        evidence_lineage_ids=evidence,
        deletion_lineage_ids=deletion,
        disposition=disposition,
        disposition_reason=reason,
    )


def build_synthetic_stage_d_report() -> dict[str, object]:
    scope = _scope()
    manifest = build_synthetic_stage_d_manifest(scope=scope, created_at=TS)
    sink = InMemoryIdempotentBackfillSink()

    records = (
        _record(
            manifest_id=manifest.manifest_id,
            record_id="phase2.record.1",
            checkpoint="000001",
            evidence=("evidence.1",),
        ),
        _record(
            manifest_id=manifest.manifest_id,
            record_id="phase2.record.2",
            checkpoint="000002",
            evidence=("evidence.2",),
            deletion=("deletion.2",),
        ),
        _record(
            manifest_id=manifest.manifest_id,
            record_id="phase2.record.3",
            checkpoint="000003",
            disposition="quarantined",
            provenance_state="missing",
            reason="missing_provenance",
        ),
        _record(
            manifest_id=manifest.manifest_id,
            record_id="phase2.record.4",
            checkpoint="000004",
            disposition="ambiguous",
            provenance_state="partial",
            reason="ambiguous_relation_target",
            evidence=("evidence.4",),
        ),
        _record(
            manifest_id=manifest.manifest_id,
            record_id="phase2.record.5",
            checkpoint="000005",
            disposition="rejected",
            provenance_state="partial",
            reason="unsupported_semantics",
            evidence=("evidence.5",),
        ),
    )

    first = run_historical_backfill_batch(
        manifest=manifest,
        records=records,
        write_accepted_record=sink.write,
        completed_at=TS,
    )
    replay = run_historical_backfill_batch(
        manifest=manifest,
        records=records,
        write_accepted_record=sink.write,
        completed_at=TS,
    )
    checkpoint = HistoricalBackfillCheckpoint.from_receipts(
        manifest=manifest,
        receipts=(first, replay),
    )

    report: dict[str, object] = {
        "evaluation_id": "phase2-shadow-migration-stage-d-synthetic-1",
        "schema_version": "1.0.0",
        "material_state": {
            "stage_d_deterministic_backfill_prototype": "operational",
            "historical_private_payload_execution": "not_performed_by_synthetic_evaluation",
            "canonical_authority": "phase2_released_profile_remains_current",
            "destination_role": "shadow_backfill_candidate",
            "production_serving_effect": "none",
            "p5_1e_state": "paused",
        },
        "manifest": manifest.metadata_record(),
        "first_batch": first.metadata_record(),
        "replay_batch": replay.metadata_record(),
        "checkpoint": checkpoint.metadata_record(),
        "synthetic_sink_record_count": sink.record_count,
        "next_gate": (
            "owner_authorized_stage_d_batch_execution_or_separately_evaluated_"
            "successor_research_profile"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _assert_report_outside_repo(report: Path, repo_root: Path) -> None:
    report_resolved = report.resolve()
    repo_resolved = repo_root.resolve()
    if report_resolved == repo_resolved or repo_resolved in report_resolved.parents:
        raise ValueError("Stage D evaluation report must remain outside public Git")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report_path = Path(args.report).resolve()
    _assert_report_outside_repo(report_path, repo_root)

    report = build_synthetic_stage_d_report()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"stage_d_evaluation_id={report['evaluation_id']}")
    print(f"stage_d_report_sha256={report['report_sha256']}")
    print(
        "stage_d_state="
        + str(report["material_state"]["stage_d_deterministic_backfill_prototype"])
    )
    print(
        "historical_private_payload_execution="
        + str(report["material_state"]["historical_private_payload_execution"])
    )
    print(
        "phase2_released_profile_remains_current="
        + str(
            report["material_state"]["canonical_authority"]
            == "phase2_released_profile_remains_current"
        )
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
