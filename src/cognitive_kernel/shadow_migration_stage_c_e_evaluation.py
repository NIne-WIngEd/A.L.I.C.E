"""Deterministic synthetic Stage C+E destination-candidate evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from .canonical import canonical_json_bytes, canonical_sha256
from .contracts import ProductHostScope
from .shadow_migration_stage_c_e import (
    DestinationCandidateEvaluation,
    ShadowReadComparisonReceipt,
    ShadowReadObservation,
    ShadowReadWorkload,
    build_m2_destination_candidate_profile,
)


def _assert_report_outside_repo(report_path: Path, repo_root: Path) -> None:
    report = report_path.resolve()
    repo = repo_root.resolve()
    if report == repo or repo in report.parents:
        raise ValueError("Stage C+E evaluation reports must remain outside public Git")


def build_synthetic_stage_c_e_report() -> dict[str, object]:
    timestamp = "2026-08-07T04:30:00Z"
    scope = ProductHostScope.create(
        product_id="alice",
        host_instance_id="stage-ce-synthetic-host",
        schema_version="1.0.0",
        encryption_domain="stage-ce-synthetic-domain",
    )
    profile = build_m2_destination_candidate_profile(scope=scope, created_at=timestamp)
    workloads = (
        ShadowReadWorkload.create(
            scope=scope,
            workload_id="shadow.workload.identity",
            workload_class="synthetic",
            query_sha256=canonical_sha256({"query_class": "identity"}),
            expected_record_ids=("claim.identity.1", "claim.identity.2"),
            expected_conflict_record_ids=("conflict.identity.1",),
            expected_correction_record_ids=("correction.identity.1",),
            expected_deleted_record_ids=("claim.identity.deleted",),
            created_at=timestamp,
        ),
        ShadowReadWorkload.create(
            scope=scope,
            workload_id="shadow.workload.project",
            workload_class="synthetic",
            query_sha256=canonical_sha256({"query_class": "project"}),
            expected_record_ids=("claim.project.1",),
            expected_deleted_record_ids=("claim.project.deleted",),
            created_at=timestamp,
        ),
    )
    comparisons: list[ShadowReadComparisonReceipt] = []
    for index, workload in enumerate(workloads):
        baseline = ShadowReadObservation.create(
            scope=scope,
            workload_id=workload.workload_id,
            candidate_id="phase2.released.baseline",
            result_record_ids=workload.expected_record_ids,
            conflict_record_ids=workload.expected_conflict_record_ids,
            correction_record_ids=workload.expected_correction_record_ids,
            deleted_record_ids_returned=(),
            latency_ms=18 + index,
            staleness_ms=5 + index,
            product_isolation_passed=True,
            private_payload_exposed=False,
            explanation_trace_sha256=canonical_sha256(
                {"source": "phase2", "workload": workload.workload_id}
            ),
            observed_at=timestamp,
        )
        candidate = ShadowReadObservation.create(
            scope=scope,
            workload_id=workload.workload_id,
            candidate_id=profile.candidate_id,
            result_record_ids=workload.expected_record_ids,
            conflict_record_ids=workload.expected_conflict_record_ids,
            correction_record_ids=workload.expected_correction_record_ids,
            deleted_record_ids_returned=(),
            latency_ms=14 + index,
            staleness_ms=3 + index,
            product_isolation_passed=True,
            private_payload_exposed=False,
            explanation_trace_sha256=canonical_sha256(
                {"source": "m2-candidate", "workload": workload.workload_id}
            ),
            observed_at=timestamp,
        )
        comparisons.append(
            ShadowReadComparisonReceipt.create(
                workload=workload,
                baseline=baseline,
                candidate=candidate,
                compared_at=timestamp,
            )
        )
    evaluation = DestinationCandidateEvaluation.create(
        profile=profile,
        comparisons=comparisons,
        evaluated_at=timestamp,
    )
    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "evaluation_id": "phase2-shadow-migration-stage-c-e-synthetic-1",
        "scope": scope.metadata_record(),
        "candidate_profile": profile.metadata_record(),
        "workloads": [item.metadata_record() for item in workloads],
        "comparisons": [item.metadata_record() for item in comparisons],
        "evaluation": evaluation.metadata_record(),
        "material_state": {
            "stage_a_b_operational": True,
            "stage_c_e_prototype_operational": True,
            "phase2_released_profile_remains_current": True,
            "private_payload_read": False,
            "historical_private_backfill": False,
            "production_write_mirroring": False,
            "canonical_authority_transfer": False,
            "production_influence": False,
            "cutover": False,
            "phase2_retired": False,
            "p5_1e_unblocked": False,
            "destination_capability_ceiling": False,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root)
    report_path = Path(args.report)
    _assert_report_outside_repo(report_path, repo_root)
    report = build_synthetic_stage_c_e_report()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    evaluation = report["evaluation"]
    assert isinstance(evaluation, dict)
    print(f"stage_c_e_evaluation_id={report['evaluation_id']}")
    print(f"stage_c_e_report_sha256={report['report_sha256']}")
    print(f"candidate_recommendation={evaluation['recommendation']}")
    print("phase2_released_profile_remains_current=True")
    print("production_influence=False")
    print(f"report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
