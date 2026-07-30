"""Metadata-only inspection for P4.9 Phase 4 release records."""

from __future__ import annotations

from dataclasses import dataclass

from .release_audit import (
    Phase4ReleaseDecision,
    verify_phase4_release_decision,
)


@dataclass(frozen=True)
class Phase4ReleaseInspection:
    audit_version: str
    release_id: str
    approved: bool
    decision_reasons: tuple[str, ...]
    repository_commit: str
    rollback_commit: str
    package_version: str
    evaluation_version: str
    benchmark_id: str
    test_set_version: str
    runtime_version: str
    runtime_manifest_id: str
    runtime_target_file_count: int
    runtime_collected_test_count: int
    runtime_passed_test_count: int
    runtime_skipped_test_count: int
    runtime_network_guard_active: bool
    case_count: int
    passed_case_count: int
    failed_metric_ids: tuple[str, ...]
    evaluation_report_digest: str
    runtime_evidence_digest: str
    runtime_backed_report_digest: str
    record_digest: str


def inspect_phase4_release(decision: Phase4ReleaseDecision) -> Phase4ReleaseInspection:
    verify_phase4_release_decision(decision)
    return Phase4ReleaseInspection(
        audit_version=decision.audit_version,
        release_id=decision.release_id,
        approved=decision.approved,
        decision_reasons=decision.decision_reasons,
        repository_commit=decision.repository_commit,
        rollback_commit=decision.rollback_commit,
        package_version=decision.package_version,
        evaluation_version=decision.evaluation_version,
        benchmark_id=decision.benchmark_id,
        test_set_version=decision.test_set_version,
        runtime_version=decision.runtime_version,
        runtime_manifest_id=decision.runtime_manifest_id,
        runtime_target_file_count=decision.runtime_target_file_count,
        runtime_collected_test_count=decision.runtime_collected_test_count,
        runtime_passed_test_count=decision.runtime_passed_test_count,
        runtime_skipped_test_count=decision.runtime_skipped_test_count,
        runtime_network_guard_active=decision.runtime_network_guard_active,
        case_count=decision.case_count,
        passed_case_count=decision.passed_case_count,
        failed_metric_ids=tuple(
            str(item["metric_id"])
            for item in decision.metric_results
            if not bool(item["passed"])
        ),
        evaluation_report_digest=decision.evaluation_report_digest,
        runtime_evidence_digest=decision.runtime_evidence_digest,
        runtime_backed_report_digest=decision.runtime_backed_report_digest,
        record_digest=decision.record_digest,
    )


def render_phase4_release_inspection(value: Phase4ReleaseInspection) -> str:
    return "\n".join(
        (
            f"audit_version={value.audit_version}",
            f"release_id={value.release_id}",
            f"approved={str(value.approved).lower()}",
            f"decision_reasons={','.join(value.decision_reasons)}",
            f"repository_commit={value.repository_commit}",
            f"rollback_commit={value.rollback_commit}",
            f"package_version={value.package_version}",
            f"evaluation_version={value.evaluation_version}",
            f"benchmark_id={value.benchmark_id}",
            f"test_set_version={value.test_set_version}",
            f"runtime_version={value.runtime_version}",
            f"runtime_manifest_id={value.runtime_manifest_id}",
            f"runtime_target_file_count={value.runtime_target_file_count}",
            f"runtime_collected_test_count={value.runtime_collected_test_count}",
            f"runtime_passed_test_count={value.runtime_passed_test_count}",
            f"runtime_skipped_test_count={value.runtime_skipped_test_count}",
            "runtime_network_guard_active="
            + str(value.runtime_network_guard_active).lower(),
            f"case_count={value.case_count}",
            f"passed_case_count={value.passed_case_count}",
            f"failed_metric_ids={','.join(value.failed_metric_ids)}",
            f"evaluation_report_digest={value.evaluation_report_digest}",
            f"runtime_evidence_digest={value.runtime_evidence_digest}",
            f"runtime_backed_report_digest={value.runtime_backed_report_digest}",
            f"record_digest={value.record_digest}",
        )
    )
