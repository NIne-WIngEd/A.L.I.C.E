"""Metadata-only inspection for P3.11 Phase 3 release records."""

from __future__ import annotations

from dataclasses import dataclass

from .release_audit import Phase3ReleaseDecision, verify_phase3_release_decision


@dataclass(frozen=True)
class Phase3ReleaseInspection:
    audit_version: str
    release_id: str
    approved: bool
    decision_reasons: tuple[str, ...]
    repository_commit: str
    rollback_commit: str
    package_version: str
    evaluation_version: str
    evidence_manifest_id: str
    evidence_target_count: int
    evidence_passed_target_count: int
    benchmark_id: str
    test_set_version: str
    case_count: int
    passed_case_count: int
    failed_metric_ids: tuple[str, ...]
    record_digest: str


def inspect_phase3_release(decision: Phase3ReleaseDecision) -> Phase3ReleaseInspection:
    verify_phase3_release_decision(decision)
    return Phase3ReleaseInspection(
        audit_version=decision.audit_version,
        release_id=decision.release_id,
        approved=decision.approved,
        decision_reasons=decision.decision_reasons,
        repository_commit=decision.repository_commit,
        rollback_commit=decision.rollback_commit,
        package_version=decision.package_version,
        evaluation_version=decision.evaluation_version,
        evidence_manifest_id=decision.evidence_manifest_id,
        evidence_target_count=decision.evidence_target_count,
        evidence_passed_target_count=decision.evidence_passed_target_count,
        benchmark_id=decision.benchmark_id,
        test_set_version=decision.test_set_version,
        case_count=decision.case_count,
        passed_case_count=decision.passed_case_count,
        failed_metric_ids=tuple(
            str(item["metric_id"])
            for item in decision.metric_results
            if not bool(item["passed"])
        ),
        record_digest=decision.record_digest,
    )


def render_phase3_release_inspection(value: Phase3ReleaseInspection) -> str:
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
            f"evidence_manifest_id={value.evidence_manifest_id}",
            f"evidence_target_count={value.evidence_target_count}",
            f"evidence_passed_target_count={value.evidence_passed_target_count}",
            f"benchmark_id={value.benchmark_id}",
            f"test_set_version={value.test_set_version}",
            f"case_count={value.case_count}",
            f"passed_case_count={value.passed_case_count}",
            f"failed_metric_ids={','.join(value.failed_metric_ids)}",
            f"record_digest={value.record_digest}",
        )
    )
