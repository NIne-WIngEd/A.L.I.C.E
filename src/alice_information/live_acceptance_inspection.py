"""Metadata-safe inspection for private P4.10c live acceptance records."""

from __future__ import annotations

from dataclasses import dataclass

from .live_acceptance import InformationLiveAcceptanceRecord


@dataclass(frozen=True)
class InformationLiveAcceptanceInspection:
    audit_version: str
    release_id: str
    approved: bool
    repository_commit: str
    rollback_commit: str
    package_version: str
    deterministic_tests: str
    repository_regression: str
    repository_subtests_passed: int
    live_outcome: str
    live_search_results: int
    live_fetch_attempts: int
    live_fetches: int
    live_fetch_failures: int
    grounded_sources: int
    p36_pre_commit_validations: int
    p45b_validation_outcome: str
    p45b_validation_sha256: str
    decision_reasons: tuple[str, ...]
    record_sha256: str


def inspect_live_acceptance(
    record: InformationLiveAcceptanceRecord,
) -> InformationLiveAcceptanceInspection:
    record.validate()
    live = dict(record.live_research_receipt)
    return InformationLiveAcceptanceInspection(
        audit_version=record.audit_version,
        release_id=record.release_id,
        approved=record.approved,
        repository_commit=record.repository_commit,
        rollback_commit=record.rollback_commit,
        package_version=record.package_version,
        deterministic_tests=(
            f"{record.deterministic_test_passed}/"
            f"{record.deterministic_test_collected}"
        ),
        repository_regression=(
            f"{record.repository_regression_passed}/"
            f"{record.repository_regression_collected}"
        ),
        repository_subtests_passed=record.repository_regression_subtests_passed,
        live_outcome=str(live["outcome"]),
        live_search_results=int(live["search_result_count"]),
        live_fetch_attempts=int(live["fetch_attempt_count"]),
        live_fetches=len(live["fetch_receipt_sha256s"]),
        live_fetch_failures=len(live["fetch_failure_sha256s"]),
        grounded_sources=len(live["grounded_source_sha256s"]),
        p36_pre_commit_validations=int(live["pre_commit_validation_count"]),
        p45b_validation_outcome=str(live["citation_validation_outcome"]),
        p45b_validation_sha256=str(live["validation_sha256"]),
        decision_reasons=record.decision_reasons,
        record_sha256=record.record_sha256,
    )


def render_live_acceptance_inspection(
    inspection: InformationLiveAcceptanceInspection,
) -> str:
    return "\n".join(
        (
            f"audit_version={inspection.audit_version}",
            f"release_id={inspection.release_id}",
            f"approved={str(inspection.approved).lower()}",
            f"repository_commit={inspection.repository_commit}",
            f"rollback_commit={inspection.rollback_commit}",
            f"package_version={inspection.package_version}",
            f"deterministic_tests={inspection.deterministic_tests}",
            f"repository_regression={inspection.repository_regression}",
            f"repository_subtests_passed={inspection.repository_subtests_passed}",
            f"live_outcome={inspection.live_outcome}",
            f"live_search_results={inspection.live_search_results}",
            f"live_fetch_attempts={inspection.live_fetch_attempts}",
            f"live_fetches={inspection.live_fetches}",
            f"live_fetch_failures={inspection.live_fetch_failures}",
            f"grounded_sources={inspection.grounded_sources}",
            f"p36_pre_commit_validations={inspection.p36_pre_commit_validations}",
            f"p45b_validation_outcome={inspection.p45b_validation_outcome}",
            f"p45b_validation_sha256={inspection.p45b_validation_sha256}",
            f"decision_reasons={','.join(inspection.decision_reasons)}",
            f"record_sha256={inspection.record_sha256}",
        )
    )
