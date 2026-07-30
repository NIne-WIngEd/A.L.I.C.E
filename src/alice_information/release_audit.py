"""Exact-commit private release audit for A.L.I.C.E. Phase 4 P4.9."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .final_evaluation_contract import (
    CANONICAL_BENCHMARK_ID,
    CANONICAL_POLICY_ID,
    CANONICAL_TEST_SET_VERSION,
    INFORMATION_FINAL_EVALUATION_VERSION,
    canonical_json,
    sha256_canonical,
)
from .final_evaluation_runtime import (
    CANONICAL_RUNTIME_MANIFEST_ID,
    INFORMATION_FINAL_EVALUATION_RUNTIME_VERSION,
    InformationRuntimeBackedEvaluationReport,
    verify_information_runtime_backed_evaluation_report,
)

PHASE4_RELEASE_AUDIT_VERSION = "p4.9-v1"
RELEASE_POLICY_SCHEMA_VERSION = 1
CANONICAL_RELEASE_POLICY_ID = "phase4-information-release-audit-v1"
CANONICAL_RELEASE_POLICY_DIGEST = (
    "9961633dc758c213894aaab5be0bf15303b16c5e75b90fb6798960e81f8b32a9"
)
CANONICAL_PACKAGE_VERSION = "0.15.0"
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[0-9a-f]{32}$")


class Phase4ReleaseAuditError(RuntimeError):
    """Raised when a Phase 4 release decision is invalid or unsafe."""


@dataclass(frozen=True)
class InformationReleaseAuditPolicy:
    policy_id: str
    required_evaluation_version: str
    required_evaluation_policy_id: str
    required_benchmark_id: str
    required_test_set_version: str
    required_runtime_version: str
    required_runtime_manifest_id: str
    required_package_version: str
    minimum_case_count: int
    minimum_runtime_target_file_count: int
    minimum_runtime_collected_test_count: int
    exact_head_commit_required: bool
    clean_working_tree_required: bool
    rollback_commit_required: bool
    rollback_must_be_ancestor: bool
    synthetic_only: bool
    private_output_only: bool
    repository_output_allowed: bool
    raw_query_text_allowed: bool
    raw_source_content_allowed: bool
    live_network_allowed: bool
    real_private_query_allowed: bool
    source_body_persistence_allowed: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    repository_write_allowed: bool
    background_execution_allowed: bool
    digest: str
    source_path: Path


@dataclass(frozen=True)
class Phase4ReleaseMetadata:
    repository_commit: str
    repository_head_commit: str
    repository_clean: bool
    evaluated_at: str
    policy_versions: tuple[str, ...]
    package_version: str
    known_limitations: tuple[str, ...] = ()
    rollback_commit: str | None = None


@dataclass(frozen=True)
class Phase4ReleaseDecision:
    audit_version: str
    release_id: str
    approved: bool
    decision_reasons: tuple[str, ...]
    repository_commit: str
    repository_head_commit: str
    repository_clean: bool
    evaluated_at: str
    rollback_commit: str
    package_version: str
    release_policy_id: str
    release_policy_digest: str
    evaluation_version: str
    benchmark_id: str
    benchmark_digest: str
    test_set_version: str
    evaluation_policy_id: str
    evaluation_policy_digest: str
    evaluation_report_digest: str
    evaluation_passed: bool
    runtime_version: str
    runtime_manifest_id: str
    runtime_manifest_digest: str
    runtime_repository_snapshot_digest: str
    runtime_collection_summary_digest: str
    runtime_execution_summary_digest: str
    runtime_evidence_digest: str
    runtime_backed_report_digest: str
    runtime_target_file_count: int
    runtime_case_evidence_count: int
    runtime_collected_test_count: int
    runtime_passed_test_count: int
    runtime_skipped_test_count: int
    runtime_network_guard_active: bool
    runtime_evidence_passed: bool
    runtime_backed_report_passed: bool
    case_count: int
    passed_case_count: int
    critical_case_failure_count: int
    metric_results: tuple[dict[str, Any], ...]
    known_limitations: tuple[str, ...]
    synthetic_only: bool
    private_output_only: bool
    repository_output_allowed: bool
    raw_query_text_allowed: bool
    raw_source_content_allowed: bool
    live_network_allowed: bool
    real_private_query_allowed: bool
    source_body_persistence_allowed: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    repository_write_allowed: bool
    background_execution_allowed: bool
    record_digest: str


def default_release_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "information_release_audit_policy.json"
    )


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Phase4ReleaseAuditError(
                "Release policy or record contains a duplicate object key."
            )
        value[key] = item
    return value


def _load_object(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
        )
    except Phase4ReleaseAuditError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase4ReleaseAuditError(f"{label} JSON could not be loaded.") from exc
    if not isinstance(value, dict):
        raise Phase4ReleaseAuditError(f"{label} root must be an object.")
    return source, value


def _exact_keys(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise Phase4ReleaseAuditError(
            f"{label} fields do not match the versioned schema."
        )


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise Phase4ReleaseAuditError(f"{field} must be boolean.")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase4ReleaseAuditError(f"{field} must be a positive integer.")
    return value


def load_phase4_release_audit_policy(
    path: Path | None = None,
) -> InformationReleaseAuditPolicy:
    source, value = _load_object(
        path or default_release_policy_path(),
        label="Release policy",
    )
    _exact_keys(
        value,
        {
            "information_release_audit_policy_schema_version",
            "policy_id",
            "phase",
            "milestone",
            "status",
            "required_evaluation_version",
            "required_evaluation_policy_id",
            "required_benchmark_id",
            "required_test_set_version",
            "required_runtime_version",
            "required_runtime_manifest_id",
            "required_package_version",
            "minimum_case_count",
            "minimum_runtime_target_file_count",
            "minimum_runtime_collected_test_count",
            "repository_requirements",
            "boundaries",
        },
        label="Release policy",
    )
    if (
        value["information_release_audit_policy_schema_version"]
        != RELEASE_POLICY_SCHEMA_VERSION
        or value["phase"] != "4"
        or value["milestone"] != "P4.9"
        or value["status"] != "final_release_audit_and_closure"
    ):
        raise Phase4ReleaseAuditError("Unsupported P4.9 release policy version.")

    required_values = {
        "policy_id": CANONICAL_RELEASE_POLICY_ID,
        "required_evaluation_version": INFORMATION_FINAL_EVALUATION_VERSION,
        "required_evaluation_policy_id": CANONICAL_POLICY_ID,
        "required_benchmark_id": CANONICAL_BENCHMARK_ID,
        "required_test_set_version": CANONICAL_TEST_SET_VERSION,
        "required_runtime_version": INFORMATION_FINAL_EVALUATION_RUNTIME_VERSION,
        "required_runtime_manifest_id": CANONICAL_RUNTIME_MANIFEST_ID,
        "required_package_version": CANONICAL_PACKAGE_VERSION,
    }
    for field, expected in required_values.items():
        if value[field] != expected:
            raise Phase4ReleaseAuditError(
                f"P4.9 release policy {field} cannot be changed."
            )

    requirements = value["repository_requirements"]
    boundaries = value["boundaries"]
    if not isinstance(requirements, dict) or not isinstance(boundaries, dict):
        raise Phase4ReleaseAuditError(
            "Release policy requirements and boundaries must be objects."
        )
    requirement_keys = {
        "exact_head_commit_required",
        "clean_working_tree_required",
        "rollback_commit_required",
        "rollback_must_be_ancestor",
    }
    boundary_keys = {
        "synthetic_only",
        "private_output_only",
        "repository_output_allowed",
        "raw_query_text_allowed",
        "raw_source_content_allowed",
        "live_network_allowed",
        "real_private_query_allowed",
        "source_body_persistence_allowed",
        "memory_write_allowed",
        "external_action_allowed",
        "repository_write_allowed",
        "background_execution_allowed",
    }
    _exact_keys(requirements, requirement_keys, label="Repository requirements")
    _exact_keys(boundaries, boundary_keys, label="Release boundaries")
    parsed_requirements = {
        key: _require_bool(requirements[key], field=key) for key in requirement_keys
    }
    parsed_boundaries = {
        key: _require_bool(boundaries[key], field=key) for key in boundary_keys
    }
    if not all(parsed_requirements.values()):
        raise Phase4ReleaseAuditError(
            "P4.9 repository requirements cannot be weakened."
        )
    if (
        parsed_boundaries["synthetic_only"] is not True
        or parsed_boundaries["private_output_only"] is not True
        or any(
            parsed_boundaries[key]
            for key in boundary_keys
            - {"synthetic_only", "private_output_only"}
        )
    ):
        raise Phase4ReleaseAuditError(
            "P4.9 release boundaries cannot be weakened."
        )

    minimum_case_count = _require_positive_int(
        value["minimum_case_count"], field="minimum_case_count"
    )
    minimum_target_count = _require_positive_int(
        value["minimum_runtime_target_file_count"],
        field="minimum_runtime_target_file_count",
    )
    minimum_test_count = _require_positive_int(
        value["minimum_runtime_collected_test_count"],
        field="minimum_runtime_collected_test_count",
    )
    if minimum_case_count < 24 or minimum_target_count < 28 or minimum_test_count < 640:
        raise Phase4ReleaseAuditError("P4.9 release thresholds cannot be weakened.")

    digest = sha256_canonical(value)
    if digest != CANONICAL_RELEASE_POLICY_DIGEST:
        raise Phase4ReleaseAuditError(
            "P4.9 release policy does not match the canonical policy digest."
        )
    return InformationReleaseAuditPolicy(
        policy_id=CANONICAL_RELEASE_POLICY_ID,
        required_evaluation_version=INFORMATION_FINAL_EVALUATION_VERSION,
        required_evaluation_policy_id=CANONICAL_POLICY_ID,
        required_benchmark_id=CANONICAL_BENCHMARK_ID,
        required_test_set_version=CANONICAL_TEST_SET_VERSION,
        required_runtime_version=INFORMATION_FINAL_EVALUATION_RUNTIME_VERSION,
        required_runtime_manifest_id=CANONICAL_RUNTIME_MANIFEST_ID,
        required_package_version=CANONICAL_PACKAGE_VERSION,
        minimum_case_count=minimum_case_count,
        minimum_runtime_target_file_count=minimum_target_count,
        minimum_runtime_collected_test_count=minimum_test_count,
        exact_head_commit_required=True,
        clean_working_tree_required=True,
        rollback_commit_required=True,
        rollback_must_be_ancestor=True,
        synthetic_only=True,
        private_output_only=True,
        repository_output_allowed=False,
        raw_query_text_allowed=False,
        raw_source_content_allowed=False,
        live_network_allowed=False,
        real_private_query_allowed=False,
        source_body_persistence_allowed=False,
        memory_write_allowed=False,
        external_action_allowed=False,
        repository_write_allowed=False,
        background_execution_allowed=False,
        digest=digest,
        source_path=source,
    )


def _utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Phase4ReleaseAuditError(
            "Release evaluation time must be ISO-8601."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Phase4ReleaseAuditError(
            "Release evaluation time must be explicitly UTC."
        )
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validated_commit(value: str, *, field: str) -> str:
    commit = value.strip().lower()
    if _FULL_COMMIT.fullmatch(commit) is None:
        raise Phase4ReleaseAuditError(
            f"{field} must be a full 40-character hexadecimal Git commit."
        )
    return commit


def _validated_limitations(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise Phase4ReleaseAuditError(
            "Release metadata contains duplicate limitations."
        )
    normalized: list[str] = []
    for value in values:
        text = value.strip()
        if not text or len(text) > 512 or "\n" in text or "\r" in text:
            raise Phase4ReleaseAuditError(
                "Release limitations must be compact single-line metadata."
            )
        normalized.append(text)
    return tuple(normalized)


def _validate_metadata(
    metadata: Phase4ReleaseMetadata,
    policy: InformationReleaseAuditPolicy,
) -> Phase4ReleaseMetadata:
    commit = _validated_commit(metadata.repository_commit, field="repository_commit")
    head = _validated_commit(
        metadata.repository_head_commit,
        field="repository_head_commit",
    )
    if head != commit:
        raise Phase4ReleaseAuditError(
            "repository_head_commit must exactly match repository_commit."
        )
    if not metadata.repository_clean:
        raise Phase4ReleaseAuditError(
            "The repository working tree must be clean for release audit."
        )
    if metadata.rollback_commit is None:
        raise Phase4ReleaseAuditError(
            "A rollback commit is required for Phase 4 release audit."
        )
    rollback = _validated_commit(metadata.rollback_commit, field="rollback_commit")
    if rollback == commit:
        raise Phase4ReleaseAuditError(
            "rollback_commit must differ from repository_commit."
        )
    evaluated_at = _utc_timestamp(metadata.evaluated_at)
    if not metadata.policy_versions or any(
        not item.strip() for item in metadata.policy_versions
    ):
        raise Phase4ReleaseAuditError(
            "Release metadata requires explicit policy versions."
        )
    if len(set(metadata.policy_versions)) != len(metadata.policy_versions):
        raise Phase4ReleaseAuditError(
            "Release metadata contains duplicate policy versions."
        )
    required = {policy.policy_id, policy.required_evaluation_policy_id}
    if not required.issubset(set(metadata.policy_versions)):
        raise Phase4ReleaseAuditError(
            "Release metadata is missing a governing policy version."
        )
    if metadata.package_version != policy.required_package_version:
        raise Phase4ReleaseAuditError(
            "Release package version does not match the governing policy."
        )
    return replace(
        metadata,
        repository_commit=commit,
        repository_head_commit=head,
        evaluated_at=evaluated_at,
        rollback_commit=rollback,
        known_limitations=_validated_limitations(metadata.known_limitations),
    )


def _decision_material(decision: Phase4ReleaseDecision) -> dict[str, Any]:
    return {
        "audit_version": decision.audit_version,
        "release_id": decision.release_id,
        "approved": decision.approved,
        "decision_reasons": list(decision.decision_reasons),
        "repository_commit": decision.repository_commit,
        "repository_head_commit": decision.repository_head_commit,
        "repository_clean": decision.repository_clean,
        "evaluated_at": decision.evaluated_at,
        "rollback_commit": decision.rollback_commit,
        "package_version": decision.package_version,
        "release_policy_id": decision.release_policy_id,
        "release_policy_digest": decision.release_policy_digest,
        "evaluation": {
            "evaluation_version": decision.evaluation_version,
            "benchmark_id": decision.benchmark_id,
            "benchmark_digest": decision.benchmark_digest,
            "test_set_version": decision.test_set_version,
            "evaluation_policy_id": decision.evaluation_policy_id,
            "evaluation_policy_digest": decision.evaluation_policy_digest,
            "evaluation_report_digest": decision.evaluation_report_digest,
            "evaluation_passed": decision.evaluation_passed,
            "case_count": decision.case_count,
            "passed_case_count": decision.passed_case_count,
            "critical_case_failure_count": decision.critical_case_failure_count,
            "metric_results": list(decision.metric_results),
        },
        "runtime_evidence": {
            "runtime_version": decision.runtime_version,
            "runtime_manifest_id": decision.runtime_manifest_id,
            "runtime_manifest_digest": decision.runtime_manifest_digest,
            "repository_snapshot_digest": (
                decision.runtime_repository_snapshot_digest
            ),
            "collection_summary_digest": (
                decision.runtime_collection_summary_digest
            ),
            "execution_summary_digest": decision.runtime_execution_summary_digest,
            "runtime_evidence_digest": decision.runtime_evidence_digest,
            "runtime_backed_report_digest": decision.runtime_backed_report_digest,
            "target_file_count": decision.runtime_target_file_count,
            "case_evidence_count": decision.runtime_case_evidence_count,
            "collected_test_count": decision.runtime_collected_test_count,
            "passed_test_count": decision.runtime_passed_test_count,
            "skipped_test_count": decision.runtime_skipped_test_count,
            "network_guard_active": decision.runtime_network_guard_active,
            "runtime_evidence_passed": decision.runtime_evidence_passed,
            "runtime_backed_report_passed": (
                decision.runtime_backed_report_passed
            ),
        },
        "known_limitations": list(decision.known_limitations),
        "boundaries": {
            "synthetic_only": decision.synthetic_only,
            "private_output_only": decision.private_output_only,
            "repository_output_allowed": decision.repository_output_allowed,
            "raw_query_text_allowed": decision.raw_query_text_allowed,
            "raw_source_content_allowed": decision.raw_source_content_allowed,
            "live_network_allowed": decision.live_network_allowed,
            "real_private_query_allowed": decision.real_private_query_allowed,
            "source_body_persistence_allowed": (
                decision.source_body_persistence_allowed
            ),
            "memory_write_allowed": decision.memory_write_allowed,
            "external_action_allowed": decision.external_action_allowed,
            "repository_write_allowed": decision.repository_write_allowed,
            "background_execution_allowed": (
                decision.background_execution_allowed
            ),
        },
    }


def phase4_release_record_digest(decision: Phase4ReleaseDecision) -> str:
    return hashlib.sha256(canonical_json(_decision_material(decision))).hexdigest()


def _phase4_release_id(
    *,
    repository_commit: str,
    rollback_commit: str,
    evaluated_at: str,
    runtime_backed_report_digest: str,
    release_policy_digest: str,
) -> str:
    release_seed = {
        "audit_version": PHASE4_RELEASE_AUDIT_VERSION,
        "repository_commit": repository_commit,
        "rollback_commit": rollback_commit,
        "evaluated_at": evaluated_at,
        "runtime_backed_report_digest": runtime_backed_report_digest,
        "release_policy_digest": release_policy_digest,
    }
    return hashlib.sha256(canonical_json(release_seed)).hexdigest()[:32]


def audit_phase4_release(
    report: InformationRuntimeBackedEvaluationReport,
    *,
    metadata: Phase4ReleaseMetadata,
    release_policy: InformationReleaseAuditPolicy | None = None,
) -> Phase4ReleaseDecision:
    """Create a deterministic P4.9 decision from verified P4.8 runtime evidence."""
    verify_information_runtime_backed_evaluation_report(report)
    policy = release_policy or load_phase4_release_audit_policy()
    metadata = _validate_metadata(metadata, policy)
    evaluation = report.evaluation_report
    runtime = report.runtime_evidence
    reasons: list[str] = []

    if evaluation.evaluation_version != policy.required_evaluation_version:
        reasons.append("evaluation_version_mismatch")
    if evaluation.policy_id != policy.required_evaluation_policy_id:
        reasons.append("evaluation_policy_mismatch")
    if evaluation.benchmark_id != policy.required_benchmark_id:
        reasons.append("benchmark_mismatch")
    if evaluation.test_set_version != policy.required_test_set_version:
        reasons.append("test_set_version_mismatch")
    if runtime.runtime_version != policy.required_runtime_version:
        reasons.append("runtime_version_mismatch")
    if runtime.manifest_id != policy.required_runtime_manifest_id:
        reasons.append("runtime_manifest_mismatch")
    if evaluation.case_count < policy.minimum_case_count:
        reasons.append("case_count_below_release_floor")
    if len(runtime.target_files) < policy.minimum_runtime_target_file_count:
        reasons.append("runtime_target_count_below_release_floor")
    if (
        len(runtime.case_evidence) < policy.minimum_case_count
        or len(runtime.case_evidence) != evaluation.case_count
    ):
        reasons.append("runtime_case_evidence_incomplete")
    if runtime.collected_test_count < policy.minimum_runtime_collected_test_count:
        reasons.append("runtime_test_count_below_release_floor")
    if runtime.passed_test_count != runtime.collected_test_count:
        reasons.append("runtime_test_failure")
    if runtime.skipped_test_count != 0:
        reasons.append("runtime_test_skip_observed")
    if not runtime.network_guard_active:
        reasons.append("runtime_network_guard_missing")
    if not runtime.passed:
        reasons.append("runtime_evidence_failed")
    if not evaluation.passed:
        reasons.append("final_information_evaluation_failed")
    if not report.passed:
        reasons.append("runtime_backed_evaluation_failed")
    for metric in evaluation.metric_results:
        if not metric.passed:
            reasons.append(f"metric_failed:{metric.metric_id}")
    if evaluation.critical_case_failure_count:
        reasons.append("critical_case_failure")
    if evaluation.passed_case_count != evaluation.case_count:
        reasons.append("incomplete_case_pass")
    if (
        not evaluation.synthetic_only
        or not evaluation.private_output_only
        or evaluation.raw_query_text_allowed
        or evaluation.raw_source_content_allowed
        or evaluation.live_network_allowed
        or evaluation.real_private_query_allowed
        or evaluation.source_body_persistence_allowed
        or evaluation.memory_write_allowed
        or evaluation.external_action_allowed
        or evaluation.repository_write_allowed
        or evaluation.background_execution_allowed
    ):
        reasons.append("evaluation_capability_boundary_failed")

    approved = not reasons
    release_id = _phase4_release_id(
        repository_commit=metadata.repository_commit,
        rollback_commit=str(metadata.rollback_commit),
        evaluated_at=metadata.evaluated_at,
        runtime_backed_report_digest=report.report_digest,
        release_policy_digest=policy.digest,
    )
    metrics = tuple(
        {
            "metric_id": item.metric_id,
            "value": item.value,
            "direction": item.direction,
            "threshold": item.threshold,
            "critical": item.critical,
            "passed": item.passed,
        }
        for item in evaluation.metric_results
    )
    decision = Phase4ReleaseDecision(
        audit_version=PHASE4_RELEASE_AUDIT_VERSION,
        release_id=release_id,
        approved=approved,
        decision_reasons=tuple(sorted(set(reasons))),
        repository_commit=metadata.repository_commit,
        repository_head_commit=metadata.repository_head_commit,
        repository_clean=metadata.repository_clean,
        evaluated_at=metadata.evaluated_at,
        rollback_commit=str(metadata.rollback_commit),
        package_version=metadata.package_version,
        release_policy_id=policy.policy_id,
        release_policy_digest=policy.digest,
        evaluation_version=evaluation.evaluation_version,
        benchmark_id=evaluation.benchmark_id,
        benchmark_digest=evaluation.benchmark_digest,
        test_set_version=evaluation.test_set_version,
        evaluation_policy_id=evaluation.policy_id,
        evaluation_policy_digest=evaluation.policy_digest,
        evaluation_report_digest=evaluation.report_digest,
        evaluation_passed=evaluation.passed,
        runtime_version=runtime.runtime_version,
        runtime_manifest_id=runtime.manifest_id,
        runtime_manifest_digest=runtime.manifest_digest,
        runtime_repository_snapshot_digest=runtime.repository_snapshot_digest,
        runtime_collection_summary_digest=runtime.collection_summary_digest,
        runtime_execution_summary_digest=runtime.execution_summary_digest,
        runtime_evidence_digest=runtime.evidence_digest,
        runtime_backed_report_digest=report.report_digest,
        runtime_target_file_count=len(runtime.target_files),
        runtime_case_evidence_count=len(runtime.case_evidence),
        runtime_collected_test_count=runtime.collected_test_count,
        runtime_passed_test_count=runtime.passed_test_count,
        runtime_skipped_test_count=runtime.skipped_test_count,
        runtime_network_guard_active=runtime.network_guard_active,
        runtime_evidence_passed=runtime.passed,
        runtime_backed_report_passed=report.passed,
        case_count=evaluation.case_count,
        passed_case_count=evaluation.passed_case_count,
        critical_case_failure_count=evaluation.critical_case_failure_count,
        metric_results=metrics,
        known_limitations=metadata.known_limitations,
        synthetic_only=True,
        private_output_only=True,
        repository_output_allowed=False,
        raw_query_text_allowed=False,
        raw_source_content_allowed=False,
        live_network_allowed=False,
        real_private_query_allowed=False,
        source_body_persistence_allowed=False,
        memory_write_allowed=False,
        external_action_allowed=False,
        repository_write_allowed=False,
        background_execution_allowed=False,
        record_digest="",
    )
    return replace(decision, record_digest=phase4_release_record_digest(decision))


def _valid_metric_results(items: tuple[dict[str, Any], ...]) -> bool:
    if not items:
        return False
    ids: list[str] = []
    expected = {
        "metric_id",
        "value",
        "direction",
        "threshold",
        "critical",
        "passed",
    }
    for item in items:
        if not isinstance(item, dict) or set(item) != expected:
            return False
        if (
            not isinstance(item["metric_id"], str)
            or not item["metric_id"]
            or isinstance(item["value"], bool)
            or not isinstance(item["value"], (int, float))
            or item["direction"] not in {"minimum", "maximum"}
            or isinstance(item["threshold"], bool)
            or not isinstance(item["threshold"], (int, float))
            or not isinstance(item["critical"], bool)
            or not isinstance(item["passed"], bool)
        ):
            return False
        ids.append(item["metric_id"])
    return len(ids) == len(set(ids))


def verify_phase4_release_decision(decision: Phase4ReleaseDecision) -> None:
    if decision.audit_version != PHASE4_RELEASE_AUDIT_VERSION:
        raise Phase4ReleaseAuditError("Phase 4 release audit version is invalid.")
    if _RELEASE_ID.fullmatch(decision.release_id) is None:
        raise Phase4ReleaseAuditError("Phase 4 release ID is invalid.")
    expected_release_id = _phase4_release_id(
        repository_commit=decision.repository_commit,
        rollback_commit=decision.rollback_commit,
        evaluated_at=decision.evaluated_at,
        runtime_backed_report_digest=decision.runtime_backed_report_digest,
        release_policy_digest=decision.release_policy_digest,
    )
    if decision.release_id != expected_release_id:
        raise Phase4ReleaseAuditError("Phase 4 release ID binding is invalid.")
    if decision.record_digest != phase4_release_record_digest(decision):
        raise Phase4ReleaseAuditError("Phase 4 release record digest is invalid.")
    if decision.approved != (not decision.decision_reasons):
        raise Phase4ReleaseAuditError(
            "Phase 4 release decision is internally inconsistent."
        )
    if (
        tuple(sorted(set(decision.decision_reasons)))
        != decision.decision_reasons
        or any(
            not item or len(item) > 160 or "\n" in item or "\r" in item
            for item in decision.decision_reasons
        )
    ):
        raise Phase4ReleaseAuditError(
            "Phase 4 release decision reasons are invalid."
        )
    commit = _validated_commit(decision.repository_commit, field="repository_commit")
    head = _validated_commit(
        decision.repository_head_commit,
        field="repository_head_commit",
    )
    rollback = _validated_commit(decision.rollback_commit, field="rollback_commit")
    if head != commit or rollback == commit or not decision.repository_clean:
        raise Phase4ReleaseAuditError(
            "Phase 4 release repository binding is invalid."
        )
    _utc_timestamp(decision.evaluated_at)
    if (
        decision.package_version != CANONICAL_PACKAGE_VERSION
        or decision.release_policy_id != CANONICAL_RELEASE_POLICY_ID
        or decision.release_policy_digest != CANONICAL_RELEASE_POLICY_DIGEST
        or decision.evaluation_version != INFORMATION_FINAL_EVALUATION_VERSION
        or decision.benchmark_id != CANONICAL_BENCHMARK_ID
        or decision.test_set_version != CANONICAL_TEST_SET_VERSION
        or decision.evaluation_policy_id != CANONICAL_POLICY_ID
        or decision.runtime_version != INFORMATION_FINAL_EVALUATION_RUNTIME_VERSION
        or decision.runtime_manifest_id != CANONICAL_RUNTIME_MANIFEST_ID
    ):
        raise Phase4ReleaseAuditError(
            "Phase 4 release version binding is invalid."
        )
    for digest in (
        decision.release_policy_digest,
        decision.benchmark_digest,
        decision.evaluation_policy_digest,
        decision.evaluation_report_digest,
        decision.runtime_manifest_digest,
        decision.runtime_repository_snapshot_digest,
        decision.runtime_collection_summary_digest,
        decision.runtime_execution_summary_digest,
        decision.runtime_evidence_digest,
        decision.runtime_backed_report_digest,
        decision.record_digest,
    ):
        if _SHA256.fullmatch(digest) is None:
            raise Phase4ReleaseAuditError(
                "Phase 4 release record contains an invalid digest."
            )
    if (
        decision.runtime_target_file_count != 28
        or decision.runtime_case_evidence_count != 24
        or decision.runtime_collected_test_count < 640
        or not (
            0
            <= decision.runtime_passed_test_count
            <= decision.runtime_collected_test_count
        )
        or decision.runtime_skipped_test_count < 0
        or decision.case_count < 24
        or not (0 <= decision.passed_case_count <= decision.case_count)
        or decision.critical_case_failure_count < 0
    ):
        raise Phase4ReleaseAuditError(
            "Phase 4 release evidence counts are invalid."
        )
    if not _valid_metric_results(decision.metric_results):
        raise Phase4ReleaseAuditError(
            "Phase 4 release metric results are invalid."
        )
    if decision.approved and (
        not decision.evaluation_passed
        or not decision.runtime_evidence_passed
        or not decision.runtime_backed_report_passed
        or not decision.runtime_network_guard_active
        or decision.runtime_passed_test_count
        != decision.runtime_collected_test_count
        or decision.runtime_skipped_test_count != 0
        or decision.runtime_case_evidence_count != decision.case_count
        or decision.passed_case_count != decision.case_count
        or decision.critical_case_failure_count != 0
        or any(not bool(item["passed"]) for item in decision.metric_results)
    ):
        raise Phase4ReleaseAuditError(
            "Approved Phase 4 release evidence is incomplete."
        )
    _validated_limitations(decision.known_limitations)
    if (
        not decision.synthetic_only
        or not decision.private_output_only
        or decision.repository_output_allowed
        or decision.raw_query_text_allowed
        or decision.raw_source_content_allowed
        or decision.live_network_allowed
        or decision.real_private_query_allowed
        or decision.source_body_persistence_allowed
        or decision.memory_write_allowed
        or decision.external_action_allowed
        or decision.repository_write_allowed
        or decision.background_execution_allowed
    ):
        raise Phase4ReleaseAuditError("Phase 4 release boundaries are invalid.")


def phase4_release_record_json(decision: Phase4ReleaseDecision) -> str:
    verify_phase4_release_decision(decision)
    payload = _decision_material(decision)
    payload["record_digest"] = decision.record_digest
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _resolved_private_output(
    output_path: str | Path,
    *,
    private_root: str | Path,
    repository_root: str | Path | None,
) -> Path:
    private = Path(private_root).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)
    try:
        output.relative_to(private)
    except ValueError as exc:
        raise Phase4ReleaseAuditError(
            "Phase 4 release records must stay under the private output root."
        ) from exc
    if repository_root is not None:
        repository = Path(repository_root).expanduser().resolve(strict=True)
        try:
            output.relative_to(repository)
        except ValueError:
            pass
        else:
            raise Phase4ReleaseAuditError(
                "Phase 4 release records cannot be written into the repository."
            )
    if output.suffix.lower() != ".json":
        raise Phase4ReleaseAuditError(
            "Phase 4 release record output must use a .json suffix."
        )
    return output


def write_phase4_release_record(
    decision: Phase4ReleaseDecision,
    output_path: str | Path,
    *,
    private_root: str | Path,
    repository_root: str | Path | None = None,
) -> Path:
    verify_phase4_release_decision(decision)
    output = _resolved_private_output(
        output_path,
        private_root=private_root,
        repository_root=repository_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = phase4_release_record_json(decision)
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != payload:
            raise Phase4ReleaseAuditError(
                "Refusing to overwrite a different Phase 4 release record."
            )
        return output
    output.write_text(payload, encoding="utf-8", newline="\n")
    return output


def _record_decision(value: dict[str, Any]) -> Phase4ReleaseDecision:
    expected = {
        "audit_version",
        "release_id",
        "approved",
        "decision_reasons",
        "repository_commit",
        "repository_head_commit",
        "repository_clean",
        "evaluated_at",
        "rollback_commit",
        "package_version",
        "release_policy_id",
        "release_policy_digest",
        "evaluation",
        "runtime_evidence",
        "known_limitations",
        "boundaries",
        "record_digest",
    }
    _exact_keys(value, expected, label="Phase 4 release record")
    evaluation = value["evaluation"]
    runtime = value["runtime_evidence"]
    boundaries = value["boundaries"]
    if not all(isinstance(item, dict) for item in (evaluation, runtime, boundaries)):
        raise Phase4ReleaseAuditError(
            "Phase 4 release record sections are invalid."
        )
    _exact_keys(
        evaluation,
        {
            "evaluation_version",
            "benchmark_id",
            "benchmark_digest",
            "test_set_version",
            "evaluation_policy_id",
            "evaluation_policy_digest",
            "evaluation_report_digest",
            "evaluation_passed",
            "case_count",
            "passed_case_count",
            "critical_case_failure_count",
            "metric_results",
        },
        label="Phase 4 evaluation record",
    )
    _exact_keys(
        runtime,
        {
            "runtime_version",
            "runtime_manifest_id",
            "runtime_manifest_digest",
            "repository_snapshot_digest",
            "collection_summary_digest",
            "execution_summary_digest",
            "runtime_evidence_digest",
            "runtime_backed_report_digest",
            "target_file_count",
            "case_evidence_count",
            "collected_test_count",
            "passed_test_count",
            "skipped_test_count",
            "network_guard_active",
            "runtime_evidence_passed",
            "runtime_backed_report_passed",
        },
        label="Phase 4 runtime record",
    )
    _exact_keys(
        boundaries,
        {
            "synthetic_only",
            "private_output_only",
            "repository_output_allowed",
            "raw_query_text_allowed",
            "raw_source_content_allowed",
            "live_network_allowed",
            "real_private_query_allowed",
            "source_body_persistence_allowed",
            "memory_write_allowed",
            "external_action_allowed",
            "repository_write_allowed",
            "background_execution_allowed",
        },
        label="Phase 4 release boundaries",
    )
    bool_fields = (
        (value, "approved"),
        (value, "repository_clean"),
        (evaluation, "evaluation_passed"),
        (runtime, "network_guard_active"),
        (runtime, "runtime_evidence_passed"),
        (runtime, "runtime_backed_report_passed"),
        *((boundaries, field) for field in boundaries),
    )
    for container, field in bool_fields:
        _require_bool(container[field], field=field)
    list_fields = (
        (value, "decision_reasons"),
        (value, "known_limitations"),
        (evaluation, "metric_results"),
    )
    for container, field in list_fields:
        if not isinstance(container[field], list):
            raise Phase4ReleaseAuditError(f"{field} must be an array.")
    if any(not isinstance(item, str) for item in value["decision_reasons"]):
        raise Phase4ReleaseAuditError("decision_reasons must contain strings.")
    if any(not isinstance(item, str) for item in value["known_limitations"]):
        raise Phase4ReleaseAuditError("known_limitations must contain strings.")
    if any(not isinstance(item, dict) for item in evaluation["metric_results"]):
        raise Phase4ReleaseAuditError("metric_results must contain objects.")
    string_fields = (
        (value, "audit_version"),
        (value, "release_id"),
        (value, "repository_commit"),
        (value, "repository_head_commit"),
        (value, "evaluated_at"),
        (value, "rollback_commit"),
        (value, "package_version"),
        (value, "release_policy_id"),
        (value, "release_policy_digest"),
        (value, "record_digest"),
        (evaluation, "evaluation_version"),
        (evaluation, "benchmark_id"),
        (evaluation, "benchmark_digest"),
        (evaluation, "test_set_version"),
        (evaluation, "evaluation_policy_id"),
        (evaluation, "evaluation_policy_digest"),
        (evaluation, "evaluation_report_digest"),
        (runtime, "runtime_version"),
        (runtime, "runtime_manifest_id"),
        (runtime, "runtime_manifest_digest"),
        (runtime, "repository_snapshot_digest"),
        (runtime, "collection_summary_digest"),
        (runtime, "execution_summary_digest"),
        (runtime, "runtime_evidence_digest"),
        (runtime, "runtime_backed_report_digest"),
    )
    for container, field in string_fields:
        if not isinstance(container[field], str):
            raise Phase4ReleaseAuditError(f"{field} must be a string.")
    integer_fields = (
        (evaluation, "case_count"),
        (evaluation, "passed_case_count"),
        (evaluation, "critical_case_failure_count"),
        (runtime, "target_file_count"),
        (runtime, "case_evidence_count"),
        (runtime, "collected_test_count"),
        (runtime, "passed_test_count"),
        (runtime, "skipped_test_count"),
    )
    for container, field in integer_fields:
        if isinstance(container[field], bool) or not isinstance(container[field], int):
            raise Phase4ReleaseAuditError(f"{field} must be an integer.")
    return Phase4ReleaseDecision(
        audit_version=str(value["audit_version"]),
        release_id=str(value["release_id"]),
        approved=value["approved"],
        decision_reasons=tuple(str(item) for item in value["decision_reasons"]),
        repository_commit=str(value["repository_commit"]),
        repository_head_commit=str(value["repository_head_commit"]),
        repository_clean=value["repository_clean"],
        evaluated_at=str(value["evaluated_at"]),
        rollback_commit=str(value["rollback_commit"]),
        package_version=str(value["package_version"]),
        release_policy_id=str(value["release_policy_id"]),
        release_policy_digest=str(value["release_policy_digest"]),
        evaluation_version=str(evaluation["evaluation_version"]),
        benchmark_id=str(evaluation["benchmark_id"]),
        benchmark_digest=str(evaluation["benchmark_digest"]),
        test_set_version=str(evaluation["test_set_version"]),
        evaluation_policy_id=str(evaluation["evaluation_policy_id"]),
        evaluation_policy_digest=str(evaluation["evaluation_policy_digest"]),
        evaluation_report_digest=str(evaluation["evaluation_report_digest"]),
        evaluation_passed=evaluation["evaluation_passed"],
        runtime_version=str(runtime["runtime_version"]),
        runtime_manifest_id=str(runtime["runtime_manifest_id"]),
        runtime_manifest_digest=str(runtime["runtime_manifest_digest"]),
        runtime_repository_snapshot_digest=str(
            runtime["repository_snapshot_digest"]
        ),
        runtime_collection_summary_digest=str(
            runtime["collection_summary_digest"]
        ),
        runtime_execution_summary_digest=str(runtime["execution_summary_digest"]),
        runtime_evidence_digest=str(runtime["runtime_evidence_digest"]),
        runtime_backed_report_digest=str(runtime["runtime_backed_report_digest"]),
        runtime_target_file_count=runtime["target_file_count"],
        runtime_case_evidence_count=runtime["case_evidence_count"],
        runtime_collected_test_count=runtime["collected_test_count"],
        runtime_passed_test_count=runtime["passed_test_count"],
        runtime_skipped_test_count=runtime["skipped_test_count"],
        runtime_network_guard_active=runtime["network_guard_active"],
        runtime_evidence_passed=runtime["runtime_evidence_passed"],
        runtime_backed_report_passed=runtime["runtime_backed_report_passed"],
        case_count=evaluation["case_count"],
        passed_case_count=evaluation["passed_case_count"],
        critical_case_failure_count=evaluation["critical_case_failure_count"],
        metric_results=tuple(dict(item) for item in evaluation["metric_results"]),
        known_limitations=tuple(str(item) for item in value["known_limitations"]),
        synthetic_only=boundaries["synthetic_only"],
        private_output_only=boundaries["private_output_only"],
        repository_output_allowed=boundaries["repository_output_allowed"],
        raw_query_text_allowed=boundaries["raw_query_text_allowed"],
        raw_source_content_allowed=boundaries["raw_source_content_allowed"],
        live_network_allowed=boundaries["live_network_allowed"],
        real_private_query_allowed=boundaries["real_private_query_allowed"],
        source_body_persistence_allowed=boundaries[
            "source_body_persistence_allowed"
        ],
        memory_write_allowed=boundaries["memory_write_allowed"],
        external_action_allowed=boundaries["external_action_allowed"],
        repository_write_allowed=boundaries["repository_write_allowed"],
        background_execution_allowed=boundaries[
            "background_execution_allowed"
        ],
        record_digest=str(value["record_digest"]),
    )


def load_phase4_release_record(path: str | Path) -> Phase4ReleaseDecision:
    _source, value = _load_object(Path(path), label="Phase 4 release record")
    decision = _record_decision(value)
    verify_phase4_release_decision(decision)
    return decision
