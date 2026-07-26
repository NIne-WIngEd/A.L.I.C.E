"""Exact-commit private release audit for A.L.I.C.E. Phase 3 P3.11."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .final_evaluation import (
    ConversationFinalEvaluationReport,
    verify_conversation_final_report,
)
from .final_evaluation_contract import canonical_json, sha256_canonical

PHASE3_RELEASE_AUDIT_VERSION = "p3.11-v1"
RELEASE_POLICY_SCHEMA_VERSION = 1
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class Phase3ReleaseAuditError(RuntimeError):
    """Raised when a Phase 3 release decision is invalid or unsafe."""


@dataclass(frozen=True)
class ConversationReleaseAuditPolicy:
    policy_id: str
    required_evaluation_version: str
    required_evaluation_policy_id: str
    required_package_version: str
    required_evidence_manifest_id: str
    exact_head_commit_required: bool
    clean_working_tree_required: bool
    rollback_commit_required: bool
    rollback_must_be_ancestor: bool
    private_output_only: bool
    repository_output_allowed: bool
    raw_conversation_content_allowed: bool
    web_access_allowed: bool
    tool_calling_allowed: bool
    external_action_allowed: bool
    memory_write_allowed: bool
    digest: str
    source_path: Path


@dataclass(frozen=True)
class Phase3ReleaseMetadata:
    repository_commit: str
    repository_head_commit: str
    repository_clean: bool
    evaluated_at: str
    policy_versions: tuple[str, ...]
    package_version: str
    evidence_manifest_id: str = ""
    evidence_manifest_digest: str = ""
    evidence_run_digest: str = ""
    evidence_target_count: int = 0
    evidence_passed_target_count: int = 0
    known_limitations: tuple[str, ...] = ()
    rollback_commit: str | None = None


@dataclass(frozen=True)
class Phase3ReleaseDecision:
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
    evidence_manifest_id: str
    evidence_manifest_digest: str
    evidence_run_digest: str
    evidence_target_count: int
    evidence_passed_target_count: int
    case_count: int
    passed_case_count: int
    critical_case_failure_count: int
    metric_results: tuple[dict[str, Any], ...]
    known_limitations: tuple[str, ...]
    private_output_only: bool
    raw_conversation_content_allowed: bool
    web_access_allowed: bool
    tool_calling_allowed: bool
    external_action_allowed: bool
    memory_write_allowed: bool
    repository_output_allowed: bool
    record_digest: str


def default_release_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "conversation_release_audit_policy.json"


def _load_object(path: Path) -> tuple[Path, dict[str, Any]]:
    source = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3ReleaseAuditError("Release policy JSON could not be loaded.") from exc
    if not isinstance(value, dict):
        raise Phase3ReleaseAuditError("Release policy root must be an object.")
    return source, value


def _exact_keys(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise Phase3ReleaseAuditError(f"{label} fields do not match the versioned schema.")


def load_phase3_release_audit_policy(path: Path | None = None) -> ConversationReleaseAuditPolicy:
    source, value = _load_object(path or default_release_policy_path())
    _exact_keys(
        value,
        {
            "conversation_release_audit_policy_schema_version",
            "policy_id",
            "phase",
            "milestone",
            "status",
            "required_evaluation_version",
            "required_evaluation_policy_id",
            "required_package_version",
            "required_evidence_manifest_id",
            "repository_requirements",
            "boundaries",
        },
        label="Release policy",
    )
    if (
        value["conversation_release_audit_policy_schema_version"]
        != RELEASE_POLICY_SCHEMA_VERSION
        or value["phase"] != "3"
        or value["milestone"] != "P3.11"
    ):
        raise Phase3ReleaseAuditError("Unsupported P3.11 release policy version.")
    requirements = value["repository_requirements"]
    boundaries = value["boundaries"]
    requirement_keys = {
        "exact_head_commit_required",
        "clean_working_tree_required",
        "rollback_commit_required",
        "rollback_must_be_ancestor",
    }
    boundary_keys = {
        "private_output_only",
        "repository_output_allowed",
        "raw_conversation_content_allowed",
        "web_access_allowed",
        "tool_calling_allowed",
        "external_action_allowed",
        "memory_write_allowed",
    }
    if not isinstance(requirements, dict) or not isinstance(boundaries, dict):
        raise Phase3ReleaseAuditError("Release policy requirements and boundaries must be objects.")
    _exact_keys(requirements, requirement_keys, label="Repository requirements")
    _exact_keys(boundaries, boundary_keys, label="Release boundaries")
    if any(not isinstance(requirements[key], bool) for key in requirement_keys):
        raise Phase3ReleaseAuditError("Repository requirements must be boolean.")
    if any(not isinstance(boundaries[key], bool) for key in boundary_keys):
        raise Phase3ReleaseAuditError("Release boundaries must be boolean.")
    if not all(requirements.values()):
        raise Phase3ReleaseAuditError("P3.11 repository requirements cannot be weakened.")
    if (
        boundaries["private_output_only"] is not True
        or boundaries["repository_output_allowed"] is not False
        or boundaries["raw_conversation_content_allowed"] is not False
        or boundaries["web_access_allowed"] is not False
        or boundaries["tool_calling_allowed"] is not False
        or boundaries["external_action_allowed"] is not False
        or boundaries["memory_write_allowed"] is not False
    ):
        raise Phase3ReleaseAuditError("P3.11 release boundaries cannot be weakened.")
    required_strings = (
        "policy_id",
        "required_evaluation_version",
        "required_evaluation_policy_id",
        "required_package_version",
        "required_evidence_manifest_id",
    )
    if any(not str(value[key]).strip() for key in required_strings):
        raise Phase3ReleaseAuditError("Release policy identifiers cannot be empty.")
    return ConversationReleaseAuditPolicy(
        policy_id=str(value["policy_id"]),
        required_evaluation_version=str(value["required_evaluation_version"]),
        required_evaluation_policy_id=str(value["required_evaluation_policy_id"]),
        required_package_version=str(value["required_package_version"]),
        required_evidence_manifest_id=str(value["required_evidence_manifest_id"]),
        exact_head_commit_required=True,
        clean_working_tree_required=True,
        rollback_commit_required=True,
        rollback_must_be_ancestor=True,
        private_output_only=True,
        repository_output_allowed=False,
        raw_conversation_content_allowed=False,
        web_access_allowed=False,
        tool_calling_allowed=False,
        external_action_allowed=False,
        memory_write_allowed=False,
        digest=sha256_canonical(value),
        source_path=source,
    )


def _utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Phase3ReleaseAuditError("Release evaluation time must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Phase3ReleaseAuditError("Release evaluation time must be explicitly UTC.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validated_commit(value: str, *, field: str) -> str:
    commit = value.strip().lower()
    if not _FULL_COMMIT.fullmatch(commit):
        raise Phase3ReleaseAuditError(f"{field} must be a full 40-character hexadecimal Git commit.")
    return commit


def _validate_metadata(
    metadata: Phase3ReleaseMetadata,
    policy: ConversationReleaseAuditPolicy,
) -> Phase3ReleaseMetadata:
    commit = _validated_commit(metadata.repository_commit, field="repository_commit")
    head = _validated_commit(metadata.repository_head_commit, field="repository_head_commit")
    if head != commit:
        raise Phase3ReleaseAuditError("repository_head_commit must exactly match repository_commit.")
    if not metadata.repository_clean:
        raise Phase3ReleaseAuditError("The repository working tree must be clean for release audit.")
    if metadata.rollback_commit is None:
        raise Phase3ReleaseAuditError("A rollback commit is required for Phase 3 release audit.")
    rollback = _validated_commit(metadata.rollback_commit, field="rollback_commit")
    if rollback == commit:
        raise Phase3ReleaseAuditError("rollback_commit must differ from repository_commit.")
    evaluated_at = _utc_timestamp(metadata.evaluated_at)
    if not metadata.policy_versions or any(not item.strip() for item in metadata.policy_versions):
        raise Phase3ReleaseAuditError("Release metadata requires explicit policy versions.")
    if len(set(metadata.policy_versions)) != len(metadata.policy_versions):
        raise Phase3ReleaseAuditError("Release metadata contains duplicate policy versions.")
    required = {policy.policy_id, policy.required_evaluation_policy_id}
    if not required.issubset(set(metadata.policy_versions)):
        raise Phase3ReleaseAuditError("Release metadata is missing a governing policy version.")
    if metadata.package_version != policy.required_package_version:
        raise Phase3ReleaseAuditError("Release package version does not match the governing policy.")
    digest_pattern = re.compile(r"^[0-9a-f]{64}$")
    if metadata.evidence_manifest_id != policy.required_evidence_manifest_id:
        raise Phase3ReleaseAuditError("Release evidence manifest does not match the governing policy.")
    if not digest_pattern.fullmatch(metadata.evidence_manifest_digest) or not digest_pattern.fullmatch(metadata.evidence_run_digest):
        raise Phase3ReleaseAuditError("Release evidence digests must be lowercase SHA-256 values.")
    if metadata.evidence_target_count <= 0 or not (0 <= metadata.evidence_passed_target_count <= metadata.evidence_target_count):
        raise Phase3ReleaseAuditError("Release evidence target counts are invalid.")
    if len(set(metadata.known_limitations)) != len(metadata.known_limitations):
        raise Phase3ReleaseAuditError("Release metadata contains duplicate limitations.")
    return replace(
        metadata,
        repository_commit=commit,
        repository_head_commit=head,
        evaluated_at=evaluated_at,
        rollback_commit=rollback,
    )


def _decision_material(decision: Phase3ReleaseDecision) -> dict[str, Any]:
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
        "evaluation_version": decision.evaluation_version,
        "benchmark_id": decision.benchmark_id,
        "benchmark_digest": decision.benchmark_digest,
        "test_set_version": decision.test_set_version,
        "evaluation_policy_id": decision.evaluation_policy_id,
        "evaluation_policy_digest": decision.evaluation_policy_digest,
        "evaluation_report_digest": decision.evaluation_report_digest,
        "evidence_manifest_id": decision.evidence_manifest_id,
        "evidence_manifest_digest": decision.evidence_manifest_digest,
        "evidence_run_digest": decision.evidence_run_digest,
        "evidence_target_count": decision.evidence_target_count,
        "evidence_passed_target_count": decision.evidence_passed_target_count,
        "case_count": decision.case_count,
        "passed_case_count": decision.passed_case_count,
        "critical_case_failure_count": decision.critical_case_failure_count,
        "metric_results": list(decision.metric_results),
        "known_limitations": list(decision.known_limitations),
        "boundaries": {
            "private_output_only": decision.private_output_only,
            "raw_conversation_content_allowed": decision.raw_conversation_content_allowed,
            "web_access_allowed": decision.web_access_allowed,
            "tool_calling_allowed": decision.tool_calling_allowed,
            "external_action_allowed": decision.external_action_allowed,
            "memory_write_allowed": decision.memory_write_allowed,
            "repository_output_allowed": decision.repository_output_allowed,
        },
    }


def phase3_release_record_digest(decision: Phase3ReleaseDecision) -> str:
    return hashlib.sha256(canonical_json(_decision_material(decision))).hexdigest()


def audit_phase3_release(
    report: ConversationFinalEvaluationReport,
    *,
    metadata: Phase3ReleaseMetadata,
    release_policy: ConversationReleaseAuditPolicy | None = None,
) -> Phase3ReleaseDecision:
    """Create a deterministic P3.11 release decision from a verified P3.10 report."""
    verify_conversation_final_report(report)
    policy = release_policy or load_phase3_release_audit_policy()
    metadata = _validate_metadata(metadata, policy)
    reasons: list[str] = []
    if report.evaluation_version != policy.required_evaluation_version:
        reasons.append("evaluation_version_mismatch")
    if report.policy_id != policy.required_evaluation_policy_id:
        reasons.append("evaluation_policy_mismatch")
    if metadata.evidence_passed_target_count != metadata.evidence_target_count:
        reasons.append("release_evidence_test_failed")
    if not report.passed:
        reasons.append("final_conversation_evaluation_failed")
    for metric in report.metric_results:
        if not metric.passed:
            reasons.append(f"metric_failed:{metric.metric_id}")
    if report.critical_case_failure_count:
        reasons.append("critical_case_failure")
    if report.passed_case_count != report.case_count:
        reasons.append("incomplete_case_pass")
    if (
        not report.synthetic_only
        or not report.private_output_only
        or report.raw_conversation_content_allowed
        or report.web_access_allowed
        or report.tool_calling_allowed
        or report.external_action_allowed
        or report.memory_write_allowed
        or report.repository_write_allowed
    ):
        reasons.append("evaluation_capability_boundary_failed")
    approved = not reasons
    release_seed = {
        "audit_version": PHASE3_RELEASE_AUDIT_VERSION,
        "repository_commit": metadata.repository_commit,
        "evaluated_at": metadata.evaluated_at,
        "evaluation_report_digest": report.report_digest,
        "release_policy_digest": policy.digest,
        "evidence_run_digest": metadata.evidence_run_digest,
    }
    release_id = hashlib.sha256(canonical_json(release_seed)).hexdigest()[:32]
    metrics = tuple(
        {
            "metric_id": item.metric_id,
            "value": item.value,
            "direction": item.direction,
            "threshold": item.threshold,
            "critical": item.critical,
            "passed": item.passed,
        }
        for item in report.metric_results
    )
    decision = Phase3ReleaseDecision(
        audit_version=PHASE3_RELEASE_AUDIT_VERSION,
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
        evaluation_version=report.evaluation_version,
        benchmark_id=report.benchmark_id,
        benchmark_digest=report.benchmark_digest,
        test_set_version=report.test_set_version,
        evaluation_policy_id=report.policy_id,
        evaluation_policy_digest=report.policy_digest,
        evaluation_report_digest=report.report_digest,
        evidence_manifest_id=metadata.evidence_manifest_id,
        evidence_manifest_digest=metadata.evidence_manifest_digest,
        evidence_run_digest=metadata.evidence_run_digest,
        evidence_target_count=metadata.evidence_target_count,
        evidence_passed_target_count=metadata.evidence_passed_target_count,
        case_count=report.case_count,
        passed_case_count=report.passed_case_count,
        critical_case_failure_count=report.critical_case_failure_count,
        metric_results=metrics,
        known_limitations=metadata.known_limitations,
        private_output_only=True,
        raw_conversation_content_allowed=False,
        web_access_allowed=False,
        tool_calling_allowed=False,
        external_action_allowed=False,
        memory_write_allowed=False,
        repository_output_allowed=False,
        record_digest="",
    )
    return replace(decision, record_digest=phase3_release_record_digest(decision))


def verify_phase3_release_decision(decision: Phase3ReleaseDecision) -> None:
    if decision.audit_version != PHASE3_RELEASE_AUDIT_VERSION:
        raise Phase3ReleaseAuditError("Phase 3 release audit version is invalid.")
    if decision.record_digest != phase3_release_record_digest(decision):
        raise Phase3ReleaseAuditError("Phase 3 release record digest is invalid.")
    if decision.approved != (not decision.decision_reasons):
        raise Phase3ReleaseAuditError("Phase 3 release decision is internally inconsistent.")
    commit = _validated_commit(decision.repository_commit, field="repository_commit")
    head = _validated_commit(decision.repository_head_commit, field="repository_head_commit")
    rollback = _validated_commit(decision.rollback_commit, field="rollback_commit")
    if head != commit or rollback == commit or not decision.repository_clean:
        raise Phase3ReleaseAuditError("Phase 3 release repository binding is invalid.")
    _utc_timestamp(decision.evaluated_at)
    digest_pattern = re.compile(r"^[0-9a-f]{64}$")
    if not digest_pattern.fullmatch(decision.evidence_manifest_digest) or not digest_pattern.fullmatch(decision.evidence_run_digest):
        raise Phase3ReleaseAuditError("Phase 3 release evidence digests are invalid.")
    if decision.evidence_target_count <= 0 or not (0 <= decision.evidence_passed_target_count <= decision.evidence_target_count):
        raise Phase3ReleaseAuditError("Phase 3 release evidence counts are invalid.")
    if decision.approved and decision.evidence_passed_target_count != decision.evidence_target_count:
        raise Phase3ReleaseAuditError("Approved Phase 3 release evidence is incomplete.")
    metric_ids = [str(item.get("metric_id", "")) for item in decision.metric_results]
    if not metric_ids or len(set(metric_ids)) != len(metric_ids):
        raise Phase3ReleaseAuditError("Phase 3 release metric results are invalid.")
    if (
        not decision.private_output_only
        or decision.raw_conversation_content_allowed
        or decision.web_access_allowed
        or decision.tool_calling_allowed
        or decision.external_action_allowed
        or decision.memory_write_allowed
        or decision.repository_output_allowed
    ):
        raise Phase3ReleaseAuditError("Phase 3 release boundaries are invalid.")


def phase3_release_record_json(decision: Phase3ReleaseDecision) -> str:
    verify_phase3_release_decision(decision)
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
        raise Phase3ReleaseAuditError("Phase 3 release records must stay under the private output root.") from exc
    if repository_root is not None:
        repository = Path(repository_root).expanduser().resolve(strict=True)
        try:
            output.relative_to(repository)
        except ValueError:
            pass
        else:
            raise Phase3ReleaseAuditError("Phase 3 release records cannot be written into the repository.")
    if output.suffix.lower() != ".json":
        raise Phase3ReleaseAuditError("Phase 3 release record output must use a .json suffix.")
    return output


def write_phase3_release_record(
    decision: Phase3ReleaseDecision,
    output_path: str | Path,
    *,
    private_root: str | Path,
    repository_root: str | Path | None = None,
) -> Path:
    verify_phase3_release_decision(decision)
    output = _resolved_private_output(
        output_path,
        private_root=private_root,
        repository_root=repository_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = phase3_release_record_json(decision)
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != payload:
            raise Phase3ReleaseAuditError("Refusing to overwrite a different Phase 3 release record.")
        return output
    output.write_text(payload, encoding="utf-8", newline="\n")
    return output


def load_phase3_release_record(path: str | Path) -> Phase3ReleaseDecision:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3ReleaseAuditError("Phase 3 release record could not be loaded.") from exc
    expected = {
        "audit_version", "release_id", "approved", "decision_reasons",
        "repository_commit", "repository_head_commit", "repository_clean",
        "evaluated_at", "rollback_commit", "package_version",
        "release_policy_id", "release_policy_digest", "evaluation_version",
        "benchmark_id", "benchmark_digest", "test_set_version",
        "evaluation_policy_id", "evaluation_policy_digest",
        "evaluation_report_digest", "case_count", "passed_case_count",
        "critical_case_failure_count", "metric_results", "known_limitations",
        "evidence_manifest_id", "evidence_manifest_digest", "evidence_run_digest",
        "evidence_target_count", "evidence_passed_target_count",
        "boundaries", "record_digest",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise Phase3ReleaseAuditError("Phase 3 release record has an invalid structure.")
    boundaries = value["boundaries"]
    if not isinstance(boundaries, dict):
        raise Phase3ReleaseAuditError("Phase 3 release record boundaries are invalid.")
    try:
        decision = Phase3ReleaseDecision(
            audit_version=str(value["audit_version"]),
            release_id=str(value["release_id"]),
            approved=bool(value["approved"]),
            decision_reasons=tuple(str(item) for item in value["decision_reasons"]),
            repository_commit=str(value["repository_commit"]),
            repository_head_commit=str(value["repository_head_commit"]),
            repository_clean=bool(value["repository_clean"]),
            evaluated_at=str(value["evaluated_at"]),
            rollback_commit=str(value["rollback_commit"]),
            package_version=str(value["package_version"]),
            release_policy_id=str(value["release_policy_id"]),
            release_policy_digest=str(value["release_policy_digest"]),
            evaluation_version=str(value["evaluation_version"]),
            benchmark_id=str(value["benchmark_id"]),
            benchmark_digest=str(value["benchmark_digest"]),
            test_set_version=str(value["test_set_version"]),
            evaluation_policy_id=str(value["evaluation_policy_id"]),
            evaluation_policy_digest=str(value["evaluation_policy_digest"]),
            evaluation_report_digest=str(value["evaluation_report_digest"]),
            evidence_manifest_id=str(value["evidence_manifest_id"]),
            evidence_manifest_digest=str(value["evidence_manifest_digest"]),
            evidence_run_digest=str(value["evidence_run_digest"]),
            evidence_target_count=int(value["evidence_target_count"]),
            evidence_passed_target_count=int(value["evidence_passed_target_count"]),
            case_count=int(value["case_count"]),
            passed_case_count=int(value["passed_case_count"]),
            critical_case_failure_count=int(value["critical_case_failure_count"]),
            metric_results=tuple(dict(item) for item in value["metric_results"]),
            known_limitations=tuple(str(item) for item in value["known_limitations"]),
            private_output_only=bool(boundaries["private_output_only"]),
            raw_conversation_content_allowed=bool(boundaries["raw_conversation_content_allowed"]),
            web_access_allowed=bool(boundaries["web_access_allowed"]),
            tool_calling_allowed=bool(boundaries["tool_calling_allowed"]),
            external_action_allowed=bool(boundaries["external_action_allowed"]),
            memory_write_allowed=bool(boundaries["memory_write_allowed"]),
            repository_output_allowed=bool(boundaries["repository_output_allowed"]),
            record_digest=str(value["record_digest"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase3ReleaseAuditError("Phase 3 release record has an invalid structure.") from exc
    verify_phase3_release_decision(decision)
    return decision
