"""Final Phase 2 Memory Core release audit for A.L.I.C.E. P2.9d."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation_contract import canonical_json
from .final_evaluation import (
    MemoryCoreFinalEvaluationReport,
    verify_memory_core_final_report,
)

PHASE2_RELEASE_AUDIT_VERSION = "p2.9d-v1"
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


class Phase2ReleaseAuditError(RuntimeError):
    """Raised when a Phase 2 release record is invalid or unsafe."""


@dataclass(frozen=True)
class Phase2ReleaseMetadata:
    repository_commit: str
    evaluated_at: str
    policy_versions: tuple[str, ...]
    model: str = "not_applicable"
    provider: str = "not_applicable"
    system_prompt_version: str = "not_applicable"
    tool_versions: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    rollback_commit: str | None = None


@dataclass(frozen=True)
class Phase2ReleaseDecision:
    audit_version: str
    release_id: str
    approved: bool
    decision_reasons: tuple[str, ...]
    repository_commit: str
    evaluated_at: str
    benchmark_id: str
    benchmark_digest: str
    test_set_version: str
    policy_id: str
    policy_digest: str
    fixture_snapshot_id: str
    evaluation_report_digest: str
    metric_results: tuple[dict[str, Any], ...]
    known_limitations: tuple[str, ...]
    rollback_commit: str | None
    record_digest: str


def _utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Phase2ReleaseAuditError(
            "Release evaluation time must be ISO-8601."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Phase2ReleaseAuditError(
            "Release evaluation time must be explicitly UTC."
        )
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_metadata(metadata: Phase2ReleaseMetadata) -> Phase2ReleaseMetadata:
    commit = metadata.repository_commit.strip().lower()
    if not _COMMIT.fullmatch(commit):
        raise Phase2ReleaseAuditError(
            "Release metadata requires a 7-40 character hexadecimal commit."
        )
    rollback = metadata.rollback_commit
    if rollback is not None:
        rollback = rollback.strip().lower()
        if not _COMMIT.fullmatch(rollback):
            raise Phase2ReleaseAuditError(
                "rollback_commit must be a hexadecimal Git commit."
            )
        if rollback == commit:
            raise Phase2ReleaseAuditError(
                "rollback_commit must differ from repository_commit."
            )
    if not metadata.policy_versions or any(
        not value.strip() for value in metadata.policy_versions
    ):
        raise Phase2ReleaseAuditError(
            "Release metadata requires explicit policy versions."
        )
    if len(set(metadata.policy_versions)) != len(metadata.policy_versions):
        raise Phase2ReleaseAuditError(
            "Release metadata contains duplicate policy versions."
        )
    if (
        metadata.model != "not_applicable"
        or metadata.provider != "not_applicable"
        or metadata.system_prompt_version != "not_applicable"
    ):
        raise Phase2ReleaseAuditError(
            "Phase 2 release audit must not claim a model, provider, or prompt."
        )
    if metadata.tool_versions:
        raise Phase2ReleaseAuditError(
            "Phase 2 release audit must not depend on tool execution."
        )
    if len(set(metadata.known_limitations)) != len(metadata.known_limitations):
        raise Phase2ReleaseAuditError(
            "Release metadata contains duplicate limitations."
        )
    return replace(
        metadata,
        repository_commit=commit,
        evaluated_at=_utc_timestamp(metadata.evaluated_at),
        rollback_commit=rollback,
    )


def _decision_material(decision: Phase2ReleaseDecision) -> dict[str, Any]:
    return {
        "audit_version": decision.audit_version,
        "release_id": decision.release_id,
        "approved": decision.approved,
        "decision_reasons": list(decision.decision_reasons),
        "repository_commit": decision.repository_commit,
        "evaluated_at": decision.evaluated_at,
        "benchmark_id": decision.benchmark_id,
        "benchmark_digest": decision.benchmark_digest,
        "test_set_version": decision.test_set_version,
        "policy_id": decision.policy_id,
        "policy_digest": decision.policy_digest,
        "fixture_snapshot_id": decision.fixture_snapshot_id,
        "evaluation_report_digest": decision.evaluation_report_digest,
        "metric_results": list(decision.metric_results),
        "known_limitations": list(decision.known_limitations),
        "rollback_commit": decision.rollback_commit,
    }


def phase2_release_record_digest(decision: Phase2ReleaseDecision) -> str:
    return hashlib.sha256(canonical_json(_decision_material(decision))).hexdigest()


def audit_phase2_release(
    report: MemoryCoreFinalEvaluationReport,
    *,
    metadata: Phase2ReleaseMetadata,
) -> Phase2ReleaseDecision:
    """Create a deterministic release decision from a verified final report."""
    verify_memory_core_final_report(report)
    metadata = _validate_metadata(metadata)

    reasons: list[str] = []
    if not report.passed:
        reasons.append("final_memory_evaluation_failed")
    failed_metrics = tuple(
        item.metric_id for item in report.metric_results if not item.passed
    )
    if failed_metrics:
        reasons.extend(f"metric_failed:{item}" for item in failed_metrics)
    if report.critical_case_failure_count:
        reasons.append("critical_case_failure")
    if report.passed_case_count != report.case_count:
        reasons.append("incomplete_case_pass")
    if not report.citation_summary.passes_all_p29b_gates:
        reasons.append("citation_evaluation_failed")
    if (
        report.memory_write_allowed
        or report.external_action_allowed
        or report.tool_calling_allowed
        or report.web_access_allowed
        or not report.private_output_only
    ):
        reasons.append("evaluation_capability_boundary_failed")

    approved = not reasons
    release_seed = {
        "audit_version": PHASE2_RELEASE_AUDIT_VERSION,
        "repository_commit": metadata.repository_commit,
        "evaluated_at": metadata.evaluated_at,
        "evaluation_report_digest": report.report_digest,
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
    decision = Phase2ReleaseDecision(
        audit_version=PHASE2_RELEASE_AUDIT_VERSION,
        release_id=release_id,
        approved=approved,
        decision_reasons=tuple(sorted(set(reasons))),
        repository_commit=metadata.repository_commit,
        evaluated_at=metadata.evaluated_at,
        benchmark_id=report.benchmark_id,
        benchmark_digest=report.benchmark_digest,
        test_set_version=report.test_set_version,
        policy_id=report.policy_id,
        policy_digest=report.policy_digest,
        fixture_snapshot_id=report.fixture_snapshot_id,
        evaluation_report_digest=report.report_digest,
        metric_results=metrics,
        known_limitations=metadata.known_limitations,
        rollback_commit=metadata.rollback_commit,
        record_digest="",
    )
    return replace(
        decision,
        record_digest=phase2_release_record_digest(decision),
    )


def verify_phase2_release_decision(decision: Phase2ReleaseDecision) -> None:
    if decision.record_digest != phase2_release_record_digest(decision):
        raise Phase2ReleaseAuditError(
            "Phase 2 release record digest is invalid."
        )
    if decision.approved != (not decision.decision_reasons):
        raise Phase2ReleaseAuditError(
            "Phase 2 release decision is internally inconsistent."
        )
    if not _COMMIT.fullmatch(decision.repository_commit):
        raise Phase2ReleaseAuditError(
            "Phase 2 release record contains an invalid repository commit."
        )
    _utc_timestamp(decision.evaluated_at)
    metric_ids = [str(item.get("metric_id", "")) for item in decision.metric_results]
    if not metric_ids or len(set(metric_ids)) != len(metric_ids):
        raise Phase2ReleaseAuditError(
            "Phase 2 release record metric results are invalid."
        )


def phase2_release_record_json(decision: Phase2ReleaseDecision) -> str:
    verify_phase2_release_decision(decision)
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
        raise Phase2ReleaseAuditError(
            "Phase 2 release records must stay under the private output root."
        ) from exc
    if repository_root is not None:
        repository = Path(repository_root).expanduser().resolve(strict=True)
        try:
            output.relative_to(repository)
        except ValueError:
            pass
        else:
            raise Phase2ReleaseAuditError(
                "Phase 2 release records cannot be written into the repository."
            )
    if output.suffix.lower() != ".json":
        raise Phase2ReleaseAuditError(
            "Phase 2 release record output must use a .json suffix."
        )
    return output


def write_phase2_release_record(
    decision: Phase2ReleaseDecision,
    output_path: str | Path,
    *,
    private_root: str | Path,
    repository_root: str | Path | None = None,
) -> Path:
    """Write a verified release record only inside an approved private root."""
    verify_phase2_release_decision(decision)
    output = _resolved_private_output(
        output_path,
        private_root=private_root,
        repository_root=repository_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = phase2_release_record_json(decision)
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != payload:
            raise Phase2ReleaseAuditError(
                "Refusing to overwrite a different Phase 2 release record."
            )
        return output
    output.write_text(payload, encoding="utf-8", newline="\n")
    return output


def load_phase2_release_record(path: str | Path) -> Phase2ReleaseDecision:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2ReleaseAuditError(
            "Phase 2 release record could not be loaded."
        ) from exc
    try:
        decision = Phase2ReleaseDecision(
            audit_version=str(value["audit_version"]),
            release_id=str(value["release_id"]),
            approved=bool(value["approved"]),
            decision_reasons=tuple(str(item) for item in value["decision_reasons"]),
            repository_commit=str(value["repository_commit"]),
            evaluated_at=str(value["evaluated_at"]),
            benchmark_id=str(value["benchmark_id"]),
            benchmark_digest=str(value["benchmark_digest"]),
            test_set_version=str(value["test_set_version"]),
            policy_id=str(value["policy_id"]),
            policy_digest=str(value["policy_digest"]),
            fixture_snapshot_id=str(value["fixture_snapshot_id"]),
            evaluation_report_digest=str(value["evaluation_report_digest"]),
            metric_results=tuple(dict(item) for item in value["metric_results"]),
            known_limitations=tuple(str(item) for item in value["known_limitations"]),
            rollback_commit=(
                None
                if value.get("rollback_commit") is None
                else str(value["rollback_commit"])
            ),
            record_digest=str(value["record_digest"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase2ReleaseAuditError(
            "Phase 2 release record has an invalid structure."
        ) from exc
    verify_phase2_release_decision(decision)
    return decision
