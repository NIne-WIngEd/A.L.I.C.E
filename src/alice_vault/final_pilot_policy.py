from __future__ import annotations

import csv
import json
import os
import random
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_POLICY_SCHEMA_VERSION = 1

POSITIVE_CATEGORIES = {
    "life_event",
    "education",
    "research_project",
    "work",
    "goal_or_plan",
    "personality_or_values",
    "communication_style",
    "workflow",
}

NEGATIVE_CATEGORIES = {
    "generic_export",
    "advertisement",
    "unrelated",
}

PILOT_EXCLUDE_CATEGORIES = {
    "financial",
    "medical",
    "legal_or_immigration",
    "relationship",
    "third_party",
}

HARD_EXCLUDE_PREFIXES = (
    "Deterministic secret indicator detected",
    "Identity-document indicator detected",
    "Prompt-injection-like text detected",
    "Sensitive topic indicators:",
    "Context-supported high-risk PII:",
    "High-risk PII entities:",
    "Model detected identity document or secrets",
    "Substantial third-party private data",
    "Sensitive category requires human review:",
    "Extraction status:",
    "Semantic review error:",
)

AUTO_COLUMNS = [
    "final_policy_run_id",
    "final_policy_decision",
    "final_policy_reason",
    "final_policy_confidence",
    "final_policy_category",
    "final_policy_source",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_path(root: Path, pattern: str) -> Path:
    matches = sorted(
        root.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No file matched {pattern!r}")
    return matches[0]


def _latest_proposal(exports: Path) -> tuple[str, Path]:
    summary_path = _latest_path(
        exports,
        "pilot-proposal-summary-*.json",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    proposal_id = str(summary["proposal_id"])
    proposal_path = exports / f"pilot-proposal-{proposal_id}.csv"
    if not proposal_path.is_file():
        raise FileNotFoundError(
            f"Pilot proposal CSV not found: {proposal_path}"
        )
    return proposal_id, proposal_path


def _read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _bool_value(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def decide_for_pilot(
    result: dict[str, Any],
    *,
    approve_threshold: float = 0.85,
    truncated_approve_threshold: float = 0.92,
    reject_threshold: float = 0.80,
) -> dict[str, Any]:
    """Make a conservative final decision for the parser pilot.

    A rejection only excludes an item from pilot-v1. It does not delete,
    quarantine, or alter the source archive.
    """
    existing = str(result.get("decision", "")).strip().lower()
    reason = str(result.get("reason", "")).strip()
    category = str(result.get("category", "")).strip()
    confidence = float(result.get("confidence", 0) or 0)
    recommendation = str(
        result.get("semantic_recommended_decision", "")
    ).strip().lower()
    third_party = _bool_value(
        result.get("semantic_contains_third_party_private_data")
    )
    extraction_status = str(
        result.get("extraction_status", "")
    ).strip().lower()
    truncated = bool(result.get("extraction_truncated", False))
    identity_flag = str(
        result.get("identity_flag", "")
    ).strip().lower()
    credential_flag = str(
        result.get("credential_flag", "")
    ).strip().lower()

    if existing == "approve":
        return {
            "decision": "approve",
            "reason": "Preserved calibrated approval",
            "source": "existing_calibrated_decision",
        }
    if existing == "reject":
        return {
            "decision": "reject",
            "reason": "Preserved calibrated rejection",
            "source": "existing_calibrated_decision",
        }

    if extraction_status not in {"ok", ""}:
        return {
            "decision": "reject",
            "reason": (
                f"Excluded from parser pilot: extraction status "
                f"{extraction_status}"
            ),
            "source": "deterministic_exclusion",
        }

    if (
        identity_flag == "yes"
        or credential_flag == "yes"
        or third_party is True
        or reason.startswith(HARD_EXCLUDE_PREFIXES)
        or category in PILOT_EXCLUDE_CATEGORIES
    ):
        return {
            "decision": "reject",
            "reason": (
                "Conservatively excluded from pilot-v1 because it is "
                "sensitive, third-party, identity/credential-related, "
                "or unsuitable for automatic ingestion"
            ),
            "source": "risk_exclusion",
        }

    if category in NEGATIVE_CATEGORIES:
        if confidence >= reject_threshold or recommendation == "reject":
            return {
                "decision": "reject",
                "reason": (
                    f"Excluded low-value pilot category: {category}"
                ),
                "source": "semantic_rejection",
            }

    effective_approve_threshold = (
        truncated_approve_threshold
        if truncated
        else approve_threshold
    )
    if (
        recommendation == "approve"
        and category in POSITIVE_CATEGORIES
        and confidence >= effective_approve_threshold
    ):
        return {
            "decision": "approve",
            "reason": (
                "High-confidence positive category and explicit model "
                "approval; contradictory relevance boolean treated as "
                "an advisory field"
            ),
            "source": "semantic_consensus",
        }

    # Older results did not always preserve recommendation. Require stronger
    # evidence when inferring approval from category alone.
    if (
        not recommendation
        and category in POSITIVE_CATEGORIES
        and confidence >= max(0.92, effective_approve_threshold)
    ):
        return {
            "decision": "approve",
            "reason": (
                "High-confidence positive category from an older semantic "
                "result"
            ),
            "source": "legacy_semantic_inference",
        }

    if recommendation == "reject" and confidence >= reject_threshold:
        return {
            "decision": "reject",
            "reason": "High-confidence semantic rejection",
            "source": "semantic_rejection",
        }

    # The initial pilot should be small and clean. Ambiguous records are
    # excluded rather than forcing the owner to review them all. They remain
    # available for later pilot versions.
    return {
        "decision": "reject",
        "reason": (
            "Excluded from pilot-v1 because evidence was ambiguous or "
            "below the automatic approval threshold"
        ),
        "source": "conservative_ambiguity_exclusion",
    }


def _promote(
    *,
    source: Path,
    target: Path,
    run_id: str,
) -> tuple[bool, str]:
    temporary = target.with_name(f".{target.name}.{run_id}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        return True, ""
    except PermissionError as exc:
        temporary.unlink(missing_ok=True)
        return (
            False,
            "The canonical review CSV is locked. Close Excel and retry. "
            f"Windows error: {exc}",
        )


def apply_final_pilot_policy(
    *,
    vault_root: Path,
    approve_threshold: float = 0.85,
    truncated_approve_threshold: float = 0.92,
    reject_threshold: float = 0.80,
    audit_approved: int = 10,
    audit_rejected: int = 5,
    audit_seed: str = "alice-pilot-v1-audit",
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    exports = vault_root / "manifests" / "exports"
    proposal_id, proposal_path = _latest_proposal(exports)
    details_path = _latest_path(
        exports,
        "pilot-policy-calibration-details-*.json",
    )
    details = json.loads(details_path.read_text(encoding="utf-8"))
    results = {
        str(key): dict(value)
        for key, value in details["content_results"].items()
    }

    fieldnames, rows = _read_csv(proposal_path)
    output_fields = list(fieldnames)
    for column in AUTO_COLUMNS:
        if column not in output_fields:
            output_fields.append(column)

    run_id = str(uuid.uuid4())
    started_at = utc_now()
    output_rows: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    decision_sources: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_buckets: set[str] = set()
    years: set[str] = set()
    contradiction_groups: dict[str, int] = defaultdict(int)
    duplicate_groups: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        content_key = row["content_key"]
        result = results.get(content_key)
        if result is None:
            raise RuntimeError(
                f"Calibration result missing for content: {content_key}"
            )

        final = decide_for_pilot(
            result,
            approve_threshold=approve_threshold,
            truncated_approve_threshold=(
                truncated_approve_threshold
            ),
            reject_threshold=reject_threshold,
        )
        updated: dict[str, Any] = dict(row)
        updated["decision"] = final["decision"]
        updated["review_notes"] = (
            f"FINAL POLICY: {final['reason']}"
        )[:1500]
        updated["known_contradiction_group"] = result.get(
            "contradiction_topic",
            "",
        )
        if final["decision"] == "approve":
            updated["contains_identity_document"] = "no"
            updated["contains_credentials_or_secrets"] = "no"
        else:
            # Preserve any positive flags for the exclusion audit.
            updated["contains_identity_document"] = result.get(
                "identity_flag",
                row.get("contains_identity_document", ""),
            )
            updated["contains_credentials_or_secrets"] = result.get(
                "credential_flag",
                row.get("contains_credentials_or_secrets", ""),
            )

        updated.update(
            {
                "final_policy_run_id": run_id,
                "final_policy_decision": final["decision"],
                "final_policy_reason": final["reason"],
                "final_policy_confidence": (
                    f"{float(result.get('confidence', 0) or 0):.4f}"
                ),
                "final_policy_category": result.get("category", ""),
                "final_policy_source": final["source"],
            }
        )
        output_rows.append(updated)
        decisions[final["decision"]] += 1
        decision_sources[final["source"]] += 1
        categories[str(result.get("category", ""))] += 1

        duplicate_group = row.get("duplicate_control_group", "").strip()
        if duplicate_group:
            duplicate_groups[duplicate_group].append(final["decision"])

        if final["decision"] == "approve":
            family_counts[row.get("family", "")] += 1
            source_buckets.add(row.get("source_bucket", ""))
            year = row.get("year_hint", "").strip()
            if year and year != "[unknown]":
                years.add(year)
            contradiction = str(
                result.get("contradiction_topic", "")
            ).strip()
            if contradiction:
                contradiction_groups[contradiction] += 1

    # Duplicate controls must be all approved or all rejected.
    mixed_duplicate_groups = {
        group: values
        for group, values in duplicate_groups.items()
        if len(set(values)) > 1
    }
    if mixed_duplicate_groups:
        raise RuntimeError(
            "Final policy produced mixed decisions in duplicate controls: "
            + ", ".join(sorted(mixed_duplicate_groups))
        )

    run_review_path = (
        exports / f"pilot-review-final-policy-{run_id}.csv"
    )
    canonical_path = exports / f"pilot-review-{proposal_id}.csv"
    audit_path = exports / f"pilot-policy-audit-{run_id}.csv"
    summary_path = (
        exports / f"pilot-final-policy-summary-{run_id}.json"
    )

    _write_csv(run_review_path, output_fields, output_rows)

    rng = random.Random(audit_seed)
    approved_rows = [
        row for row in output_rows if row["decision"] == "approve"
    ]
    rejected_rows = [
        row for row in output_rows if row["decision"] == "reject"
    ]
    audit_rows = (
        rng.sample(
            approved_rows,
            min(audit_approved, len(approved_rows)),
        )
        + rng.sample(
            rejected_rows,
            min(audit_rejected, len(rejected_rows)),
        )
    )
    audit_fields = [
        "file_id",
        "relative_path",
        "filename",
        "family",
        "source_bucket",
        "year_hint",
        "decision",
        "review_notes",
        "final_policy_confidence",
        "final_policy_category",
        "final_policy_source",
        "known_contradiction_group",
    ]
    _write_csv(audit_path, audit_fields, audit_rows)

    promoted, promotion_error = _promote(
        source=run_review_path,
        target=canonical_path,
        run_id=run_id,
    )

    valid_contradiction_groups = sum(
        count >= 2 for count in contradiction_groups.values()
    )
    summary = {
        "final_policy_schema_version": FINAL_POLICY_SCHEMA_VERSION,
        "run_id": run_id,
        "proposal_id": proposal_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "source_calibration_details": str(details_path),
        "selected_file_records": len(rows),
        "auto_approved": decisions["approve"],
        "auto_rejected": decisions["reject"],
        "manual_review_required": decisions["pending"],
        "decision_source_counts": dict(decision_sources),
        "approved_family_counts": dict(family_counts),
        "approved_source_bucket_count": len(source_buckets),
        "approved_known_year_count": len(years),
        "complete_duplicate_control_groups": sum(
            len(values) == 2 and len(set(values)) == 1
            for values in duplicate_groups.values()
        ),
        "valid_contradiction_groups": valid_contradiction_groups,
        "audit_sample_count": len(audit_rows),
        "canonical_review_updated": promoted,
        "canonical_review_error": promotion_error,
        "run_review_csv_path": str(run_review_path),
        "canonical_review_csv_path": str(canonical_path),
        "audit_csv_path": str(audit_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary
