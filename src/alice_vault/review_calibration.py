from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .auto_review import (
    AUTO_COLUMNS,
    _promote_canonical,
    _read_csv,
    _result_record,
    _review_with_fallback,
    _write_csv,
)
from .content_extraction import extract_text
from .privacy_scan import (
    presidio_blocking_entities,
    scan_privacy,
)
from .semantic_review import OllamaLocalClient


CALIBRATION_SCHEMA_VERSION = 1
SAFE_RELEVANT_CATEGORIES = {
    "life_event",
    "education",
    "research_project",
    "work",
    "goal_or_plan",
    "personality_or_values",
    "communication_style",
    "workflow",
}
REJECT_CATEGORIES = {
    "generic_export",
    "advertisement",
    "unrelated",
}
MANUAL_CATEGORIES = {
    "financial",
    "medical",
    "legal_or_immigration",
    "relationship",
    "third_party",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_path(exports: Path, pattern: str) -> Path:
    candidates = sorted(
        exports.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No file matched {pattern!r}")
    return candidates[0]


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


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_checkpoint(
    path: Path,
    config_hash: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("config_hash") != config_hash:
            raise RuntimeError(
                "Calibration checkpoint settings do not match"
            )
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            results[str(item["content_key"])] = dict(item["result"])
    return results


def _append_checkpoint(
    path: Path,
    *,
    config_hash: str,
    config: dict[str, Any],
    content_key: str,
    result: dict[str, Any],
) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(
                json.dumps(
                    {
                        "config_hash": config_hash,
                        "config": config,
                        "created_at": utc_now(),
                    }
                )
                + "\n"
            )
        handle.write(
            json.dumps(
                {
                    "content_key": content_key,
                    "result": result,
                    "saved_at": utc_now(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _existing_policy_decision(
    result: dict[str, Any],
    *,
    approve_threshold: float,
    reject_threshold: float,
) -> dict[str, Any]:
    updated = dict(result)
    reason = str(result.get("reason", ""))
    category = str(result.get("category", ""))
    confidence = float(result.get("confidence", 0.0) or 0.0)

    hard_pending_prefixes = (
        "Deterministic secret indicator detected",
        "Identity-document indicator detected",
        "Prompt-injection-like text detected",
        "Sensitive topic indicators:",
        "Context-supported high-risk PII:",
        "Substantial third-party private data",
        "Model detected identity document or secrets",
        "Extraction status:",
        "Semantic review error:",
    )
    if reason.startswith(hard_pending_prefixes):
        updated["decision"] = "pending"
        return updated

    if category in MANUAL_CATEGORIES:
        updated["decision"] = "pending"
        updated["reason"] = (
            f"Sensitive category requires human review: {category}"
        )
        return updated

    if category in REJECT_CATEGORIES and confidence >= reject_threshold:
        updated["decision"] = "reject"
        updated["reason"] = (
            f"Calibrated rejection from existing semantic category: "
            f"{category}"
        )
        return updated

    # Older details did not retain the raw recommended_decision. Require a
    # higher confidence for inferred approval than new model results.
    inferred_approval_threshold = max(approve_threshold, 0.92)
    if (
        category in SAFE_RELEVANT_CATEGORIES
        and confidence >= inferred_approval_threshold
    ):
        updated["decision"] = "approve"
        updated["reason"] = (
            f"Calibrated approval from existing semantic category: "
            f"{category}"
        )
        return updated

    updated["decision"] = "pending"
    return updated


def _needs_targeted_semantic_retry(result: dict[str, Any]) -> bool:
    if str(result.get("category", "")):
        return False
    if str(result.get("extraction_status", "")) != "ok":
        return False
    reason = str(result.get("reason", ""))
    return (
        bool(result.get("extraction_truncated"))
        or reason.startswith("High-risk PII entities:")
        or reason.startswith("Local semantic review was unavailable")
    )


def _batched(values: list[dict[str, Any]], size: int):
    return [
        values[index:index + size]
        for index in range(0, len(values), size)
    ]


def recalibrate_auto_review(
    *,
    vault_root: Path,
    model: str = "qwen3:4b-instruct",
    base_url: str = "http://127.0.0.1:11434",
    use_presidio: bool = True,
    approve_threshold: float = 0.85,
    reject_threshold: float = 0.85,
    batch_size: int = 4,
    max_chars: int = 1200,
    timeout_seconds: int = 210,
    single_item_retries: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    exports = vault_root / "manifests" / "exports"
    database = vault_root / "manifests" / "inventory.sqlite3"

    proposal_id, proposal_path = _latest_proposal(exports)
    details_path = _latest_path(
        exports,
        "pilot-auto-review-details-*.json",
    )
    old_details = json.loads(details_path.read_text(encoding="utf-8"))
    old_results: dict[str, dict[str, Any]] = {
        str(key): dict(value)
        for key, value in old_details["content_results"].items()
    }

    fieldnames, rows = _read_csv(proposal_path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["content_key"]].append(row)

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    file_records = {
        row["file_id"]: row
        for row in connection.execute(
            "SELECT file_id, original_path, size_bytes, sha256 FROM files"
        )
    }
    connection.close()

    client = OllamaLocalClient(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        num_ctx=6144,
        num_predict=500,
    )
    client.verify_model()

    run_id = str(uuid.uuid4())
    started_at = utc_now()
    config = {
        "proposal_id": proposal_id,
        "source_details": details_path.name,
        "model": model,
        "base_url": base_url,
        "use_presidio": use_presidio,
        "approve_threshold": approve_threshold,
        "reject_threshold": reject_threshold,
        "batch_size": batch_size,
        "max_chars": max_chars,
        "timeout_seconds": timeout_seconds,
        "single_item_retries": single_item_retries,
        "schema_version": CALIBRATION_SCHEMA_VERSION,
    }
    config_hash = _config_hash(config)
    checkpoint_path = (
        exports
        / f"pilot-policy-calibration-checkpoint-"
        f"{proposal_id}-{config_hash}.jsonl"
    )
    retry_results = (
        _load_checkpoint(checkpoint_path, config_hash)
        if resume
        else {}
    )

    final_results: dict[str, dict[str, Any]] = {}
    retry_queue: list[dict[str, Any]] = []
    deterministic_retry_blocked = 0

    for content_key, members in groups.items():
        old = old_results.get(content_key)
        if old is None:
            raise RuntimeError(
                f"Existing details missing content key: {content_key}"
            )

        if content_key in retry_results:
            final_results[content_key] = retry_results[content_key]
            continue

        if not _needs_targeted_semantic_retry(old):
            final_results[content_key] = _existing_policy_decision(
                old,
                approve_threshold=approve_threshold,
                reject_threshold=reject_threshold,
            )
            continue

        representative = members[0]
        file_record = file_records.get(representative["file_id"])
        if file_record is None:
            raise RuntimeError(
                f"Missing file inventory record: "
                f"{representative['file_id']}"
            )

        extraction = extract_text(
            Path(file_record["original_path"]),
            representative["family"],
            max_chars=max_chars,
        )
        privacy = scan_privacy(
            extraction.text,
            metadata_text=(
                f"{representative['relative_path']} "
                f"{representative['filename']}"
            ),
            use_presidio=use_presidio,
        )

        hard_blocked = (
            extraction.status != "ok"
            or privacy.has_secret
            or privacy.has_identity_document
            or privacy.has_prompt_injection
            or bool(privacy.sensitive_topics)
            or bool(
                presidio_blocking_entities(
                    privacy,
                    extraction.text,
                )
            )
        )
        if hard_blocked:
            result = _result_record(
                extraction,
                privacy,
                None,
                approve_threshold=approve_threshold,
                reject_threshold=reject_threshold,
            )
            final_results[content_key] = result
            deterministic_retry_blocked += 1
            _append_checkpoint(
                checkpoint_path,
                config_hash=config_hash,
                config=config,
                content_key=content_key,
                result=result,
            )
            continue

        retry_queue.append(
            {
                "content_key": content_key,
                "members": members,
                "representative": representative,
                "extraction": extraction,
                "privacy": privacy,
            }
        )

    unresolved: dict[str, str] = {}
    request_batches = _batched(retry_queue, batch_size)
    for batch_index, batch in enumerate(request_batches, start=1):
        semantic_results, errors = _review_with_fallback(
            client,
            batch,
            "",
            single_item_retries=single_item_retries,
        )
        for item in batch:
            content_key = item["content_key"]
            semantic = semantic_results.get(content_key)
            if semantic is None:
                unresolved[content_key] = errors.get(
                    content_key,
                    "Ollama omitted targeted calibration item",
                )
                continue
            result = _result_record(
                item["extraction"],
                item["privacy"],
                semantic,
                approve_threshold=approve_threshold,
                reject_threshold=reject_threshold,
            )
            final_results[content_key] = result
            _append_checkpoint(
                checkpoint_path,
                config_hash=config_hash,
                config=config,
                content_key=content_key,
                result=result,
            )
        print(
            f"Completed targeted calibration batch "
            f"{batch_index}/{len(request_batches)} "
            f"({min(batch_index * batch_size, len(retry_queue))}/"
            f"{len(retry_queue)} targeted items)"
        )

    for item in retry_queue:
        content_key = item["content_key"]
        if content_key in final_results:
            continue
        final_results[content_key] = _result_record(
            item["extraction"],
            item["privacy"],
            None,
            approve_threshold=approve_threshold,
            reject_threshold=reject_threshold,
            semantic_error=unresolved.get(
                content_key,
                "Targeted semantic calibration unavailable",
            ),
        )

    if set(final_results) != set(groups):
        missing = set(groups).difference(final_results)
        raise RuntimeError(
            f"Calibration incomplete; missing {len(missing)} contents"
        )

    output_fields = list(fieldnames)
    for column in AUTO_COLUMNS:
        if column not in output_fields:
            output_fields.append(column)

    output_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    categories: Counter[str] = Counter()

    for row in rows:
        result = final_results[row["content_key"]]
        updated: dict[str, Any] = dict(row)
        updated["decision"] = result["decision"]
        updated["review_notes"] = (
            f"CALIBRATED: {result['reason']}"
        )[:1500]
        updated["known_contradiction_group"] = result.get(
            "contradiction_topic",
            "",
        )
        updated["contains_identity_document"] = result.get(
            "identity_flag",
            "",
        )
        updated["contains_credentials_or_secrets"] = result.get(
            "credential_flag",
            "",
        )
        updated.update(
            {
                "auto_review_run_id": run_id,
                "auto_decision": result["decision"],
                "auto_confidence": f"{float(result.get('confidence', 0)):.4f}",
                "auto_category": result.get("category", ""),
                "auto_sensitivity": result.get("sensitivity", ""),
                "auto_reason": result.get("reason", ""),
                "auto_summary": result.get("summary", ""),
                "extraction_status": result.get(
                    "extraction_status",
                    "",
                ),
                "extraction_truncated": str(
                    result.get("extraction_truncated", False)
                ).lower(),
                "privacy_flags": json.dumps(
                    result.get("privacy_flags", [])
                ),
                "needs_human_review": str(
                    result["decision"] == "pending"
                ).lower(),
                "semantic_recommended_decision": result.get(
                    "semantic_recommended_decision",
                    "",
                ),
                "semantic_relevant_to_alice": (
                    ""
                    if result.get("semantic_relevant_to_alice") is None
                    else str(
                        result["semantic_relevant_to_alice"]
                    ).lower()
                ),
                "semantic_contains_third_party_private_data": (
                    ""
                    if result.get(
                        "semantic_contains_third_party_private_data"
                    ) is None
                    else str(
                        result[
                            "semantic_contains_third_party_private_data"
                        ]
                    ).lower()
                ),
            }
        )
        output_rows.append(updated)
        decisions[result["decision"]] += 1
        categories[str(result.get("category", ""))] += 1
        if result["decision"] == "pending":
            manual_rows.append(updated)

    run_review_path = exports / f"pilot-review-calibrated-{run_id}.csv"
    canonical_path = exports / f"pilot-review-{proposal_id}.csv"
    manual_path = exports / f"pilot-manual-review-calibrated-{run_id}.csv"
    details_output = (
        exports / f"pilot-policy-calibration-details-{run_id}.json"
    )
    summary_path = (
        exports / f"pilot-policy-calibration-summary-{run_id}.json"
    )

    _write_csv(run_review_path, output_fields, output_rows)
    _write_csv(manual_path, output_fields, manual_rows)
    details_output.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_details_path": str(details_path),
                "content_results": final_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    promoted, promotion_error = _promote_canonical(
        source=run_review_path,
        target=canonical_path,
        run_id=run_id,
    )

    summary = {
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "run_id": run_id,
        "proposal_id": proposal_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "source_details_path": str(details_path),
        "model": model,
        "selected_file_records": len(rows),
        "unique_contents": len(groups),
        "reused_existing_semantic_results": (
            len(groups) - len(retry_queue) - deterministic_retry_blocked
        ),
        "targeted_semantic_retry_items": len(retry_queue),
        "targeted_semantic_batches": len(request_batches),
        "targeted_semantic_unresolved": len(unresolved),
        "deterministic_retry_blocked": deterministic_retry_blocked,
        "auto_approved": decisions["approve"],
        "auto_rejected": decisions["reject"],
        "manual_review_required": decisions["pending"],
        "category_counts": dict(categories),
        "ollama_metrics": client.metrics(),
        "canonical_review_updated": promoted,
        "canonical_review_error": promotion_error,
        "run_review_csv_path": str(run_review_path),
        "manual_review_csv_path": str(manual_path),
        "details_path": str(details_output),
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary
