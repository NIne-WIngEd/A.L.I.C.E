from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .content_extraction import ExtractionResult, extract_text
from .privacy_scan import (
    PrivacyScanResult,
    presidio_blocking_entities,
    scan_privacy,
)
from .semantic_review import OllamaLocalClient, SemanticReview


AUTO_REVIEW_SCHEMA_VERSION = 4
AUTO_COLUMNS = [
    "auto_review_run_id",
    "auto_decision",
    "auto_confidence",
    "auto_category",
    "auto_sensitivity",
    "auto_reason",
    "auto_summary",
    "extraction_status",
    "extraction_truncated",
    "privacy_flags",
    "needs_human_review",
    "semantic_recommended_decision",
    "semantic_relevant_to_alice",
    "semantic_contains_third_party_private_data",
]
MANUAL_EDIT_COLUMNS = [
    "decision",
    "review_notes",
    "known_contradiction_group",
    "contains_identity_document",
    "contains_credentials_or_secrets",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _latest_proposal(exports: Path) -> tuple[str, Path]:
    summaries = sorted(
        exports.glob("pilot-proposal-summary-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        raise FileNotFoundError("No pilot proposal summary found")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    proposal_id = str(summary["proposal_id"])
    csv_path = exports / f"pilot-proposal-{proposal_id}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Pilot proposal CSV not found: {csv_path}"
        )
    return proposal_id, csv_path


def _privacy_flags(result: PrivacyScanResult) -> list[str]:
    flags: list[str] = []
    flags.extend(f"secret:{item}" for item in result.secret_types)
    flags.extend(
        f"identity:{item}" for item in result.identity_document_types
    )
    flags.extend(
        f"prompt_injection:{item}"
        for item in result.prompt_injection_types
    )
    flags.extend(f"sensitive:{item}" for item in result.sensitive_topics)
    flags.extend(
        f"pii:{key}:{value}"
        for key, value in sorted(result.pii_counts.items())
    )
    flags.extend(
        f"presidio:{key}:{value}"
        for key, value in sorted(result.presidio_counts.items())
    )
    return flags


def _decide(
    extraction: ExtractionResult,
    privacy: PrivacyScanResult,
    semantic: SemanticReview | None,
    *,
    approve_threshold: float,
    reject_threshold: float,
) -> tuple[str, float, str, str, str]:
    if extraction.status == "empty":
        return "reject", 1.0, "Empty file", "no", "no"
    if extraction.status in {
        "unsupported",
        "too_large",
        "error",
        "no_text",
    }:
        return (
            "pending",
            0.0,
            f"Extraction status: {extraction.status}",
            "",
            "",
        )
    if privacy.has_secret:
        return (
            "pending",
            1.0,
            "Deterministic secret indicator detected",
            "no",
            "yes",
        )
    if privacy.has_identity_document:
        return (
            "pending",
            1.0,
            "Identity-document indicator detected",
            "yes",
            "no",
        )
    if privacy.has_prompt_injection:
        return (
            "pending",
            1.0,
            "Prompt-injection-like text detected",
            "no",
            "no",
        )
    if privacy.sensitive_topics:
        topics = ", ".join(privacy.sensitive_topics)
        return (
            "pending",
            1.0,
            f"Sensitive topic indicators: {topics}",
            "no",
            "no",
        )

    blocking_presidio = presidio_blocking_entities(
        privacy,
        extraction.text,
    )
    if blocking_presidio:
        entities = ", ".join(sorted(blocking_presidio))
        return (
            "pending",
            1.0,
            f"Context-supported high-risk PII: {entities}",
            "no",
            "no",
        )

    if semantic is None:
        return (
            "pending",
            0.0,
            "Local semantic review was unavailable",
            "no",
            "no",
        )

    identity = "yes" if semantic.contains_identity_document else "no"
    credentials = (
        "yes" if semantic.contains_credentials_or_secrets else "no"
    )
    if (
        semantic.contains_identity_document
        or semantic.contains_credentials_or_secrets
    ):
        return (
            "pending",
            semantic.relevance_score,
            "Model detected identity document or secrets",
            identity,
            credentials,
        )
    if semantic.contains_third_party_private_data:
        return (
            "pending",
            semantic.relevance_score,
            "Substantial third-party private data",
            identity,
            credentials,
        )

    manual_categories = {
        "financial",
        "medical",
        "legal_or_immigration",
        "relationship",
    }
    if semantic.document_category in manual_categories:
        return (
            "pending",
            semantic.relevance_score,
            f"Sensitive category requires human review: "
            f"{semantic.document_category}",
            identity,
            credentials,
        )

    # A single document cannot establish a contradiction by itself. Preserve
    # the topic as an annotation for later cross-document comparison, but do
    # not make it a blanket blocker.
    if semantic.recommended_decision == "manual":
        return (
            "pending",
            semantic.relevance_score,
            semantic.reason,
            identity,
            credentials,
        )

    effective_approve_threshold = (
        max(approve_threshold, 0.92)
        if extraction.truncated
        else approve_threshold
    )
    if (
        semantic.recommended_decision == "approve"
        and semantic.relevant_to_alice
        and semantic.relevance_score >= effective_approve_threshold
    ):
        suffix = " (truncated preview)" if extraction.truncated else ""
        return (
            "approve",
            semantic.relevance_score,
            f"{semantic.reason}{suffix}",
            identity,
            credentials,
        )

    if (
        semantic.recommended_decision == "reject"
        and not semantic.relevant_to_alice
        and semantic.relevance_score >= reject_threshold
        and semantic.document_category
        in {
            "generic_export",
            "advertisement",
            "third_party",
            "unrelated",
        }
    ):
        return (
            "reject",
            semantic.relevance_score,
            semantic.reason,
            identity,
            credentials,
        )

    return (
        "pending",
        semantic.relevance_score,
        semantic.reason or "Low-confidence semantic decision",
        identity,
        credentials,
    )


def _result_record(
    extraction: ExtractionResult,
    privacy: PrivacyScanResult,
    semantic: SemanticReview | None,
    *,
    approve_threshold: float,
    reject_threshold: float,
    semantic_error: str = "",
) -> dict[str, Any]:
    decision, confidence, reason, identity_flag, credential_flag = _decide(
        extraction,
        privacy,
        semantic,
        approve_threshold=approve_threshold,
        reject_threshold=reject_threshold,
    )
    if semantic_error:
        decision = "pending"
        confidence = 0.0
        reason = f"Semantic review error: {semantic_error}"
    return {
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "category": semantic.document_category if semantic else "",
        "sensitivity": semantic.sensitivity if semantic else "",
        "summary": semantic.summary if semantic else "",
        "identity_flag": identity_flag,
        "credential_flag": credential_flag,
        "contradiction_topic": (
            semantic.contradiction_topic if semantic else ""
        ),
        "semantic_recommended_decision": (
            semantic.recommended_decision if semantic else ""
        ),
        "semantic_relevant_to_alice": (
            semantic.relevant_to_alice if semantic else None
        ),
        "semantic_contains_third_party_private_data": (
            semantic.contains_third_party_private_data
            if semantic
            else None
        ),
        "extraction_status": extraction.status,
        "extraction_truncated": extraction.truncated,
        "privacy_flags": _privacy_flags(privacy),
    }


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
        header = handle.readline()
        if not header:
            return {}
        first = json.loads(header)
        if first.get("type") != "config":
            raise RuntimeError("Invalid auto-review checkpoint header")
        if first.get("config_hash") != config_hash:
            raise RuntimeError(
                "Existing checkpoint belongs to different settings. "
                "Use --no-resume or delete the checkpoint."
            )
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") == "result":
                result = dict(entry["result"])
                # Transient model failures must be retried, not treated as
                # completed work during resume.
                reason = str(result.get("reason", ""))
                if reason.startswith("Semantic review error:"):
                    continue
                results[str(entry["content_key"])] = result
    return results


def _append_checkpoint(
    path: Path,
    *,
    config_hash: str,
    config: dict[str, Any],
    content_key: str,
    result: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(
                json.dumps(
                    {
                        "type": "config",
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
                    "type": "result",
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


def _batched(
    values: list[dict[str, Any]],
    size: int,
) -> list[list[dict[str, Any]]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _review_batch_compat(
    client: Any,
    batch: list[dict[str, Any]],
    private_profile: str,
) -> dict[str, SemanticReview]:
    if hasattr(client, "review_batch"):
        request_items: list[dict[str, str]] = []
        id_to_key: dict[str, str] = {}
        for index, item in enumerate(batch, start=1):
            request_id = f"i{index}"
            id_to_key[request_id] = item["content_key"]
            request_items.append(
                {
                    "item_id": request_id,
                    "filename": item["representative"]["filename"],
                    "family": item["representative"]["family"],
                    "source_bucket": item["representative"]["source_bucket"],
                    "year_hint": item["representative"]["year_hint"],
                    "text": item["extraction"].text,
                }
            )
        raw = client.review_batch(
            items=request_items,
            private_profile=private_profile,
        )
        return {id_to_key[item_id]: review for item_id, review in raw.items()}

    # Compatibility for tests and older fake clients.
    result: dict[str, SemanticReview] = {}
    for item in batch:
        representative = item["representative"]
        result[item["content_key"]] = client.review(
            filename=representative["filename"],
            family=representative["family"],
            source_bucket=representative["source_bucket"],
            year_hint=representative["year_hint"],
            text=item["extraction"].text,
            private_profile=private_profile,
        )
    return result


def _review_with_fallback(
    client: Any,
    batch: list[dict[str, Any]],
    private_profile: str,
    *,
    single_item_retries: int,
) -> tuple[dict[str, SemanticReview], dict[str, str]]:
    """Retry failed batches by splitting them down to individual items."""
    try:
        return _review_batch_compat(client, batch, private_profile), {}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if len(batch) > 1:
        midpoint = len(batch) // 2
        left_results, left_errors = _review_with_fallback(
            client,
            batch[:midpoint],
            private_profile,
            single_item_retries=single_item_retries,
        )
        right_results, right_errors = _review_with_fallback(
            client,
            batch[midpoint:],
            private_profile,
            single_item_retries=single_item_retries,
        )
        left_results.update(right_results)
        left_errors.update(right_errors)
        return left_results, left_errors

    item = batch[0]
    last_error = error
    for _ in range(single_item_retries):
        # A shorter final preview reduces timeout/truncation risk.
        original = item["extraction"]
        shortened_text = original.text[:900]
        shortened = type(original)(
            status=original.status,
            text=shortened_text,
            chars=len(shortened_text),
            truncated=False,
            parser=original.parser,
            error=original.error,
        )
        retry_item = dict(item)
        retry_item["extraction"] = shortened
        try:
            return _review_batch_compat(
                client, [retry_item], private_profile
            ), {}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return {}, {item["content_key"]: last_error}


def _promote_canonical(
    *,
    source: Path,
    target: Path,
    run_id: str,
) -> tuple[bool, str]:
    temporary = target.with_name(
        f".{target.name}.{run_id}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        return True, ""
    except PermissionError as exc:
        temporary.unlink(missing_ok=True)
        return (
            False,
            "The canonical review CSV is locked by another program. "
            "Close Excel or any CSV viewer, then run promote_auto_review.py. "
            f"Windows error: {exc}",
        )


def auto_review_pilot(
    *,
    vault_root: Path,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    use_ollama: bool = True,
    use_presidio: bool = False,
    approve_threshold: float = 0.85,
    reject_threshold: float = 0.85,
    profile_path: Path | None = None,
    max_chars: int = 1800,
    batch_size: int = 3,
    resume: bool = True,
    num_ctx: int = 8192,
    num_predict: int = 600,
    timeout_seconds: int = 240,
    single_item_retries: int = 1,
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 16:
        raise ValueError("batch_size must be between 1 and 16")
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500")

    vault_root = vault_root.expanduser().resolve(strict=True)
    database = vault_root / "manifests" / "inventory.sqlite3"
    exports = vault_root / "manifests" / "exports"
    proposal_id, proposal_path = _latest_proposal(exports)
    fieldnames, rows = _read_csv(proposal_path)
    required = {
        "file_id",
        "content_key",
        "relative_path",
        "filename",
        "family",
        "source_bucket",
        "year_hint",
        "decision",
        "review_notes",
        "known_contradiction_group",
        "contains_identity_document",
        "contains_credentials_or_secrets",
    }
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"Proposal CSV missing columns: {sorted(missing)}"
        )

    private_profile = ""
    profile_hash = ""
    if profile_path is not None:
        profile_text = profile_path.expanduser().read_text(
            encoding="utf-8"
        )
        private_profile = profile_text[:6000]
        profile_hash = hashlib.sha256(
            profile_text.encode("utf-8")
        ).hexdigest()

    client: OllamaLocalClient | None = None
    if use_ollama:
        client = OllamaLocalClient(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        client.verify_model()

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    file_records = {
        row["file_id"]: row
        for row in connection.execute(
            "SELECT file_id, original_path, size_bytes, sha256 FROM files"
        )
    }
    connection.close()

    run_id = str(uuid.uuid4())
    started_at = utc_now()
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["content_key"]].append(row)

    config = {
        "proposal_id": proposal_id,
        "model": model if use_ollama else None,
        "base_url": base_url if use_ollama else None,
        "use_presidio": use_presidio,
        "approve_threshold": approve_threshold,
        "reject_threshold": reject_threshold,
        "max_chars": max_chars,
        "batch_size": batch_size,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "timeout_seconds": timeout_seconds,
        "single_item_retries": single_item_retries,
        "profile_hash": profile_hash,
        "schema_version": AUTO_REVIEW_SCHEMA_VERSION,
    }
    config_hash = _config_hash(config)
    checkpoint_path = (
        exports
        / f"pilot-auto-review-checkpoint-{proposal_id}-{config_hash}.jsonl"
    )
    if not resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    content_results = (
        _load_checkpoint(checkpoint_path, config_hash)
        if resume
        else {}
    )
    resumed_content_count = len(content_results)

    extraction_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    privacy_flag_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    model_error_count = 0
    semantic_queue: list[dict[str, Any]] = []

    for index, (content_key, members) in enumerate(groups.items(), start=1):
        if content_key in content_results:
            continue

        representative = members[0]
        file_record = file_records.get(representative["file_id"])
        if file_record is None:
            raise RuntimeError(
                f"Missing inventory record: {representative['file_id']}"
            )
        path = Path(file_record["original_path"])
        extraction = extract_text(
            path,
            representative["family"],
            max_chars=max_chars,
        )
        extraction_counts[extraction.status] += 1
        metadata_text = (
            f"{representative['relative_path']} "
            f"{representative['filename']}"
        )
        privacy = scan_privacy(
            extraction.text,
            metadata_text=metadata_text,
            use_presidio=use_presidio,
        )
        flags = _privacy_flags(privacy)
        for flag in flags:
            privacy_flag_counts[flag.split(":", 1)[0]] += 1

        deterministic = _result_record(
            extraction,
            privacy,
            None,
            approve_threshold=approve_threshold,
            reject_threshold=reject_threshold,
        )
        needs_semantic = (
            client is not None
            and extraction.status == "ok"
            and not privacy.has_secret
            and not privacy.has_identity_document
            and not privacy.has_prompt_injection
            and not privacy.sensitive_topics
            and not presidio_blocking_entities(
                privacy,
                extraction.text,
            )
        )

        if needs_semantic:
            semantic_queue.append(
                {
                    "content_key": content_key,
                    "members": members,
                    "representative": representative,
                    "extraction": extraction,
                    "privacy": privacy,
                }
            )
        else:
            content_results[content_key] = deterministic
            _append_checkpoint(
                checkpoint_path,
                config_hash=config_hash,
                config=config,
                content_key=content_key,
                result=deterministic,
            )

        if index % 20 == 0:
            print(
                f"Prepared {index}/{len(groups)} unique contents "
                f"({len(content_results)} resolved without model)"
            )

    semantic_batches = _batched(semantic_queue, batch_size)
    semantic_failure_counts: Counter[str] = Counter()
    unresolved_semantic_errors: dict[str, str] = {}

    for batch_index, batch in enumerate(semantic_batches, start=1):
        semantic_results, batch_errors = (
            _review_with_fallback(
                client,
                batch,
                private_profile,
                single_item_retries=single_item_retries,
            )
            if client is not None
            else ({}, {item["content_key"]: "Ollama disabled" for item in batch})
        )

        for item in batch:
            content_key = item["content_key"]
            semantic = semantic_results.get(content_key)
            if semantic is not None:
                category_counts[semantic.document_category] += 1
                result = _result_record(
                    item["extraction"],
                    item["privacy"],
                    semantic,
                    approve_threshold=approve_threshold,
                    reject_threshold=reject_threshold,
                )
                content_results[content_key] = result
                _append_checkpoint(
                    checkpoint_path,
                    config_hash=config_hash,
                    config=config,
                    content_key=content_key,
                    result=result,
                )
            else:
                error = batch_errors.get(
                    content_key, "Ollama omitted this item"
                )
                unresolved_semantic_errors[content_key] = error
                semantic_failure_counts[error] += 1

        print(
            f"Completed root model batch {batch_index}/{len(semantic_batches)} "
            f"({min(batch_index * batch_size, len(semantic_queue))}/"
            f"{len(semantic_queue)} semantic items; "
            f"{len(unresolved_semantic_errors)} unresolved)"
        )

    model_error_count = len(unresolved_semantic_errors)
    # Unresolved model failures are emitted as pending for this run but are not
    # checkpointed, so the same command retries them next time.
    for item in semantic_queue:
        content_key = item["content_key"]
        if content_key in content_results:
            continue
        error = unresolved_semantic_errors.get(
            content_key, "Local semantic review was unavailable"
        )
        content_results[content_key] = _result_record(
            item["extraction"],
            item["privacy"],
            None,
            approve_threshold=approve_threshold,
            reject_threshold=reject_threshold,
            semantic_error=error,
        )

    if len(content_results) != len(groups):
        missing_keys = set(groups).difference(content_results)
        raise RuntimeError(
            f"Auto-review incomplete; missing {len(missing_keys)} items"
        )

    output_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    output_fields = list(fieldnames)
    for column in AUTO_COLUMNS:
        if column not in output_fields:
            output_fields.append(column)

    for row in rows:
        result = content_results[row["content_key"]]
        updated: dict[str, Any] = dict(row)
        updated["decision"] = result["decision"]
        updated["review_notes"] = (
            f"AUTO: {result['reason']}"
        )[:1500]
        updated["known_contradiction_group"] = (
            result["contradiction_topic"]
        )
        updated["contains_identity_document"] = (
            result["identity_flag"]
        )
        updated["contains_credentials_or_secrets"] = (
            result["credential_flag"]
        )
        updated.update(
            {
                "auto_review_run_id": run_id,
                "auto_decision": result["decision"],
                "auto_confidence": f"{result['confidence']:.4f}",
                "auto_category": result["category"],
                "auto_sensitivity": result["sensitivity"],
                "auto_reason": result["reason"],
                "auto_summary": result["summary"],
                "extraction_status": result["extraction_status"],
                "extraction_truncated": str(
                    result["extraction_truncated"]
                ).lower(),
                "privacy_flags": json.dumps(
                    result["privacy_flags"]
                ),
                "needs_human_review": str(
                    result["decision"] == "pending"
                ).lower(),
                "semantic_recommended_decision": result.get(
                    "semantic_recommended_decision", ""
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
        decision_counts[result["decision"]] += 1
        if result["decision"] == "pending":
            manual_rows.append(updated)

    run_review_path = exports / f"pilot-review-auto-{run_id}.csv"
    canonical_review_path = exports / f"pilot-review-{proposal_id}.csv"
    manual_path = exports / f"pilot-manual-review-{run_id}.csv"
    details_path = (
        exports / f"pilot-auto-review-details-{run_id}.json"
    )
    summary_path = (
        exports / f"pilot-auto-review-summary-{run_id}.json"
    )

    _write_csv(run_review_path, output_fields, output_rows)
    _write_csv(manual_path, output_fields, manual_rows)
    details_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "content_results": content_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    promoted, promotion_error = _promote_canonical(
        source=run_review_path,
        target=canonical_review_path,
        run_id=run_id,
    )

    ollama_metrics = (
        client.metrics()
        if client is not None and hasattr(client, "metrics")
        else {}
    )
    summary = {
        "auto_review_schema_version": AUTO_REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "proposal_id": proposal_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "model": model if use_ollama else None,
        "ollama_endpoint": base_url if use_ollama else None,
        "thinking_disabled": bool(use_ollama),
        "batch_size": batch_size,
        "max_chars_per_item": max_chars,
        "presidio_enabled": use_presidio,
        "selected_file_records": len(rows),
        "unique_contents_reviewed": len(groups),
        "resumed_unique_contents": resumed_content_count,
        "semantic_items": len(semantic_queue),
        "semantic_batches": len(semantic_batches),
        "semantic_failure_reason_counts": dict(semantic_failure_counts),
        "auto_approved": decision_counts["approve"],
        "auto_rejected": decision_counts["reject"],
        "manual_review_required": decision_counts["pending"],
        "model_error_count": model_error_count,
        "extraction_status_counts": dict(extraction_counts),
        "document_category_counts": dict(category_counts),
        "privacy_flag_class_counts": dict(privacy_flag_counts),
        "ollama_metrics": ollama_metrics,
        "checkpoint_path": str(checkpoint_path),
        "run_review_csv_path": str(run_review_path),
        "canonical_review_csv_path": str(canonical_review_path),
        "canonical_review_updated": promoted,
        "canonical_review_error": promotion_error,
        "manual_review_csv_path": str(manual_path),
        "private_details_path": str(details_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def promote_auto_review(
    *,
    vault_root: Path,
    review_csv: Path | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    exports = vault_root / "manifests" / "exports"
    proposal_id, _ = _latest_proposal(exports)

    if review_csv is None:
        candidates = sorted(
            exports.glob("pilot-review-auto-*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "No run-specific auto-review CSV was found"
            )
        review_csv = candidates[0]
    else:
        review_csv = review_csv.expanduser().resolve(strict=True)

    target = exports / f"pilot-review-{proposal_id}.csv"
    promoted, error = _promote_canonical(
        source=review_csv,
        target=target,
        run_id=uuid.uuid4().hex,
    )
    if not promoted:
        raise PermissionError(error)
    return {
        "proposal_id": proposal_id,
        "source_review_csv": str(review_csv),
        "canonical_review_csv": str(target),
        "promoted": True,
    }


def apply_manual_review(
    *,
    vault_root: Path,
    manual_csv: Path,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    exports = vault_root / "manifests" / "exports"
    manual_csv = manual_csv.expanduser().resolve(strict=True)
    manual_fields, manual_rows = _read_csv(manual_csv)
    if not manual_rows:
        return {"updated": 0, "remaining_pending": 0}

    run_ids = {
        row.get("auto_review_run_id", "") for row in manual_rows
    }
    if len(run_ids) != 1 or "" in run_ids:
        raise ValueError(
            "Manual CSV must contain one valid auto_review_run_id"
        )

    target_candidates = []
    manual_file_ids = {row["file_id"] for row in manual_rows}
    for path in exports.glob("pilot-review-*.csv"):
        if path.name.startswith("pilot-review-auto-"):
            continue
        fields, target_rows = _read_csv(path)
        if manual_file_ids.issubset(
            {row.get("file_id", "") for row in target_rows}
        ):
            target_candidates.append((path, fields, target_rows))

    if len(target_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify the canonical pilot review CSV"
        )

    target_path, target_fields, target_rows = target_candidates[0]
    manual_by_id = {row["file_id"]: row for row in manual_rows}
    updated_count = 0
    for row in target_rows:
        manual = manual_by_id.get(row["file_id"])
        if manual is None:
            continue
        decision = manual.get("decision", "").strip().lower()
        if decision not in {"approve", "reject", "pending"}:
            raise ValueError(
                f"Invalid decision for {row['file_id']}: {decision}"
            )
        for column in MANUAL_EDIT_COLUMNS:
            row[column] = manual.get(column, "")
        updated_count += 1

    temporary = target_path.with_name(
        f".{target_path.name}.{uuid.uuid4().hex}.tmp"
    )
    _write_csv(temporary, target_fields, target_rows)
    try:
        os.replace(temporary, target_path)
    except PermissionError as exc:
        temporary.unlink(missing_ok=True)
        raise PermissionError(
            "The canonical review CSV is open in another program. "
            "Close Excel and retry."
        ) from exc

    remaining_pending = sum(
        row.get("decision", "").strip().lower() == "pending"
        for row in target_rows
    )
    return {
        "updated": updated_count,
        "remaining_pending": remaining_pending,
        "review_csv_path": str(target_path),
    }
