from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "grounded_context_policy.json"


def load_policy(path: Path | None = None) -> dict[str, Any]:
    source = (path or default_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("grounded_context_policy_schema_version", -1)) != 1:
        raise ValueError("Unsupported grounded-context policy schema")
    if data.get("memory_write_allowed") is not False:
        raise ValueError("P1.10 may not write memories")
    if data.get("answer_generation_allowed") is not False:
        raise ValueError("P1.10 may not generate final answers")
    if data.get("external_action_allowed") is not False:
        raise ValueError("P1.10 may not perform external actions")
    if data.get("auto_resolve_contradictions") is not False:
        raise ValueError("P1.10 may not auto-resolve contradictions")
    if data.get("private_output_only") is not True:
        raise ValueError("Context packages must remain private")
    data["_digest"] = _sha(_canonical({k: v for k, v in data.items() if k != "_digest"}))
    data["_path"] = str(source)
    return data


def _default_search(**kwargs):
    from .semantic_retrieval import hybrid_search
    return hybrid_search(**kwargs)


def _compact(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit - 2].rstrip() + " …"


def _dedupe_provenance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        key = (
            str(item.get("file_id", "")),
            str(item.get("original_relative_path", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "file_id": str(item.get("file_id", "")),
            "original_relative_path": str(item.get("original_relative_path", "")),
            "filename": str(item.get("filename", "")),
            "role": str(item.get("role", "")),
            "source_bucket": str(item.get("source_bucket", "")),
            "year_hint": str(item.get("year_hint", "")),
            "duplicate_control_group": str(item.get("duplicate_control_group", "")),
            "known_contradiction_group": str(
                item.get("known_contradiction_group", "")
            ),
        })
    output.sort(key=lambda x: (x["original_relative_path"], x["file_id"]))
    return output


def _label(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.casefold() in {"", "none", "null", "[none]"} else text


def _fingerprint(package: dict[str, Any]) -> str:
    material = {
        k: v for k, v in package.items()
        if k not in {"package_id", "created_at", "package_fingerprint", "package_path"}
    }
    return _sha(_canonical(material))


def build_grounded_context(
    *,
    vault_root: Path,
    query: str,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    semantic_policy_path: Path | None = None,
    lexical_policy_path: Path | None = None,
    max_sources: int | None = None,
    device: str = "auto",
    save: bool = True,
    search_fn: Callable[..., dict[str, Any]] | None = None,
    model_loader=None,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query may not be empty")

    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    max_sources = max_sources or int(policy["default_max_sources"])
    if not 1 <= max_sources <= int(policy["maximum_max_sources"]):
        raise ValueError("max_sources outside policy range")

    search = search_fn or _default_search
    result = search(
        vault_root=vault_root,
        query=query,
        pilot_name=pilot_name,
        semantic_policy_path=semantic_policy_path,
        lexical_policy_path=lexical_policy_path,
        limit=max_sources,
        device=device,
        model_loader=model_loader,
    )

    by_source: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in result.get("results", []):
        source = str(item["source_content_sha256"])
        if source not in by_source:
            by_source[source] = dict(item)
            order.append(source)
        else:
            merged = by_source[source]
            merged["provenance"] = list(merged.get("provenance", [])) + list(
                item.get("provenance", [])
            )

    evidence = []
    contradictions: dict[str, list[str]] = defaultdict(list)
    agreement_counts: Counter[str] = Counter()

    for i, source in enumerate(order[:max_sources], start=1):
        item = by_source[source]
        provenance = _dedupe_provenance(list(item.get("provenance", [])))
        labels = sorted({
            value
            for value in (_label(p.get("known_contradiction_group")) for p in provenance)
            if value
        })
        citation_id = f"S{i}"
        for label in labels:
            contradictions[label].append(citation_id)

        lexical = item.get("lexical_rank") is not None
        semantic = item.get("semantic_rank") is not None
        agreement = (
            "lexical_and_semantic" if lexical and semantic
            else "lexical_only" if lexical
            else "semantic_only" if semantic
            else "unknown"
        )
        agreement_counts[agreement] += 1

        warnings = []
        if item.get("source_extraction_truncated"):
            warnings.append("source_extraction_truncated")
        if agreement != "lexical_and_semantic":
            warnings.append("single_channel_retrieval")
        if labels:
            warnings.append("known_contradiction_group_present")
        if not provenance:
            warnings.append("missing_provenance")

        evidence.append({
            "citation_id": citation_id,
            "citation": f"[{citation_id}]",
            "rank": i,
            "source_content_sha256": source,
            "chunk_id": str(item.get("chunk_id", "")),
            "chunk_index": item.get("chunk_index"),
            "family": str(item.get("family", "")),
            "retrieval_agreement": agreement,
            "rrf_score": item.get("rrf_score"),
            "lexical_rank": item.get("lexical_rank"),
            "semantic_rank": item.get("semantic_rank"),
            "source_extraction_truncated": bool(
                item.get("source_extraction_truncated", False)
            ),
            "context_text": _compact(
                item.get("snippet", ""),
                int(policy["snippet_characters_per_source"]),
            ),
            "provenance": provenance,
            "contradiction_labels": labels,
            "warnings": warnings,
        })

    package_id = str(uuid.uuid4())
    package = {
        "context_package_schema_version": SCHEMA_VERSION,
        "package_id": package_id,
        "pilot_name": pilot_name,
        "created_at": _now(),
        "query": query,
        "query_sha256": _sha(query.encode("utf-8")),
        "policy_id": policy["policy_id"],
        "policy_digest": policy["_digest"],
        "source_count": len(evidence),
        "evidence": evidence,
        "contradiction_groups": [
            {
                "label": label,
                "citations": citations,
                "unresolved": True,
                "resolution": None,
            }
            for label, citations in sorted(contradictions.items())
        ],
        "guardrails": {
            "memory_write_allowed": False,
            "answer_generation_allowed": False,
            "external_action_allowed": False,
            "contradictions_auto_resolved": False,
            "source_text_is_untrusted_data": True,
            "private_output_only": True,
        },
    }
    package["package_fingerprint"] = _fingerprint(package)

    summary = {
        "context_summary_schema_version": 1,
        "package_id": package_id,
        "pilot_name": pilot_name,
        "query_sha256": package["query_sha256"],
        "policy_id": policy["policy_id"],
        "policy_digest": policy["_digest"],
        "source_count": len(evidence),
        "retrieval_agreement_counts": dict(agreement_counts),
        "contradiction_group_count": len(package["contradiction_groups"]),
        "duplicate_provenance_source_count": sum(
            len(item["provenance"]) > 1 for item in evidence
        ),
        "truncated_source_count": sum(
            item["source_extraction_truncated"] for item in evidence
        ),
        "memory_write_allowed": False,
        "answer_generation_allowed": False,
        "external_action_allowed": False,
        "private_output_only": True,
        "package_fingerprint": package["package_fingerprint"],
    }

    if save:
        private_root = vault_root / "manifests" / "context" / pilot_name
        exports = vault_root / "manifests" / "exports"
        package_path = private_root / f"context-package-{package_id}.json"
        summary_path = exports / f"context-summary-{package_id}.json"
        package["package_path"] = str(package_path)
        _atomic_json(package_path, package)
        summary["package_path"] = str(package_path)
        _atomic_json(summary_path, summary)
        summary["summary_path"] = str(summary_path)

    return {"package": package, "summary": summary}


def verify_grounded_context_package(
    *,
    package_path: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    package_path = package_path.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    errors = []

    if package.get("policy_digest") != policy["_digest"]:
        errors.append("Grounding-policy digest mismatch")
    if package.get("package_fingerprint") != _fingerprint(package):
        errors.append("Package fingerprint mismatch")

    evidence = list(package.get("evidence", []))
    citations = [str(x.get("citation_id", "")) for x in evidence]
    if citations != [f"S{i}" for i in range(1, len(evidence) + 1)]:
        errors.append("Citation IDs are not contiguous")
    sources = [str(x.get("source_content_sha256", "")) for x in evidence]
    if len(sources) != len(set(sources)):
        errors.append("Duplicate source-content hashes")
    if any(not x.get("provenance") for x in evidence):
        errors.append("Evidence missing provenance")

    surfaced: dict[str, set[str]] = defaultdict(set)
    for item in evidence:
        for label in item.get("contradiction_labels", []):
            surfaced[str(label)].add(str(item["citation_id"]))

    top = {}
    for group in package.get("contradiction_groups", []):
        label = str(group.get("label", ""))
        top[label] = {str(x) for x in group.get("citations", [])}
        if group.get("unresolved") is not True or group.get("resolution") is not None:
            errors.append(f"Contradiction group {label!r} was resolved")
    if dict(surfaced) != top:
        errors.append("Contradiction labels are not fully surfaced")

    guardrails = dict(package.get("guardrails", {}))
    for key in (
        "memory_write_allowed",
        "answer_generation_allowed",
        "external_action_allowed",
        "contradictions_auto_resolved",
    ):
        if guardrails.get(key) is not False:
            errors.append(f"Guardrail {key} is not false")
    if guardrails.get("source_text_is_untrusted_data") is not True:
        errors.append("Source text is not marked untrusted")
    if guardrails.get("private_output_only") is not True:
        errors.append("Package is not private-output-only")

    return {
        "context_verification_schema_version": 1,
        "package_id": package.get("package_id"),
        "source_count": len(evidence),
        "citation_count": len(citations),
        "contradiction_group_count": len(top),
        "error_count": len(errors),
        "errors": errors,
        "memory_write_allowed": False,
        "answer_generation_allowed": False,
        "external_action_allowed": False,
        "ready_for_llm_context": not errors,
    }
