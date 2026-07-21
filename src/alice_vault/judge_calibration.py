from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class JudgeCalibrationPolicy:
    policy_id: str
    default_sample_size: int
    selection_seed: str
    fever_high_confidence_threshold: float
    minimum_labeled_items_for_recommendation: int
    minimum_accuracy_margin_for_preference: float
    minimum_support_precision_for_hard_gate: float
    maximum_evidence_windows_per_item: int
    review_host: str
    review_port: int
    blind_review: bool
    private_output_only: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    source_path: Path


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "judge_calibration_policy.json"


def load_judge_calibration_policy(path: Path | None = None) -> JudgeCalibrationPolicy:
    source = (path or default_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("judge_calibration_policy_schema_version", -1)) != 1:
        raise ValueError("Unsupported judge-calibration policy schema")

    policy = JudgeCalibrationPolicy(
        policy_id=str(data["policy_id"]),
        default_sample_size=int(data["default_sample_size"]),
        selection_seed=str(data["selection_seed"]),
        fever_high_confidence_threshold=float(data["fever_high_confidence_threshold"]),
        minimum_labeled_items_for_recommendation=int(
            data["minimum_labeled_items_for_recommendation"]
        ),
        minimum_accuracy_margin_for_preference=float(
            data["minimum_accuracy_margin_for_preference"]
        ),
        minimum_support_precision_for_hard_gate=float(
            data["minimum_support_precision_for_hard_gate"]
        ),
        maximum_evidence_windows_per_item=int(
            data["maximum_evidence_windows_per_item"]
        ),
        review_host=str(data["review_host"]),
        review_port=int(data["review_port"]),
        blind_review=bool(data["blind_review"]),
        private_output_only=bool(data["private_output_only"]),
        memory_write_allowed=bool(data["memory_write_allowed"]),
        external_action_allowed=bool(data["external_action_allowed"]),
        tool_calling_allowed=bool(data["tool_calling_allowed"]),
        web_access_allowed=bool(data["web_access_allowed"]),
        source_path=source,
    )
    if policy.default_sample_size < 1:
        raise ValueError("default_sample_size must be positive")
    if not 0 <= policy.fever_high_confidence_threshold <= 1:
        raise ValueError("Invalid FEVER high-confidence threshold")
    if policy.maximum_evidence_windows_per_item < 1:
        raise ValueError("maximum_evidence_windows_per_item must be positive")
    if policy.review_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Calibration review must bind to loopback only")
    if any(
        (
            policy.memory_write_allowed,
            policy.external_action_allowed,
            policy.tool_calling_allowed,
            policy.web_access_allowed,
        )
    ):
        raise ValueError("Judge calibration must remain read-only and offline")
    if not policy.private_output_only:
        raise ValueError("Judge calibration data must remain private")
    return policy


def latest_claim_support_audit_details(vault_root: Path, pilot_name: str) -> Path:
    directory = vault_root / "manifests" / "audits" / pilot_name
    candidates = sorted(
        directory.glob("claim-support-audit-details-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No claim-support audit details were found")
    return candidates[0]


def _benchmark_case_map(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("query_id", "")): case
        for case in benchmark.get("cases", [])
        if str(case.get("query_id", ""))
    }


def _cached_embedding_loader(
    *,
    vault_root: Path,
    semantic_policy_path: Path | None,
    device: str,
):
    from .semantic_retrieval import _load_local_model, load_semantic_policy

    semantic_policy = load_semantic_policy(semantic_policy_path)
    model, _ = _load_local_model(
        vault_root=vault_root,
        policy=semantic_policy,
        device=device,
    )

    def loader(*args, **kwargs):
        return model

    return loader


def _build_current_context(
    *,
    vault_root: Path,
    question: str,
    pilot_name: str,
    cached_embedding_loader,
    device: str,
) -> dict[str, Any]:
    from .grounded_context import build_grounded_context
    from .grounded_response import load_grounded_response_policy
    from .owner_attribution import annotate_context_owner_relation
    from .response_context_enrichment import enrich_response_context

    response_policy = load_grounded_response_policy()
    result = build_grounded_context(
        vault_root=vault_root,
        query=question,
        pilot_name=pilot_name,
        device=device,
        save=False,
        model_loader=cached_embedding_loader,
    )
    context = result["package"]
    expansion = response_policy.evidence_expansion
    if expansion.get("enabled"):
        context = enrich_response_context(
            vault_root=vault_root,
            context_package=context,
            passages_per_source=int(expansion["passages_per_source"]),
            maximum_characters_per_source=int(
                expansion["maximum_characters_per_source"]
            ),
            lexical_overlap_weight=float(expansion["lexical_overlap_weight"]),
            minimum_passage_characters=int(
                expansion["minimum_passage_characters"]
            ),
            device=device,
            model_loader=cached_embedding_loader,
        )
    return annotate_context_owner_relation(
        vault_root=vault_root,
        context_package=context,
        require_identity=True,
    )


def _normalize_qwen_verdict(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"supported", "partially_supported", "unsupported"}:
        return text
    return "unsupported"


def _fever_bucket(probability: float, high_threshold: float, keep_threshold: float) -> str:
    if probability >= high_threshold:
        return "high"
    if probability >= keep_threshold:
        return "borderline"
    return "low"


def _stratum(fever_bucket: str, qwen_verdict: str) -> str:
    qwen = "qwen_supported" if qwen_verdict == "supported" else "qwen_not_supported"
    return f"fever_{fever_bucket}__{qwen}"


def _source_metadata(evidence: dict[str, Any]) -> dict[str, Any]:
    provenance = list(evidence.get("provenance", []))
    first = provenance[0] if provenance else {}
    return {
        "source_content_sha256": str(evidence.get("source_content_sha256", "")),
        "filename": str(first.get("filename", "")),
        "original_relative_path": str(first.get("original_relative_path", "")),
        "family": str(evidence.get("family", "")),
        "owner_relation": str(evidence.get("owner_relation", "")),
        "owner_relation_confidence": str(
            evidence.get("owner_relation_confidence", "")
        ),
    }


def _review_evidence_windows(
    *,
    claim: dict[str, Any],
    context_package: dict[str, Any],
    maximum_windows: int,
    nli_policy,
) -> list[dict[str, Any]]:
    from .claim_entailment_gate import cited_passages_for_claim

    evidence_map = {
        str(item.get("citation")): item
        for item in context_package.get("evidence", [])
    }
    windows = cited_passages_for_claim(
        claim=claim,
        context_package=context_package,
        limit=nli_policy.maximum_evidence_passages_per_claim,
        sentence_window_size=nli_policy.sentence_window_size,
        sentence_window_stride=nli_policy.sentence_window_stride,
        maximum_window_characters=nli_policy.maximum_window_characters,
        maximum_windows=max(maximum_windows, 1),
    )
    result = []
    for window in windows[:maximum_windows]:
        citation = str(window["citation"])
        result.append(
            {
                "citation": citation,
                "text": str(window["premise"]),
                "lexical_score": round(float(window.get("lexical_score", 0.0)), 6),
                **_source_metadata(evidence_map.get(citation, {})),
            }
        )
    return result


def _candidate_id(query_id: str, claim_index: int, claim_text: str) -> str:
    material = f"{query_id}\0{claim_index}\0{claim_text}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _score_candidates(
    *,
    vault_root: Path,
    audit_details: dict[str, Any],
    benchmark: dict[str, Any],
    pilot_name: str,
    policy: JudgeCalibrationPolicy,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .claim_entailment_gate import (
        load_local_claim_entailment_model,
        score_claim_support,
    )

    benchmark_cases = _benchmark_case_map(benchmark)
    embedding_loader = _cached_embedding_loader(
        vault_root=vault_root,
        semantic_policy_path=None,
        device=device,
    )
    nli_model, nli_policy = load_local_claim_entailment_model(
        vault_root=vault_root,
        device=device,
    )

    candidates: list[dict[str, Any]] = []
    context_errors: list[dict[str, str]] = []

    for case in audit_details.get("cases", []):
        query_id = str(case.get("query_id", ""))
        benchmark_case = benchmark_cases.get(query_id)
        if benchmark_case is None:
            context_errors.append(
                {"query_id": query_id, "error": "Query ID not present in benchmark"}
            )
            continue
        question = str(benchmark_case.get("question", ""))

        try:
            context = _build_current_context(
                vault_root=vault_root,
                question=question,
                pilot_name=pilot_name,
                cached_embedding_loader=embedding_loader,
                device=device,
            )
        except Exception as exc:
            context_errors.append(
                {
                    "query_id": query_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        available_citations = {
            str(item.get("citation", ""))
            for item in context.get("evidence", [])
        }

        for index, claim in enumerate(case.get("claims", []), start=1):
            claim_text = str(claim.get("text", "")).strip()
            citations = [str(value) for value in claim.get("citations", [])]
            unresolved = [
                citation for citation in citations if citation not in available_citations
            ]
            if unresolved:
                context_errors.append(
                    {
                        "query_id": query_id,
                        "error": (
                            "Reconstructed context does not contain claim citations: "
                            + ", ".join(unresolved)
                        ),
                    }
                )
                continue

            normalized_claim = {
                "text": claim_text,
                "claim_type": str(claim.get("claim_type", "fact")),
                "citations": citations,
            }
            fever = score_claim_support(
                claim=normalized_claim,
                context_package=context,
                model=nli_model,
                policy=nli_policy,
            )
            qwen_verdict = _normalize_qwen_verdict(claim.get("verdict"))
            fever_probability = float(fever["best_entailment_probability"])
            bucket = _fever_bucket(
                fever_probability,
                policy.fever_high_confidence_threshold,
                nli_policy.entailment_threshold,
            )
            candidates.append(
                {
                    "item_id": _candidate_id(query_id, index, claim_text),
                    "query_id": query_id,
                    "question": question,
                    "claim_index": index,
                    "claim_text": claim_text,
                    "citations": citations,
                    "evidence_windows": _review_evidence_windows(
                        claim=normalized_claim,
                        context_package=context,
                        maximum_windows=policy.maximum_evidence_windows_per_item,
                        nli_policy=nli_policy,
                    ),
                    "qwen_auditor": {
                        "verdict": qwen_verdict,
                        "confidence": float(claim.get("confidence", 0.0) or 0.0),
                        "rationale": str(claim.get("rationale", "")),
                    },
                    "fever_nli": {
                        **fever,
                        "label_order": list(nli_policy.label_order),
                        "entailment_threshold": nli_policy.entailment_threshold,
                    },
                    "fever_confidence_bucket": bucket,
                    "stratum": _stratum(bucket, qwen_verdict),
                    "human_label": "",
                    "human_labeled_at": "",
                }
            )

    metadata = {
        "candidate_count": len(candidates),
        "context_error_count": len(context_errors),
        "context_errors": context_errors,
        "current_nli_model_id": nli_policy.model_id,
        "current_nli_policy_id": nli_policy.policy_id,
        "current_nli_entailment_threshold": nli_policy.entailment_threshold,
    }
    return candidates, metadata


_STRATUM_PRIORITY = [
    "fever_high__qwen_not_supported",
    "fever_borderline__qwen_not_supported",
    "fever_low__qwen_supported",
    "fever_high__qwen_supported",
    "fever_borderline__qwen_supported",
    "fever_low__qwen_not_supported",
]


def select_stratified_sample(
    *,
    candidates: list[dict[str, Any]],
    sample_size: int,
    seed: str,
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        buckets.setdefault(str(item["stratum"]), []).append(item)
    for values in buckets.values():
        rng.shuffle(values)

    ordered_strata = [s for s in _STRATUM_PRIORITY if s in buckets]
    ordered_strata.extend(sorted(s for s in buckets if s not in ordered_strata))

    selected: list[dict[str, Any]] = []
    used_queries: Counter[str] = Counter()

    while len(selected) < sample_size and any(buckets.get(s) for s in ordered_strata):
        made_progress = False
        for stratum in ordered_strata:
            values = buckets.get(stratum, [])
            if not values:
                continue
            best_index = min(
                range(len(values)),
                key=lambda i: (used_queries[str(values[i]["query_id"])], i),
            )
            item = values.pop(best_index)
            selected.append(item)
            used_queries[str(item["query_id"])] += 1
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break
    return selected


def prepare_judge_calibration(
    *,
    vault_root: Path,
    benchmark_path: Path,
    audit_details_path: Path | None = None,
    pilot_name: str = "pilot-v1",
    sample_size: int | None = None,
    policy_path: Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = benchmark_path.expanduser().resolve(strict=True)
    policy = load_judge_calibration_policy(policy_path)
    audit_details_path = (
        latest_claim_support_audit_details(vault_root, pilot_name)
        if audit_details_path is None
        else audit_details_path.expanduser().resolve(strict=True)
    )

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit_details = json.loads(audit_details_path.read_text(encoding="utf-8"))
    candidates, scoring_metadata = _score_candidates(
        vault_root=vault_root,
        audit_details=audit_details,
        benchmark=benchmark,
        pilot_name=pilot_name,
        policy=policy,
        device=device,
    )

    requested = sample_size if sample_size is not None else policy.default_sample_size
    selected = select_stratified_sample(
        candidates=candidates,
        sample_size=min(requested, len(candidates)),
        seed=policy.selection_seed,
    )

    calibration_id = str(uuid.uuid4())
    private_root = vault_root / "manifests" / "calibration" / pilot_name
    export_root = vault_root / "manifests" / "exports"
    bundle_path = private_root / f"judge-calibration-{calibration_id}.json"
    summary_path = export_root / f"judge-calibration-summary-{calibration_id}.json"

    stratum_counts = Counter(str(item["stratum"]) for item in selected)
    bundle = {
        "judge_calibration_bundle_schema_version": 1,
        "calibration_id": calibration_id,
        "created_at": _now(),
        "pilot_name": pilot_name,
        "policy_id": policy.policy_id,
        "benchmark_id": str(benchmark.get("benchmark_id", "")),
        "benchmark_path": str(benchmark_path),
        "source_audit_details_path": str(audit_details_path),
        "blind_review": policy.blind_review,
        "requested_sample_size": requested,
        "selected_sample_size": len(selected),
        "candidate_count": len(candidates),
        "selection_seed": policy.selection_seed,
        "stratum_counts": dict(stratum_counts),
        "scoring_metadata": scoring_metadata,
        "items": selected,
    }
    _atomic_json(bundle_path, bundle)

    summary = {
        "judge_calibration_summary_schema_version": 1,
        "calibration_id": calibration_id,
        "pilot_name": pilot_name,
        "selected_sample_size": len(selected),
        "candidate_count": len(candidates),
        "stratum_counts": dict(stratum_counts),
        "context_error_count": int(scoring_metadata["context_error_count"]),
        "current_nli_model_id": scoring_metadata["current_nli_model_id"],
        "current_nli_policy_id": scoring_metadata["current_nli_policy_id"],
        "blind_review": policy.blind_review,
        "private_bundle_path": str(bundle_path),
    }
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def load_calibration_bundle(path: Path) -> dict[str, Any]:
    return json.loads(
        path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )


def save_human_label(*, bundle_path: Path, item_id: str, label: str) -> dict[str, Any]:
    if label not in {"supported", "partially_supported", "unsupported", ""}:
        raise ValueError("Invalid human label")

    bundle_path = bundle_path.expanduser().resolve(strict=True)
    bundle = load_calibration_bundle(bundle_path)
    found = False
    for item in bundle.get("items", []):
        if str(item.get("item_id", "")) == item_id:
            item["human_label"] = label
            item["human_labeled_at"] = _now() if label else ""
            found = True
            break
    if not found:
        raise KeyError("Calibration item was not found")
    _atomic_json(bundle_path, bundle)
    labeled = sum(bool(str(item.get("human_label", ""))) for item in bundle.get("items", []))
    return {
        "item_id": item_id,
        "human_label": label,
        "labeled_count": labeled,
        "total_count": len(bundle.get("items", [])),
    }


_LABELS = ["supported", "partially_supported", "unsupported"]


def _accuracy(expected: list[str], predicted: list[str]) -> float:
    if not expected:
        return 0.0
    return sum(a == b for a, b in zip(expected, predicted)) / len(expected)


def _confusion_matrix(
    expected: list[str],
    predicted: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {
        actual: {pred: 0 for pred in _LABELS}
        for actual in _LABELS
    }
    for actual, pred in zip(expected, predicted):
        if actual in matrix and pred in matrix[actual]:
            matrix[actual][pred] += 1
    return matrix


def _support_metrics(expected: list[str], predicted: list[str]) -> dict[str, float]:
    tp = sum(a == "supported" and p == "supported" for a, p in zip(expected, predicted))
    fp = sum(a != "supported" and p == "supported" for a, p in zip(expected, predicted))
    fn = sum(a == "supported" and p != "supported" for a, p in zip(expected, predicted))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "support_precision": round(precision, 6),
        "support_recall": round(recall, 6),
        "support_f1": round(f1, 6),
    }


def _binary_full_support_accuracy(expected: list[str], predicted: list[str]) -> float:
    if not expected:
        return 0.0
    return sum(
        (a == "supported") == (p == "supported")
        for a, p in zip(expected, predicted)
    ) / len(expected)


def _cohen_kappa(expected: list[str], predicted: list[str]) -> float:
    if not expected:
        return 0.0
    observed = _accuracy(expected, predicted)
    total = len(expected)
    expected_counts = Counter(expected)
    predicted_counts = Counter(predicted)
    chance = sum(
        (expected_counts[label] / total) * (predicted_counts[label] / total)
        for label in _LABELS
    )
    if math.isclose(chance, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - chance) / (1.0 - chance)


def evaluate_judge_calibration(
    *,
    vault_root: Path,
    bundle_path: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    bundle_path = bundle_path.expanduser().resolve(strict=True)
    policy = load_judge_calibration_policy(policy_path)
    bundle = load_calibration_bundle(bundle_path)

    labeled = [
        item
        for item in bundle.get("items", [])
        if str(item.get("human_label", "")) in _LABELS
    ]
    expected = [str(item["human_label"]) for item in labeled]
    qwen = [
        _normalize_qwen_verdict(item["qwen_auditor"]["verdict"])
        for item in labeled
    ]
    fever = [
        "supported"
        if str(item["fever_nli"]["decision"]) == "keep_entailment"
        else "unsupported"
        for item in labeled
    ]

    qwen_exact = _accuracy(expected, qwen)
    fever_exact = _accuracy(expected, fever)
    qwen_binary = _binary_full_support_accuracy(expected, qwen)
    fever_binary = _binary_full_support_accuracy(expected, fever)
    qwen_support = _support_metrics(expected, qwen)
    fever_support = _support_metrics(expected, fever)

    recommendation = "insufficient_calibration"
    if len(labeled) >= policy.minimum_labeled_items_for_recommendation:
        margin = policy.minimum_accuracy_margin_for_preference
        if (
            fever_binary >= qwen_binary + margin
            and fever_support["support_precision"]
            >= policy.minimum_support_precision_for_hard_gate
        ):
            recommendation = "prefer_fever_nli_hard_gate"
        elif (
            qwen_binary >= fever_binary + margin
            and qwen_support["support_precision"]
            >= policy.minimum_support_precision_for_hard_gate
        ):
            recommendation = "prefer_qwen_auditor_hard_gate"
        else:
            recommendation = "use_disagreement_escalation"

    run_id = str(uuid.uuid4())
    pilot_name = str(bundle.get("pilot_name", "pilot-v1"))
    private_root = vault_root / "manifests" / "calibration" / pilot_name
    export_root = vault_root / "manifests" / "exports"
    details_path = private_root / f"judge-calibration-evaluation-details-{run_id}.json"
    summary_path = export_root / f"judge-calibration-evaluation-summary-{run_id}.json"

    details = {
        "judge_calibration_evaluation_schema_version": 1,
        "run_id": run_id,
        "calibration_id": str(bundle.get("calibration_id", "")),
        "labeled_item_count": len(labeled),
        "total_sample_item_count": len(bundle.get("items", [])),
        "human_label_counts": dict(Counter(expected)),
        "qwen_auditor": {
            "exact_3class_accuracy": round(qwen_exact, 6),
            "binary_full_support_accuracy": round(qwen_binary, 6),
            "cohen_kappa": round(_cohen_kappa(expected, qwen), 6),
            "confusion_matrix": _confusion_matrix(expected, qwen),
            **qwen_support,
        },
        "fever_nli": {
            "exact_3class_accuracy": round(fever_exact, 6),
            "binary_full_support_accuracy": round(fever_binary, 6),
            "cohen_kappa": round(_cohen_kappa(expected, fever), 6),
            "confusion_matrix": _confusion_matrix(expected, fever),
            **fever_support,
        },
        "judge_disagreement_count": sum(a != b for a, b in zip(qwen, fever)),
        "recommendation": recommendation,
        "items": [
            {
                "item_id": str(item["item_id"]),
                "query_id": str(item["query_id"]),
                "human_label": str(item["human_label"]),
                "qwen_verdict": _normalize_qwen_verdict(
                    item["qwen_auditor"]["verdict"]
                ),
                "fever_verdict": (
                    "supported"
                    if str(item["fever_nli"]["decision"]) == "keep_entailment"
                    else "unsupported"
                ),
                "fever_entailment_probability": float(
                    item["fever_nli"]["best_entailment_probability"]
                ),
                "stratum": str(item["stratum"]),
            }
            for item in labeled
        ],
    }
    _atomic_json(details_path, details)

    summary = {key: value for key, value in details.items() if key != "items"}
    summary["private_details_path"] = str(details_path)
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
