from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grounded_response import atomic_json


@dataclass(frozen=True)
class QueryClaimRelevanceHoldoutPolicy:
    policy_id: str
    frozen_threshold: float
    objective: str
    positive_labels: tuple[str, ...]
    negative_labels: tuple[str, ...]
    default_sample_size: int
    max_per_query: int
    minimum_labeled_items: int
    minimum_precision: float
    minimum_recall: float
    excluded_regression_query_ids: tuple[str, ...]
    private_output_only: bool
    private_text_uploaded: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    source_path: Path


def default_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "query_claim_relevance_holdout_policy.json"
    )


def load_query_claim_relevance_holdout_policy(
    path: Path | None = None,
) -> QueryClaimRelevanceHoldoutPolicy:
    source = (path or default_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("query_claim_relevance_holdout_policy_schema_version", -1)) != 1:
        raise ValueError("Unsupported query-claim relevance holdout policy schema")

    policy = QueryClaimRelevanceHoldoutPolicy(
        policy_id=str(data["policy_id"]),
        frozen_threshold=float(data["frozen_threshold"]),
        objective=str(data["objective"]),
        positive_labels=tuple(str(v) for v in data["positive_labels"]),
        negative_labels=tuple(str(v) for v in data["negative_labels"]),
        default_sample_size=int(data["default_sample_size"]),
        max_per_query=int(data["max_per_query"]),
        minimum_labeled_items=int(data["minimum_labeled_items"]),
        minimum_precision=float(data["minimum_precision"]),
        minimum_recall=float(data["minimum_recall"]),
        excluded_regression_query_ids=tuple(
            str(v) for v in data["excluded_regression_query_ids"]
        ),
        private_output_only=bool(data["private_output_only"]),
        private_text_uploaded=bool(data["private_text_uploaded"]),
        memory_write_allowed=bool(data["memory_write_allowed"]),
        external_action_allowed=bool(data["external_action_allowed"]),
        tool_calling_allowed=bool(data["tool_calling_allowed"]),
        web_access_allowed=bool(data["web_access_allowed"]),
        source_path=source,
    )

    if policy.objective != "non_irrelevance_filter":
        raise ValueError("Only the calibrated non-irrelevance objective is supported")
    if policy.default_sample_size < 1 or policy.max_per_query < 1:
        raise ValueError("Sample size and query cap must be positive")
    if policy.minimum_labeled_items < 1:
        raise ValueError("minimum_labeled_items must be positive")
    if not 0.0 <= policy.minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be between 0 and 1")
    if not 0.0 <= policy.minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")
    if not policy.private_output_only or policy.private_text_uploaded:
        raise ValueError("Relevance holdout must remain private")
    if any(
        (
            policy.memory_write_allowed,
            policy.external_action_allowed,
            policy.tool_calling_allowed,
            policy.web_access_allowed,
        )
    ):
        raise ValueError("Relevance holdout must remain read-only and offline")
    return policy


def stable_item_id(query_id: str, claim_text: str) -> str:
    return hashlib.sha256(
        f"{query_id}\0{claim_text}".encode("utf-8")
    ).hexdigest()[:20]


def calibration_item_ids(bundle: dict[str, Any]) -> set[str]:
    return {
        str(item.get("item_id", ""))
        for item in bundle.get("items", [])
        if str(item.get("item_id", ""))
    }


def select_with_query_cap(
    candidates: list[dict[str, Any]],
    *,
    sample_size: int,
    max_per_query: int,
) -> list[dict[str, Any]]:
    # Prefer broad query coverage first, then higher score within each query.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault(str(item.get("query_id", "")), []).append(item)
    for items in grouped.values():
        items.sort(key=lambda x: float(x["relevance_score"]), reverse=True)

    query_ids = sorted(
        grouped,
        key=lambda q: (
            -len(grouped[q]),
            q,
        ),
    )
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    while len(selected) < sample_size:
        progress = False
        for query_id in query_ids:
            if counts[query_id] >= max_per_query or not grouped[query_id]:
                continue
            selected.append(grouped[query_id].pop(0))
            counts[query_id] += 1
            progress = True
            if len(selected) >= sample_size:
                break
        if not progress:
            break
    return selected


def binary_metrics(expected: list[int], predicted: list[int]) -> dict[str, Any]:
    tp = sum(y == 1 and p == 1 for y, p in zip(expected, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(expected, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(expected, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(expected, predicted))
    accuracy = (tp + tn) / len(expected) if expected else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def evaluate_frozen_holdout(
    *,
    vault_root: Path,
    holdout_path: Path,
    policy_path: Path | None = None,
    pilot_name: str = "pilot-v1",
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    holdout_path = holdout_path.expanduser().resolve(strict=True)
    policy = load_query_claim_relevance_holdout_policy(policy_path)
    bundle = json.loads(holdout_path.read_text(encoding="utf-8"))

    if int(bundle.get("query_claim_relevance_holdout_bundle_schema_version", -1)) != 1:
        raise ValueError("Input is not a relevance holdout bundle")
    if not bool(bundle.get("threshold_frozen_before_human_review", False)):
        raise ValueError("Threshold was not frozen before human review")
    if abs(
        float(bundle.get("frozen_threshold", 999.0))
        - policy.frozen_threshold
    ) > 1e-9:
        raise ValueError("Holdout threshold differs from repository policy")

    valid_labels = set(policy.positive_labels) | set(policy.negative_labels)
    items = [
        item
        for item in bundle.get("items", [])
        if str(item.get("relevance_human_label", "")) in valid_labels
    ]
    if len(items) < policy.minimum_labeled_items:
        raise ValueError(
            f"Not enough labeled holdout items: {len(items)} < "
            f"{policy.minimum_labeled_items}"
        )

    expected = [
        1 if str(item["relevance_human_label"]) in policy.positive_labels else 0
        for item in items
    ]
    predicted = [
        1 if float(item["relevance_score"]) >= policy.frozen_threshold else 0
        for item in items
    ]
    metrics = binary_metrics(expected, predicted)
    passes = (
        metrics["precision"] >= policy.minimum_precision
        and metrics["recall"] >= policy.minimum_recall
    )

    if passes:
        recommendation = (
            "non_irrelevance_filter_holdout_passed_candidate_for_gate_integration_review"
        )
    else:
        recommendation = (
            "non_irrelevance_filter_holdout_failed_keep_diagnostic_only"
        )

    run_id = str(uuid.uuid4())
    private_root = vault_root / "manifests" / "calibration" / pilot_name
    export_root = vault_root / "manifests" / "exports"
    details_path = (
        private_root / f"query-claim-relevance-holdout-details-{run_id}.json"
    )
    summary_path = (
        export_root / f"query-claim-relevance-holdout-summary-{run_id}.json"
    )

    details = {
        "query_claim_relevance_holdout_evaluation_schema_version": 1,
        "run_id": run_id,
        "holdout_id": str(bundle.get("holdout_id", "")),
        "calibration_id": str(bundle.get("source_calibration_id", "")),
        "pilot_name": pilot_name,
        "objective": policy.objective,
        "frozen_threshold": policy.frozen_threshold,
        "threshold_sweep_performed_on_holdout": False,
        "threshold_frozen_before_human_review": True,
        "labeled_item_count": len(items),
        "human_label_counts": dict(
            Counter(str(item["relevance_human_label"]) for item in items)
        ),
        "binary_definition": (
            "relevant and partially_relevant=positive; irrelevant=negative"
        ),
        "metrics": metrics,
        "passes_minimum_precision": (
            metrics["precision"] >= policy.minimum_precision
        ),
        "passes_minimum_recall": (
            metrics["recall"] >= policy.minimum_recall
        ),
        "passes_holdout_gate": passes,
        "diagnostic_recommendation": recommendation,
        "production_gate_changed": False,
        "human_decision_required_before_production_change": True,
        "private_text_uploaded": False,
        "items": [
            {
                "item_id": str(item.get("item_id", "")),
                "query_id": str(item.get("query_id", "")),
                "relevance_human_label": str(
                    item.get("relevance_human_label", "")
                ),
                "relevance_score": round(
                    float(item.get("relevance_score", 0.0)), 6
                ),
                "predicted_non_irrelevant": bool(
                    float(item.get("relevance_score", 0.0))
                    >= policy.frozen_threshold
                ),
            }
            for item in items
        ],
    }
    atomic_json(details_path, details)
    summary = {k: v for k, v in details.items() if k != "items"}
    summary["private_details_path"] = str(details_path)
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
