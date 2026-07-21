from __future__ import annotations

import json
import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LABELS = ("relevant", "partially_relevant", "irrelevant")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class QueryClaimRelevanceCalibrationPolicy:
    policy_id: str
    default_sample_size: int
    selection_seed: str
    excluded_regression_query_ids: tuple[str, ...]
    minimum_labeled_items: int
    high_precision_target: float
    minimum_recall_for_promising_gate: float
    review_host: str
    review_port: int
    private_output_only: bool
    private_text_uploaded: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    source_path: Path


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "query_claim_relevance_calibration_policy.json"


def load_query_claim_relevance_calibration_policy(path: Path | None = None) -> QueryClaimRelevanceCalibrationPolicy:
    source = (path or default_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("query_claim_relevance_calibration_policy_schema_version", -1)) != 1:
        raise ValueError("Unsupported query-claim relevance calibration policy schema")
    policy = QueryClaimRelevanceCalibrationPolicy(
        policy_id=str(data["policy_id"]),
        default_sample_size=int(data["default_sample_size"]),
        selection_seed=str(data["selection_seed"]),
        excluded_regression_query_ids=tuple(str(x) for x in data.get("excluded_regression_query_ids", [])),
        minimum_labeled_items=int(data["minimum_labeled_items"]),
        high_precision_target=float(data["high_precision_target"]),
        minimum_recall_for_promising_gate=float(data["minimum_recall_for_promising_gate"]),
        review_host=str(data["review_host"]),
        review_port=int(data["review_port"]),
        private_output_only=bool(data["private_output_only"]),
        private_text_uploaded=bool(data["private_text_uploaded"]),
        memory_write_allowed=bool(data["memory_write_allowed"]),
        external_action_allowed=bool(data["external_action_allowed"]),
        tool_calling_allowed=bool(data["tool_calling_allowed"]),
        web_access_allowed=bool(data["web_access_allowed"]),
        source_path=source,
    )
    if policy.review_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Relevance review must bind to loopback only")
    if not policy.private_output_only or policy.private_text_uploaded:
        raise ValueError("Relevance calibration must remain private")
    if any((policy.memory_write_allowed, policy.external_action_allowed, policy.tool_calling_allowed, policy.web_access_allowed)):
        raise ValueError("Relevance calibration must remain read-only and offline")
    return policy


def _rank_stratified_sample(*, candidates: list[dict[str, Any]], sample_size: int, seed: str) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: float(item["relevance_score"]))
    bucket_count = min(4, len(ordered))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(bucket_count)]
    for index, item in enumerate(ordered):
        bucket_index = min(bucket_count - 1, int(index * bucket_count / len(ordered)))
        buckets[bucket_index].append(item)
    rng = random.Random(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    used_queries: Counter[str] = Counter()
    while len(selected) < sample_size and any(buckets):
        progress = False
        for bucket in buckets:
            if not bucket:
                continue
            best_index = min(range(len(bucket)), key=lambda i: (used_queries[str(bucket[i].get("query_id", ""))], i))
            item = bucket.pop(best_index)
            selected.append(item)
            used_queries[str(item.get("query_id", ""))] += 1
            progress = True
            if len(selected) >= sample_size:
                break
        if not progress:
            break
    return selected


def prepare_query_claim_relevance_calibration(*, vault_root: Path, benchmark_path: Path, audit_details_path: Path | None = None,
        pilot_name: str = "pilot-v1", sample_size: int | None = None, policy_path: Path | None = None,
        judge_policy_path: Path | None = None, device: str = "auto", candidate_builder=None) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = benchmark_path.expanduser().resolve(strict=True)
    policy = load_query_claim_relevance_calibration_policy(policy_path)
    from .judge_calibration import _score_candidates, latest_claim_support_audit_details, load_judge_calibration_policy
    from .response_reranker import load_local_response_reranker
    audit_details_path = latest_claim_support_audit_details(vault_root, pilot_name) if audit_details_path is None else audit_details_path.expanduser().resolve(strict=True)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit_details = json.loads(audit_details_path.read_text(encoding="utf-8"))
    judge_policy = load_judge_calibration_policy(judge_policy_path)
    builder = candidate_builder or _score_candidates
    candidates, source_metadata = builder(vault_root=vault_root, audit_details=audit_details, benchmark=benchmark, pilot_name=pilot_name, policy=judge_policy, device=device)
    excluded = set(policy.excluded_regression_query_ids)
    eligible = [item for item in candidates if str(item.get("query_id", "")) not in excluded and str(item.get("question", "")).strip() and str(item.get("claim_text", "")).strip()]
    reranker, reranker_policy = load_local_response_reranker(vault_root=vault_root, device=device)
    pairs = [[str(item["question"]), str(item["claim_text"])] for item in eligible]
    scores = reranker.predict(pairs, batch_size=reranker_policy.batch_size, show_progress_bar=False)
    scored = []
    for item, score in zip(eligible, scores):
        current = dict(item)
        current["relevance_score"] = float(score)
        current["relevance_human_label"] = ""
        current["relevance_human_labeled_at"] = ""
        scored.append(current)
    requested = sample_size if sample_size is not None else policy.default_sample_size
    selected = _rank_stratified_sample(candidates=scored, sample_size=min(requested, len(scored)), seed=policy.selection_seed)
    calibration_id = str(uuid.uuid4())
    private_root = vault_root / "manifests" / "calibration" / pilot_name
    export_root = vault_root / "manifests" / "exports"
    bundle_path = private_root / f"query-claim-relevance-calibration-{calibration_id}.json"
    summary_path = export_root / f"query-claim-relevance-calibration-summary-{calibration_id}.json"
    bundle = {
        "query_claim_relevance_calibration_bundle_schema_version": 1,
        "calibration_id": calibration_id,
        "created_at": _now(),
        "pilot_name": pilot_name,
        "policy_id": policy.policy_id,
        "benchmark_id": str(benchmark.get("benchmark_id", "")),
        "benchmark_path": str(benchmark_path),
        "source_audit_details_path": str(audit_details_path),
        "excluded_regression_query_ids": list(policy.excluded_regression_query_ids),
        "blind_review": True,
        "requested_sample_size": requested,
        "selected_sample_size": len(selected),
        "candidate_count_before_exclusion": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selection_seed": policy.selection_seed,
        "selection_method": "rank_stratified_cross_encoder_score_with_query_diversity",
        "reranker": {"model_id": reranker_policy.model_id, "revision": reranker_policy.revision, "policy_id": reranker_policy.policy_id, "batch_size": reranker_policy.batch_size},
        "source_candidate_metadata": source_metadata,
        "items": selected,
    }
    _atomic_json(bundle_path, bundle)
    summary = {
        "query_claim_relevance_calibration_summary_schema_version": 1,
        "calibration_id": calibration_id,
        "pilot_name": pilot_name,
        "selected_sample_size": len(selected),
        "candidate_count_before_exclusion": len(candidates),
        "eligible_candidate_count": len(eligible),
        "excluded_regression_query_ids": list(policy.excluded_regression_query_ids),
        "model_id": reranker_policy.model_id,
        "private_bundle_path": str(bundle_path),
        "private_text_uploaded": False,
    }
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def load_relevance_calibration_bundle(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))


def save_relevance_human_label(*, bundle_path: Path, item_id: str, label: str) -> dict[str, Any]:
    if label not in {*_LABELS, ""}:
        raise ValueError("Invalid relevance human label")
    bundle_path = bundle_path.expanduser().resolve(strict=True)
    bundle = load_relevance_calibration_bundle(bundle_path)
    found = False
    for item in bundle.get("items", []):
        if str(item.get("item_id", "")) == item_id:
            item["relevance_human_label"] = label
            item["relevance_human_labeled_at"] = _now() if label else ""
            found = True
            break
    if not found:
        raise KeyError("Relevance calibration item was not found")
    _atomic_json(bundle_path, bundle)
    labeled = sum(str(item.get("relevance_human_label", "")) in _LABELS for item in bundle.get("items", []))
    return {"item_id": item_id, "relevance_human_label": label, "labeled_count": labeled, "total_count": len(bundle.get("items", []))}


def _binary_metrics(expected: list[int], predicted: list[int]) -> dict[str, Any]:
    tp = sum(y == 1 and p == 1 for y, p in zip(expected, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(expected, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(expected, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(expected, predicted))
    accuracy = (tp + tn) / len(expected) if expected else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "accuracy": round(accuracy, 6), "relevance_precision": round(precision, 6), "relevance_recall": round(recall, 6), "relevance_f1": round(f1, 6)}


def _threshold_candidates(scores: list[float]) -> list[float]:
    unique = sorted(set(scores))
    if not unique:
        return []
    return [unique[0] - 1e-6] + [(a + b) / 2.0 for a, b in zip(unique, unique[1:])] + [unique[-1] + 1e-6]


def evaluate_query_claim_relevance_calibration(*, vault_root: Path, bundle_path: Path, policy_path: Path | None = None) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    bundle_path = bundle_path.expanduser().resolve(strict=True)
    policy = load_query_claim_relevance_calibration_policy(policy_path)
    bundle = load_relevance_calibration_bundle(bundle_path)
    if int(bundle.get("query_claim_relevance_calibration_bundle_schema_version", -1)) != 1:
        raise ValueError("Input is not a relevance calibration bundle")
    labeled = [item for item in bundle.get("items", []) if str(item.get("relevance_human_label", "")) in _LABELS]
    if len(labeled) < policy.minimum_labeled_items:
        raise ValueError(f"Not enough labeled relevance items: {len(labeled)} < {policy.minimum_labeled_items}")
    scores = [float(item["relevance_score"]) for item in labeled]
    expected = [1 if str(item["relevance_human_label"]) == "relevant" else 0 for item in labeled]
    sweeps = []
    for threshold in _threshold_candidates(scores):
        sweeps.append({"threshold": round(float(threshold), 6), **_binary_metrics(expected, [1 if score >= threshold else 0 for score in scores])})
    best_accuracy = max(sweeps, key=lambda row: (row["accuracy"], row["relevance_precision"], row["relevance_recall"], row["threshold"]))
    best_f1 = max(sweeps, key=lambda row: (row["relevance_f1"], row["relevance_precision"], row["relevance_recall"], row["threshold"]))
    high_precision = [row for row in sweeps if row["relevance_precision"] >= policy.high_precision_target and row["relevance_recall"] >= policy.minimum_recall_for_promising_gate]
    best_high_precision = max(high_precision, key=lambda row: (row["relevance_recall"], row["accuracy"], row["relevance_precision"], row["threshold"])) if high_precision else None
    recommendation = "cross_encoder_relevance_promising_requires_holdout_validation" if best_high_precision else "cross_encoder_relevance_not_ready_for_gate"
    run_id = str(uuid.uuid4())
    pilot_name = str(bundle.get("pilot_name", "pilot-v1"))
    private_root = vault_root / "manifests" / "calibration" / pilot_name
    export_root = vault_root / "manifests" / "exports"
    details_path = private_root / f"query-claim-relevance-calibration-details-{run_id}.json"
    summary_path = export_root / f"query-claim-relevance-calibration-evaluation-summary-{run_id}.json"
    details = {
        "query_claim_relevance_calibration_evaluation_schema_version": 1,
        "run_id": run_id,
        "calibration_id": str(bundle.get("calibration_id", "")),
        "pilot_name": pilot_name,
        "model_id": str(bundle.get("reranker", {}).get("model_id", "")),
        "labeled_item_count": len(labeled),
        "human_label_counts": dict(Counter(str(item["relevance_human_label"]) for item in labeled)),
        "human_binary_definition": "relevant=positive; partially_relevant and irrelevant=negative",
        "best_accuracy_threshold": best_accuracy,
        "best_f1_threshold": best_f1,
        "best_high_precision_threshold": best_high_precision,
        "diagnostic_recommendation": recommendation,
        "production_gate_changed": False,
        "holdout_validation_required_before_production": True,
        "private_text_uploaded": False,
        "items": [{"item_id": str(item.get("item_id", "")), "query_id": str(item.get("query_id", "")), "relevance_human_label": str(item.get("relevance_human_label", "")), "relevance_score": round(float(item.get("relevance_score", 0.0)), 6)} for item in labeled],
        "threshold_sweep": sweeps,
    }
    _atomic_json(details_path, details)
    summary = {key: value for key, value in details.items() if key not in {"items", "threshold_sweep"}}
    summary["private_details_path"] = str(details_path)
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
