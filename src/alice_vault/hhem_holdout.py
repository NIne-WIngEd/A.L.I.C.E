from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from alice_vault.hhem_calibration import (
    _binary_labels,
    _binary_metrics,
    _existing_judge_metrics,
    build_hhem_premise,
    load_local_hhem_model,
    score_hhem_pairs,
)
from alice_vault.judge_calibration import (
    _score_candidates,
    latest_claim_support_audit_details,
    load_judge_calibration_policy,
    select_stratified_sample,
)


@dataclass(frozen=True)
class HHEMHoldoutPolicy:
    policy_id: str
    frozen_threshold: float
    default_sample_size: int
    selection_seed: str
    minimum_labeled_items: int
    minimum_accuracy: float
    minimum_support_precision: float
    minimum_support_recall: float
    private_text_uploaded: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    source_path: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def default_holdout_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "hhem_holdout_policy.json"
    )


def load_hhem_holdout_policy(
    path: Path | None = None,
) -> HHEMHoldoutPolicy:
    source = (path or default_holdout_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))

    if int(data.get("hhem_holdout_policy_schema_version", -1)) != 1:
        raise ValueError("Unsupported HHEM holdout policy schema")

    policy = HHEMHoldoutPolicy(
        policy_id=str(data["policy_id"]),
        frozen_threshold=float(data["frozen_threshold"]),
        default_sample_size=int(data["default_sample_size"]),
        selection_seed=str(data["selection_seed"]),
        minimum_labeled_items=int(data["minimum_labeled_items"]),
        minimum_accuracy=float(data["minimum_accuracy"]),
        minimum_support_precision=float(data["minimum_support_precision"]),
        minimum_support_recall=float(data["minimum_support_recall"]),
        private_text_uploaded=bool(data["private_text_uploaded"]),
        memory_write_allowed=bool(data["memory_write_allowed"]),
        external_action_allowed=bool(data["external_action_allowed"]),
        tool_calling_allowed=bool(data["tool_calling_allowed"]),
        web_access_allowed=bool(data["web_access_allowed"]),
        source_path=source,
    )

    if not 0.0 <= policy.frozen_threshold <= 1.0:
        raise ValueError("frozen_threshold must be between 0 and 1")
    if policy.default_sample_size < 1:
        raise ValueError("default_sample_size must be positive")
    if policy.minimum_labeled_items < 1:
        raise ValueError("minimum_labeled_items must be positive")
    for value in (
        policy.minimum_accuracy,
        policy.minimum_support_precision,
        policy.minimum_support_recall,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("Holdout metric thresholds must be between 0 and 1")
    if policy.private_text_uploaded:
        raise ValueError("Private text may not be uploaded")
    if any(
        (
            policy.memory_write_allowed,
            policy.external_action_allowed,
            policy.tool_calling_allowed,
            policy.web_access_allowed,
        )
    ):
        raise ValueError("HHEM holdout validation must remain read-only and offline")

    return policy


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))


def _calibration_item_ids(bundle: dict[str, Any]) -> set[str]:
    return {
        str(item.get("item_id", ""))
        for item in bundle.get("items", [])
        if str(item.get("item_id", ""))
    }


def prepare_hhem_holdout(
    *,
    vault_root: Path,
    benchmark_path: Path,
    calibration_bundle_path: Path,
    audit_details_path: Path | None = None,
    pilot_name: str = "pilot-v1",
    sample_size: int | None = None,
    holdout_policy_path: Path | None = None,
    judge_policy_path: Path | None = None,
    device: str = "auto",
    candidate_builder: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Create a blind claim-level holdout that excludes calibration item IDs.

    The output deliberately retains judge_calibration_bundle_schema_version=1 so
    the existing run_judge_calibration_review.py UI can label the holdout when
    invoked with --calibration <holdout-path>.
    """

    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = benchmark_path.expanduser().resolve(strict=True)
    calibration_bundle_path = calibration_bundle_path.expanduser().resolve(strict=True)

    holdout_policy = load_hhem_holdout_policy(holdout_policy_path)
    judge_policy = load_judge_calibration_policy(judge_policy_path)

    audit_details_path = (
        latest_claim_support_audit_details(vault_root, pilot_name)
        if audit_details_path is None
        else audit_details_path.expanduser().resolve(strict=True)
    )

    benchmark = _load_json(benchmark_path)
    audit_details = _load_json(audit_details_path)
    calibration_bundle = _load_json(calibration_bundle_path)
    excluded_ids = _calibration_item_ids(calibration_bundle)

    builder = candidate_builder or _score_candidates
    candidates, scoring_metadata = builder(
        vault_root=vault_root,
        audit_details=audit_details,
        benchmark=benchmark,
        pilot_name=pilot_name,
        policy=judge_policy,
        device=device,
    )

    eligible = [
        item
        for item in candidates
        if str(item.get("item_id", "")) not in excluded_ids
    ]

    requested = sample_size if sample_size is not None else holdout_policy.default_sample_size
    if requested < 1:
        raise ValueError("sample_size must be positive")
    if not eligible:
        raise ValueError(
            "No holdout candidates remain after excluding calibration item IDs"
        )

    selected = select_stratified_sample(
        candidates=eligible,
        sample_size=min(requested, len(eligible)),
        seed=holdout_policy.selection_seed,
    )

    overlap = {
        str(item.get("item_id", ""))
        for item in selected
        if str(item.get("item_id", "")) in excluded_ids
    }
    if overlap:
        raise RuntimeError(
            "Holdout selection overlaps calibration items: "
            + ", ".join(sorted(overlap))
        )

    holdout_id = str(uuid.uuid4())
    private_root = vault_root / "manifests" / "calibration" / pilot_name
    export_root = vault_root / "manifests" / "exports"
    bundle_path = private_root / f"judge-holdout-{holdout_id}.json"
    summary_path = export_root / f"judge-holdout-summary-{holdout_id}.json"

    stratum_counts = Counter(str(item.get("stratum", "")) for item in selected)
    calibration_query_ids = {
        str(item.get("query_id", ""))
        for item in calibration_bundle.get("items", [])
        if str(item.get("query_id", ""))
    }
    holdout_query_ids = {
        str(item.get("query_id", ""))
        for item in selected
        if str(item.get("query_id", ""))
    }

    bundle = {
        # Keep schema compatibility with the existing blind review UI.
        "judge_calibration_bundle_schema_version": 1,
        "holdout_schema_version": 1,
        "holdout_id": holdout_id,
        "holdout_kind": "hhem_frozen_threshold_claim_level",
        "created_at": _now(),
        "pilot_name": pilot_name,
        "policy_id": holdout_policy.policy_id,
        "benchmark_id": str(benchmark.get("benchmark_id", "")),
        "benchmark_path": str(benchmark_path),
        "source_audit_details_path": str(audit_details_path),
        "source_calibration_bundle_path": str(calibration_bundle_path),
        "source_calibration_id": str(calibration_bundle.get("calibration_id", "")),
        "blind_review": True,
        "threshold_frozen_before_human_holdout_review": True,
        "frozen_hhem_threshold": holdout_policy.frozen_threshold,
        "requested_sample_size": requested,
        "selected_sample_size": len(selected),
        "candidate_count_before_exclusion": len(candidates),
        "excluded_calibration_item_count": len(excluded_ids),
        "eligible_candidate_count": len(eligible),
        "selection_seed": holdout_policy.selection_seed,
        "stratum_counts": dict(stratum_counts),
        "query_overlap_with_calibration_count": len(
            calibration_query_ids.intersection(holdout_query_ids)
        ),
        "independence_scope": (
            "claim-level: calibration item_ids are excluded; benchmark query_ids "
            "may overlap because the pilot benchmark is small"
        ),
        "scoring_metadata": scoring_metadata,
        "items": selected,
    }
    _atomic_json(bundle_path, bundle)

    summary = {
        "hhem_holdout_prepare_schema_version": 1,
        "holdout_id": holdout_id,
        "pilot_name": pilot_name,
        "selected_sample_size": len(selected),
        "candidate_count_before_exclusion": len(candidates),
        "excluded_calibration_item_count": len(excluded_ids),
        "eligible_candidate_count": len(eligible),
        "calibration_overlap_item_count": 0,
        "query_overlap_with_calibration_count": bundle[
            "query_overlap_with_calibration_count"
        ],
        "frozen_hhem_threshold": holdout_policy.frozen_threshold,
        "threshold_frozen_before_human_holdout_review": True,
        "private_bundle_path": str(bundle_path),
        "private_text_uploaded": False,
    }
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def evaluate_hhem_holdout(
    *,
    vault_root: Path,
    holdout_bundle_path: Path,
    pilot_name: str = "pilot-v1",
    holdout_policy_path: Path | None = None,
    hhem_policy_path: Path | None = None,
    device: str | None = None,
    model_loader=None,
) -> dict[str, Any]:
    """Evaluate a human-labeled holdout at the pre-frozen HHEM threshold.

    This function never sweeps or optimizes the threshold on holdout labels.
    """

    vault_root = vault_root.expanduser().resolve(strict=True)
    holdout_bundle_path = holdout_bundle_path.expanduser().resolve(strict=True)
    holdout_policy = load_hhem_holdout_policy(holdout_policy_path)
    bundle = _load_json(holdout_bundle_path)

    if int(bundle.get("holdout_schema_version", -1)) != 1:
        raise ValueError("Input is not an HHEM holdout bundle")
    if not bool(bundle.get("threshold_frozen_before_human_holdout_review", False)):
        raise ValueError("Holdout threshold was not recorded as frozen before review")

    bundle_threshold = float(bundle.get("frozen_hhem_threshold", -1.0))
    if abs(bundle_threshold - holdout_policy.frozen_threshold) > 1e-12:
        raise ValueError(
            "Holdout bundle threshold does not match the repository holdout policy"
        )

    labeled_items = [
        item
        for item in bundle.get("items", [])
        if str(item.get("human_label", ""))
        in {"supported", "partially_supported", "unsupported"}
    ]
    if len(labeled_items) < holdout_policy.minimum_labeled_items:
        raise ValueError(
            "Not enough human-labeled holdout items: "
            f"{len(labeled_items)} < {holdout_policy.minimum_labeled_items}"
        )

    pairs: list[tuple[str, str]] = []
    scored_items: list[dict[str, Any]] = []
    for item in labeled_items:
        premise = build_hhem_premise(item)
        hypothesis = str(item.get("claim_text", "")).strip()
        if not premise or not hypothesis:
            continue
        pairs.append((premise, hypothesis))
        scored_items.append(item)

    if len(scored_items) < holdout_policy.minimum_labeled_items:
        raise ValueError("Too few holdout items have usable premise-hypothesis pairs")

    if model_loader is None:
        model, hhem_policy = load_local_hhem_model(
            vault_root=vault_root,
            policy_path=hhem_policy_path,
            device=device,
        )
    else:
        loaded = model_loader(
            vault_root=vault_root,
            policy_path=hhem_policy_path,
            device=device,
        )
        if isinstance(loaded, tuple):
            model, hhem_policy = loaded
        else:
            model = loaded
            hhem_policy = None

    batch_size = int(getattr(hhem_policy, "batch_size", 4))
    scores = score_hhem_pairs(
        model=model,
        pairs=pairs,
        batch_size=batch_size,
    )

    human_labels = [str(item["human_label"]) for item in scored_items]
    expected = _binary_labels(human_labels)
    threshold = holdout_policy.frozen_threshold
    predicted = [1 if score >= threshold else 0 for score in scores]
    hhem_metrics = _binary_metrics(expected=expected, predicted=predicted)
    existing = _existing_judge_metrics(scored_items)

    passes_accuracy = (
        float(hhem_metrics["accuracy"]) >= holdout_policy.minimum_accuracy
    )
    passes_precision = (
        float(hhem_metrics["support_precision"])
        >= holdout_policy.minimum_support_precision
    )
    passes_recall = (
        float(hhem_metrics["support_recall"])
        >= holdout_policy.minimum_support_recall
    )
    passes_holdout = passes_accuracy and passes_precision and passes_recall

    best_existing_accuracy = max(
        float(existing["qwen_auditor"]["accuracy"]),
        float(existing["fever_nli"]["accuracy"]),
    )
    beats_existing_accuracy = (
        float(hhem_metrics["accuracy"]) > best_existing_accuracy
    )

    if passes_holdout and beats_existing_accuracy:
        recommendation = (
            "hhem_holdout_passed_candidate_for_production_gate_review"
        )
    elif passes_holdout:
        recommendation = (
            "hhem_holdout_passed_but_does_not_beat_existing_accuracy"
        )
    else:
        recommendation = "hhem_holdout_failed_keep_diagnostic_only"

    run_id = str(uuid.uuid4())
    private_root = vault_root / "manifests" / "calibration" / pilot_name
    export_root = vault_root / "manifests" / "exports"
    details_path = private_root / f"hhem-holdout-details-{run_id}.json"
    summary_path = export_root / f"hhem-holdout-summary-{run_id}.json"

    details = {
        "hhem_holdout_evaluation_schema_version": 1,
        "run_id": run_id,
        "holdout_id": str(bundle.get("holdout_id", "")),
        "pilot_name": pilot_name,
        "holdout_bundle_path": str(holdout_bundle_path),
        "source_calibration_bundle_path": str(
            bundle.get("source_calibration_bundle_path", "")
        ),
        "labeled_item_count": len(scored_items),
        "human_label_counts": dict(Counter(human_labels)),
        "human_binary_definition": (
            "supported=positive; partially_supported and unsupported=negative"
        ),
        "threshold_selection_rule": (
            "threshold frozen on calibration set before holdout labels were reviewed"
        ),
        "threshold_frozen": True,
        "frozen_hhem_threshold": threshold,
        "threshold_sweep_performed_on_holdout": False,
        "hhem": {
            "true_positive": hhem_metrics["true_positive"],
            "true_negative": hhem_metrics["true_negative"],
            "false_positive": hhem_metrics["false_positive"],
            "false_negative": hhem_metrics["false_negative"],
            "accuracy": hhem_metrics["accuracy"],
            "support_precision": hhem_metrics["support_precision"],
            "support_recall": hhem_metrics["support_recall"],
            "support_f1": hhem_metrics["support_f1"],
            "passes_minimum_accuracy": passes_accuracy,
            "passes_minimum_support_precision": passes_precision,
            "passes_minimum_support_recall": passes_recall,
            "passes_holdout_gate": passes_holdout,
        },
        "existing_judges": existing,
        "beats_best_existing_judge_accuracy": beats_existing_accuracy,
        "recommendation": recommendation,
        "production_gate_changed": False,
        "human_decision_required_before_production_change": True,
        "private_text_uploaded": False,
        "items": [
            {
                "item_id": str(item.get("item_id", "")),
                "query_id": str(item.get("query_id", "")),
                "human_label": str(item.get("human_label", "")),
                "hhem_score": round(float(score), 6),
                "hhem_predicted_supported": bool(score >= threshold),
                "qwen_verdict": str(
                    item.get("qwen_auditor", {}).get("verdict", "")
                ),
                "fever_decision": str(
                    item.get("fever_nli", {}).get("decision", "")
                ),
            }
            for item, score in zip(scored_items, scores)
        ],
    }
    _atomic_json(details_path, details)

    summary = {key: value for key, value in details.items() if key != "items"}
    summary["private_details_path"] = str(details_path)
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
