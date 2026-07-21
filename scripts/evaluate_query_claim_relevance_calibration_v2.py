from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import atomic_json


LABELS = {"relevant", "partially_relevant", "irrelevant"}


def metrics(expected, predicted):
    tp = sum(y == 1 and p == 1 for y, p in zip(expected, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(expected, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(expected, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(expected, predicted))
    accuracy = (tp + tn) / len(expected) if expected else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
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


def thresholds(scores):
    unique = sorted(set(scores))
    out = [unique[0] - 1e-6]
    for left, right in zip(unique, unique[1:]):
        out.append((left + right) / 2.0)
    out.append(unique[-1] + 1e-6)
    return out


def evaluate_objective(items, positive_labels):
    scores = [float(item["relevance_score"]) for item in items]
    expected = [
        1 if item["relevance_human_label"] in positive_labels else 0
        for item in items
    ]
    sweep = []
    for threshold in thresholds(scores):
        predicted = [1 if score >= threshold else 0 for score in scores]
        sweep.append({
            "threshold": round(float(threshold), 6),
            **metrics(expected, predicted),
        })

    best_accuracy = max(
        sweep,
        key=lambda row: (
            row["accuracy"],
            row["precision"],
            row["recall"],
            row["threshold"],
        ),
    )
    best_f1 = max(
        sweep,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            row["threshold"],
        ),
    )
    high_precision = [
        row for row in sweep
        if row["precision"] >= 0.9 and row["recall"] >= 0.5
    ]
    best_high_precision = (
        max(
            high_precision,
            key=lambda row: (
                row["recall"],
                row["accuracy"],
                row["precision"],
                row["threshold"],
            ),
        )
        if high_precision else None
    )
    return {
        "positive_labels": sorted(positive_labels),
        "positive_count": sum(expected),
        "negative_count": len(expected) - sum(expected),
        "best_accuracy_threshold": best_accuracy,
        "best_f1_threshold": best_f1,
        "best_high_precision_threshold": best_high_precision,
        "threshold_sweep": sweep,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve(strict=True)
    path = args.calibration.expanduser().resolve(strict=True)
    bundle = json.loads(path.read_text(encoding="utf-8"))

    items = [
        item for item in bundle.get("items", [])
        if str(item.get("relevance_human_label", "")) in LABELS
    ]
    if len(items) < 8:
        raise ValueError("At least 8 human-labeled relevance items are required")

    strict = evaluate_objective(items, {"relevant"})
    non_irrelevant = evaluate_objective(
        items,
        {"relevant", "partially_relevant"},
    )

    if non_irrelevant["best_high_precision_threshold"] is not None:
        recommendation = (
            "cross_encoder_promising_as_irrelevance_rejection_filter_"
            "requires_fresh_holdout"
        )
    elif strict["best_high_precision_threshold"] is not None:
        recommendation = (
            "cross_encoder_promising_for_strict_relevance_"
            "requires_fresh_holdout"
        )
    else:
        recommendation = "cross_encoder_relevance_not_ready_for_gate"

    run_id = str(uuid.uuid4())
    private_root = vault / "manifests" / "calibration" / args.pilot_name
    export_root = vault / "manifests" / "exports"
    details_path = private_root / f"query-claim-relevance-calibration-v2-details-{run_id}.json"
    summary_path = export_root / f"query-claim-relevance-calibration-v2-evaluation-summary-{run_id}.json"

    details = {
        "query_claim_relevance_calibration_v2_evaluation_schema_version": 1,
        "run_id": run_id,
        "calibration_id": str(bundle.get("calibration_id", "")),
        "labeled_item_count": len(items),
        "human_label_counts": dict(
            Counter(item["relevance_human_label"] for item in items)
        ),
        "selected_query_counts": bundle.get("selected_query_counts", {}),
        "strict_relevance_objective": strict,
        "non_irrelevance_objective": non_irrelevant,
        "diagnostic_recommendation": recommendation,
        "production_gate_changed": False,
        "holdout_validation_required_before_production": True,
        "private_text_uploaded": False,
    }
    atomic_json(details_path, details)

    summary = {
        key: value for key, value in details.items()
        if key not in {
            "strict_relevance_objective",
            "non_irrelevance_objective",
        }
    }
    summary["strict_relevance_objective"] = {
        key: value for key, value in strict.items()
        if key != "threshold_sweep"
    }
    summary["non_irrelevance_objective"] = {
        key: value for key, value in non_irrelevant.items()
        if key != "threshold_sweep"
    }
    summary["private_details_path"] = str(details_path)
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
