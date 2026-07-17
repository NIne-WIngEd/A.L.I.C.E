from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _binary_metrics(expected: list[int], predicted: list[int]) -> dict[str, Any]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    if not expected:
        raise ValueError("No items available for ensemble analysis")

    tp = sum(1 for y, p in zip(expected, predicted) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(expected, predicted) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(expected, predicted) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(expected, predicted) if y == 1 and p == 0)

    accuracy = (tp + tn) / len(expected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_supported_count": tp + fp,
        "accuracy": round(accuracy, 6),
        "support_precision": round(precision, 6),
        "support_recall": round(recall, 6),
        "support_f1": round(f1, 6),
    }


def _qwen_supported(item: dict[str, Any]) -> bool:
    return str(item.get("qwen_verdict", "")).strip().lower() == "supported"


def _hhem_supported(item: dict[str, Any]) -> bool:
    return bool(item.get("hhem_predicted_supported", False))


def _fever_supported(item: dict[str, Any]) -> bool:
    return str(item.get("fever_decision", "")).strip() == "keep_entailment"


def _human_supported(item: dict[str, Any]) -> bool:
    return str(item.get("human_label", "")).strip() == "supported"


def _rules() -> dict[str, Callable[[bool, bool, bool], bool]]:
    return {
        "qwen_only": lambda q, h, f: q,
        "hhem_only": lambda q, h, f: h,
        "fever_only": lambda q, h, f: f,
        "qwen_and_hhem": lambda q, h, f: q and h,
        "qwen_and_fever": lambda q, h, f: q and f,
        "hhem_and_fever": lambda q, h, f: h and f,
        "all_three": lambda q, h, f: q and h and f,
        "any_two_of_three": lambda q, h, f: (int(q) + int(h) + int(f)) >= 2,
    }


def analyze_verifier_ensembles(
    *,
    details_path: Path,
    vault_root: Path | None = None,
    pilot_name: str = "pilot-v1",
) -> dict[str, Any]:
    details_path = details_path.expanduser().resolve(strict=True)
    data = json.loads(details_path.read_text(encoding="utf-8"))

    if int(data.get("hhem_holdout_evaluation_schema_version", -1)) != 1:
        raise ValueError("Input must be an HHEM holdout evaluation details file")

    items = list(data.get("items", []))
    if not items:
        raise ValueError("Holdout details contain no items")

    expected = [1 if _human_supported(item) else 0 for item in items]
    triples = [
        (
            _qwen_supported(item),
            _hhem_supported(item),
            _fever_supported(item),
        )
        for item in items
    ]

    rule_results: dict[str, Any] = {}
    for name, rule in _rules().items():
        predicted = [
            1 if rule(qwen, hhem, fever) else 0
            for qwen, hhem, fever in triples
        ]
        rule_results[name] = _binary_metrics(expected, predicted)

    pairwise_disagreements = {
        "qwen_vs_hhem": sum(q != h for q, h, _ in triples),
        "qwen_vs_fever": sum(q != f for q, _, f in triples),
        "hhem_vs_fever": sum(h != f for _, h, f in triples),
        "all_three_disagree_or_partial_agreement": sum(
            len({q, h, f}) > 1 for q, h, f in triples
        ),
        "all_three_agree": sum(len({q, h, f}) == 1 for q, h, f in triples),
    }

    unanimous_supported = [
        i for i, (q, h, f) in enumerate(triples) if q and h and f
    ]
    unanimous_rejected = [
        i for i, (q, h, f) in enumerate(triples) if not q and not h and not f
    ]

    best_precision = max(
        result["support_precision"] for result in rule_results.values()
    )
    highest_precision_rules = [
        name
        for name, result in rule_results.items()
        if result["support_precision"] == best_precision
    ]

    run_id = str(uuid.uuid4())
    summary: dict[str, Any] = {
        "verifier_ensemble_diagnostics_schema_version": 1,
        "run_id": run_id,
        "created_at": _now(),
        "pilot_name": pilot_name,
        "source_holdout_run_id": str(data.get("run_id", "")),
        "source_holdout_id": str(data.get("holdout_id", "")),
        "source_details_path": str(details_path),
        "item_count": len(items),
        "human_supported_count": sum(expected),
        "human_not_fully_supported_count": len(expected) - sum(expected),
        "semantics": {
            "human_positive": "human_label == supported",
            "qwen_positive": "qwen_verdict == supported",
            "hhem_positive": "hhem_predicted_supported == true",
            "fever_positive": "fever_decision == keep_entailment",
        },
        "rules": rule_results,
        "pairwise_disagreements": pairwise_disagreements,
        "unanimous_supported_count": len(unanimous_supported),
        "unanimous_rejected_count": len(unanimous_rejected),
        "highest_observed_support_precision": best_precision,
        "highest_precision_rules": highest_precision_rules,
        "diagnostic_only": True,
        "production_gate_changed": False,
        "selection_warning": (
            "Choosing a production ensemble rule based on this holdout would tune "
            "the rule to the holdout. Any selected rule requires validation on a "
            "fresh independent set before production use."
        ),
        "private_text_uploaded": False,
    }

    if vault_root is not None:
        vault_root = vault_root.expanduser().resolve(strict=True)
        export_root = vault_root / "manifests" / "exports"
        private_root = vault_root / "manifests" / "calibration" / pilot_name

        private_details = {
            **summary,
            "items": [
                {
                    "item_id": str(item.get("item_id", "")),
                    "query_id": str(item.get("query_id", "")),
                    "human_label": str(item.get("human_label", "")),
                    "qwen_supported": q,
                    "hhem_supported": h,
                    "fever_supported": f,
                    "rule_predictions": {
                        name: bool(rule(q, h, f))
                        for name, rule in _rules().items()
                    },
                }
                for item, (q, h, f) in zip(items, triples)
            ],
        }

        private_path = (
            private_root / f"verifier-ensemble-diagnostics-{run_id}.json"
        )
        summary_path = (
            export_root / f"verifier-ensemble-diagnostics-summary-{run_id}.json"
        )

        _atomic_json(private_path, private_details)
        summary["private_details_path"] = str(private_path)
        _atomic_json(summary_path, summary)
        summary["summary_path"] = str(summary_path)

    return summary
