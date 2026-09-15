#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

TEACHER_TOLERANCE = 0.005
EPS = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failed_confidence(path: Path) -> float:
    rows = read_jsonl(path)
    failed = [
        row for row in rows
        if not bool(row.get("top_supported")) or not bool(row.get("supported_set_separated"))
    ]
    values = [abs(float(score)) for row in failed for score in row.get("scores", [])]
    return statistics.fmean(values) if values else 0.0


def load_result(root: Path, key: str) -> dict[str, Any]:
    base = root / key
    repair = read_json(base / "repair-dev" / "dev_metrics.json")
    teacher = read_json(base / "teacher-dev" / "teacher_dev_metrics.json")
    novel = read_json(base / "novel" / "fixed_metrics.json")
    return {
        "repair_dev": {
            "top1_accuracy": repair["top1_accuracy"],
            "separation_rate": repair["supported_set_separation_rate"],
            "mean_margin": repair["mean_separation_margin"],
            "failures": repair["failures"],
            "mean_abs_score_on_failed_examples": failed_confidence(base / "repair-dev" / "dev_predictions.jsonl"),
        },
        "teacher_dev": {
            "top1_accuracy": teacher["top1_accuracy"],
            "separation_rate": teacher["separation_rate"],
            "full_invariance_pass_rate": teacher["full_order_invariance_pass_rate"],
            "mean_margin": teacher["mean_margin"],
            "failures": teacher["full_order_invariance_failures"],
            "mean_abs_score_on_failed_examples": failed_confidence(base / "teacher-dev" / "teacher_dev_predictions.jsonl"),
        },
        "novel": {
            "top1_accuracy": novel["top1_accuracy"],
            "separation_rate": novel["separation_rate"],
            "full_invariance_pass_rate": novel["full_invariance_pass_rate"],
            "mean_margin": novel["mean_margin"],
            "failures": novel["full_invariance_failures"],
            "mean_abs_score_on_failed_examples": failed_confidence(base / "novel" / "fixed_predictions.jsonl"),
        },
    }


def gate(candidate: dict[str, Any], parent: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate["teacher_dev"]["top1_accuracy"] < parent["teacher_dev"]["top1_accuracy"] - TEACHER_TOLERANCE:
        reasons.append("teacher_dev_top1_regression_gt_0.5pp")
    if candidate["teacher_dev"]["full_invariance_pass_rate"] < parent["teacher_dev"]["full_invariance_pass_rate"] - TEACHER_TOLERANCE:
        reasons.append("teacher_dev_invariance_regression_gt_0.5pp")
    if candidate["novel"]["top1_accuracy"] < parent["novel"]["top1_accuracy"] - EPS:
        reasons.append("novel_top1_regression")
    if candidate["novel"]["full_invariance_pass_rate"] < parent["novel"]["full_invariance_pass_rate"] - EPS:
        reasons.append("novel_invariance_regression")
    if candidate["repair_dev"]["top1_accuracy"] < parent["repair_dev"]["top1_accuracy"] - EPS:
        reasons.append("repair_dev_top1_regression")

    improved = (
        candidate["novel"]["top1_accuracy"] > parent["novel"]["top1_accuracy"] + EPS
        or candidate["novel"]["full_invariance_pass_rate"] > parent["novel"]["full_invariance_pass_rate"] + EPS
        or candidate["repair_dev"]["top1_accuracy"] > parent["repair_dev"]["top1_accuracy"] + EPS
    )
    if not improved:
        reasons.append("no_target_generalization_gain_over_parent")
    return not reasons, reasons


def rank_tuple(result: dict[str, Any]) -> tuple[float, ...]:
    failed_conf = sum(
        float(result[name]["mean_abs_score_on_failed_examples"])
        for name in ("novel", "repair_dev", "teacher_dev")
    )
    return (
        float(result["novel"]["full_invariance_pass_rate"]),
        float(result["novel"]["top1_accuracy"]),
        float(result["repair_dev"]["top1_accuracy"]),
        float(result["teacher_dev"]["full_invariance_pass_rate"]),
        float(result["teacher_dev"]["top1_accuracy"]),
        -failed_conf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select N0 v0.2 parent/repair checkpoint from compact held-out evidence.")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--previous-teacher-comparison", required=True)
    parser.add_argument("--previous-novel-comparison", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.evaluation_root).resolve()
    keys = ["parent-step250", "repair-step040", "repair-step080", "repair-step120"]
    results = {key: load_result(root, key) for key in keys}
    parent = results["parent-step250"]

    previous_teacher = read_json(Path(args.previous_teacher_comparison).resolve())
    previous_novel = read_json(Path(args.previous_novel_comparison).resolve())
    old_teacher = previous_teacher["checkpoints"]["step-00000250"]
    old_novel = previous_novel["checkpoints"]["step-00000250"]
    reproducibility = {
        "teacher_top1": abs(parent["teacher_dev"]["top1_accuracy"] - old_teacher["top1_accuracy"]) <= 1e-9,
        "teacher_invariance": abs(parent["teacher_dev"]["full_invariance_pass_rate"] - old_teacher["full_order_invariance_pass_rate"]) <= 1e-9,
        "novel_top1": abs(parent["novel"]["top1_accuracy"] - old_novel["top1_accuracy"]) <= 1e-9,
        "novel_invariance": abs(parent["novel"]["full_invariance_pass_rate"] - old_novel["full_invariance_pass_rate"]) <= 1e-9,
    }
    if not all(reproducibility.values()):
        raise SystemExit(f"parent reproducibility mismatch: {reproducibility}")

    gates: dict[str, Any] = {}
    eligible: list[str] = []
    for key in keys[1:]:
        passed, reasons = gate(results[key], parent)
        gates[key] = {"passed": passed, "reasons": reasons}
        if passed:
            eligible.append(key)

    if eligible:
        selected = max(eligible, key=lambda key: rank_tuple(results[key]))
        repair_selected = True
        next_action = "ratify_selected_repair_as_n0_v02_semantic_base_and_continue_model_building"
    else:
        selected = "parent-step250"
        repair_selected = False
        next_action = "retain_parent_step250_and_continue_model_building_without_blind_retraining"

    output = {
        "schema": "alice.eipm.n0.v02-post-repair-selection.v0.1",
        "status": "PASS",
        "status_meaning": "checkpoint_selection_completed_not_production_promotion",
        "eval_only": True,
        "private_identity_gradient": False,
        "additional_gradient_authorized": False,
        "production_promotion_authorized": False,
        "teacher_regression_tolerance": TEACHER_TOLERANCE,
        "reproducibility": reproducibility,
        "selection_policy": "no novel or repair-dev regression; teacher aggregate regression <=0.5pp; require a target gain; then novel invariance, novel top1, repair-dev top1, teacher invariance, teacher top1, lower failed-example confidence",
        "results": results,
        "repair_gates": gates,
        "eligible_repairs": eligible,
        "selected_checkpoint": selected,
        "repair_selected": repair_selected,
        "next_action": next_action,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
