"""Precommit or run a public frozen-Gemma semantic experiment, unqualified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alice_personality.gemma_n0.public_semantic_experiment import (
    ExperimentError, create_plan, run_experiment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--source-admission", type=Path, required=True)
    plan.add_argument("--preparation", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--train-rows-per-family", type=int, default=1)
    plan.add_argument("--dev-rows-per-family", type=int, default=2)
    plan.add_argument("--style-rows-per-dev-family", type=int, default=1)
    plan.add_argument("--width", type=int, default=64)
    plan.add_argument("--steps", type=int, default=256)
    plan.add_argument("--seed", type=int, default=20261002)
    plan.add_argument("--learning-rate", type=float, default=0.0003)
    plan.add_argument("--weight-decay", type=float, default=0.01)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--cache-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            recipe = {name: getattr(args, name) for name in ("train_rows_per_family", "dev_rows_per_family",
                      "style_rows_per_dev_family", "width", "steps", "seed", "learning_rate", "weight_decay")}
            receipt = create_plan(source_admission=args.source_admission, preparation_receipt=args.preparation,
                                  output_plan=args.output, recipe=recipe)
            summary = {"state": receipt["state"], "receipt_sha256": receipt["receipt_sha256"],
                       "planned_feature_inputs": receipt["unique_complete_feature_inputs"],
                       "recipe": recipe, "model_geometry": receipt["model_geometry"], "n0_approved": False}
        else:
            receipt = run_experiment(plan_path=args.plan, output_directory=args.output_directory,
                                     cache_directory=args.cache_directory)
            summary = {"state": receipt["state"], "receipt_sha256": receipt["receipt_sha256"],
                       "learned": receipt["learned"], "n0_approved": False}
    except (ExperimentError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
