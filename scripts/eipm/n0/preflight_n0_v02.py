#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum import validate_curriculum_manifest, validate_curriculum_rows
from alice_personality.n0.model import build_masked_lm, count_parameters
from alice_personality.n0.v02_objectives import validate_teacher_row_for_v02


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only production preflight for N0 v0.2.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--skip-model-build", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[3]
    config_path = root / "configs/eipm/n0/alice_n0_semantic_v0.2.json"
    mixture_path = root / "configs/eipm/n0/public_corpus_v0.2.plan.json"
    benchmark_path = root / "evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl"
    compiler = root / "scripts/eipm/n0/compile_n0_v02_fixed_eval.py"

    config = load_n0_config(config_path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))

    if config.model_id != "alice-n0-semantic-v0.2":
        raise SystemExit("wrong model_id for N0 v0.2 preflight")
    if raw_config["architecture"].get("third_party_weight_initialization") is not False:
        raise SystemExit("N0 v0.2 must remain native random-init")
    if raw_config["governance"].get("private_identity_gradient") is not False:
        raise SystemExit("N0 v0.2 preflight must remain public-only")
    if abs(config.mlm_probability - 0.30) > 1e-9:
        raise SystemExit("N0 v0.2 expected 30% MLM corruption")

    category_share = sum(float(group["target_share"]) for group in mixture["mixture"])
    source_share = sum(
        float(source["target_share"])
        for group in mixture["mixture"]
        for source in group["sources"]
    )
    if abs(category_share - 1.0) > 1e-9 or abs(source_share - 1.0) > 1e-9:
        raise SystemExit("N0 v0.2 corpus mixture weights must sum to 1.0")
    if mixture.get("status") != "planned_not_activated":
        raise SystemExit("corpus plan must stay non-active until source rights/schema freeze")

    benchmark_rows = [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(benchmark_rows) != 43 or len({row["competency"] for row in benchmark_rows}) != 43:
        raise SystemExit("fixed N0 v0.2 base suite must contain exactly one base for all 43 competencies")
    if any(row.get("training_authorized") is not False for row in benchmark_rows):
        raise SystemExit("fixed readiness benchmark must never authorize training")

    curricula = [
        (
            root / "training/eipm/n0/sol_curriculum_seed_v0.1.jsonl",
            root / "training/eipm/n0/sol_curriculum_seed_v0.1.origin.json",
        ),
        (
            root / "training/eipm/n0/sol_curriculum_coverage_v0.2.jsonl",
            root / "training/eipm/n0/sol_curriculum_coverage_v0.2.origin.json",
        ),
        (
            root / "training/eipm/n0/sol_curriculum_repair_v0.3.jsonl",
            root / "training/eipm/n0/sol_curriculum_repair_v0.3.origin.json",
        ),
    ]
    teacher_rows = 0
    for curriculum, manifest in curricula:
        validate_curriculum_manifest(curriculum, manifest)
        summary = validate_curriculum_rows(curriculum)
        teacher_rows += int(summary["row_count"])
        for line in curriculum.read_text(encoding="utf-8").splitlines():
            if line.strip():
                validate_teacher_row_for_v02(json.loads(line))

    with tempfile.TemporaryDirectory(prefix="alice-n0-v02-preflight-") as tmp:
        compiled = Path(tmp) / "fixed.jsonl"
        receipt = Path(tmp) / "receipt.json"
        subprocess.run(
            [
                sys.executable,
                str(compiler),
                "--base",
                str(benchmark_path),
                "--output",
                str(compiled),
                "--manifest",
                str(receipt),
            ],
            check=True,
        )
        compiled_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        if int(compiled_receipt["compiled_rows"]) != 258:
            raise SystemExit("fixed suite should compile to 258 invariance-scored rows")

    model_parameters = None
    trainable_parameters = None
    if not args.skip_model_build:
        model = build_masked_lm(config)
        model_parameters, trainable_parameters = count_parameters(model)
        if not 110_000_000 <= model_parameters <= 180_000_000:
            raise SystemExit(
                f"exact N0 v0.2 parameter count {model_parameters} outside approved 110M-180M range"
            )

    result = {
        "status": "PASS",
        "model_id": config.model_id,
        "exact_parameters": model_parameters,
        "trainable_parameters": trainable_parameters,
        "planned_parameter_reference": config.planned_parameter_count,
        "mlm_probability": config.mlm_probability,
        "teacher_rows_available_now": teacher_rows,
        "teacher_rows_minimum_for_full_multitask": raw_config["training"]["curriculum_policy"]["principle_rows_minimum_before_full_multitask_training"],
        "fixed_base_rows": len(benchmark_rows),
        "fixed_compiled_rows": 258,
        "corpus_plan_status": mixture["status"],
        "corpus_source_count": sum(len(group["sources"]) for group in mixture["mixture"]),
        "private_identity_gradient": False,
        "gpu_training_authorized_by_this_preflight": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
