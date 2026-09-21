#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from alice_personality.n0.qsre_production_core import (
    QSREProductionExecutor,
    QSREProductionSchemaEncoder,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import config_from_plan, load_plan
from train_n0_v02_qsre_production_p1_v1 import eligible, evaluate


CAUSAL_CORRECTION = (
    "path_frontier_stop_tail_persistence:"
    "inactive_relation_steps_preserve_the_last_reached_frontier"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--prepared-cache", required=True)
    p.add_argument("--schema-cache", required=True)
    p.add_argument("--failed-p1-result", required=True)
    p.add_argument("--failed-p1-root", required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--selected-output-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    plan_path = Path(args.plan)
    prepared_path = Path(args.prepared_cache)
    schema_cache_path = Path(args.schema_cache)
    failed_result_path = Path(args.failed_p1_result)
    failed_root = Path(args.failed_p1_root)
    receipt_path = Path(args.receipt)
    selected_output = Path(args.selected_output_dir)

    if selected_output.exists():
        raise SystemExit("refusing to overwrite recovered P1 evidence")

    failed = json.loads(failed_result_path.read_text(encoding="utf-8"))
    if failed.get("status") != "FAIL_QSRE_PRODUCTION_P1_EXECUTOR":
        raise SystemExit("source P1 result is not the preserved failed production run")
    if failed.get("selected") is not None:
        raise SystemExit("failed P1 unexpectedly contains a selected checkpoint")
    if failed.get("p2_authorized") is not False:
        raise SystemExit("failed P1 unexpectedly authorized P2")

    exact_hashes = {
        "plan_sha256": sha256(plan_path),
        "prepared_cache_sha256": sha256(prepared_path),
        "schema_cache_sha256": sha256(schema_cache_path),
    }
    for key, value in exact_hashes.items():
        if failed.get(key) != value:
            raise SystemExit(
                f"preserved P1 lineage drift for {key}: "
                f"failed={failed.get(key)} current={value}"
            )

    prepared = torch.load(prepared_path, map_location="cpu")
    if (
        prepared.get("private_identity_data") is not False
        or prepared.get("test_present") is not False
    ):
        raise SystemExit("P1 recovery crossed the governed public boundary")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("P1 checkpoint requalification requested CUDA but CUDA is unavailable")

    plan = load_plan(plan_path)
    stage = plan["stages"]["P1"]
    config = config_from_plan(plan)
    schema, _ = load_dynamic_schema_cache(schema_cache_path, device=device)
    dev = prepared["dev"]

    history = failed.get("history")
    if not isinstance(history, list) or not history:
        raise SystemExit("preserved P1 history missing")
    steps = sorted({int(row["step"]) for row in history})
    requalified = []
    selected = None

    for step in steps:
        checkpoint = failed_root / f"step-{step:08d}" / "qsre_production_p1.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"preserved P1 checkpoint missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("schema") != "alice.eipm.n0.qsre-production-p1-checkpoint.v1":
            raise SystemExit(f"P1 checkpoint schema drift at step {step}")
        if int(payload.get("step", -1)) != step:
            raise SystemExit(f"P1 checkpoint step drift at step {step}")
        for key, value in exact_hashes.items():
            if payload.get(key) != value:
                raise SystemExit(
                    f"P1 checkpoint lineage drift at step {step} for {key}"
                )

        schema_encoder = QSREProductionSchemaEncoder(config)
        executor = QSREProductionExecutor(config)
        schema_encoder.load_state_dict(payload["schema_encoder"], strict=True)
        executor.load_state_dict(payload["executor"], strict=True)
        for module in (schema_encoder, executor):
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.to(device).eval()

        metrics = evaluate(
            split=dev,
            schema=schema,
            schema_encoder=schema_encoder,
            executor=executor,
            config=config,
            device=device,
            batch_size=int(stage["training"]["batch_size"]),
        )
        passes = eligible(metrics, stage["eligibility"])
        record = {
            "step": step,
            "checkpoint_sha256": sha256(checkpoint),
            "eligible_under_corrected_runtime": bool(passes),
            "dev": metrics,
        }
        requalified.append(record)
        print("P1_REQUAL_EVAL=" + json.dumps(record, sort_keys=True), flush=True)

        del schema_encoder, executor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if passes:
            selected = record
            selected_dir = selected_output / f"step-{step:08d}"
            selected_dir.mkdir(parents=True, exist_ok=False)
            target = selected_dir / "qsre_production_p1.pt"
            shutil.copy2(checkpoint, target)
            copied_sha = sha256(target)
            if copied_sha != record["checkpoint_sha256"]:
                raise SystemExit("copied recovered P1 checkpoint hash drift")

            result = {
                "schema": "alice.eipm.n0.qsre-production-p1-result.v1",
                "status": "PASS_QSRE_PRODUCTION_P1_EXECUTOR",
                "selected": {
                    "step": step,
                    "checkpoint_sha256": copied_sha,
                    "metrics": metrics,
                },
                "best_observed": {
                    "step": step,
                    "checkpoint_sha256": copied_sha,
                    "metrics": metrics,
                },
                "history": requalified,
                **exact_hashes,
                "train_relation_count": int(failed["train_relation_count"]),
                "dev_relation_count": int(failed["dev_relation_count"]),
                "semantic_backbone_gradient": False,
                "operator_gradient": False,
                "binder_gradient": False,
                "schema_encoder_gradient": False,
                "executor_gradient": False,
                "private_identity_gradient": False,
                "automatic_rerun": False,
                "automatic_hotfix": False,
                "p2_authorized": True,
                "recovered_from_preserved_failed_p1": True,
                "source_failed_result_sha256": sha256(failed_result_path),
                "source_checkpoint_sha256": record["checkpoint_sha256"],
                "causal_runtime_correction": CAUSAL_CORRECTION,
                "checkpoint_weights_changed": False,
                "gradient_performed_during_requalification": False,
            }
            (selected_output / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            break

    receipt = {
        "schema": "alice.eipm.n0.qsre-production-p1-stop-tail-requalification.v1",
        "status": (
            "PASS_EXISTING_P1_CHECKPOINT_REQUALIFIED"
            if selected is not None
            else "NO_EXISTING_P1_CHECKPOINT_ELIGIBLE_CORRECTED_P1_REQUIRED"
        ),
        "causal_runtime_correction": CAUSAL_CORRECTION,
        "source_failed_result_sha256": sha256(failed_result_path),
        "source_failed_status": failed["status"],
        "source_steps_evaluated": len(requalified),
        "selected": selected,
        "requalified": requalified,
        **exact_hashes,
        "optimizer": False,
        "gradient": False,
        "checkpoint_weights_changed": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "automatic_hyperparameter_search": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("P1_REQUAL_RESULT=" + json.dumps(receipt, sort_keys=True), flush=True)

    if selected is None:
        raise SystemExit(10)


if __name__ == "__main__":
    main()
