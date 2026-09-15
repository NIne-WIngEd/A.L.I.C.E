#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from alice_personality.n0.curriculum_data import CurriculumDataset
from alice_personality.n0.v02_training import verify_teacher_registry

EXPECTED_CHECKPOINT_SCHEMA = "alice.eipm.n0.v02-multitask-checkpoint-receipt.v0.1"
EXPECTED_MODEL_ID = "alice-n0-semantic-v0.2"
EXPECTED_PARAMETERS = 136_594_435
EXPECTED_TRAIN_ROWS = 51 * 15
EXPECTED_DEV_ROWS = 51 * 5
EXPECTED_COMPETENCIES = 51


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify_checkpoint(
    *,
    checkpoint: Path,
    step: int,
    config_sha256: str,
    tokenizer_sha256: str,
    teacher_registry_sha256: str,
    teacher_audit_sha256: str,
) -> dict[str, Any]:
    receipt_path = checkpoint / "receipt.json"
    ranker_path = checkpoint / "ranker.safetensors"
    full_model_path = checkpoint / "alice_n0_v02.safetensors"
    mlm_dir = checkpoint / "mlm"

    for required in (receipt_path, ranker_path, full_model_path, mlm_dir / "config.json"):
        require(required.exists(), f"missing checkpoint artifact: {required}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == EXPECTED_CHECKPOINT_SCHEMA, f"unexpected checkpoint schema at step {step}")
    require(receipt.get("model_id") == EXPECTED_MODEL_ID, f"model_id mismatch at step {step}")
    require(int(receipt.get("step", -1)) == step, f"receipt step mismatch: expected {step}")
    require(receipt.get("random_initialization_only") is True, f"step {step} is not random-init lineage")
    require(receipt.get("v01_weights_used") is False, f"step {step} unexpectedly used v0.1 weights")
    require(int(receipt.get("exact_parameter_count", 0)) == EXPECTED_PARAMETERS, f"parameter-count drift at step {step}")
    require(receipt.get("config_sha256") == config_sha256, f"config hash mismatch at step {step}")
    require(receipt.get("tokenizer_sha256") == tokenizer_sha256, f"tokenizer hash mismatch at step {step}")
    require(receipt.get("teacher_registry_sha256") == teacher_registry_sha256, f"teacher registry hash mismatch at step {step}")
    require(receipt.get("teacher_audit_sha256") == teacher_audit_sha256, f"teacher audit hash mismatch at step {step}")
    require(int(receipt.get("teacher_registered_rows", 0)) == 1020, f"teacher row count mismatch at step {step}")
    require(int(receipt.get("teacher_competencies", 0)) == EXPECTED_COMPETENCIES, f"teacher competency mismatch at step {step}")
    require(receipt.get("teacher_coverage_gate_open") is True, f"teacher coverage gate closed at step {step}")
    require(receipt.get("private_identity_data") is False, f"private identity data present at step {step}")
    require(receipt.get("private_identity_gradient") is False, f"private identity gradient present at step {step}")
    require(receipt.get("private_identity_gradient_authorized") is False, f"private identity gradient authorization drift at step {step}")
    require(receipt.get("model_training_performed") is True, f"step {step} is not a trained checkpoint")
    require(receipt.get("training_stage") == "N0_public_native_multitask", f"training-stage mismatch at step {step}")

    require(receipt.get("ranker_sha256") == sha256_file(ranker_path), f"ranker hash mismatch at step {step}")
    require(receipt.get("full_model_sha256") == sha256_file(full_model_path), f"full-model hash mismatch at step {step}")

    recorded_mlm = receipt.get("mlm_artifact_sha256")
    require(isinstance(recorded_mlm, dict) and recorded_mlm, f"missing MLM artifact manifest at step {step}")
    observed_mlm: dict[str, str] = {}
    for rel, expected_hash in sorted(recorded_mlm.items()):
        rel_path = Path(str(rel))
        require(not rel_path.is_absolute() and ".." not in rel_path.parts, f"unsafe MLM artifact path at step {step}: {rel_path}")
        path = mlm_dir / rel_path
        require(path.is_file(), f"missing MLM artifact at step {step}: {path}")
        observed = sha256_file(path)
        require(observed == str(expected_hash), f"MLM artifact hash mismatch at step {step}: {rel_path}")
        observed_mlm[str(rel_path)] = observed

    return {
        "step": step,
        "receipt_sha256": sha256_file(receipt_path),
        "ranker_sha256": receipt["ranker_sha256"],
        "full_model_sha256": receipt["full_model_sha256"],
        "mlm_artifact_count": len(observed_mlm),
        "git_revision": receipt.get("git_revision"),
        "world_size": receipt.get("world_size"),
        "mixed_precision": receipt.get("mixed_precision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed lineage and split preflight for the N0 v0.2 teacher-dev challenge."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--teacher-registry", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--tranche-root", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    registry_path = Path(args.teacher_registry).resolve()
    audit_path = Path(args.teacher_audit).resolve()
    tranche_root = Path(args.tranche_root).resolve()

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    for required in (config_path, tokenizer_path, registry_path, audit_path):
        require(required.is_file(), f"missing challenge preflight input: {required}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config.get("model_id") == EXPECTED_MODEL_ID, "challenge config model_id mismatch")
    architecture = config.get("architecture") or {}
    require(int(architecture.get("exact_parameter_count", 0)) == EXPECTED_PARAMETERS, "challenge config parameter-count drift")
    governance = config.get("governance") or {}
    require(governance.get("private_identity_gradient") is False, "challenge config unexpectedly enables private identity gradient")

    curriculum_paths, teacher_report = verify_teacher_registry(repo_root, registry_path, audit_path)
    require(int(teacher_report.get("registered_rows", 0)) == 1020, "teacher registry no longer binds exactly 1020 rows")
    require(int(teacher_report.get("competency_count", 0)) == EXPECTED_COMPETENCIES, "teacher registry competency count drift")

    train = CurriculumDataset(curriculum_paths, "train")
    dev = CurriculumDataset(curriculum_paths, "dev")
    require(len(train) == EXPECTED_TRAIN_ROWS, f"expected {EXPECTED_TRAIN_ROWS} train rows, observed {len(train)}")
    require(len(dev) == EXPECTED_DEV_ROWS, f"expected {EXPECTED_DEV_ROWS} dev rows, observed {len(dev)}")

    train_ids = {str(row["id"]) for row in train.rows}
    dev_ids = {str(row["id"]) for row in dev.rows}
    overlap = sorted(train_ids & dev_ids)
    require(not overlap, f"teacher train/dev id overlap detected: {overlap[:10]}")

    train_counts = Counter(str(row["competency"]) for row in train.rows)
    dev_counts = Counter(str(row["competency"]) for row in dev.rows)
    require(len(train_counts) == EXPECTED_COMPETENCIES and all(value == 15 for value in train_counts.values()), "teacher train split must contain 15 rows for every competency")
    require(len(dev_counts) == EXPECTED_COMPETENCIES and all(value == 5 for value in dev_counts.values()), "teacher dev split must contain 5 rows for every competency")
    require(set(train_counts) == set(dev_counts), "teacher train/dev competency sets differ")

    config_sha256 = sha256_file(config_path)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    registry_sha256 = sha256_file(registry_path)
    audit_sha256 = sha256_file(audit_path)

    checkpoints = {}
    for step in (250, 500):
        checkpoint = tranche_root / "checkpoints" / f"step-{step:08d}"
        checkpoints[f"step-{step:08d}"] = verify_checkpoint(
            checkpoint=checkpoint,
            step=step,
            config_sha256=config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            teacher_registry_sha256=registry_sha256,
            teacher_audit_sha256=audit_sha256,
        )

    summary = {
        "schema": "alice.eipm.n0.v02-teacher-dev-challenge-preflight.v0.1",
        "status": "PASS",
        "status_meaning": "lineage_split_and_artifact_integrity_only",
        "model_id": EXPECTED_MODEL_ID,
        "config_sha256": config_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "teacher_registry_sha256": registry_sha256,
        "teacher_audit_sha256": audit_sha256,
        "teacher_train_rows": len(train),
        "teacher_dev_rows": len(dev),
        "teacher_competencies": len(dev_counts),
        "train_dev_id_overlap": 0,
        "checkpoints": checkpoints,
        "eval_only": True,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "full_expanded_challenge_satisfied": False,
        "additional_gradient_authorized": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
