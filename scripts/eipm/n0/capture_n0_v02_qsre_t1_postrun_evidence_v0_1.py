from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_RESULT_SHA256 = (
    "ba6b82c10fcb13f403569ad24dbd77cf0b3657dfc3eaf9cb9d10b6690b2d32be"
)
EXPECTED_PREPARED_CACHE_SHA256 = (
    "03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564"
)
EXPECTED_CURRICULUM_SHA256 = (
    "155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063"
)
EXPECTED_SELECTED_STEP = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result_path = Path(args.result).resolve()
    output_dir = Path(args.output).resolve()

    if not result_path.is_file():
        raise SystemExit(f"T1 result missing: {result_path}")

    actual_result_sha = sha256(result_path)
    if actual_result_sha != EXPECTED_RESULT_SHA256:
        raise SystemExit(
            "T1 result hash drift: "
            f"expected {EXPECTED_RESULT_SHA256} got {actual_result_sha}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != "alice.eipm.n0.qsre-t1-training-result.v0.1":
        raise SystemExit("T1 result schema drift")
    if result.get("status") != "PASS_QSRE_T1_EXECUTOR_DEV_CONTRACT":
        raise SystemExit("T1 result is not a governed PASS")
    if result.get("selected_checkpoint_step") != EXPECTED_SELECTED_STEP:
        raise SystemExit("unexpected selected T1 checkpoint step")
    if result.get("prepared_cache_sha256") != EXPECTED_PREPARED_CACHE_SHA256:
        raise SystemExit("prepared-cache lineage drift")
    if result.get("curriculum_sha256") != EXPECTED_CURRICULUM_SHA256:
        raise SystemExit("curriculum lineage drift")

    forbidden_true = (
        "learned_operator",
        "learned_support",
        "causal_test_opened",
        "frozen_challenge_opened",
        "private_identity_gradient",
        "automatic_rerun_authorized",
    )
    for key in forbidden_true:
        if result.get(key) is not False:
            raise SystemExit(f"T1 governance drift: {key} must be false")

    key = f"step-{EXPECTED_SELECTED_STEP:08d}"
    checkpoint_info = result.get("checkpoints", {}).get(key)
    if not isinstance(checkpoint_info, dict):
        raise SystemExit(f"selected checkpoint record missing: {key}")
    if checkpoint_info.get("eligible") is not True:
        raise SystemExit("selected checkpoint is not marked eligible")

    dev = checkpoint_info.get("dev")
    if not isinstance(dev, dict):
        raise SystemExit("selected checkpoint DEV metrics missing")

    exact_one = (
        "causal_pair_completion",
        "control_accuracy",
        "family_min_success",
        "row_success_accuracy",
        "single_target_top1_accuracy",
    )
    for metric in exact_one:
        if float(dev.get(metric, -1.0)) != 1.0:
            raise SystemExit(f"selected DEV metric drift: {metric}")

    family_success = dev.get("family_success")
    if not isinstance(family_success, dict) or not family_success:
        raise SystemExit("family_success missing")
    if any(float(value) != 1.0 for value in family_success.values()):
        raise SystemExit("not every T1 DEV family passed")

    if float(dev.get("outside_support_mass_max", 1.0)) != 0.0:
        raise SystemExit("outside-support mass is not zero")
    if float(dev.get("outside_support_invariance_max_delta", 1.0)) > 1.0e-7:
        raise SystemExit("outside-support invariance drift")
    if float(dev.get("plural_l1", 1.0)) > 0.1:
        raise SystemExit("plurality DEV threshold failed")

    recorded_checkpoint_sha = checkpoint_info.get("checkpoint_sha256")
    if not isinstance(recorded_checkpoint_sha, str) or len(recorded_checkpoint_sha) != 64:
        raise SystemExit("selected checkpoint SHA-256 missing from T1 result")

    checkpoint_path = (
        result_path.parent / key / "qsre_t1_executor.pt"
    ).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"selected T1 checkpoint missing: {checkpoint_path}")

    actual_checkpoint_sha = sha256(checkpoint_path)
    if actual_checkpoint_sha != recorded_checkpoint_sha:
        raise SystemExit(
            "selected T1 checkpoint hash mismatch between file and result receipt"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "alice.eipm.n0.qsre-t1-postrun-evidence.v0.1",
        "status": "PASS_QSRE_T1_POSTRUN_CHECKPOINT_BOUND",
        "t1_authorized_source_revision": (
            "f4a9edb1915e39d893224f4503746bfd61fc6424"
        ),
        "magnolia_job_id": 575914,
        "result_path": str(result_path),
        "result_sha256": actual_result_sha,
        "selected_checkpoint_step": EXPECTED_SELECTED_STEP,
        "selected_checkpoint_path": str(checkpoint_path),
        "selected_checkpoint_sha256": actual_checkpoint_sha,
        "prepared_cache_sha256": result["prepared_cache_sha256"],
        "source_cache_sha256": result["source_cache_sha256"],
        "curriculum_sha256": result["curriculum_sha256"],
        "selected_dev": dev,
        "learned_operator": False,
        "learned_support": False,
        "causal_test_opened": False,
        "frozen_challenge_opened": False,
        "private_identity_gradient": False,
        "optimizer_created": False,
        "gradient_performed": False,
        "gpu_required": False,
        "t1_rerun_authorized": False,
        "t2_gradient_authorized": False,
        "next_action": (
            "Use this exact frozen T1 checkpoint as the executor boundary for "
            "T2 learned-operator CPU preparation; do not open T1 TEST or rerun T1."
        ),
    }

    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("status=" + receipt["status"])
    print("result_sha256=" + actual_result_sha)
    print("selected_checkpoint_step=" + str(EXPECTED_SELECTED_STEP))
    print("selected_checkpoint_sha256=" + actual_checkpoint_sha)
    print("postrun_receipt_sha256=" + sha256(receipt_path))
    print("causal_test_opened=False")
    print("frozen_challenge_opened=False")
    print("t2_gradient_authorized=False")


if __name__ == "__main__":
    main()
