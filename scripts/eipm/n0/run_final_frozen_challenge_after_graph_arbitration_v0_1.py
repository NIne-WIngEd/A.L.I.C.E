#!/usr/bin/env python3
"""Run the one final frozen N0 latent challenge authorized by graph arbitration.

This is execution binding, not a new evaluation design. It reuses the immutable
v0.2 latent challenge/spec/evaluator and preselected step-360 latent checkpoint.
The only model-path change is the already-arbitrated step-200 repaired evidence
graph. No training, threshold change, checkpoint selection, promotion, or scaling
is permitted here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "n0_final_frozen_challenge_after_graph_arbitration_v0.1"
ARBITRATION_PROTOCOL_VERSION = "n0_downstream_causal_arbitration_v0.2"
GATE_PROTOCOL_VERSION = "n0_post_arbitration_decision_gate_v0.1"
FINALIZER_PROTOCOL_VERSION = "n0_downstream_causal_arbitration_finalizer_v0.1"
EXPECTED_DECISION = "AUTHORIZE_ONE_FINAL_FROZEN_CHALLENGE"
EXPECTED_REPAIRED_GRAPH_SHA256 = "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_LATENT_SHA256 = "503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4"
EXPECTED_INTERVENTION = "before_parent_cache_and_before_cross_context_fusion"


class FinalChallengeError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory_tree(path: Path) -> str:
    if not path.is_dir():
        raise FinalChallengeError(f"directory artifact is not a directory: {path}")
    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise FinalChallengeError(f"directory artifact contains symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise FinalChallengeError(
                f"directory artifact contains non-regular entry: {item}"
            )
        rows.append(
            {
                "path": item.relative_to(path).as_posix(),
                "sha256": sha256_file(item),
                "size": item.stat().st_size,
            }
        )
    if not rows:
        raise FinalChallengeError(f"directory artifact is empty: {path}")
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalChallengeError(f"unable to load JSON {path}: {exc}") from exc


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalChallengeError(f"{label} must be an object")
    return value


def require_false(root: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    for key in keys:
        if root.get(key) is not False:
            raise FinalChallengeError(f"{label}.{key} must be false")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalChallengeError(f"unable to resolve git HEAD: {exc}") from exc


def require_clean_tracked_worktree(repo_root: Path) -> None:
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalChallengeError(f"unable to inspect git worktree: {exc}") from exc
    if output.strip():
        raise FinalChallengeError(
            "tracked repository worktree must be clean for final frozen challenge"
        )


def checked_file(path: Path, expected_sha: str, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FinalChallengeError(f"{label} is not a regular file: {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise FinalChallengeError(
            f"{label} sha256 mismatch: expected {expected_sha}, got {actual}"
        )
    return actual


def checked_tree(path: Path, expected_sha: str, label: str) -> str:
    path = path.resolve()
    actual = sha256_directory_tree(path)
    if actual != expected_sha:
        raise FinalChallengeError(
            f"{label} tree sha256 mismatch: expected {expected_sha}, got {actual}"
        )
    return actual


def validate_authorization(
    *,
    arbitration_dir: Path,
    repaired_graph: Path,
) -> dict[str, Any]:
    result_path = (arbitration_dir / "arbitration_result.json").resolve()
    gate_path = (arbitration_dir / "post_arbitration_gate_receipt.json").resolve()
    final_path = (arbitration_dir / "finalization_receipt.json").resolve()
    stack_path = (arbitration_dir / "full_stack_manifest.json").resolve()

    for label, path in (
        ("arbitration result", result_path),
        ("post-arbitration gate receipt", gate_path),
        ("finalization receipt", final_path),
        ("full-stack manifest", stack_path),
    ):
        if not path.is_file():
            raise FinalChallengeError(f"{label} missing: {path}")

    result_sha = sha256_file(result_path)
    gate_sha = sha256_file(gate_path)
    final_sha = sha256_file(final_path)
    stack_sha = sha256_file(stack_path)

    result = require_mapping(load_json(result_path), "arbitration_result")
    gate = require_mapping(load_json(gate_path), "gate_receipt")
    final = require_mapping(load_json(final_path), "finalization_receipt")
    stack = require_mapping(load_json(stack_path), "full_stack_manifest")

    if result.get("protocol_version") != ARBITRATION_PROTOCOL_VERSION:
        raise FinalChallengeError("arbitration protocol drift")
    if result.get("status") != "COMPLETE":
        raise FinalChallengeError("arbitration result is not COMPLETE")
    if result.get("classification") != "IMPROVEMENT":
        raise FinalChallengeError("final challenge requires arbitration IMPROVEMENT")
    require_false(
        result,
        ("ratifies_repair", "scale_authorized", "promotion_authorized", "n0_complete"),
        "arbitration_result",
    )

    comparisons = result.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise FinalChallengeError("arbitration comparisons missing")
    states = [item.get("result") for item in comparisons if isinstance(item, Mapping)]
    if "HARM" in states or "IMPROVEMENT" not in states:
        raise FinalChallengeError(
            "final challenge requires at least one frozen improvement and no harm"
        )

    arms = require_mapping(result.get("arms"), "arbitration_result.arms")
    candidate = require_mapping(arms.get("candidate"), "arbitration_result.arms.candidate")
    candidate_sha = str(candidate.get("sha256", "")).lower()
    if candidate_sha != EXPECTED_REPAIRED_GRAPH_SHA256:
        raise FinalChallengeError(
            "arbitration candidate is not the selected step-200 repaired graph"
        )
    if Path(str(candidate.get("path"))).resolve() != repaired_graph.resolve():
        raise FinalChallengeError(
            "repaired graph path differs from frozen arbitration candidate path"
        )
    checked_file(
        repaired_graph.resolve(),
        EXPECTED_REPAIRED_GRAPH_SHA256,
        "selected repaired graph",
    )

    if gate.get("protocol_version") != GATE_PROTOCOL_VERSION:
        raise FinalChallengeError("post-arbitration gate protocol drift")
    if gate.get("status") != "COMPLETE":
        raise FinalChallengeError("post-arbitration gate is not COMPLETE")
    if gate.get("classification") != "IMPROVEMENT":
        raise FinalChallengeError("post-arbitration gate classification drift")
    if gate.get("decision") != EXPECTED_DECISION:
        raise FinalChallengeError("post-arbitration gate did not authorize final challenge")
    if gate.get("final_frozen_challenge_authorized") is not True:
        raise FinalChallengeError("final frozen challenge is not authorized")
    if gate.get("final_frozen_challenge_limit") != 1:
        raise FinalChallengeError("final frozen challenge limit must equal one")
    require_false(
        gate,
        (
            "candidate_graph_promoted",
            "ratifies_repair",
            "training_authorized",
            "promotion_authorized",
            "scale_authorized",
            "n0_ready",
            "n0_complete",
        ),
        "gate_receipt",
    )
    gate_source = require_mapping(gate.get("source"), "gate_receipt.source")
    if gate_source.get("arbitration_receipt_sha256") != result_sha:
        raise FinalChallengeError("gate receipt no longer binds arbitration result")

    if final.get("protocol_version") != FINALIZER_PROTOCOL_VERSION:
        raise FinalChallengeError("finalization protocol drift")
    if final.get("status") != "COMPLETE":
        raise FinalChallengeError("finalization receipt is not COMPLETE")
    if final.get("classification") != "IMPROVEMENT":
        raise FinalChallengeError("finalization classification drift")
    if final.get("decision") != EXPECTED_DECISION:
        raise FinalChallengeError("finalization decision drift")
    if final.get("final_frozen_challenge_authorized") is not True:
        raise FinalChallengeError("finalization did not preserve challenge authorization")
    if final.get("final_frozen_challenge_limit") != 1:
        raise FinalChallengeError("finalization challenge limit must equal one")
    invariants = require_mapping(final.get("invariants"), "finalization.invariants")
    require_false(
        invariants,
        (
            "training_enabled",
            "candidate_graph_promoted",
            "ratifies_repair",
            "promotion_authorized",
            "scale_authorized",
            "n0_ready",
            "n0_complete",
        ),
        "finalization.invariants",
    )

    frozen = require_mapping(final.get("frozen_inputs"), "finalization.frozen_inputs")
    frozen_result = require_mapping(
        frozen.get("arbitration_result"), "finalization.frozen_inputs.arbitration_result"
    )
    if frozen_result.get("sha256") != result_sha:
        raise FinalChallengeError("finalization no longer binds arbitration result")
    materialized = require_mapping(
        final.get("materialized_outputs"), "finalization.materialized_outputs"
    )
    frozen_gate = require_mapping(
        materialized.get("gate_receipt"),
        "finalization.materialized_outputs.gate_receipt",
    )
    if frozen_gate.get("sha256") != gate_sha:
        raise FinalChallengeError("finalization no longer binds gate receipt")

    common_inputs = require_mapping(result.get("common_inputs"), "arbitration.common_inputs")
    eval_config = require_mapping(
        common_inputs.get("evaluation_config"), "arbitration.common_inputs.evaluation_config"
    )
    if eval_config.get("sha256") != stack_sha:
        raise FinalChallengeError("full-stack manifest hash differs from arbitration")

    return {
        "source_revision": str(result.get("source_revision")),
        "experiment_id": str(result.get("experiment_id")),
        "result_path": result_path,
        "result_sha256": result_sha,
        "gate_path": gate_path,
        "gate_sha256": gate_sha,
        "finalization_path": final_path,
        "finalization_sha256": final_sha,
        "stack_path": stack_path,
        "stack_sha256": stack_sha,
        "stack": stack,
    }


def validate_stack_inputs(
    *,
    stack: Mapping[str, Any],
    challenge: Path,
    latent_checkpoint: Path,
    latent_config: Path,
    semantic_config: Path,
    semantic_checkpoint: Path,
    tokenizer_dir: Path,
    structured_config: Path,
    structured_checkpoint: Path,
    evidence_adapter: Path,
    fusion_checkpoint: Path,
    fusion_config: Path,
    fusion_ratification: Path,
) -> None:
    external = require_mapping(stack.get("common_external_inputs"), "stack.common_external_inputs")
    artifacts = require_mapping(stack.get("common_stack_artifacts"), "stack.common_stack_artifacts")

    eval_spec = require_mapping(external.get("evaluation_set"), "stack.evaluation_set")
    checked_file(challenge.resolve(), str(eval_spec.get("sha256")), "frozen challenge")

    latent_spec = require_mapping(external.get("latent_checkpoint"), "stack.latent_checkpoint")
    checked_file(
        latent_checkpoint.resolve(),
        str(latent_spec.get("sha256")),
        "latent checkpoint",
    )
    if str(latent_spec.get("sha256")) != EXPECTED_LATENT_SHA256:
        raise FinalChallengeError("unexpected latent checkpoint lineage")

    file_roles = {
        "latent_config": latent_config,
        "semantic_config": semantic_config,
        "structured_config": structured_config,
        "evidence_adapter": evidence_adapter,
        "fusion_config": fusion_config,
        "fusion_ratification": fusion_ratification,
    }
    tree_roles = {
        "semantic_checkpoint": semantic_checkpoint,
        "tokenizer_dir": tokenizer_dir,
        "structured_checkpoint": structured_checkpoint,
        "fusion_checkpoint": fusion_checkpoint,
    }
    for role, path in file_roles.items():
        record = require_mapping(artifacts.get(role), f"stack.{role}")
        checked_file(path.resolve(), str(record.get("sha256")), role)
    for role, path in tree_roles.items():
        record = require_mapping(artifacts.get(role), f"stack.{role}")
        checked_tree(path.resolve(), str(record.get("sha256")), role)


def validate_original_freeze(
    *,
    challenge_root: Path,
    spec: Path,
    evaluator: Path,
    latent_checkpoint: Path,
) -> dict[str, Any]:
    challenge = (challenge_root / "challenge.jsonl").resolve()
    manifest = (challenge_root / "manifest.json").resolve()
    freeze_path = (challenge_root / "freeze_receipt.json").resolve()

    for label, path in (
        ("challenge", challenge),
        ("challenge manifest", manifest),
        ("original freeze receipt", freeze_path),
        ("challenge spec", spec.resolve()),
        ("challenge evaluator", evaluator.resolve()),
        ("latent checkpoint", latent_checkpoint.resolve()),
    ):
        if not path.is_file():
            raise FinalChallengeError(f"{label} missing: {path}")

    freeze = require_mapping(load_json(freeze_path), "original_freeze_receipt")
    if freeze.get("status") != "FROZEN_CONFIRMATORY_BEFORE_EVALUATION":
        raise FinalChallengeError("original challenge freeze status drift")

    expected = (
        (challenge, "challenge_sha256"),
        (manifest, "manifest_sha256"),
        (spec.resolve(), "spec_sha256"),
        (evaluator.resolve(), "evaluator_sha256"),
        (latent_checkpoint.resolve(), "candidate_latent_pool_sha256"),
    )
    for path, key in expected:
        digest = str(freeze.get(key, "")).lower()
        if not digest:
            raise FinalChallengeError(f"original freeze receipt missing {key}")
        checked_file(path, digest, f"original frozen {key}")

    if freeze.get("candidate_latent_pool_sha256") != EXPECTED_LATENT_SHA256:
        raise FinalChallengeError("original freeze bound a different latent checkpoint")
    if freeze.get("candidate_preselected_before_challenge") is not True:
        raise FinalChallengeError("candidate was not preselected before challenge")
    if freeze.get("checkpoint_selection_on_challenge_forbidden") is not True:
        raise FinalChallengeError("challenge checkpoint-selection prohibition drift")
    if freeze.get("candidate_weights_changed_after_v0_1") is not False:
        raise FinalChallengeError("latent candidate changed after prior challenge")
    if freeze.get("counterfactual_intervention_point") != EXPECTED_INTERVENTION:
        raise FinalChallengeError("counterfactual intervention contract drift")
    if freeze.get("challenge_rows_used_for_training") is not False:
        raise FinalChallengeError("challenge rows were marked training-authorized")
    if freeze.get("gradient_performed") is not False:
        raise FinalChallengeError("original freeze receipt indicates a gradient")

    challenge_manifest = require_mapping(load_json(manifest), "challenge_manifest")
    if challenge_manifest.get("training_authorized") is not False:
        raise FinalChallengeError("challenge manifest training_authorized drift")
    if challenge_manifest.get("results_observed_at_compile_time") is not False:
        raise FinalChallengeError("challenge was not compiled blind to results")
    if challenge_manifest.get("counterfactual_intervention_point") != EXPECTED_INTERVENTION:
        raise FinalChallengeError("challenge manifest intervention-point drift")

    return {
        "challenge": challenge,
        "manifest": manifest,
        "freeze_path": freeze_path,
        "freeze": freeze,
    }


def run_final_challenge(
    *,
    repo_root: Path,
    arbitration_dir: Path,
    challenge_root: Path,
    spec: Path,
    evaluator: Path,
    latent_checkpoint: Path,
    latent_config: Path,
    semantic_config: Path,
    semantic_checkpoint: Path,
    tokenizer_dir: Path,
    structured_config: Path,
    structured_checkpoint: Path,
    evidence_adapter: Path,
    repaired_graph: Path,
    fusion_checkpoint: Path,
    fusion_config: Path,
    fusion_ratification: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FinalChallengeError(f"refusing to reuse output directory: {output_dir}")
    require_clean_tracked_worktree(repo_root)
    execution_revision = git_head(repo_root)

    authorization = validate_authorization(
        arbitration_dir=arbitration_dir.resolve(),
        repaired_graph=repaired_graph.resolve(),
    )
    original = validate_original_freeze(
        challenge_root=challenge_root.resolve(),
        spec=spec.resolve(),
        evaluator=evaluator.resolve(),
        latent_checkpoint=latent_checkpoint.resolve(),
    )
    validate_stack_inputs(
        stack=authorization["stack"],
        challenge=original["challenge"],
        latent_checkpoint=latent_checkpoint.resolve(),
        latent_config=latent_config.resolve(),
        semantic_config=semantic_config.resolve(),
        semantic_checkpoint=semantic_checkpoint.resolve(),
        tokenizer_dir=tokenizer_dir.resolve(),
        structured_config=structured_config.resolve(),
        structured_checkpoint=structured_checkpoint.resolve(),
        evidence_adapter=evidence_adapter.resolve(),
        fusion_checkpoint=fusion_checkpoint.resolve(),
        fusion_config=fusion_config.resolve(),
        fusion_ratification=fusion_ratification.resolve(),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    derived_freeze_path = output_dir / "derived_freeze_receipt.json"
    result_path = output_dir / "result.json"
    execution_receipt_path = output_dir / "execution_receipt.json"

    derived = copy.deepcopy(dict(original["freeze"]))
    derived["git_revision"] = execution_revision
    derived["derived_for_final_frozen_challenge"] = {
        "protocol_version": PROTOCOL_VERSION,
        "original_freeze_receipt_path": str(original["freeze_path"]),
        "original_freeze_receipt_sha256": sha256_file(original["freeze_path"]),
        "original_git_revision": original["freeze"].get("git_revision"),
        "execution_git_revision": execution_revision,
        "arbitration_source_revision": authorization["source_revision"],
        "arbitration_experiment_id": authorization["experiment_id"],
        "arbitration_result_sha256": authorization["result_sha256"],
        "post_arbitration_gate_receipt_sha256": authorization["gate_sha256"],
        "finalization_receipt_sha256": authorization["finalization_sha256"],
        "selected_repaired_graph_path": str(repaired_graph.resolve()),
        "selected_repaired_graph_sha256": EXPECTED_REPAIRED_GRAPH_SHA256,
        "thresholds_changed": False,
        "challenge_changed": False,
        "evaluator_changed": False,
        "latent_checkpoint_changed": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "scale_authorized": False,
        "n0_complete": False,
    }
    atomic_write_json(derived_freeze_path, derived)

    command = [
        sys.executable,
        str(evaluator.resolve()),
        "--challenge",
        str(original["challenge"]),
        "--manifest",
        str(original["manifest"]),
        "--spec",
        str(spec.resolve()),
        "--freeze-receipt",
        str(derived_freeze_path),
        "--candidate",
        str(latent_checkpoint.resolve()),
        "--latent-config",
        str(latent_config.resolve()),
        "--semantic-config",
        str(semantic_config.resolve()),
        "--semantic-checkpoint",
        str(semantic_checkpoint.resolve()),
        "--tokenizer-dir",
        str(tokenizer_dir.resolve()),
        "--structured-config",
        str(structured_config.resolve()),
        "--structured-checkpoint",
        str(structured_checkpoint.resolve()),
        "--evidence-adapter",
        str(evidence_adapter.resolve()),
        "--evidence-graph",
        str(repaired_graph.resolve()),
        "--fusion-checkpoint",
        str(fusion_checkpoint.resolve()),
        "--fusion-config",
        str(fusion_config.resolve()),
        "--fusion-ratification",
        str(fusion_ratification.resolve()),
        "--output",
        str(result_path),
    ]

    launch_receipt = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "AUTHORIZED_EXECUTION_START",
        "execution_git_revision": execution_revision,
        "arbitration_source_revision": authorization["source_revision"],
        "arbitration_experiment_id": authorization["experiment_id"],
        "arbitration_result_sha256": authorization["result_sha256"],
        "gate_receipt_sha256": authorization["gate_sha256"],
        "finalization_receipt_sha256": authorization["finalization_sha256"],
        "derived_freeze_receipt_sha256": sha256_file(derived_freeze_path),
        "selected_repaired_graph_sha256": EXPECTED_REPAIRED_GRAPH_SHA256,
        "latent_checkpoint_sha256": EXPECTED_LATENT_SHA256,
        "evaluator_sha256": sha256_file(evaluator.resolve()),
        "challenge_sha256": sha256_file(original["challenge"]),
        "spec_sha256": sha256_file(spec.resolve()),
        "evaluator_argv_sha256": hashlib.sha256(
            canonical_json_bytes(command)
        ).hexdigest(),
        "training_authorized": False,
        "promotion_authorized": False,
        "scale_authorized": False,
        "n0_complete": False,
    }
    atomic_write_json(output_dir / "launch_receipt.json", launch_receipt)

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        failure = dict(launch_receipt)
        failure.update(
            {
                "status": "EXECUTION_FAILED_NO_MODEL_CONCLUSION",
                "returncode": completed.returncode,
                "infrastructure_failure_is_not_model_evidence": True,
            }
        )
        atomic_write_json(execution_receipt_path, failure)
        raise FinalChallengeError(
            f"frozen evaluator exited with status {completed.returncode}"
        )

    result = require_mapping(load_json(result_path), "final_challenge_result")
    if result.get("git_revision") != execution_revision:
        raise FinalChallengeError("final challenge result source revision drift")
    if result.get("candidate_latent_pool_sha256") != EXPECTED_LATENT_SHA256:
        raise FinalChallengeError("final challenge changed latent checkpoint")
    if result.get("candidate_preselected_before_challenge") is not True:
        raise FinalChallengeError("final challenge violated preselection contract")
    if result.get("checkpoint_selection_performed_on_challenge") is not False:
        raise FinalChallengeError("final challenge selected checkpoint on challenge")
    if result.get("challenge_rows_used_for_training") is not False:
        raise FinalChallengeError("final challenge used challenge rows for training")
    if result.get("gradient_performed") is not False:
        raise FinalChallengeError("final challenge performed a gradient")
    if result.get("private_identity_data") is not False:
        raise FinalChallengeError("final challenge used private identity data")
    if result.get("private_identity_gradient") is not False:
        raise FinalChallengeError("final challenge used private identity gradient")
    if result.get("production_promotion_authorized") is not False:
        raise FinalChallengeError("evaluator unexpectedly authorized promotion")
    if result.get("n0_complete") is not False:
        raise FinalChallengeError("evaluator unexpectedly marked N0 complete")

    status = str(result.get("status"))
    passed = result.get("gate_pass")
    if status == "PASS_PRESELECTED_CANDIDATE_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION":
        if passed is not True:
            raise FinalChallengeError("PASS result has gate_pass != true")
        conclusion = "PASS_ELIGIBLE_FOR_RATIFICATION_DECISION"
    elif status == "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION":
        if passed is not False:
            raise FinalChallengeError("FAIL result has gate_pass != false")
        conclusion = "FAIL_STOP_AND_LOCALIZE_WITHOUT_AUTOMATIC_HOTFIX"
    else:
        raise FinalChallengeError(f"unexpected final challenge status: {status}")

    receipt = {
        **launch_receipt,
        "status": "COMPLETE",
        "model_conclusion": conclusion,
        "result_status": status,
        "gate_pass": passed,
        "result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "selected_repaired_graph": {
            "path": str(repaired_graph.resolve()),
            "sha256": EXPECTED_REPAIRED_GRAPH_SHA256,
            "promoted": False,
        },
        "invariants": {
            "diagnostic_only": True,
            "training_enabled": False,
            "gradient_performed": False,
            "candidate_graph_promoted": False,
            "ratifies_repair": False,
            "promotion_authorized": False,
            "scale_authorized": False,
            "private_identity_data": False,
            "private_identity_gradient": False,
            "n0_complete": False,
        },
        "next_action_from_frozen_evaluator": result.get("next_action"),
    }
    atomic_write_json(execution_receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--arbitration-dir", required=True, type=Path)
    parser.add_argument("--challenge-root", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--latent-checkpoint", required=True, type=Path)
    parser.add_argument("--latent-config", required=True, type=Path)
    parser.add_argument("--semantic-config", required=True, type=Path)
    parser.add_argument("--semantic-checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--structured-config", required=True, type=Path)
    parser.add_argument("--structured-checkpoint", required=True, type=Path)
    parser.add_argument("--evidence-adapter", required=True, type=Path)
    parser.add_argument("--repaired-graph", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--fusion-config", required=True, type=Path)
    parser.add_argument("--fusion-ratification", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        receipt = run_final_challenge(
            repo_root=args.repo_root,
            arbitration_dir=args.arbitration_dir,
            challenge_root=args.challenge_root,
            spec=args.spec,
            evaluator=args.evaluator,
            latent_checkpoint=args.latent_checkpoint,
            latent_config=args.latent_config,
            semantic_config=args.semantic_config,
            semantic_checkpoint=args.semantic_checkpoint,
            tokenizer_dir=args.tokenizer_dir,
            structured_config=args.structured_config,
            structured_checkpoint=args.structured_checkpoint,
            evidence_adapter=args.evidence_adapter,
            repaired_graph=args.repaired_graph,
            fusion_checkpoint=args.fusion_checkpoint,
            fusion_config=args.fusion_config,
            fusion_ratification=args.fusion_ratification,
            output_dir=args.output_dir,
        )
    except FinalChallengeError as exc:
        print(f"FINAL_FROZEN_CHALLENGE_ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
