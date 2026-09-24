#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PASS_DEV="PASS_DEV_STAGE_GATE"
PASS_FREEZE="FROZEN_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_BEFORE_GRADIENT"
J3="J3_full_public_n0_coadaptation"
STATUS="AUTHORIZED_N0_FULL_ENVELOPE_FINAL_V2_OPENING_FROM_DEV_SELECTED_J3"


def read_json(path: str | Path) -> dict[str,Any]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_revision() -> str:
    return subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True
    ).strip()


def require_clean_tracked_worktree() -> None:
    status=subprocess.check_output(
        ["git","status","--porcelain"],text=True
    )
    if status.strip():
        raise SystemExit(
            "FINAL opening authorizer requires a clean exact-source worktree"
        )


def main() -> None:
    p=argparse.ArgumentParser(
        description=(
            "Create the one-way authorization receipt that opens sealed N0 "
            "FINAL-v2 only for an already DEV-qualified J3 candidate. "
            "This script performs no candidate ranking, training, repair, or FINAL evaluation."
        )
    )
    p.add_argument("--candidate-system",required=True)
    p.add_argument("--candidate-receipt",required=True)
    p.add_argument("--dev-receipt",required=True)
    p.add_argument("--selection-receipt",required=True)
    p.add_argument("--freeze-receipt",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite FINAL opening authorization")

    require_clean_tracked_worktree()
    revision=current_git_revision()

    candidate=read_json(args.candidate_receipt)
    dev=read_json(args.dev_receipt)
    selection=read_json(args.selection_receipt)
    freeze=read_json(args.freeze_receipt)

    candidate_system_sha256=sha256_file(args.candidate_system)
    candidate_receipt_sha256=sha256_file(args.candidate_receipt)
    dev_receipt_sha256=sha256_file(args.dev_receipt)
    selection_receipt_sha256=sha256_file(args.selection_receipt)
    freeze_receipt_sha256=sha256_file(args.freeze_receipt)

    if candidate.get("schema")!="alice.eipm.n0.full-envelope-checkpoint-receipt.v1":
        raise SystemExit("FINAL opening candidate checkpoint receipt schema drift")
    if candidate.get("status")!="TRAINED_PUBLIC_N0_CANDIDATE_REQUIRES_DEV_SELECTION":
        raise SystemExit("FINAL opening candidate checkpoint receipt status drift")
    if candidate.get("stage")!=J3:
        raise SystemExit("FINAL opening requires a J3 candidate")
    if candidate.get("source_revision")!=revision:
        raise SystemExit("FINAL opening candidate source revision drift")
    if candidate.get("full_system_sha256")!=candidate_system_sha256:
        raise SystemExit("FINAL opening candidate system hash drift")
    if candidate.get("final_results_observed") is not False:
        raise SystemExit("FINAL opening candidate already observed FINAL")
    if candidate.get("final_opening_authorized") is not False:
        raise SystemExit("candidate receipt already authorizes FINAL")
    if candidate.get("private_identity_data") is not False:
        raise SystemExit("private identity data forbidden in N0 FINAL opening")

    if dev.get("schema")!="alice.eipm.n0.full-envelope-dev-evaluation.v1":
        raise SystemExit("FINAL opening DEV receipt schema drift")
    dev_evaluator=Path(__file__).resolve().with_name(
        "evaluate_n0_v02_full_envelope_dev_v1.py"
    )
    if dev.get("dev_evaluator_sha256")!=sha256_file(dev_evaluator):
        raise SystemExit("FINAL opening DEV evaluator implementation hash drift")
    if dev.get("status")!=PASS_DEV:
        raise SystemExit("FINAL opening requires PASS_DEV_STAGE_GATE")
    if dev.get("stage")!=J3:
        raise SystemExit("FINAL opening DEV receipt is not J3")
    if dev.get("source_revision")!=revision:
        raise SystemExit("FINAL opening DEV source revision drift")
    if dev.get("checkpoint_selection_surface")!="DEV_ONLY":
        raise SystemExit("FINAL opening requires DEV_ONLY checkpoint selection surface")
    if dev.get("stage_gate_pass") is not True:
        raise SystemExit("FINAL opening DEV stage gate not passed")
    if dev.get("stage_gate_coverage_complete") is not True:
        raise SystemExit("FINAL opening DEV gate coverage incomplete")
    if dev.get("registry_matches_declared_stage_gates") is not True:
        raise SystemExit("FINAL opening DEV gate mapping incomplete")
    if dev.get("final_results_observed") is not False:
        raise SystemExit("FINAL opening DEV receipt observed FINAL")
    if dev.get("final_opening_authorized") is not False:
        raise SystemExit("DEV receipt may not itself authorize FINAL")
    if dev.get("candidate_checkpoint_receipt_sha256")!=candidate_receipt_sha256:
        raise SystemExit("DEV selection/candidate checkpoint receipt drift")
    if dev.get("candidate_system_sha256")!=candidate_system_sha256:
        raise SystemExit("DEV selection/candidate system drift")
    if dev.get("training_authorization_sha256")!=candidate.get(
        "training_authorization_sha256"
    ):
        raise SystemExit("FINAL opening DEV/training authorization lineage drift")

    if selection.get("schema")!="alice.eipm.n0.full-envelope-dev-checkpoint-selection.v1":
        raise SystemExit("FINAL opening selection receipt drift: schema")
    if selection.get("status")!="SELECTED_FIRST_PASSING_N0_DEV_CHECKPOINT":
        raise SystemExit("FINAL opening selection receipt drift: status")
    if selection.get("source_revision")!=revision or selection.get("stage")!=J3:
        raise SystemExit("FINAL opening selection receipt drift: source/stage")
    if selection.get("first_passing_checkpoint") is not True:
        raise SystemExit("FINAL opening selection receipt drift: not first passing")
    if selection.get("selected_checkpoint_receipt_sha256")!=candidate_receipt_sha256:
        raise SystemExit("FINAL opening selection receipt drift: checkpoint")
    if selection.get("selected_system_sha256")!=candidate_system_sha256:
        raise SystemExit("FINAL opening selection receipt drift: system")
    if selection.get("selected_dev_receipt_sha256")!=dev_receipt_sha256:
        raise SystemExit("FINAL opening selection receipt drift: DEV")
    if selection.get("selected_dev_evaluator_sha256")!=dev.get(
        "dev_evaluator_sha256"
    ):
        raise SystemExit("FINAL opening selection receipt drift: DEV evaluator")
    selector=Path(__file__).resolve().with_name(
        "select_n0_v02_full_envelope_dev_checkpoint_v1.py"
    )
    if selection.get("selection_authorizer_sha256")!=sha256_file(selector):
        raise SystemExit("FINAL opening selection receipt drift: selector")

    if freeze.get("schema")!="alice.eipm.n0.full-envelope-final-v2-freeze-receipt.v1":
        raise SystemExit("FINAL opening freeze receipt schema drift")
    if freeze.get("status")!=PASS_FREEZE:
        raise SystemExit("FINAL opening requires valid pre-gradient freeze receipt")
    if freeze.get("source_revision")!=revision:
        raise SystemExit("FINAL opening freeze source revision drift")
    if freeze.get("results_observed") is not False:
        raise SystemExit("FINAL freeze already observed results")
    if freeze.get("final_opening_authorized") is not False:
        raise SystemExit("freeze receipt unexpectedly authorizes FINAL")

    opening_authorizer_sha256=sha256_file(Path(__file__).resolve())
    expected_authorizer_sha256=freeze.get("hashes",{}).get("opening_authorizer")
    if opening_authorizer_sha256!=expected_authorizer_sha256:
        raise SystemExit("frozen FINAL opening authorizer hash drift")

    result={
        "schema":"alice.eipm.n0.full-envelope-final-opening-authorization.v1",
        "status":STATUS,
        "source_revision":revision,
        "selected_stage":J3,
        "checkpoint_selection_surface":"DEV_ONLY",
        "dev_selection_pass":True,
        "stage_gate_pass":True,
        "stage_gate_coverage_complete":True,
        "registry_matches_declared_stage_gates":True,
        "candidate_checkpoint_sha256":candidate_system_sha256,
        "candidate_system_sha256":candidate_system_sha256,
        "candidate_checkpoint_receipt_sha256":candidate_receipt_sha256,
        "dev_receipt_sha256":dev_receipt_sha256,
        "dev_evaluator_sha256":dev["dev_evaluator_sha256"],
        "training_authorization_sha256":candidate.get(
            "training_authorization_sha256"
        ),
        "selection_receipt_sha256":selection_receipt_sha256,
        "freeze_receipt_sha256":freeze_receipt_sha256,
        "opening_authorizer_sha256":opening_authorizer_sha256,
        "automatic_checkpoint_selection":False,
        "automatic_stage_transition":False,
        "automatic_repair_or_rerun":False,
        "threshold_changes_after_results":False,
        "training_authorized":False,
        "model_selection_after_final":False,
        "final_results_observed":False,
        "final_opening_authorized":True,
        "private_identity_data":False,
        "n0_complete":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
