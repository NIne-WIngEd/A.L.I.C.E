#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PASS_DEV="PASS_DEV_STAGE_GATE"
STATUS="SELECTED_FIRST_PASSING_N0_DEV_CHECKPOINT"


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
        ["git","status","--porcelain","--untracked-files=no"],text=True
    )
    if status.strip():
        raise SystemExit("DEV selector requires a clean tracked-source worktree")


def main() -> None:
    p=argparse.ArgumentParser(
        description=(
            "Authorize the precommitted first DEV-passing checkpoint in one "
            "stage-local chained checkpoint history. FINAL is never opened here."
        )
    )
    p.add_argument("--stage",required=True)
    p.add_argument("--checkpoint-root",required=True)
    p.add_argument("--candidate-receipt",required=True)
    p.add_argument("--dev-results-dir",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite DEV checkpoint selection receipt")
    require_clean_tracked_worktree()
    revision=current_git_revision()

    checkpoint_root=Path(args.checkpoint_root).resolve()
    dev_root=Path(args.dev_results_dir).resolve()
    candidate_path=Path(args.candidate_receipt).resolve()
    if not checkpoint_root.is_dir():
        raise SystemExit("checkpoint root missing")
    if not dev_root.is_dir():
        raise SystemExit("DEV results directory missing")

    checkpoints={}
    for path in sorted(checkpoint_root.rglob("receipt.json")):
        try:
            receipt=read_json(path)
        except Exception:
            continue
        if receipt.get("schema")!="alice.eipm.n0.full-envelope-checkpoint-receipt.v1":
            continue
        if receipt.get("stage")!=args.stage:
            continue
        if receipt.get("source_revision")!=revision:
            continue
        digest=sha256_file(path)
        if digest in checkpoints:
            raise SystemExit("duplicate checkpoint receipt hash")
        checkpoints[digest]=(path,receipt)

    candidate_hash=sha256_file(candidate_path)
    if candidate_hash not in checkpoints:
        raise SystemExit("candidate receipt is not in exact stage checkpoint root")
    if checkpoints[candidate_hash][0] != candidate_path:
        raise SystemExit("candidate checkpoint receipt path/hash alias drift")

    dev_by_checkpoint={}
    for path in sorted(dev_root.rglob("*.json")):
        try:
            receipt=read_json(path)
        except Exception:
            continue
        if receipt.get("schema")!="alice.eipm.n0.full-envelope-dev-evaluation.v1":
            continue
        checkpoint_hash=str(receipt.get("candidate_checkpoint_receipt_sha256",""))
        if not checkpoint_hash:
            continue
        if checkpoint_hash in dev_by_checkpoint:
            raise SystemExit("multiple DEV receipts for one checkpoint chain member")
        dev_by_checkpoint[checkpoint_hash]=(path,receipt)

    chain=[]
    seen=set()
    current=candidate_hash
    while current is not None:
        if current in seen:
            raise SystemExit("checkpoint parent chain cycle")
        seen.add(current)
        item=checkpoints.get(current)
        if item is None:
            raise SystemExit("checkpoint parent chain member missing from checkpoint root")
        path,receipt=item
        chain.append((current,path,receipt))
        parent=receipt.get("stage_checkpoint_parent_receipt_sha256")
        current=str(parent) if parent else None
    chain.reverse()

    last_step=0
    evaluated=[]
    for checkpoint_hash,path,receipt in chain:
        step=int(receipt.get("optimizer_step",0))
        if step<=last_step:
            raise SystemExit("checkpoint chain optimizer steps are not strictly increasing")
        last_step=step
        dev_item=dev_by_checkpoint.get(checkpoint_hash)
        if dev_item is None:
            raise SystemExit("missing DEV receipt for checkpoint chain member")
        dev_path,dev=dev_item
        if dev.get("stage")!=args.stage:
            raise SystemExit("DEV receipt stage drift in checkpoint chain")
        if dev.get("source_revision")!=revision:
            raise SystemExit("DEV receipt source revision drift in checkpoint chain")
        if dev.get("checkpoint_selection_surface")!="DEV_ONLY":
            raise SystemExit("DEV receipt selection surface drift")
        if dev.get("candidate_system_sha256")!=receipt.get("full_system_sha256"):
            raise SystemExit("DEV receipt/checkpoint system hash drift")
        if dev.get("final_results_observed") is not False:
            raise SystemExit("DEV checkpoint chain observed FINAL")
        passed=(
            dev.get("status")==PASS_DEV
            and dev.get("stage_gate_pass") is True
            and dev.get("stage_gate_coverage_complete") is True
            and dev.get("registry_matches_declared_stage_gates") is True
        )
        evaluated.append({
            "checkpoint_hash":checkpoint_hash,
            "checkpoint_path":str(path),
            "dev_path":str(dev_path),
            "dev_sha256":sha256_file(dev_path),
            "optimizer_step":step,
            "passed":bool(passed),
        })

    if not evaluated or evaluated[-1]["checkpoint_hash"]!=candidate_hash:
        raise SystemExit("candidate is not checkpoint-chain head")
    if not evaluated[-1]["passed"]:
        raise SystemExit("candidate checkpoint does not pass DEV stage gate")
    earlier=[item for item in evaluated[:-1] if item["passed"]]
    if earlier:
        raise SystemExit(
            "earlier passing checkpoint exists; candidate violates first-pass selection"
        )

    candidate_receipt=checkpoints[candidate_hash][1]
    selected_dev=dev_by_checkpoint[candidate_hash]
    selector_sha256=sha256_file(Path(__file__).resolve())
    result={
        "schema":"alice.eipm.n0.full-envelope-dev-checkpoint-selection.v1",
        "status":STATUS,
        "source_revision":revision,
        "stage":args.stage,
        "checkpoint_selection_surface":"DEV_ONLY",
        "first_passing_checkpoint":True,
        "selected_checkpoint_receipt_sha256":candidate_hash,
        "selected_system_sha256":candidate_receipt["full_system_sha256"],
        "selected_dev_receipt_sha256":sha256_file(selected_dev[0]),
        "selected_optimizer_step":int(candidate_receipt["optimizer_step"]),
        "selection_authorizer_sha256":selector_sha256,
        "checkpoint_chain_length":len(evaluated),
        "checkpoint_chain_evaluated_completely":True,
        "automatic_stage_transition":False,
        "final_results_observed":False,
        "final_opening_authorized":False,
        "private_identity_data":False,
        "n0_complete":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
