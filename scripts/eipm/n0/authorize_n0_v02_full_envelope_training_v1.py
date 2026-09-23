#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alice_personality.n0.v02_training import (
    sha256_file,
    verify_public_corpus_v021,
    verify_teacher_registry,
)
from alice_personality.n0.source_authority_v1 import require_canonical_source_file
from train_n0_v02_full_envelope_joint_v1 import (
    current_git_revision,
    read_json,
    require_clean_worktree,
    verify_optimizer_lane_bindings,
    verify_pre_gradient_runtime,
)


STATUS="AUTHORIZED_N0_FULL_ENVELOPE_TRAINING_FROM_EXACT_RUNTIME_RECEIPTS"


def main() -> None:
    p=argparse.ArgumentParser(
        description=(
            "Create a one-way runtime authorization receipt for N0 successor "
            "gradient execution after all exact-head pre-gradient qualification "
            "has passed. This script performs no training and does not open FINAL."
        )
    )
    p.add_argument("--training-plan",required=True)
    p.add_argument("--topology-config",required=True)
    p.add_argument("--semantic-config",required=True)
    p.add_argument("--semantic-checkpoint",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--source-config",required=True)
    p.add_argument("--corpus-dir",required=True)
    p.add_argument("--teacher-registry",required=True)
    p.add_argument("--teacher-audit",required=True)
    p.add_argument("--mixture-manifest",required=True)
    p.add_argument("--mixture-audit",required=True)
    p.add_argument("--tokenizer-audit",required=True)
    p.add_argument("--operator-evidence-token-receipt",required=True)
    p.add_argument("--cpu-runtime-receipt",required=True)
    p.add_argument("--gpu-memory-receipt",required=True)
    p.add_argument("--long-boundary-receipt",required=True)
    p.add_argument("--semantic-long-token-receipt",required=True)
    p.add_argument("--static-proof-receipt",required=True)
    p.add_argument("--semantic-rows",required=True)
    p.add_argument("--semantic-long-rows",required=True)
    p.add_argument("--behavioral-rows",required=True)
    p.add_argument("--runtime-view-rows",required=True)
    p.add_argument("--long-context-rows",required=True)
    p.add_argument("--natural-rows",required=True)
    p.add_argument("--natural-bank",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite runtime training authorization")

    require_clean_worktree()
    revision=current_git_revision()

    require_canonical_source_file(
        args.training_plan,
        "configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json",
        label="joint training plan",
    )
    plan=read_json(args.training_plan)
    if plan.get("schema")!="alice.eipm.n0.semantic-operator-joint-training-plan.v1":
        raise SystemExit("runtime training authorization plan schema drift")
    authority=dict(plan.get("authorization") or {})
    for name in ("optimizer","gradient","gpu_training"):
        if authority.get(name) is not False:
            raise SystemExit(
                "source training authority must remain false before runtime authorization"
            )
    if authority.get("runtime_training_authorization_receipt_required") is not True:
        raise SystemExit("training plan does not require runtime authorization receipt")
    if authority.get("final_validation_opening") is not False:
        raise SystemExit("training authorization may not pre-open FINAL")
    if authority.get("private_identity_gradient") is not False:
        raise SystemExit("training authorization may not enable private identity gradient")

    corpus_dir=Path(args.corpus_dir).resolve()
    mixture,mixture_audit=verify_pre_gradient_runtime(
        mixture_manifest_path=args.mixture_manifest,
        mixture_audit_path=args.mixture_audit,
        tokenizer_audit_path=args.tokenizer_audit,
        operator_evidence_token_receipt_path=args.operator_evidence_token_receipt,
        tokenizer_dir_path=args.tokenizer_dir,
        topology_config_path=args.topology_config,
        semantic_config_path=args.semantic_config,
        semantic_checkpoint_path=args.semantic_checkpoint,
        source_config_path=args.source_config,
        corpus_receipt_path=corpus_dir/"corpus_receipt.json",
        cpu_runtime_receipt_path=args.cpu_runtime_receipt,
        gpu_memory_receipt_path=args.gpu_memory_receipt,
        long_boundary_receipt_path=args.long_boundary_receipt,
        semantic_long_token_receipt_path=args.semantic_long_token_receipt,
        static_proof_receipt_path=args.static_proof_receipt,
    )
    verify_optimizer_lane_bindings(
        mixture=mixture,
        semantic_rows=args.semantic_rows,
        semantic_long_rows=args.semantic_long_rows,
        behavioral_rows=args.behavioral_rows,
        runtime_view_rows=args.runtime_view_rows,
        long_context_rows=args.long_context_rows,
        natural_rows=args.natural_rows,
        natural_bank=args.natural_bank,
        source_config=args.source_config,
        corpus_receipt=corpus_dir/"corpus_receipt.json",
        teacher_registry=args.teacher_registry,
        teacher_audit=args.teacher_audit,
    )

    # Re-run the live public-corpus and governed teacher-bank verifiers at the
    # authorization boundary instead of trusting only previously written PASS
    # receipts. These are P41-style runtime revalidations, not model training.
    corpus_receipt,_=verify_public_corpus_v021(corpus_dir,args.source_config)
    _,teacher_report=verify_teacher_registry(
        Path(__file__).resolve().parents[3],
        args.teacher_registry,
        args.teacher_audit,
    )
    if mixture.get("source_revision")!=revision:
        raise SystemExit("runtime training authorization source revision drift")
    if corpus_receipt.get("status")!="PASS":
        raise SystemExit("runtime training authorization public corpus revalidation failed")
    if teacher_report.get("status")!="PASS":
        raise SystemExit("runtime training authorization teacher-bank revalidation failed")

    result={
        "schema":"alice.eipm.n0.full-envelope-training-authorization.v1",
        "status":STATUS,
        "source_revision":revision,
        "training_authorizer_sha256":sha256_file(Path(__file__).resolve()),
        "training_plan_sha256":sha256_file(args.training_plan),
        "topology_config_sha256":sha256_file(args.topology_config),
        "semantic_config_sha256":sha256_file(args.semantic_config),
        "semantic_checkpoint_sha256":sha256_file(args.semantic_checkpoint),
        "tokenizer_json_sha256":sha256_file(
            Path(args.tokenizer_dir)/"tokenizer.json"
        ),
        "source_config_sha256":sha256_file(args.source_config),
        "corpus_receipt_sha256":sha256_file(
            corpus_dir/"corpus_receipt.json"
        ),
        "teacher_registry_sha256":sha256_file(args.teacher_registry),
        "teacher_audit_sha256":sha256_file(args.teacher_audit),
        "mixture_manifest_sha256":sha256_file(args.mixture_manifest),
        "mixture_audit_sha256":sha256_file(args.mixture_audit),
        "tokenizer_audit_sha256":sha256_file(args.tokenizer_audit),
        "operator_evidence_token_receipt_sha256":sha256_file(
            args.operator_evidence_token_receipt
        ),
        "cpu_runtime_receipt_sha256":sha256_file(args.cpu_runtime_receipt),
        "gpu_memory_receipt_sha256":sha256_file(args.gpu_memory_receipt),
        "long_boundary_receipt_sha256":sha256_file(args.long_boundary_receipt),
        "semantic_long_token_receipt_sha256":sha256_file(
            args.semantic_long_token_receipt
        ),
        "static_proof_receipt_sha256":sha256_file(args.static_proof_receipt),
        "optimizer_lane_bindings_verified":True,
        "public_corpus_runtime_revalidated":True,
        "teacher_bank_runtime_revalidated":True,
        "authorized_stages":[
            "J1_joint_semantic_operator",
            "J2_reopened_representation_interfaces",
            "J3_full_public_n0_coadaptation",
        ],
        "optimizer":True,
        "gradient":True,
        "gpu_training":True,
        "automatic_stage_transition":False,
        "final_results_observed":False,
        "final_opening_authorized":False,
        "private_identity_data":False,
        "private_identity_gradient":False,
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
