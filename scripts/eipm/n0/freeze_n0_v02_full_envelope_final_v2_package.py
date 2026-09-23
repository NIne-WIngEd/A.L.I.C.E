#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from alice_personality.n0.source_authority_v1 import require_canonical_source_file


STATUS="FROZEN_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_BEFORE_GRADIENT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source-revision",required=True)
    p.add_argument("--package-config",required=True)
    p.add_argument("--evaluator-contract",required=True)
    p.add_argument("--evaluator-implementation",required=True)
    p.add_argument("--gate-registry",required=True)
    p.add_argument("--opening-authorizer",required=True)
    p.add_argument("--final-contract",required=True)
    p.add_argument("--synthetic-final-rows",required=True)
    p.add_argument("--semantic-final-rows",required=True)
    p.add_argument("--runtime-view-final-rows",required=True)
    p.add_argument("--long-context-final-rows",required=True)
    p.add_argument("--package-manifest",required=True)
    p.add_argument("--fewrel-final-rows",required=True)
    p.add_argument("--fewrel-final-bank",required=True)
    p.add_argument("--fewrel-manifest",required=True)
    p.add_argument("--audit",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    source_revision=str(args.source_revision).strip().lower()
    if len(source_revision)!=40 or any(ch not in "0123456789abcdef" for ch in source_revision):
        raise SystemExit("FINAL-v2 freeze source revision must be exact 40-hex")
    status=subprocess.check_output(["git","status","--porcelain"],text=True)
    if status.strip():
        raise SystemExit("FINAL-v2 freeze requires a clean exact-source worktree")
    current_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip().lower()
    if current_revision!=source_revision:
        raise SystemExit("FINAL-v2 freeze source revision drift")

    paths={name:Path(getattr(args,name)) for name in (
        "package_config","evaluator_contract","evaluator_implementation","gate_registry","opening_authorizer","final_contract","synthetic_final_rows","semantic_final_rows",
        "runtime_view_final_rows","long_context_final_rows",
        "package_manifest","fewrel_final_rows","fewrel_final_bank","fewrel_manifest","audit","output"
    )}
    canonical_source_inputs={
        "package_config":"configs/eipm/n0/n0_v02_full_envelope_final_package_v1.json",
        "evaluator_contract":"configs/eipm/n0/n0_v02_full_envelope_final_v2_evaluator_contract_v1.json",
        "evaluator_implementation":"scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py",
        "gate_registry":"configs/eipm/n0/n0_v02_full_envelope_final_v2_gate_registry_v1.json",
        "opening_authorizer":"scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py",
        "final_contract":"configs/eipm/n0/n0_v02_full_envelope_final_validation_contract_v2.json",
    }
    for name,relative in canonical_source_inputs.items():
        require_canonical_source_file(paths[name],relative,label=f"FINAL-v2 {name}")
    if paths["output"].exists():
        raise SystemExit("refusing to overwrite final-v2 freeze receipt")
    audit=json.loads(paths["audit"].read_text(encoding="utf-8"))
    package=json.loads(paths["package_config"].read_text(encoding="utf-8"))
    evaluator=json.loads(paths["evaluator_contract"].read_text(encoding="utf-8"))
    manifest=json.loads(paths["package_manifest"].read_text(encoding="utf-8"))
    package=json.loads(paths["package_config"].read_text(encoding="utf-8"))
    if package.get("evaluator_implementation")!="scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py":
        raise SystemExit("package evaluator implementation binding drift")
    if package.get("evaluator_gate_registry")!="configs/eipm/n0/n0_v02_full_envelope_final_v2_gate_registry_v1.json":
        raise SystemExit("package evaluator gate-registry binding drift")
    if package.get("opening_authorizer")!="scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py":
        raise SystemExit("package opening-authorizer binding drift")
    if evaluator.get("evaluator_implementation")!=package.get("evaluator_implementation"):
        raise SystemExit("package/evaluator implementation binding mismatch")
    if evaluator.get("gate_registry")!=package.get("evaluator_gate_registry"):
        raise SystemExit("package/evaluator gate-registry binding mismatch")
    gate_registry=json.loads(paths["gate_registry"].read_text(encoding="utf-8"))
    if gate_registry.get("schema")!="alice.eipm.n0.full-envelope-final-v2-gate-registry.v1":
        raise SystemExit("FINAL-v2 gate registry schema drift")
    if gate_registry.get("results_observed") is not False:
        raise SystemExit("FINAL-v2 gate registry already observed results")
    if audit.get("status")!="PASS_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_AUDIT_V1":
        raise SystemExit("cannot freeze final-v2 package without audit PASS")
    if any([
        audit.get("results_observed") is not False,
        package.get("results_observed") is not False,
        evaluator.get("results_observed") is not False,
        manifest.get("results_observed") is not False,
    ]):
        raise SystemExit("cannot freeze package after observing candidate results")
    if evaluator.get("final_opening_authorized") is not False:
        raise SystemExit("evaluator contract unexpectedly authorizes final opening")

    receipt={
        "schema":"alice.eipm.n0.full-envelope-final-v2-freeze-receipt.v1",
        "status":STATUS,
        "source_revision":source_revision,
        "hashes":{
            name:sha256(path)
            for name,path in paths.items()
            if name!="output"
        },
        "registered_system":"N0FullEnvelopeTrainableSystemV1",
        "legacy_final_v1_authority":False,
        "legacy_final_v3_evaluator_authority":False,
        "results_observed":False,
        "training_authorized":False,
        "model_selection_authorized":False,
        "final_opening_authorized":False,
        "gradient_performed":False,
        "optimizer_performed":False,
        "private_identity_data":False,
        "n0_complete":False,
    }
    paths["output"].parent.mkdir(parents=True,exist_ok=True)
    paths["output"].write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
