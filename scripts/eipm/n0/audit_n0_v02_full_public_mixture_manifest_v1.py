#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PASS="PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--contract",required=True)
    p.add_argument("--source-revision",required=True)
    p.add_argument("--source-config",required=True)
    p.add_argument("--corpus-receipt",required=True)
    p.add_argument("--teacher-registry",required=True)
    p.add_argument("--teacher-audit",required=True)
    p.add_argument("--semantic-rows",required=True)
    p.add_argument("--semantic-manifest",required=True)
    p.add_argument("--semantic-audit",required=True)
    p.add_argument("--behavioral-rows",required=True)
    p.add_argument("--behavioral-manifest",required=True)
    p.add_argument("--behavioral-audit",required=True)
    p.add_argument("--runtime-view-rows",required=True)
    p.add_argument("--runtime-view-manifest",required=True)
    p.add_argument("--runtime-view-audit",required=True)
    p.add_argument("--long-context-rows",required=True)
    p.add_argument("--long-context-manifest",required=True)
    p.add_argument("--long-context-audit",required=True)
    p.add_argument("--fewrel-train-dev-rows",required=True)
    p.add_argument("--fewrel-train-dev-bank",required=True)
    p.add_argument("--fewrel-manifest",required=True)
    p.add_argument("--fewrel-audit",required=True)
    p.add_argument("--final-freeze-receipt",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite full public-mixture audit")
    manifest=load_json(Path(args.manifest))
    contract=load_json(Path(args.contract))
    errors=[]

    if manifest.get("schema")!="alice.eipm.n0.full-public-mixture-manifest.v1":
        errors.append("mixture manifest schema drift")
    if manifest.get("status")!="MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT":
        errors.append("mixture manifest status drift")
    if contract.get("schema")!="alice.eipm.n0.full-public-mixture-contract.v1":
        errors.append("mixture contract schema drift")
    if manifest.get("contract_sha256")!=sha256(Path(args.contract)):
        errors.append("mixture contract hash drift")
    if manifest.get("source_revision")!=str(args.source_revision):
        errors.append("source revision drift")

    lanes=manifest.get("training_lanes") or {}
    required=list(contract.get("required_training_lanes") or [])
    if list(manifest.get("required_training_lanes") or [])!=required:
        errors.append("required lane ordering drift")
    if set(lanes)!=set(required):
        errors.append("mixture lane set drift")

    expected_lane_hashes={
        "broad_semantic_replay":{
            "source_config_sha256":sha256(Path(args.source_config)),
            "corpus_receipt_sha256":sha256(Path(args.corpus_receipt)),
        },
        "governed_judgment_replay":{
            "teacher_registry_sha256":sha256(Path(args.teacher_registry)),
            "teacher_audit_sha256":sha256(Path(args.teacher_audit)),
        },
        "semantic_operator_intervention":{
            "rows_sha256":sha256(Path(args.semantic_rows)),
            "manifest_sha256":sha256(Path(args.semantic_manifest)),
            "audit_sha256":sha256(Path(args.semantic_audit)),
        },
        "full_envelope_behavioral":{
            "rows_sha256":sha256(Path(args.behavioral_rows)),
            "manifest_sha256":sha256(Path(args.behavioral_manifest)),
            "audit_sha256":sha256(Path(args.behavioral_audit)),
        },
        "runtime_view_supplement":{
            "rows_sha256":sha256(Path(args.runtime_view_rows)),
            "manifest_sha256":sha256(Path(args.runtime_view_manifest)),
            "audit_sha256":sha256(Path(args.runtime_view_audit)),
        },
        "long_context_supplement":{
            "rows_sha256":sha256(Path(args.long_context_rows)),
            "manifest_sha256":sha256(Path(args.long_context_manifest)),
            "audit_sha256":sha256(Path(args.long_context_audit)),
        },
        "natural_relation":{
            "rows_sha256":sha256(Path(args.fewrel_train_dev_rows)),
            "bank_sha256":sha256(Path(args.fewrel_train_dev_bank)),
            "manifest_sha256":sha256(Path(args.fewrel_manifest)),
            "audit_sha256":sha256(Path(args.fewrel_audit)),
        },
    }
    for lane,hashes in expected_lane_hashes.items():
        actual=lanes.get(lane) or {}
        for key,value in hashes.items():
            if actual.get(key)!=value:
                errors.append(f"{lane} hash drift: {key}")
        if actual.get("private_identity_data") is not False:
            errors.append(f"{lane} private identity authority drift")
        if int(actual.get("final_rows",0))!=0:
            errors.append(f"{lane} contains FINAL rows")

    family_coverage=manifest.get("macro_family_coverage") or {}
    required_families=set(map(str,contract.get("required_macro_families") or []))
    if set(family_coverage)!=required_families:
        errors.append("macro-family coverage set drift")
    for family in sorted(required_families):
        owners=list(family_coverage.get(family) or [])
        if not owners:
            errors.append(f"macro family lacks training lane: {family}")
        if not set(owners)<=set(required):
            errors.append(f"macro family has undeclared lane: {family}")

    if manifest.get("final_freeze_receipt_sha256")!=sha256(Path(args.final_freeze_receipt)):
        errors.append("FINAL freeze receipt hash drift")
    freeze=load_json(Path(args.final_freeze_receipt))
    if freeze.get("results_observed") is not False:
        errors.append("FINAL results observed")
    if freeze.get("final_opening_authorized") is not False:
        errors.append("FINAL opening authorized")

    forbidden_true=(
        "private_identity_data",
        "private_identity_gradient_authorized",
        "gradient",
        "optimizer",
        "gpu_training_authorized",
        "n0_complete",
        "final_results_observed",
    )
    for key in forbidden_true:
        if manifest.get(key) is not False:
            errors.append(f"mixture manifest must keep {key}=false")
    if int(manifest.get("final_rows_in_training",-1))!=0:
        errors.append("FINAL rows entered training")
    if int(manifest.get("final_rows_in_model_selection",-1))!=0:
        errors.append("FINAL rows entered model selection")
    if manifest.get("row_count_is_capability_ceiling") is not False:
        errors.append("row count became capability ceiling")
    if manifest.get("fixed_lane_sampling_ratio_as_capability_definition") is not False:
        errors.append("lane sampling ratio became capability definition")

    result={
        "schema":"alice.eipm.n0.full-public-mixture-manifest-audit.v1",
        "status":PASS if not errors else "FAIL_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1",
        "errors":errors,
        "source_revision":str(args.source_revision),
        "training_lanes":required,
        "macro_families":sorted(required_families),
        "lane_count":len(required),
        "macro_family_count":len(required_families),
        "final_results_observed":False,
        "final_rows_in_training":0,
        "private_identity_data":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "n0_complete":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
