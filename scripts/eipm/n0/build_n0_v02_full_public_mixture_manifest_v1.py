#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA="alice.eipm.n0.full-public-mixture-manifest.v1"
PASS_STATUS="MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT"

EXPECTED={
    "semantic":{
        "manifest_schema":"alice.eipm.n0.semantic-operator-intervention-manifest.v1",
        "audit_status":"PASS_SEMANTIC_OPERATOR_CURRICULUM_AUDIT",
    },
    "behavioral":{
        "manifest_schema":"alice.eipm.n0.full-envelope-behavioral-manifest.v1",
        "audit_status":"PASS_N0_FULL_ENVELOPE_BEHAVIORAL_CURRICULUM_AUDIT_V1",
    },
    "runtime_view":{
        "manifest_schema":"alice.eipm.n0.full-envelope-runtime-view-manifest.v1",
        "audit_status":"PASS_N0_FULL_ENVELOPE_RUNTIME_VIEW_CURRICULUM_AUDIT_V1",
    },
    "long_context":{
        "manifest_schema":"alice.eipm.n0.full-envelope-long-context-manifest.v1",
        "audit_status":"PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_CURRICULUM_AUDIT_V1",
    },
    "fewrel":{
        "manifest_schema":"alice.eipm.n0.fewrel-natural-relation-manifest.v3",
        "audit_status":"PASS_FEWREL_NATURAL_RELATION_AUDIT_V3",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_train_dev_rows(path: Path, *, label: str) -> dict[str,int]:
    rows=read_jsonl(path)
    if not rows:
        raise SystemExit(f"{label} rows empty")
    counts={"train":0,"dev":0}
    for row in rows:
        split=str(row.get("split",""))
        if split not in counts:
            raise SystemExit(f"{label} contains non-TRAIN/DEV row: {row.get('id')}")
        if row.get("private_identity_data") is not False:
            raise SystemExit(f"{label} contains private identity row: {row.get('id')}")
        expected_training=split=="train"
        if row.get("training_authorized") is not expected_training:
            raise SystemExit(f"{label} training authority drift: {row.get('id')}")
        if "model_selection_authorized" in row:
            if row.get("model_selection_authorized") is not (split=="dev"):
                raise SystemExit(f"{label} model-selection authority drift: {row.get('id')}")
        if row.get("final_validation_only") is True:
            raise SystemExit(f"{label} contains FINAL-only row: {row.get('id')}")
        counts[split]+=1
    return counts


def validate_lane(
    *,
    name: str,
    rows_path: Path,
    manifest_path: Path,
    audit_path: Path,
) -> dict[str,Any]:
    spec=EXPECTED[name]
    manifest=load_json(manifest_path)
    audit=load_json(audit_path)
    if manifest.get("schema")!=spec["manifest_schema"]:
        raise SystemExit(f"{name} manifest schema drift")
    if audit.get("status")!=spec["audit_status"]:
        raise SystemExit(f"{name} audit not PASS")
    if manifest.get("sha256") is not None and manifest.get("sha256")!=sha256(rows_path):
        raise SystemExit(f"{name} row hash drift")
    counts=check_train_dev_rows(rows_path,label=name)
    if int(manifest.get("train_rows",counts["train"]))!=counts["train"]:
        raise SystemExit(f"{name} TRAIN row count drift")
    if int(manifest.get("dev_rows",counts["dev"]))!=counts["dev"]:
        raise SystemExit(f"{name} DEV row count drift")
    return {
        "rows_sha256":sha256(rows_path),
        "manifest_sha256":sha256(manifest_path),
        "audit_sha256":sha256(audit_path),
        "train_rows":counts["train"],
        "dev_rows":counts["dev"],
        "private_identity_data":False,
        "final_rows":0,
    }


def main() -> None:
    p=argparse.ArgumentParser()
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
        raise SystemExit("refusing to overwrite full public-mixture manifest")
    contract_path=Path(args.contract)
    contract=load_json(contract_path)
    if contract.get("schema")!="alice.eipm.n0.full-public-mixture-contract.v1":
        raise SystemExit("full public-mixture contract schema drift")

    source_revision=str(args.source_revision).strip()
    if len(source_revision)!=40 or any(ch not in "0123456789abcdef" for ch in source_revision.lower()):
        raise SystemExit("source revision must be exact 40-hex git commit")

    source_config_path=Path(args.source_config)
    corpus_receipt_path=Path(args.corpus_receipt)
    source_config=load_json(source_config_path)
    corpus=load_json(corpus_receipt_path)
    if source_config.get("schema")!="alice.eipm.n0.public-corpus-sources.v0.2.1":
        raise SystemExit("public source config schema drift")
    if source_config.get("status")!="activated_public_n0_v02":
        raise SystemExit("public source config not activated")
    if corpus.get("schema")!="alice.eipm.n0.derived-tokenizer-corpus-receipt.v0.2.1":
        raise SystemExit("public corpus receipt schema drift")
    if corpus.get("status")!="PASS":
        raise SystemExit("public corpus receipt not PASS")
    if corpus.get("source_config_sha256")!=sha256(source_config_path):
        raise SystemExit("public corpus/source-config hash drift")
    for key in ("private_identity_data","private_identity_gradient","model_training_performed"):
        if corpus.get(key) is not False:
            raise SystemExit(f"public corpus receipt must declare {key}=false")

    teacher_registry_path=Path(args.teacher_registry)
    teacher_audit_path=Path(args.teacher_audit)
    teacher_registry=load_json(teacher_registry_path)
    teacher_audit=load_json(teacher_audit_path)
    if teacher_audit.get("status")!="PASS":
        raise SystemExit("teacher-bank audit not PASS")
    if teacher_audit.get("coverage_gate_ok") is not True:
        raise SystemExit("teacher-bank coverage gate closed")
    if teacher_audit.get("full_multitask_gate_open") is not True:
        raise SystemExit("teacher-bank full multitask gate closed")
    if int(teacher_audit.get("registered_rows",0))<1000:
        raise SystemExit("teacher-bank registered row floor not met")
    if teacher_audit.get("private_identity_data") is not False:
        raise SystemExit("teacher-bank audit contains private identity data")
    if teacher_audit.get("private_identity_gradient_authorized") is not False:
        raise SystemExit("teacher-bank private gradient unexpectedly authorized")
    if teacher_registry.get("private_identity_data") is not False:
        raise SystemExit("teacher registry private identity flag drift")
    if teacher_registry.get("private_identity_gradient_authorized") is not False:
        raise SystemExit("teacher registry private-gradient flag drift")

    lanes={
        "semantic_operator_intervention":validate_lane(
            name="semantic",
            rows_path=Path(args.semantic_rows),
            manifest_path=Path(args.semantic_manifest),
            audit_path=Path(args.semantic_audit),
        ),
        "full_envelope_behavioral":validate_lane(
            name="behavioral",
            rows_path=Path(args.behavioral_rows),
            manifest_path=Path(args.behavioral_manifest),
            audit_path=Path(args.behavioral_audit),
        ),
        "runtime_view_supplement":validate_lane(
            name="runtime_view",
            rows_path=Path(args.runtime_view_rows),
            manifest_path=Path(args.runtime_view_manifest),
            audit_path=Path(args.runtime_view_audit),
        ),
        "long_context_supplement":validate_lane(
            name="long_context",
            rows_path=Path(args.long_context_rows),
            manifest_path=Path(args.long_context_manifest),
            audit_path=Path(args.long_context_audit),
        ),
    }

    fewrel_rows_path=Path(args.fewrel_train_dev_rows)
    fewrel_bank_path=Path(args.fewrel_train_dev_bank)
    fewrel_manifest_path=Path(args.fewrel_manifest)
    fewrel_audit_path=Path(args.fewrel_audit)
    fewrel_manifest=load_json(fewrel_manifest_path)
    fewrel_audit=load_json(fewrel_audit_path)
    if fewrel_manifest.get("schema")!=EXPECTED["fewrel"]["manifest_schema"]:
        raise SystemExit("FewRel manifest schema drift")
    if fewrel_audit.get("status")!=EXPECTED["fewrel"]["audit_status"]:
        raise SystemExit("FewRel audit not PASS")
    if fewrel_manifest.get("train_dev_rows_sha256")!=sha256(fewrel_rows_path):
        raise SystemExit("FewRel TRAIN/DEV row hash drift")
    if fewrel_manifest.get("train_dev_bank_sha256")!=sha256(fewrel_bank_path):
        raise SystemExit("FewRel TRAIN/DEV bank hash drift")
    fewrel_counts=check_train_dev_rows(fewrel_rows_path,label="fewrel")
    lanes["natural_relation"]={
        "rows_sha256":sha256(fewrel_rows_path),
        "bank_sha256":sha256(fewrel_bank_path),
        "manifest_sha256":sha256(fewrel_manifest_path),
        "audit_sha256":sha256(fewrel_audit_path),
        "train_rows":fewrel_counts["train"],
        "dev_rows":fewrel_counts["dev"],
        "private_identity_data":False,
        "final_rows":0,
    }

    lanes["broad_semantic_replay"]={
        "source_config_sha256":sha256(source_config_path),
        "corpus_receipt_sha256":sha256(corpus_receipt_path),
        "source_count":len(corpus.get("sources") or []),
        "private_identity_data":False,
        "final_rows":0,
    }
    lanes["governed_judgment_replay"]={
        "teacher_registry_sha256":sha256(teacher_registry_path),
        "teacher_audit_sha256":sha256(teacher_audit_path),
        "registered_rows":int(teacher_audit["registered_rows"]),
        "competency_count":int(teacher_audit["competency_count"]),
        "private_identity_data":False,
        "final_rows":0,
    }

    required_lanes=list(contract["required_training_lanes"])
    if list(lanes)!=required_lanes:
        # Preserve contract ordering explicitly instead of allowing a builder
        # implementation detail to redefine training-lane authority.
        lanes={name:lanes[name] for name in required_lanes}

    required_families=set(map(str,contract["required_macro_families"]))
    family_to_lanes={name:[] for name in sorted(required_families)}
    for lane in required_lanes:
        families=list(contract["lane_macro_families"].get(lane) or [])
        if not families:
            raise SystemExit(f"training lane has no macro-family authority: {lane}")
        for family in families:
            if family not in required_families:
                raise SystemExit(f"lane maps to undeclared macro family: {lane}/{family}")
            family_to_lanes[family].append(lane)
    missing=sorted(name for name,owners in family_to_lanes.items() if not owners)
    if missing:
        raise SystemExit(f"macro-family coverage incomplete: {missing}")

    freeze_path=Path(args.final_freeze_receipt)
    freeze=load_json(freeze_path)
    if freeze.get("status")!="FROZEN_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_BEFORE_GRADIENT":
        raise SystemExit("successor FINAL-v2 freeze receipt not valid")
    if freeze.get("results_observed") is not False:
        raise SystemExit("FINAL results already observed")
    if freeze.get("final_opening_authorized") is not False:
        raise SystemExit("FINAL opening unexpectedly authorized")
    if freeze.get("training_authorized") is not False:
        raise SystemExit("FINAL training unexpectedly authorized")

    manifest={
        "schema":MANIFEST_SCHEMA,
        "status":PASS_STATUS,
        "source_revision":source_revision,
        "contract_sha256":sha256(contract_path),
        "training_lanes":lanes,
        "required_training_lanes":required_lanes,
        "macro_family_coverage":family_to_lanes,
        "required_macro_families":sorted(required_families),
        "final_freeze_receipt_sha256":sha256(freeze_path),
        "final_results_observed":False,
        "final_rows_in_training":0,
        "final_rows_in_model_selection":0,
        "row_count_is_capability_ceiling":False,
        "fixed_lane_sampling_ratio_as_capability_definition":False,
        "private_identity_data":False,
        "private_identity_gradient_authorized":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "n0_complete":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
