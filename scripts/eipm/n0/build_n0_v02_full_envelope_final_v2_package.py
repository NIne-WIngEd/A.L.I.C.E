#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_full_envelope_behavioral_curriculum_v1 as behavioral

PASS_STATUS="MATERIALIZED_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_NO_RESULTS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--behavioral-train-dev-rows",required=True)
    p.add_argument("--fewrel-final-rows",required=True)
    p.add_argument("--fewrel-final-bank",required=True)
    p.add_argument("--fewrel-manifest",required=True)
    p.add_argument("--final-contract",required=True)
    p.add_argument("--synthetic-output",required=True)
    p.add_argument("--manifest-output",required=True)
    p.add_argument("--examples-per-mode",type=int,default=2)
    p.add_argument("--seed",type=int,default=20260922)
    args=p.parse_args()
    if args.examples_per_mode <= 0:
        raise SystemExit("examples-per-mode must be positive")

    behavioral_rows_path=Path(args.behavioral_train_dev_rows)
    fewrel_rows_path=Path(args.fewrel_final_rows)
    fewrel_bank_path=Path(args.fewrel_final_bank)
    fewrel_manifest_path=Path(args.fewrel_manifest)
    final_contract_path=Path(args.final_contract)
    synthetic_path=Path(args.synthetic_output)
    manifest_path=Path(args.manifest_output)
    if synthetic_path.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite final-v2 package artifacts")

    train_dev_rows=read_jsonl(behavioral_rows_path)
    if not train_dev_rows or any(str(x.get("split")) not in {"train","dev"} for x in train_dev_rows):
        raise SystemExit("behavioral reference must contain only TRAIN/DEV rows")

    fewrel_manifest=json.loads(fewrel_manifest_path.read_text(encoding="utf-8"))
    if fewrel_manifest.get("schema")!="alice.eipm.n0.fewrel-natural-relation-manifest.v3":
        raise SystemExit("FewRel final manifest schema drift")
    if fewrel_manifest.get("final_rows_sha256")!=sha256(fewrel_rows_path):
        raise SystemExit("FewRel final row hash drift")
    if fewrel_manifest.get("final_bank_sha256")!=sha256(fewrel_bank_path):
        raise SystemExit("FewRel final bank hash drift")
    if fewrel_manifest.get("final_training_authorized") is not False:
        raise SystemExit("FewRel FINAL unexpectedly authorized for training")
    if fewrel_manifest.get("final_model_selection_authorized") is not False:
        raise SystemExit("FewRel FINAL unexpectedly authorized for model selection")

    final_contract=json.loads(final_contract_path.read_text(encoding="utf-8"))
    if final_contract.get("schema")!="alice.eipm.n0.full-envelope-final-validation-contract.v2":
        raise SystemExit("final-v2 contract schema drift")
    if final_contract["authority"].get("results_observed") is not False:
        raise SystemExit("final-v2 contract already records observed results")

    rows=[]
    relation_points=(9,10)
    field_points=(18,24)
    answer_points=(7,8)
    total_modes=19
    for cycle in range(args.examples_per_mode):
        for mode in range(total_modes):
            example=cycle*total_modes+mode
            row=behavioral.materialize_row(
                split="final",
                example=example,
                seed=args.seed,
                relation_count=relation_points[(mode+cycle)%len(relation_points)],
                field_count=field_points[(mode//2+cycle)%len(field_points)],
                answer_count=answer_points[(mode//3+cycle)%len(answer_points)],
            )
            if row.get("split")!="final":
                raise RuntimeError("FINAL builder produced non-final row")
            if row.get("training_authorized") is not False:
                raise RuntimeError("FINAL synthetic row authorized for training")
            if row.get("model_selection_authorized") is not False:
                raise RuntimeError("FINAL synthetic row authorized for model selection")
            if row.get("final_validation_only") is not True:
                raise RuntimeError("FINAL synthetic row missing final-only marker")
            rows.append(row)

    synthetic_path.parent.mkdir(parents=True,exist_ok=True)
    synthetic_path.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),
        encoding="utf-8",
    )
    manifest={
        "schema":"alice.eipm.n0.full-envelope-final-v2-package-manifest.v1",
        "status":PASS_STATUS,
        "results_observed":False,
        "training_authorized":False,
        "model_selection_authorized":False,
        "final_opening_authorized":False,
        "private_identity_data":False,
        "behavioral_train_dev_reference_sha256":sha256(behavioral_rows_path),
        "synthetic_final_rows_sha256":sha256(synthetic_path),
        "fewrel_final_rows_sha256":sha256(fewrel_rows_path),
        "fewrel_final_bank_sha256":sha256(fewrel_bank_path),
        "fewrel_manifest_sha256":sha256(fewrel_manifest_path),
        "final_contract_sha256":sha256(final_contract_path),
        "synthetic_rows":len(rows),
        "natural_final_rows":int(fewrel_manifest.get("final_rows",0)),
        "scenario_histogram":dict(sorted(Counter(str(x["scenario_family"]) for x in rows).items())),
        "relation_count_points":sorted({int(x["runtime_relation_count"]) for x in rows}),
        "field_count_points":sorted({int(x["runtime_field_count"]) for x in rows}),
        "candidate_count_points":sorted({int(x["runtime_candidate_answer_count"]) for x in rows}),
        "reasoning_step_points":sorted({int(x["runtime_reasoning_steps"]) for x in rows}),
        "factor_cardinality_points":sorted({
            max(len(bank) for bank in x["factor_schemas"].values())
            for x in rows
        }),
        "row_count_is_capability_ceiling":False,
        "runtime_axis_points_are_capability_ceilings":False,
        "legacy_final_v1_authority":False,
        "legacy_final_v3_evaluator_authority":False,
        "n0_complete":False,
    }
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
