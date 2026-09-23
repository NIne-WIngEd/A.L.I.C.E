#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import build_n0_v02_semantic_operator_long_context_curriculum_v1 as builder


PASS="PASS_N0_SEMANTIC_OPERATOR_LONG_CONTEXT_CURRICULUM_AUDIT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def target_signature(row: dict[str,Any]) -> dict[str,Any]:
    keys=(
        "relation_sequence_target",
        "event_sequence_target",
        "applicability_target",
        "uncertainty_target",
        "factor_targets",
        "counterfactual_factor_targets",
        "step_factor_targets",
        "target_entities",
        "intervention",
        "runtime_relation_count",
        "runtime_reasoning_steps",
        "runtime_operator_slots",
    )
    return {key:row[key] for key in keys}


def validate_span(container: str, span: dict[str,Any]) -> bool:
    start=int(span["start"])
    end=int(span["end"])
    return (
        0<=start<end<=len(container)
        and container[start:end]==str(span["text"])
    )


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--rows",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--contract",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    rows_path=Path(args.rows)
    manifest_path=Path(args.manifest)
    contract_path=Path(args.contract)
    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite semantic long-context audit")
    rows=read_jsonl(rows_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    contract=json.loads(contract_path.read_text(encoding="utf-8"))
    errors=[]
    required=set(map(str,contract["required_surfaces"]))

    if manifest.get("schema")!="alice.eipm.n0.semantic-operator-long-context-manifest.v1":
        errors.append("manifest schema drift")
    if manifest.get("sha256")!=sha256(rows_path):
        errors.append("row hash drift")
    by_split={"train":set(),"dev":set()}
    target_equivalence=0
    evidence_exact=0
    for row in rows:
        rid=str(row.get("id","<missing>"))
        split=str(row.get("split",""))
        if split not in by_split:
            errors.append(f"{rid}: non TRAIN/DEV split")
            continue
        surface=str(row.get("long_context_surface",""))
        by_split[split].add(surface)
        if row.get("lane")!="semantic_operator_long_context":
            errors.append(f"{rid}: lane drift")
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data")
        if row.get("final_validation_only") is not False:
            errors.append(f"{rid}: FINAL authority leak")
        if row.get("training_authorized") is not (split=="train"):
            errors.append(f"{rid}: training authority drift")
        if row.get("model_selection_authorized") is not (split=="dev"):
            errors.append(f"{rid}: model-selection authority drift")
        if int(row.get("long_context_word_operating_point",0))<=4096:
            errors.append(f"{rid}: materialized row does not cross native window")
        if row.get("long_context_word_operating_point_is_product_ceiling") is not False:
            errors.append(f"{rid}: operating point became product ceiling")

        base=builder.base_row(split)
        if target_signature(row)==target_signature(base):
            target_equivalence+=1
        else:
            errors.append(f"{rid}: base semantic/operator targets changed")

        exact=True
        for span in row["query_relation_evidence_char_spans"]:
            exact &= validate_span(str(row["query"]),span)
        for span in row["relation_schema_evidence_char_spans"]:
            exact &= validate_span(
                str(row["relation_candidates"][int(span["candidate_index"])]["text"]),
                span,
            )
        for name,span in row["factor_schema_evidence_char_spans"].items():
            exact &= validate_span(
                str(row["factor_schemas"][name][int(span["candidate_index"])]["text"]),
                span,
            )
        for name,spans in row["step_factor_schema_evidence_char_spans"].items():
            for span in spans:
                exact &= validate_span(
                    str(row["factor_schemas"][name][int(span["candidate_index"])]["text"]),
                    span,
                )
        if exact:
            evidence_exact+=1
        else:
            errors.append(f"{rid}: exact evidence-span text drift")

    for split in ("train","dev"):
        if by_split[split]!=required:
            errors.append(
                f"{split}: surface coverage drift {sorted(by_split[split])}"
            )

    result={
        "schema":"alice.eipm.n0.semantic-operator-long-context-audit.v1",
        "status":PASS if not errors else "FAIL_N0_SEMANTIC_OPERATOR_LONG_CONTEXT_CURRICULUM_AUDIT_V1",
        "errors":errors,
        "rows":len(rows),
        "surface_coverage":{
            split:sorted(values) for split,values in by_split.items()
        },
        "base_target_equivalence_rows":target_equivalence,
        "exact_evidence_span_rows":evidence_exact,
        "private_identity_data":False,
        "final_rows":0,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "n0_complete":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
