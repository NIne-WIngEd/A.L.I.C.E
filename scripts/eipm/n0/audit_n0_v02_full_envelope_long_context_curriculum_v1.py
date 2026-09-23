#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import build_n0_v02_full_envelope_behavioral_curriculum_v1 as base
import build_n0_v02_full_envelope_long_context_curriculum_v1 as long_builder
import build_n0_v02_full_envelope_runtime_view_curriculum_v1 as runtime_views
from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
    INTERNAL_VIEW_DESCRIPTIONS,
)

PASS="PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_CURRICULUM_AUDIT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("long-context curriculum is empty")
    return rows


def structural_target_view(row: dict[str,Any]) -> dict[str,Any]:
    keys=(
        "public_target_index",
        "relation_sequence_target",
        "counterfactual_relation_sequence_target",
        "event_sequence_target",
        "factor_target_keys",
        "counterfactual_factor_keys",
        "step_factor_target_keys",
        "applicability_target",
        "uncertainty_target",
        "support_edge_indices",
        "endpoint_target",
        "decisive_field_indices",
        "irrelevant_field_indices",
        "recoverable_view_names",
        "runtime_reasoning_steps",
    )
    return {key:row[key] for key in keys}


def base_text_for(
    base_row: dict[str,Any],
    locator: dict[str,Any],
) -> str:
    kind=str(locator["kind"])
    if kind=="internal_view_descriptor":
        return str(INTERNAL_VIEW_DESCRIPTIONS[int(locator["index"])])
    return long_builder.get_text(base_row,locator)


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
        raise SystemExit("refusing to overwrite long-context audit")
    rows=read_jsonl(rows_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    contract=json.loads(contract_path.read_text(encoding="utf-8"))
    errors=[]

    if contract.get("schema")!="alice.eipm.n0.full-envelope-long-context-curriculum-contract.v1":
        errors.append("contract schema drift")
    if manifest.get("schema")!="alice.eipm.n0.full-envelope-long-context-manifest.v1":
        errors.append("manifest schema drift")
    if manifest.get("sha256")!=sha256(rows_path):
        errors.append("row hash drift")
    required=set(map(str,contract["required_surfaces"]))
    target_words=int(contract["operating_point"]["long_word_target"])
    if target_words <= int(contract["operating_point"]["native_window_tokens"]):
        errors.append("long operating point does not cross native window")
    if contract["operating_point"].get("long_word_target_is_product_ceiling") is not False:
        errors.append("long operating point became capability ceiling")

    by_split={split:set() for split in ("train","dev")}
    target_equivalence=0
    tail_preservation=0
    boundary_rows={}
    boundary_shift_pairs_verified=0
    for row in rows:
        rid=str(row.get("id","<missing>"))
        split=str(row.get("split",""))
        if split not in by_split:
            errors.append(f"{rid}: split must be train/dev")
            continue
        surface=str(row.get("long_context_surface",""))
        by_split[split].add(surface)
        if row.get("final_validation_only") is not False:
            errors.append(f"{rid}: FINAL row leaked into training supplement")
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data forbidden")
        if row.get("training_authorized") is not (split=="train"):
            errors.append(f"{rid}: training authorization drift")
        if row.get("model_selection_authorized") is not (split=="dev"):
            errors.append(f"{rid}: model-selection authorization drift")
        if row.get("long_context_word_operating_point_is_product_ceiling") is not False:
            errors.append(f"{rid}: long operating point marked as ceiling")

        base_info=dict(row.get("base_materialization") or {})
        try:
            kind=str(base_info.get("kind","behavioral"))
            if kind=="behavioral":
                params=dict(base_info.get("params") or base_info)
                base_row=base.materialize_row(**params)
            elif kind=="runtime_view":
                candidates=[
                    item for item in runtime_views.materialize_rows(split)
                    if str(item["id"])==str(base_info.get("row_id"))
                ]
                if len(candidates)!=1:
                    raise ValueError("runtime-view base row not uniquely recoverable")
                base_row=candidates[0]
            else:
                raise ValueError(f"unknown base materialization kind: {kind}")
        except Exception as exc:
            errors.append(f"{rid}: unable to reconstruct base row: {exc}")
            continue
        if structural_target_view(row)!=structural_target_view(base_row):
            errors.append(f"{rid}: long intervention changed structural/behavioral targets")
        else:
            target_equivalence+=1

        locator=dict(row.get("long_context_locator") or {})
        try:
            current=long_builder.get_text(row,locator)
            original=base_text_for(base_row,locator)
        except Exception as exc:
            errors.append(f"{rid}: invalid long-context locator: {exc}")
            continue
        if long_builder.word_count(current) < target_words:
            errors.append(f"{rid}: long surface below operating point")
        variant=str(row.get("long_context_placement_variant","tail"))
        if variant=="tail":
            if not current.endswith(original):
                errors.append(f"{rid}: original decisive semantic content not preserved at tail")
            else:
                tail_preservation+=1
        elif variant in {"boundary_early","boundary_late"}:
            pair_id=str(row.get("boundary_shift_pair_id") or "")
            if not pair_id:
                errors.append(f"{rid}: boundary row missing pair id")
            key=(split,surface,pair_id)
            boundary_rows.setdefault(key,{})[variant]=(row,current,original)
            start=row.get("decisive_start_word")
            end=row.get("decisive_end_word")
            if not isinstance(start,int) or not isinstance(end,int) or not (0 <= start < end):
                errors.append(f"{rid}: invalid decisive word placement receipt")
            else:
                words=current.split()
                original_words=original.split()
                if words[start:end] != original_words:
                    errors.append(f"{rid}: decisive semantics not preserved at declared position")
        else:
            errors.append(f"{rid}: unknown long-context placement variant {variant}")

        descriptions=row.get("internal_view_descriptions")
        if descriptions is not None and len(descriptions)!=len(INTERNAL_VIEW_DESCRIPTIONS):
            errors.append(f"{rid}: internal view descriptor count drift")

    for (split,surface,pair_id),variants in sorted(boundary_rows.items()):
        if set(variants)!={"boundary_early","boundary_late"}:
            errors.append(f"{pair_id}: incomplete boundary-shift pair")
            continue
        early,late=variants["boundary_early"],variants["boundary_late"]
        early_row,early_text,early_original=early
        late_row,late_text,late_original=late
        if early_original != late_original:
            errors.append(f"{pair_id}: decisive semantics differ across boundary pair")
        if long_builder.word_count(early_text) != long_builder.word_count(late_text):
            errors.append(f"{pair_id}: total length differs across boundary pair")
        if structural_target_view(early_row) != structural_target_view(late_row):
            errors.append(f"{pair_id}: structural/behavioral targets differ across boundary pair")
        early_start=int(early_row.get("decisive_start_word",-1))
        late_start=int(late_row.get("decisive_start_word",-1))
        if not (early_start < 3840 and late_start > 4096):
            errors.append(
                f"{pair_id}: decisive text did not move across the first virtualized boundary region"
            )
        if not errors or not any(pair_id in e for e in errors):
            boundary_shift_pairs_verified+=1

    for split in ("train","dev"):
        missing=sorted(required-by_split[split])
        extra=sorted(by_split[split]-required)
        if missing:
            errors.append(f"{split}: missing long surfaces {missing}")
        if extra:
            errors.append(f"{split}: undeclared long surfaces {extra}")

    result={
        "schema":"alice.eipm.n0.full-envelope-long-context-curriculum-audit.v1",
        "status":PASS if not errors else "FAIL_N0_FULL_ENVELOPE_LONG_CONTEXT_CURRICULUM_AUDIT_V1",
        "errors":errors,
        "rows":len(rows),
        "surface_coverage":{
            split:sorted(values)
            for split,values in by_split.items()
        },
        "base_target_equivalence_rows":target_equivalence,
        "tail_preservation_rows":tail_preservation,
        "boundary_shift_pairs_verified":boundary_shift_pairs_verified,
        "expected_boundary_shift_pairs":2*len(required),
        "long_word_operating_point":target_words,
        "operating_point_is_product_ceiling":False,
        "final_rows":0,
        "private_identity_data":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "final_opening_authorized":False,
        "n0_complete":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
