#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PASS="PASS_N0_FULL_ENVELOPE_RUNTIME_VIEW_CURRICULUM_AUDIT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("runtime-view curriculum is empty")
    return rows


def _pair(rows: list[dict[str,Any]], split: str, family: str) -> list[dict[str,Any]]:
    selected=[
        x for x in rows
        if x["split"]==split and x["scenario_family"]==family
    ]
    if len(selected)!=2:
        raise AssertionError(f"{split}/{family} requires exactly two pair members")
    return sorted(selected,key=lambda x:x["id"])


def _descriptors(row: dict[str,Any]) -> list[str]:
    return [str(x["descriptor_text"]) for x in row["additional_views"]]


def _availability(row: dict[str,Any]) -> list[bool]:
    return [bool(x["available"]) for x in row["additional_views"]]


def _reliability(row: dict[str,Any]) -> list[float]:
    return [float(x["reliability"]) for x in row["additional_views"]]


def _sources(row: dict[str,Any]) -> list[str]:
    return [str(x["source_text"]) for x in row["additional_views"]]


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
        raise SystemExit("refusing to overwrite runtime-view audit")
    rows=read_jsonl(rows_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    contract=json.loads(contract_path.read_text(encoding="utf-8"))
    errors=[]

    if contract.get("schema")!="alice.eipm.n0.full-envelope-runtime-view-curriculum-contract.v1":
        errors.append("contract schema drift")
    if manifest.get("schema")!="alice.eipm.n0.full-envelope-runtime-view-manifest.v1":
        errors.append("manifest schema drift")
    if manifest.get("sha256")!=sha256(rows_path):
        errors.append("row hash drift")
    if manifest.get("fixed_view_taxonomy") is not False:
        errors.append("runtime-view supplement introduced fixed taxonomy")
    if manifest.get("additional_view_count_is_product_ceiling") is not False:
        errors.append("runtime-view count became a product ceiling")

    for row in rows:
        rid=str(row.get("id","<missing>"))
        split=str(row.get("split",""))
        if split not in {"train","dev"}:
            errors.append(f"{rid}: split must be TRAIN/DEV")
            continue
        if row.get("training_authorized") is not (split=="train"):
            errors.append(f"{rid}: training authority drift")
        if row.get("model_selection_authorized") is not (split=="dev"):
            errors.append(f"{rid}: model-selection authority drift")
        if row.get("final_validation_only") is not False:
            errors.append(f"{rid}: FINAL row leaked into runtime-view supplement")
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data forbidden")
        views=list(row.get("additional_views") or [])
        if len(views)<3:
            errors.append(f"{rid}: runtime-view intervention needs at least three views")
            continue
        if len(views)!=int(row.get("runtime_additional_view_count",-1)):
            errors.append(f"{rid}: additional-view count receipt drift")
        if any(not str(x.get("source_text","")).strip() for x in views):
            errors.append(f"{rid}: source text missing")
        if any(not str(x.get("descriptor_text","")).strip() for x in views):
            errors.append(f"{rid}: descriptor text missing")
        if any(not 0.0 <= float(x.get("reliability",-1.0)) <= 1.0 for x in views):
            errors.append(f"{rid}: reliability outside [0,1]")
        for item in views:
            if not bool(item.get("available")) and bool(item.get("recoverable")):
                errors.append(f"{rid}: unavailable view marked recoverable")
            if bool(item.get("decisive")) and bool(item.get("irrelevant")):
                errors.append(f"{rid}: view both decisive and irrelevant")

    for split in ("train","dev"):
        try:
            rel=_pair(rows,split,"runtime_view_relevance_flip")
            a,b=rel
            if a["query"]!=b["query"] or a["candidate_answers"]!=b["candidate_answers"]:
                errors.append(f"{split}: relevance-flip pair changed query/candidates")
            if _descriptors(a)!=_descriptors(b):
                errors.append(f"{split}: relevance-flip descriptors changed")
            if _availability(a)!=_availability(b):
                errors.append(f"{split}: relevance-flip availability changed")
            if _reliability(a)!=_reliability(b):
                errors.append(f"{split}: relevance-flip reliability changed")
            if a["public_target_index"]==b["public_target_index"]:
                errors.append(f"{split}: relevance-flip target did not change")
            if _sources(a)==_sources(b):
                errors.append(f"{split}: relevance-flip source semantics did not change")
            for row in rel:
                views=row["additional_views"]
                available=[x for x in views if x["available"]]
                if not any(x["decisive"] and x["recoverable"] for x in available):
                    errors.append(f"{row['id']}: no decisive recoverable additional view")
                if not any(x["irrelevant"] and not x["recoverable"] for x in available):
                    errors.append(f"{row['id']}: no merely available irrelevant view")
                if not any(not x["available"] for x in views):
                    errors.append(f"{row['id']}: no unavailable-view intervention")
        except Exception as exc:
            errors.append(f"{split}: relevance-flip pair audit failed: {exc}")

        try:
            rev=_pair(rows,split,"runtime_view_reliability_reversal")
            a,b=rev
            if a["query"]!=b["query"] or a["candidate_answers"]!=b["candidate_answers"]:
                errors.append(f"{split}: reliability pair changed query/candidates")
            if _descriptors(a)!=_descriptors(b) or _sources(a)!=_sources(b):
                errors.append(f"{split}: reliability pair changed semantic source/descriptor content")
            if _availability(a)!=_availability(b):
                errors.append(f"{split}: reliability pair availability changed")
            if a["public_target_index"]==b["public_target_index"]:
                errors.append(f"{split}: reliability reversal target did not change")
            ar=_reliability(a); br=_reliability(b)
            if not (ar[0]==br[1] and ar[1]==br[0]):
                errors.append(f"{split}: first two relevant-view reliabilities did not reverse")
            for row in rev:
                views=row["additional_views"]
                relevant=[x for x in views[:2] if x["available"]]
                if len(relevant)!=2 or not all(x["recoverable"] for x in relevant):
                    errors.append(f"{row['id']}: relevant conflicting views must both be recoverable")
                irrelevant=views[2]
                if not (
                    irrelevant["available"]
                    and irrelevant["irrelevant"]
                    and not irrelevant["recoverable"]
                    and float(irrelevant["reliability"]) > max(float(x["reliability"]) for x in relevant)
                ):
                    errors.append(f"{row['id']}: high-reliability irrelevant-view control missing")
        except Exception as exc:
            errors.append(f"{split}: reliability-reversal pair audit failed: {exc}")

    result={
        "schema":"alice.eipm.n0.full-envelope-runtime-view-curriculum-audit.v1",
        "status":PASS if not errors else "FAIL_N0_FULL_ENVELOPE_RUNTIME_VIEW_CURRICULUM_AUDIT_V1",
        "errors":errors,
        "rows":len(rows),
        "train_rows":sum(1 for x in rows if x.get("split")=="train"),
        "dev_rows":sum(1 for x in rows if x.get("split")=="dev"),
        "scenario_histogram":{
            family:sum(1 for x in rows if str(x.get("scenario_family"))==family)
            for family in sorted({str(x.get("scenario_family")) for x in rows})
        },
        "relevance_flip_verified":not any("relevance-flip" in e for e in errors),
        "reliability_reversal_verified":not any("reliability" in e for e in errors),
        "recoverability_not_availability_verified":not any("recoverable" in e for e in errors),
        "fixed_view_taxonomy":False,
        "view_count_ceiling":None,
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
