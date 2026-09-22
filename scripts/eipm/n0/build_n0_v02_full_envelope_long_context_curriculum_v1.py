#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_full_envelope_behavioral_curriculum_v1 as base
from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
    INTERNAL_VIEW_DESCRIPTIONS,
)

ROW_SCHEMA="alice.eipm.n0.full-envelope-long-context-row.v1"
MANIFEST_SCHEMA="alice.eipm.n0.full-envelope-long-context-manifest.v1"
SURFACES=(
    "query",
    "relation_schema",
    "factor_schema",
    "type_schema",
    "field_text",
    "field_descriptor",
    "candidate_text",
    "internal_view_descriptor",
)
SURFACE_EXAMPLE={
    "query":1,
    "relation_schema":2,
    "factor_schema":4,
    "type_schema":3,
    "field_text":10,
    "field_descriptor":7,
    "candidate_text":15,
    "internal_view_descriptor":11,
}
BASE_SEED=20260922


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_count(text: str) -> int:
    return len(str(text).split())


def longify(text: str, *, surface: str, target_words: int) -> str:
    original=str(text).strip()
    if not original:
        raise ValueError("cannot longify empty semantic text")
    marker="decisive semantic content follows"
    filler_unit=(
        f"public neutral {surface} background context marker "
        "nondecisive archive note calibration detail"
    ).split()
    required_prefix=max(
        0,
        int(target_words)-word_count(marker)-word_count(original),
    )
    filler=(
        filler_unit
        * ((required_prefix+len(filler_unit)-1)//len(filler_unit))
    )[:required_prefix]
    pieces=filler+[marker,original]
    result=" ".join(piece for piece in pieces if piece).strip()
    if word_count(result) != int(target_words):
        raise RuntimeError(
            "long-context operating point construction drift: "
            f"expected {int(target_words)} words got {word_count(result)}"
        )
    if not result.endswith(original):
        raise RuntimeError("original semantic content not preserved at tail")
    return result


def target_locator(row: dict[str,Any], surface: str) -> dict[str,Any]:
    if surface=="query":
        return {"kind":"query"}
    if surface=="relation_schema":
        targets=list(row["relation_sequence_target"])
        index=int(targets[0]) if targets else 0
        return {"kind":"relation_schema","index":index}
    if surface=="factor_schema":
        bank_name="direction"
        target_key=str(row["factor_target_keys"][bank_name])
        bank=list(row["factor_schemas"][bank_name])
        index=next(i for i,x in enumerate(bank) if str(x["key"])==target_key)
        return {"kind":"factor_schema","bank":bank_name,"index":index}
    if surface=="type_schema":
        field_index=int((row.get("decisive_field_indices") or [0])[0])
        type_key=str(row["fields"][field_index]["type_key"])
        index=next(i for i,x in enumerate(row["type_schema"]) if str(x["key"])==type_key)
        return {"kind":"type_schema","index":index}
    if surface=="field_text":
        index=int((row.get("decisive_field_indices") or [0])[0])
        return {"kind":"field_text","index":index}
    if surface=="field_descriptor":
        index=int((row.get("decisive_field_indices") or [0])[0])
        return {"kind":"field_descriptor","index":index,"descriptor":"provenance"}
    if surface=="candidate_text":
        return {"kind":"candidate_text","index":int(row["public_target_index"])}
    if surface=="internal_view_descriptor":
        return {
            "kind":"internal_view_descriptor",
            "index":len(INTERNAL_VIEW_DESCRIPTIONS)-1,
        }
    raise ValueError(f"unknown long-context surface: {surface}")


def get_text(row: dict[str,Any], locator: dict[str,Any]) -> str:
    kind=str(locator["kind"])
    if kind=="query":
        return str(row["query"])
    if kind=="relation_schema":
        return str(row["relation_candidates"][int(locator["index"])]["text"])
    if kind=="factor_schema":
        return str(
            row["factor_schemas"][str(locator["bank"])][int(locator["index"])]["text"]
        )
    if kind=="type_schema":
        return str(row["type_schema"][int(locator["index"])]["text"])
    if kind=="field_text":
        return str(row["fields"][int(locator["index"])]["text"])
    if kind=="field_descriptor":
        return str(
            row["fields"][int(locator["index"])]["descriptors"][
                str(locator["descriptor"])
            ]
        )
    if kind=="candidate_text":
        return str(row["candidate_answers"][int(locator["index"])])
    if kind=="internal_view_descriptor":
        return str(row["internal_view_descriptions"][int(locator["index"])])
    raise ValueError(f"unknown locator kind: {kind}")


def set_text(
    row: dict[str,Any],
    locator: dict[str,Any],
    value: str,
) -> None:
    kind=str(locator["kind"])
    if kind=="query":
        row["query"]=value
    elif kind=="relation_schema":
        row["relation_candidates"][int(locator["index"])]["text"]=value
    elif kind=="factor_schema":
        row["factor_schemas"][str(locator["bank"])][int(locator["index"])]["text"]=value
    elif kind=="type_schema":
        row["type_schema"][int(locator["index"])]["text"]=value
    elif kind=="field_text":
        row["fields"][int(locator["index"])]["text"]=value
    elif kind=="field_descriptor":
        row["fields"][int(locator["index"])]["descriptors"][
            str(locator["descriptor"])
        ]=value
    elif kind=="candidate_text":
        row["candidate_answers"][int(locator["index"])]=value
    elif kind=="internal_view_descriptor":
        row["internal_view_descriptions"][int(locator["index"])]=value
    else:
        raise ValueError(f"unknown locator kind: {kind}")


def materialize(
    *,
    split: str,
    surface: str,
    target_words: int,
) -> dict[str,Any]:
    if split not in {"train","dev"}:
        raise ValueError("long-context supplement is TRAIN/DEV only")
    example=int(SURFACE_EXAMPLE[surface])
    params={
        "split":split,
        "example":example,
        "seed":BASE_SEED,
        "relation_count":4,
        "field_count":6,
        "answer_count":4,
    }
    row=base.materialize_row(**params)
    row["internal_view_descriptions"]=list(INTERNAL_VIEW_DESCRIPTIONS)
    locator=target_locator(row,surface)
    original=get_text(row,locator)
    set_text(
        row,
        locator,
        longify(original,surface=surface,target_words=target_words),
    )
    row["schema"]=ROW_SCHEMA
    row["lane"]="full_envelope_long_context_supplement"
    row["id"]=f"felc_{split}_{surface}_{example:04d}"
    row["template_id"]=f"{split}:long-context:{surface}:{row['scenario_family']}"
    row["template_signature_sha256"]=base.template_signature(
        str(row["query"]),
        list(row["entities"]),
    )
    row["causal_group"]=f"{split}:long-context:{surface}:{example:04d}"
    row["data_origin"]="deterministic_public_full_envelope_long_context_supplement_v1"
    row["long_context_surface"]=surface
    row["long_context_locator"]=locator
    row["long_context_word_operating_point"]=int(target_words)
    row["long_context_word_operating_point_is_product_ceiling"]=False
    row["base_materialization"]=params
    return row


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--long-word-target",type=int,default=4608)
    args=p.parse_args()
    if args.long_word_target <= 4096:
        raise SystemExit("long-word target must exceed native-window operating point")
    output=Path(args.output)
    manifest_path=Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite long-context curriculum artifacts")

    rows=[
        materialize(split=split,surface=surface,target_words=args.long_word_target)
        for split in ("train","dev")
        for surface in SURFACES
    ]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),
        encoding="utf-8",
    )
    manifest={
        "schema":MANIFEST_SCHEMA,
        "status":"MATERIALIZED_PUBLIC_FULL_ENVELOPE_LONG_CONTEXT_TRAIN_DEV",
        "sha256":sha256(output),
        "rows":len(rows),
        "train_rows":sum(1 for x in rows if x["split"]=="train"),
        "dev_rows":sum(1 for x in rows if x["split"]=="dev"),
        "surface_histogram":dict(sorted(Counter(x["long_context_surface"] for x in rows).items())),
        "surfaces_by_split":{
            split:sorted(x["long_context_surface"] for x in rows if x["split"]==split)
            for split in ("train","dev")
        },
        "long_word_operating_point":int(args.long_word_target),
        "long_word_operating_point_is_product_ceiling":False,
        "product_context_ceiling":None,
        "final_rows":0,
        "private_identity_data":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "n0_complete":False,
    }
    manifest_path.write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
