#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_semantic_operator_intervention_curriculum_v1 as base


ROW_SCHEMA="alice.eipm.n0.semantic-operator-long-context-row.v1"
MANIFEST_SCHEMA="alice.eipm.n0.semantic-operator-long-context-manifest.v1"
SURFACES=("query","relation_schema","factor_schema")
BASE_SEED=20260922
BASE_EXAMPLE=2
BASE_CANDIDATES=4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_count(text: str) -> int:
    return len(str(text).split())


def _prefix(*, surface: str, words: int) -> str:
    unit=(
        f"public neutral {surface} background context nondecisive archive "
        "note calibration detail unrelated filler "
    ).split()
    if words<=0:
        return ""
    return " ".join(unit[i%len(unit)] for i in range(words))


def longify(
    original: str,
    *,
    surface: str,
    target_words: int,
) -> tuple[str,int]:
    original=str(original)
    if not original:
        raise ValueError("cannot longify empty semantic text")
    prefix_words=max(1,int(target_words)-word_count(original))
    prefix=_prefix(surface=surface,words=prefix_words)
    transformed=f"{prefix} {original}"
    while word_count(transformed)<int(target_words):
        prefix=f"{prefix} neutral"
        transformed=f"{prefix} {original}"
    offset=len(prefix)+1
    if transformed[offset:]!=original:
        raise RuntimeError("long semantic prefix failed to preserve original text")
    return transformed,offset


def base_row(split: str) -> dict[str,Any]:
    if split not in {"train","dev"}:
        raise ValueError("semantic long-context supplement is TRAIN/DEV only")
    relations=base.TRAIN_RELATIONS if split=="train" else base.DEV_RELATIONS
    relation=relations[0]
    second=relations[1%len(relations)]
    rng=random.Random(BASE_SEED+(0 if split=="train" else 1_000_000))
    row=base.make_row(
        split=split,
        relation=relation,
        second=second,
        example=BASE_EXAMPLE,
        candidates=BASE_CANDIDATES,
        rng=rng,
    )
    return copy.deepcopy(row)


def _shift_span(span: dict[str,Any], offset: int) -> None:
    span["start"]=int(span["start"])+int(offset)
    span["end"]=int(span["end"])+int(offset)


def _validate_span(container: str, span: dict[str,Any]) -> None:
    start=int(span["start"])
    end=int(span["end"])
    text=str(span["text"])
    if not (0<=start<end<=len(container)):
        raise RuntimeError("long semantic evidence span outside text")
    if container[start:end]!=text:
        raise RuntimeError(
            f"long semantic evidence span drift expected={text!r} "
            f"observed={container[start:end]!r}"
        )


def materialize(
    *,
    split: str,
    surface: str,
    target_words: int,
    placement_variant: str="tail",
) -> dict[str,Any]:
    if surface not in SURFACES:
        raise ValueError(f"unknown semantic long-context surface: {surface}")
    if placement_variant!="tail":
        raise ValueError(
            "semantic-operator long-context v1 currently owns the distant-tail "
            "cross-window intervention; boundary-shift robustness is governed "
            "by the separate full-envelope long-context lane"
        )
    if int(target_words)<=0:
        raise ValueError("target_words must be positive")

    row=base_row(split)
    if surface=="query":
        original=str(row["query"])
        transformed,offset=longify(
            original,surface=surface,target_words=target_words
        )
        row["query"]=transformed
        for span in row["query_relation_evidence_char_spans"]:
            _shift_span(span,offset)
            _validate_span(transformed,span)
        locator={"kind":"query"}
    elif surface=="relation_schema":
        spans=list(row["relation_schema_evidence_char_spans"])
        if not spans:
            raise RuntimeError("base semantic row lacks relation-schema evidence")
        target_index=int(spans[0]["candidate_index"])
        original=str(row["relation_candidates"][target_index]["text"])
        transformed,offset=longify(
            original,surface=surface,target_words=target_words
        )
        row["relation_candidates"][target_index]["text"]=transformed
        for span in spans:
            if int(span["candidate_index"])==target_index:
                _shift_span(span,offset)
                _validate_span(transformed,span)
        locator={"kind":"relation_schema","candidate_index":target_index}
    else:
        bank_name="direction"
        span=dict(row["factor_schema_evidence_char_spans"][bank_name])
        target_index=int(span["candidate_index"])
        original=str(row["factor_schemas"][bank_name][target_index]["text"])
        transformed,offset=longify(
            original,surface=surface,target_words=target_words
        )
        row["factor_schemas"][bank_name][target_index]["text"]=transformed
        global_span=row["factor_schema_evidence_char_spans"][bank_name]
        _shift_span(global_span,offset)
        _validate_span(transformed,global_span)
        for step_span in row[
            "step_factor_schema_evidence_char_spans"
        ].get(bank_name,[]):
            if int(step_span["candidate_index"])==target_index:
                _shift_span(step_span,offset)
                _validate_span(transformed,step_span)
        locator={
            "kind":"factor_schema",
            "bank":bank_name,
            "candidate_index":target_index,
        }

    row["schema"]=ROW_SCHEMA
    row["lane"]="semantic_operator_long_context"
    row["id"]=f"solc_{split}_{surface}_{BASE_EXAMPLE:04d}"
    row["data_origin"]="deterministic_public_semantic_operator_long_context_v1"
    row["long_context_surface"]=surface
    row["long_context_locator"]=locator
    row["long_context_placement_variant"]="tail"
    row["long_context_word_operating_point"]=int(target_words)
    row["long_context_word_operating_point_is_product_ceiling"]=False
    row["base_semantic_row_id"]=str(base_row(split)["id"])
    row["private_identity_data"]=False
    row["training_authorized"]=split=="train"
    row["model_selection_authorized"]=split=="dev"
    row["final_validation_only"]=False
    return row


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--long-word-target",type=int,default=4608)
    args=p.parse_args()
    if args.long_word_target<=4096:
        raise SystemExit(
            "materialized long semantic operating point must exceed native window"
        )
    output=Path(args.output)
    manifest_path=Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite semantic long-context artifacts")
    rows=[
        materialize(
            split=split,
            surface=surface,
            target_words=args.long_word_target,
        )
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
        "status":"MATERIALIZED_PUBLIC_SEMANTIC_OPERATOR_LONG_CONTEXT_TRAIN_DEV",
        "sha256":sha256(output),
        "rows":len(rows),
        "train_rows":sum(1 for x in rows if x["split"]=="train"),
        "dev_rows":sum(1 for x in rows if x["split"]=="dev"),
        "surface_histogram":dict(sorted(Counter(x["long_context_surface"] for x in rows).items())),
        "surfaces_by_split":{
            split:sorted(
                x["long_context_surface"] for x in rows if x["split"]==split
            )
            for split in ("train","dev")
        },
        "long_word_operating_point":int(args.long_word_target),
        "long_word_operating_point_is_product_ceiling":False,
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
