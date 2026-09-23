#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_full_envelope_behavioral_curriculum_v1 as base
import build_n0_v02_full_envelope_runtime_view_curriculum_v1 as runtime_views
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
    "additional_view_descriptor",
    "additional_view_source",
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
    "additional_view_descriptor":15,
    "additional_view_source":15,
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


def boundary_shift_text(
    text: str,
    *,
    surface: str,
    total_words: int,
    decisive_start_word: int,
) -> tuple[str,int,int]:
    """Place identical decisive semantics at a chosen word-space position.

    This is a tokenizer-agnostic curriculum construction probe. Exact token
    boundary placement is verified later with the real tokenizer before
    gradient. Here we guarantee equal total length, equal decisive text, and a
    large enough early/late displacement to exercise different virtualized
    ownership regions under ordinary tokenizer behavior.
    """
    original=str(text).strip()
    if not original:
        raise ValueError("cannot boundary-shift empty semantic text")
    marker="decisive semantic content follows"
    marker_words=marker.split()
    original_words=original.split()
    start=int(decisive_start_word)
    if start <= len(marker_words):
        raise ValueError("decisive start must leave room for marker context")
    prefix_count=start-len(marker_words)
    suffix_count=int(total_words)-start-len(original_words)
    if suffix_count < 0:
        raise ValueError("boundary-shift total_words too small")

    filler_unit=(
        f"public neutral {surface} background context marker "
        "nondecisive archive note calibration detail"
    ).split()
    def fill(count: int) -> list[str]:
        if count <= 0:
            return []
        return (
            filler_unit
            * ((count+len(filler_unit)-1)//len(filler_unit))
        )[:count]

    pieces=fill(prefix_count)+marker_words+original_words+fill(suffix_count)
    result=" ".join(pieces)
    if word_count(result) != int(total_words):
        raise RuntimeError("boundary-shift total length drift")
    actual_start=prefix_count+len(marker_words)
    actual_end=actual_start+len(original_words)
    if actual_start != start:
        raise RuntimeError("boundary-shift decisive start drift")
    return result,actual_start,actual_end


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
    if surface in {"additional_view_descriptor","additional_view_source"}:
        views=list(row.get("additional_views") or [])
        if not views:
            raise ValueError("additional-view long-context surface requires runtime views")
        index=next(
            (
                i for i,item in enumerate(views)
                if bool(item.get("decisive")) and bool(item.get("available",True))
            ),
            0,
        )
        return {"kind":surface,"index":index}
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
    if kind=="additional_view_descriptor":
        return str(row["additional_views"][int(locator["index"])]["descriptor_text"])
    if kind=="additional_view_source":
        return str(row["additional_views"][int(locator["index"])]["source_text"])
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
    elif kind=="additional_view_descriptor":
        row["additional_views"][int(locator["index"])]["descriptor_text"]=value
    elif kind=="additional_view_source":
        row["additional_views"][int(locator["index"])]["source_text"]=value
    else:
        raise ValueError(f"unknown locator kind: {kind}")


def materialize(
    *,
    split: str,
    surface: str,
    target_words: int,
    placement_variant: str = "tail",
    example_override: int | None = None,
) -> dict[str,Any]:
    if split not in {"train","dev","final"}:
        raise ValueError("long-context split must be train/dev/final")
    example=(
        int(example_override)
        if example_override is not None
        else int(SURFACE_EXAMPLE[surface])
    )
    if surface in {"additional_view_descriptor","additional_view_source"}:
        candidates=[
            copy.deepcopy(item)
            for item in runtime_views.materialize_rows(split)
            if item["scenario_family"]=="runtime_view_relevance_flip"
            and item["id"].endswith("_a")
        ]
        if len(candidates)!=1:
            raise RuntimeError("unable to resolve one base runtime-view row")
        row=candidates[0]
        base_materialization={
            "kind":"runtime_view",
            "split":split,
            "row_id":str(row["id"]),
        }
    else:
        params={
            "split":split,
            "example":example,
            "seed":BASE_SEED,
            "relation_count":9 if split=="final" else 4,
            "field_count":18 if split=="final" else 6,
            "answer_count":7 if split=="final" else 4,
        }
        row=base.materialize_row(**params)
        base_materialization={
            "kind":"behavioral",
            "params":params,
        }
    row["internal_view_descriptions"]=list(INTERNAL_VIEW_DESCRIPTIONS)
    locator=target_locator(row,surface)
    original=get_text(row,locator)
    boundary_pair_id=None
    decisive_start_word=None
    decisive_end_word=None
    if placement_variant=="tail":
        transformed=longify(
            original,
            surface=surface,
            target_words=target_words,
        )
    elif placement_variant in {"boundary_early","boundary_late"}:
        boundary_pair_id=f"{split}:long-context-boundary:{surface}"
        # Keep total length constant while moving the identical decisive text
        # from well before to well after the first overlap-ownership boundary.
        # Use a very wide tokenizer-agnostic separation. The exact governed
        # tokenizer must still verify token-space placement before gradient,
        # but the early case is intentionally far from the first ownership
        # boundary while the late case is guaranteed to occur after at least
        # 4096 word tokens.
        requested_start=512 if placement_variant=="boundary_early" else 4300
        transformed,decisive_start_word,decisive_end_word=boundary_shift_text(
            original,
            surface=surface,
            total_words=target_words,
            decisive_start_word=requested_start,
        )
    else:
        raise ValueError(f"unknown placement_variant: {placement_variant}")
    set_text(row,locator,transformed)
    row["schema"]=ROW_SCHEMA
    row["lane"]="full_envelope_long_context_supplement"
    row["id"]=f"felc_{split}_{surface}_{placement_variant}_{example:04d}"
    row["template_id"]=(
        f"{split}:long-context:{surface}:{placement_variant}:{row['scenario_family']}"
    )
    row["template_signature_sha256"]=base.template_signature(
        str(row["query"]),
        list(row["entities"]),
    )
    row["causal_group"]=f"{split}:long-context:{surface}:{placement_variant}:{example:04d}"
    row["data_origin"]=(
        "deterministic_public_full_envelope_long_context_final_v2"
        if split=="final"
        else "deterministic_public_full_envelope_long_context_supplement_v1"
    )
    row["long_context_surface"]=surface
    row["long_context_locator"]=locator
    row["long_context_placement_variant"]=placement_variant
    row["boundary_shift_pair_id"]=boundary_pair_id
    row["decisive_start_word"]=decisive_start_word
    row["decisive_end_word"]=decisive_end_word
    row["long_context_word_operating_point"]=int(target_words)
    row["long_context_word_operating_point_is_product_ceiling"]=False
    row["base_materialization"]=base_materialization
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
        materialize(
            split=split,
            surface=surface,
            target_words=args.long_word_target,
            placement_variant=variant,
        )
        for split in ("train","dev")
        for surface in SURFACES
        for variant in ("tail","boundary_early","boundary_late")
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
        "placement_histogram":dict(sorted(Counter(x["long_context_placement_variant"] for x in rows).items())),
        "boundary_shift_pair_count":sum(
            1 for split in ("train","dev") for surface in SURFACES
        ),
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
