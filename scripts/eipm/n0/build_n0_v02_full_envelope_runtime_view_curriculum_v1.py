#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_full_envelope_behavioral_curriculum_v1 as behavioral

ROW_SCHEMA="alice.eipm.n0.full-envelope-runtime-view-row.v1"
MANIFEST_SCHEMA="alice.eipm.n0.full-envelope-runtime-view-manifest.v1"
BASE_SEED=20260922


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(split: str) -> dict[str,Any]:
    row=behavioral.materialize_row(
        split=split,
        example=15,
        seed=BASE_SEED,
        relation_count=4,
        field_count=6,
        answer_count=4,
    )
    row=copy.deepcopy(row)
    row["candidate_context_swap_pair_id"]=None
    row["candidate_context_swap_variant"]=None
    row["decisive_field_indices"]=[]
    row["irrelevant_field_indices"]=[]
    row["decisive_view_active"]=False
    row["irrelevant_view_active"]=False
    row["recoverable_view_names"]=[]
    return row


def _query(split: str) -> str:
    if split=="train":
        return (
            "Use the runtime-described public evidence views to choose between "
            "the supplied candidate conclusions. First determine which view is "
            "about this query. If two relevant views conflict, use their declared "
            "reliability as evidence quality."
        )
    return (
        "Choose among the same candidate conclusions using the supplied runtime "
        "evidence streams. Separate query-relevant streams from merely available "
        "ones, then use stated reliability only among streams that actually bear "
        "on the question."
    )


def _descriptor(split: str) -> str:
    if split=="train":
        return "runtime-described public evidence stream with no fixed learned view identity"
    return "public runtime evidence channel described at inference time without a learned channel ID"


def _relevant_text(split: str, candidate: str, label: str) -> str:
    if split=="train":
        return (
            f"Current independently checked evidence for this exact query {label} "
            f"supports the following candidate conclusion: {candidate}"
        )
    return (
        f"A current independently validated source bearing directly on the question {label} "
        f"backs this candidate conclusion: {candidate}"
    )


def _irrelevant_text(split: str, candidate: str) -> str:
    if split=="train":
        return (
            "A highly reliable public archive note concerns an unrelated facility "
            f"schedule. It incidentally repeats words from this candidate but does "
            f"not bear on the current query: {candidate}"
        )
    return (
        "A trustworthy public background bulletin addresses a separate operational "
        f"topic. It happens to mention wording from this candidate but is not "
        f"evidence for the present question: {candidate}"
    )


def _unavailable_text(split: str, candidate: str) -> str:
    prefix=(
        "Unavailable archived evidence would have supported"
        if split=="train"
        else "An unavailable historical channel would have favored"
    )
    return f"{prefix} this candidate: {candidate}"


def _decorate(
    row: dict[str,Any],
    *,
    split: str,
    family: str,
    variant: str,
    target: int,
    views: list[dict[str,Any]],
) -> dict[str,Any]:
    row=copy.deepcopy(row)
    row["schema"]=ROW_SCHEMA
    row["lane"]="full_envelope_runtime_view_supplement"
    row["scenario_family"]=family
    row["id"]=f"ferv_{split}_{family}_{variant}"
    row["query"]=_query(split)
    row["public_target_index"]=int(target)
    row["template_id"]=f"{split}:runtime-view:{family}"
    row["template_signature_sha256"]=behavioral.template_signature(
        row["query"],list(row["entities"])
    )
    row["causal_group"]=f"{split}:runtime-view:{family}:{variant}"
    row["additional_views"]=views
    row["runtime_additional_view_count"]=len(views)
    row["decisive_view_active"]=any(bool(x.get("decisive")) for x in views)
    row["irrelevant_view_active"]=any(bool(x.get("irrelevant")) for x in views)
    row["data_origin"]="deterministic_public_full_envelope_runtime_view_supplement_v1"
    row["generated_text"]=True
    row["private_identity_data"]=False
    row["training_authorized"]=split=="train"
    row["model_selection_authorized"]=split=="dev"
    row["final_validation_only"]=False
    return row


def materialize_rows(split: str) -> list[dict[str,Any]]:
    if split not in {"train","dev"}:
        raise ValueError("runtime-view supplement is TRAIN/DEV only")
    seed=_base(split)
    candidates=list(seed["candidate_answers"])
    if len(candidates)<2:
        raise RuntimeError("runtime-view supplement requires at least two candidates")
    descriptor=_descriptor(split)
    rows=[]

    # Pair 1: availability and reliability are held fixed. Only which source
    # is actually about the query changes, so target flips cannot be learned
    # from view position, availability, descriptor, or reliability.
    for variant,target in (("a",0),("b",1)):
        other=1-target
        views=[
            {
                "source_text":(
                    _relevant_text(split,candidates[0],"stream A")
                    if target==0
                    else _irrelevant_text(split,candidates[0])
                ),
                "descriptor_text":descriptor,
                "available":True,
                "reliability":0.80,
                "recoverable":target==0,
                "decisive":target==0,
                "irrelevant":target!=0,
            },
            {
                "source_text":(
                    _relevant_text(split,candidates[1],"stream B")
                    if target==1
                    else _irrelevant_text(split,candidates[1])
                ),
                "descriptor_text":descriptor,
                "available":True,
                "reliability":0.80,
                "recoverable":target==1,
                "decisive":target==1,
                "irrelevant":target!=1,
            },
            {
                "source_text":_unavailable_text(split,candidates[other]),
                "descriptor_text":descriptor,
                "available":False,
                "reliability":1.0,
                "recoverable":False,
                "decisive":False,
                "irrelevant":False,
            },
        ]
        rows.append(_decorate(
            seed,split=split,family="runtime_view_relevance_flip",
            variant=variant,target=target,views=views,
        ))

    # Pair 2: both first views are query-relevant and semantically fixed.
    # Reliability reverses which one should control. A third available source
    # is even more reliable but query-irrelevant, preventing raw reliability
    # from becoming the relevance label.
    fixed_sources=[
        _relevant_text(split,candidates[0],"stream A"),
        _relevant_text(split,candidates[1],"stream B"),
    ]
    for variant,(r0,r1,target) in (
        ("a",(0.90,0.20,0)),
        ("b",(0.20,0.90,1)),
    ):
        views=[
            {
                "source_text":fixed_sources[0],
                "descriptor_text":descriptor,
                "available":True,
                "reliability":r0,
                "recoverable":True,
                "decisive":target==0,
                "irrelevant":False,
            },
            {
                "source_text":fixed_sources[1],
                "descriptor_text":descriptor,
                "available":True,
                "reliability":r1,
                "recoverable":True,
                "decisive":target==1,
                "irrelevant":False,
            },
            {
                # Keep this high-reliability irrelevant control semantically
                # identical across the pair. The only pair intervention is the
                # reliability reversal between the two query-relevant views.
                "source_text":_irrelevant_text(split,candidates[0]),
                "descriptor_text":descriptor,
                "available":True,
                "reliability":0.95,
                "recoverable":False,
                "decisive":False,
                "irrelevant":True,
            },
        ]
        rows.append(_decorate(
            seed,split=split,family="runtime_view_reliability_reversal",
            variant=variant,target=target,views=views,
        ))
    return rows


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--manifest",required=True)
    args=p.parse_args()
    output=Path(args.output)
    manifest_path=Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite runtime-view curriculum artifacts")
    rows=materialize_rows("train")+materialize_rows("dev")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),
        encoding="utf-8",
    )
    manifest={
        "schema":MANIFEST_SCHEMA,
        "status":"MATERIALIZED_PUBLIC_FULL_ENVELOPE_RUNTIME_VIEW_TRAIN_DEV",
        "sha256":sha256(output),
        "rows":len(rows),
        "train_rows":sum(1 for x in rows if x["split"]=="train"),
        "dev_rows":sum(1 for x in rows if x["split"]=="dev"),
        "scenario_histogram":dict(sorted(Counter(x["scenario_family"] for x in rows).items())),
        "additional_view_count_points":sorted({x["runtime_additional_view_count"] for x in rows}),
        "additional_view_count_is_product_ceiling":False,
        "fixed_view_taxonomy":False,
        "final_rows":0,
        "private_identity_data":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "n0_complete":False,
    }
    manifest_path.write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
