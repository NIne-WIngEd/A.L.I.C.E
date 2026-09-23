#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_full_envelope_behavioral_curriculum_v1 as behavioral
import build_n0_v02_semantic_operator_intervention_curriculum_v1 as semantic
import build_n0_v02_full_envelope_runtime_view_curriculum_v1 as runtime_views
import build_n0_v02_full_envelope_long_context_curriculum_v1 as long_context

PASS_STATUS="MATERIALIZED_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_NO_RESULTS"

FINAL_SEMANTIC_ENTITIES=[
    "FinalAlder","FinalBasin","FinalCobalt","FinalDrift","FinalEsker","FinalFlint",
    "FinalGrove","FinalHarbor","FinalIon","FinalJasper","FinalKeel","FinalMoraine",
    "FinalNimbus","FinalOpal","FinalPrairie","FinalQuartz","FinalReef","FinalSummit",
    "FinalTundra","FinalUmber","FinalVale","FinalWarden","FinalXylem","FinalYonder",
]
FINAL_SEMANTIC_RELATIONS=[
    semantic.rel("fz001","attestation","The source artifact receives a formal attestation from the target authority.","artifact receiving attestation","attesting authority",["receives formal attestation from","is formally attested by"]),
    semantic.rel("fz002","reservation","The source resource is reserved for the target activity.","reserved resource","activity holding the reservation",["is reserved for","is held in reserve for"]),
    semantic.rel("fz003","synchronization","The source process is synchronized against the target reference clock or process.","process being synchronized","reference process or clock",["is synchronized with","is aligned in time with"],symmetric=True),
    semantic.rel("fz004","allocation","The source resource is allocated to the target recipient or task.","allocated resource","recipient or task",["is allocated to","is assigned for use by"]),
    semantic.rel("fz005","activation","The source condition activates the target mechanism or state.","activating condition","activated mechanism or state",["activates","switches on"]),
    semantic.rel("fz006","revocation","The source authority revokes the target permission or credential.","revoking authority","permission or credential revoked",["revokes","withdraws authorization for"]),
    semantic.rel("fz007","adjudication","The source decision adjudicates the target dispute or case.","decision","dispute or case decided",["adjudicates","resolves by formal decision"]),
    semantic.rel("fz008","handoff","The source process hands responsibility to the target process or actor.","originating process or actor","receiving process or actor",["hands responsibility to","transfers control to"]),
    semantic.rel("fz009","annotation","The source note annotates the target artifact with explanatory metadata.","annotating note","artifact being annotated",["annotates","adds explanatory metadata to"]),
    semantic.rel("fz010","co_location","The source and target occupy the same designated operational location.","one located item","other co-located item",["is co-located with","shares the designated location with"],symmetric=True),
    semantic.rel("fz011","eligibility","The source condition establishes eligibility for the target program or action.","qualifying condition","program or action enabled by eligibility",["establishes eligibility for","qualifies for"]),
    semantic.rel("fz012","notification","The source event notifies the target recipient of a state change.","notification event","recipient informed",["notifies","sends a state-change notice to"]),
]


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
    p.add_argument("--semantic-final-output",required=True)
    p.add_argument("--runtime-view-final-output",required=True)
    p.add_argument("--long-context-final-output",required=True)
    p.add_argument("--manifest-output",required=True)
    p.add_argument("--examples-per-mode",type=int,default=2)
    p.add_argument("--semantic-examples-per-relation",type=int,default=12)
    p.add_argument("--seed",type=int,default=20260922)
    args=p.parse_args()
    if args.examples_per_mode <= 0 or args.semantic_examples_per_relation <= 0:
        raise SystemExit("FINAL example counts must be positive")

    behavioral_rows_path=Path(args.behavioral_train_dev_rows)
    fewrel_rows_path=Path(args.fewrel_final_rows)
    fewrel_bank_path=Path(args.fewrel_final_bank)
    fewrel_manifest_path=Path(args.fewrel_manifest)
    final_contract_path=Path(args.final_contract)
    synthetic_path=Path(args.synthetic_output)
    semantic_path=Path(args.semantic_final_output)
    runtime_view_path=Path(args.runtime_view_final_output)
    long_context_path=Path(args.long_context_final_output)
    manifest_path=Path(args.manifest_output)
    if any(path.exists() for path in (synthetic_path,semantic_path,runtime_view_path,long_context_path,manifest_path)):
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


    semantic_rows=[]
    candidate_points=(4,8,12)
    for relation_index,relation in enumerate(FINAL_SEMANTIC_RELATIONS):
        second=FINAL_SEMANTIC_RELATIONS[
            (relation_index+1)%len(FINAL_SEMANTIC_RELATIONS)
        ]
        for example in range(args.semantic_examples_per_relation):
            pair_anchor=example-1 if example%12==1 else example
            rng=random.Random(
                args.seed
                + 30_000_000
                + relation_index*10007
                + pair_anchor*97
            )
            semantic_rows.append(
                semantic.make_row(
                    split="final",
                    relation=relation,
                    second=second,
                    example=example,
                    candidates=candidate_points[
                        pair_anchor%len(candidate_points)
                    ],
                    rng=rng,
                    entity_pool=FINAL_SEMANTIC_ENTITIES,
                    candidate_pool=FINAL_SEMANTIC_RELATIONS,
                    relation_partition="sealed_final_relation_family",
                    template_partition="sealed_final_templates",
                    data_origin="deterministic_public_semantic_operator_final_v2",
                )
            )
    semantic_path.parent.mkdir(parents=True,exist_ok=True)
    semantic_path.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in semantic_rows),
        encoding="utf-8",
    )


    runtime_view_rows=runtime_views.materialize_rows("final")
    runtime_view_path.parent.mkdir(parents=True,exist_ok=True)
    runtime_view_path.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in runtime_view_rows),
        encoding="utf-8",
    )

    long_context_rows=[
        long_context.materialize(
            split="final",
            surface=surface,
            target_words=4608,
            placement_variant=variant,
        )
        for surface in long_context.SURFACES
        for variant in ("tail","boundary_early","boundary_late")
    ]
    long_context_rows.append(
        long_context.materialize(
            split="final",
            surface="field_text",
            target_words=4608,
            placement_variant="tail",
            example_override=8,
        )
    )
    long_context_path.parent.mkdir(parents=True,exist_ok=True)
    long_context_path.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in long_context_rows),
        encoding="utf-8",
    )

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
        "semantic_final_rows_sha256":sha256(semantic_path),
        "runtime_view_final_rows_sha256":sha256(runtime_view_path),
        "long_context_final_rows_sha256":sha256(long_context_path),
        "fewrel_final_rows_sha256":sha256(fewrel_rows_path),
        "fewrel_final_bank_sha256":sha256(fewrel_bank_path),
        "fewrel_manifest_sha256":sha256(fewrel_manifest_path),
        "final_contract_sha256":sha256(final_contract_path),
        "synthetic_rows":len(rows),
        "semantic_final_rows":len(semantic_rows),
        "semantic_final_relation_families":sorted({str(x["relation_family"]) for x in semantic_rows}),
        "semantic_final_interventions":sorted({str(x["intervention"]) for x in semantic_rows}),
        "semantic_final_plurality_rows":sum(1 for x in semantic_rows if x.get("plurality_supervision_required")),
        "runtime_view_final_rows":len(runtime_view_rows),
        "runtime_view_final_scenarios":sorted({str(x["scenario_family"]) for x in runtime_view_rows}),
        "long_context_final_rows":len(long_context_rows),
        "long_context_final_surfaces":sorted({str(x["long_context_surface"]) for x in long_context_rows}),
        "long_context_final_placements":sorted({str(x["long_context_placement_variant"]) for x in long_context_rows}),
        "long_context_final_conflict_rows":sum(1 for x in long_context_rows if x.get("scenario_family")=="conflict_plurality"),
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
