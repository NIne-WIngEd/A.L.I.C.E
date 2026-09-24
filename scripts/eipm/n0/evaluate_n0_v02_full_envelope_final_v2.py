#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
    compile_behavioral_batch,
)
from alice_personality.n0.full_envelope_runtime_factory_v1 import (
    load_registered_full_envelope_system,
)
from alice_personality.n0.natural_relation_batch_v1 import (
    compile_natural_relation_batch,
)
from alice_personality.n0.semantic_operator_batch_v1 import (
    compile_semantic_operator_batch,
)
from alice_personality.n0.source_authority_v1 import (
    repository_root,
    require_canonical_source_file,
)
from alice_personality.n0.v02_training import (
    TeacherMultitaskCollator,
    sha256_file,
    verify_teacher_registry,
)
from evaluate_n0_v02_full_envelope_dev_v1 import (
    _candidate_subset_batch_accuracy,
    fabric_lane_metrics,
    fixed_preference_top1,
    natural_lane_metrics,
    read_fixed_eval_rows,
    semantic_lane_metrics,
    teacher_top1_rate,
    to_device,
)

PASS_STATUS="PASS_N0_FULL_ENVELOPE_FINAL_SELF_VALIDATION"
FAIL_STATUS="FAIL_N0_FULL_ENVELOPE_FINAL_SELF_VALIDATION"
COMPLETE_STATUS="FINAL_V2_EVALUATION_COMPLETE"


def read_json(path: str | Path) -> dict[str,Any]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"empty FINAL artifact: {path}")
    return rows


def nested_get(root: Mapping[str,Any], path: str) -> Any:
    current: Any=root
    for part in str(path).split("."):
        if not isinstance(current,Mapping) or part not in current:
            raise KeyError(path)
        current=current[part]
    return current


def mean_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(int(x) for x in values)/len(values))


def mean_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(x) for x in values)/len(values))


def candidate_probabilities(output: Mapping[str,Any]) -> torch.Tensor:
    judgment=output["public_judgment"]
    logits=judgment["candidate_logits"].float()
    valid=judgment["candidate_valid_mask"].bool()
    floor=torch.finfo(logits.dtype).min
    return torch.softmax(logits.masked_fill(~valid,floor),dim=-1)


def final_semantic_derived_metrics(
    *,
    rows: list[dict[str,Any]],
    semantic_metrics: Mapping[str,Any],
) -> dict[str,Any]:
    row_by_id={str(row["id"]):row for row in rows}
    records=list(semantic_metrics.get("records") or [])
    ordered_groups=defaultdict(dict)
    reverse=[]
    multistep=[]
    for rec in records:
        row=row_by_id[str(rec["id"])]
        intervention=str(rec.get("intervention",""))
        success=bool(rec["relation_exact"]) and bool(rec["event_exact"])
        if intervention in {"ordered_composition","reverse_ordered_composition"}:
            ordered_groups[str(row["relation_family"])][intervention]=success
        if intervention=="reverse_ordered_composition":
            reverse.append(success)
        if intervention in {
            "ordered_composition","reverse_ordered_composition",
            "mixed_direction_composition","mixed_step_modifier_composition",
        }:
            multistep.append(success)
    pair_success=[
        values.get("ordered_composition",False)
        and values.get("reverse_ordered_composition",False)
        for values in ordered_groups.values()
        if {
            "ordered_composition","reverse_ordered_composition"
        }<=set(values)
    ]
    ordered_family_values=[
        bool(rec["relation_exact"]) and bool(rec["event_exact"])
        for rec in records
        if rec.get("intervention") in {
            "ordered_composition","reverse_ordered_composition"
        }
    ]
    return {
        "relation_order_pair_completion":mean_bool(pair_success),
        "relation_order_pair_count":len(pair_success),
        "reverse_ordered_accuracy":mean_bool(reverse),
        "reverse_ordered_count":len(reverse),
        "multi_step_relation_exact":mean_bool(multistep),
        "multi_step_relation_count":len(multistep),
        "ordered_family_min_success":(
            1.0 if ordered_family_values and all(ordered_family_values)
            else mean_bool(ordered_family_values)
        ),
        "ordered_family_count":len(ordered_family_values),
        "nonrelational_false_assertion_rate":(
            1.0-float(semantic_metrics["unknown_defer_accuracy"])
        ),
    }


def record_subset(
    *,
    records: list[dict[str,Any]],
    rows: list[dict[str,Any]],
    predicate,
) -> tuple[float,int]:
    by_id={str(row["id"]):row for row in rows}
    values=[]
    for rec in records:
        row=by_id.get(str(rec["id"]))
        if row is None or not predicate(row,rec):
            continue
        value=rec.get("public_judgment_correct")
        if value is not None:
            values.append(bool(value))
    return mean_bool(values),len(values)


def extrapolation_metrics(
    *,
    system: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    behavioral_rows: list[dict[str,Any]],
    long_rows: list[dict[str,Any]],
    fabric: Mapping[str,Any],
) -> dict[str,Any]:
    records=list(fabric.get("records") or [])
    all_rows=behavioral_rows+long_rows
    max_candidate=max(int(x["runtime_candidate_answer_count"]) for x in behavioral_rows)
    candidate_accuracy,candidate_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: (
            row in behavioral_rows
            and int(row.get("runtime_candidate_answer_count",0))==max_candidate
        ),
    )
    factor_accuracy,factor_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row in behavioral_rows,
    )
    longer_accuracy,longer_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row.get("scenario_family")=="long_causal_chain",
    )
    type_accuracy,type_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row in behavioral_rows,
    )
    domain_accuracy,domain_count=type_accuracy,type_count
    sym_accuracy,sym_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row.get("scenario_family")=="symmetric_relation",
    )
    combo_accuracy,combo_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: (
            row.get("scenario_family")=="heldout_recency_provenance_combo"
        ),
    )
    distant_accuracy,distant_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: (
            row.get("long_context_surface") in {"field_text","additional_view_source"}
            and row.get("long_context_placement_variant") in {"tail","boundary_late"}
        ),
    )
    relation_accuracy,relation_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: (
            row.get("long_context_surface")=="relation_schema"
            and row.get("long_context_placement_variant") in {"tail","boundary_late"}
        ),
    )
    conflict_accuracy,conflict_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: (
            row.get("scenario_family")=="conflict_plurality"
            and row.get("long_context_surface") is not None
        ),
    )
    long_factor_accuracy,long_factor_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row.get("long_context_surface")=="factor_schema",
    )
    long_field_accuracy,long_field_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row.get("long_context_surface")=="field_text",
    )
    long_candidate_accuracy,long_candidate_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row.get("long_context_surface")=="candidate_text",
    )
    descriptor_surfaces={
        "field_descriptor","internal_view_descriptor","additional_view_descriptor"
    }
    long_descriptor_accuracy,long_descriptor_count=record_subset(
        records=records,rows=all_rows,
        predicate=lambda row,_: row.get("long_context_surface") in descriptor_surfaces,
    )
    candidate_subset=_candidate_subset_batch_accuracy(
        system=system,
        rows=behavioral_rows,
        tokenizer=tokenizer,
        device=device,
    )
    return {
        "candidate_cardinality_accuracy":candidate_accuracy,
        "candidate_cardinality_count":candidate_count,
        "factor_cardinality_accuracy":factor_accuracy,
        "factor_cardinality_count":factor_count,
        "longer_composition_accuracy":longer_accuracy,
        "longer_composition_count":longer_count,
        "type_schema_accuracy":type_accuracy,
        "type_schema_count":type_count,
        "domain_transfer_accuracy":domain_accuracy,
        "domain_transfer_count":domain_count,
        "candidate_subset_batch_accuracy":float(candidate_subset),
        "symmetric_relation_accuracy":sym_accuracy,
        "symmetric_relation_count":sym_count,
        "heldout_factor_combination_accuracy":combo_accuracy,
        "heldout_factor_combination_count":combo_count,
        "long_distant_accuracy":distant_accuracy,
        "long_distant_count":distant_count,
        "cross_window_relation_accuracy":relation_accuracy,
        "cross_window_relation_count":relation_count,
        "cross_window_conflict_accuracy":conflict_accuracy,
        "cross_window_conflict_count":conflict_count,
        "long_relation_accuracy":relation_accuracy,
        "long_relation_count":relation_count,
        "long_factor_accuracy":long_factor_accuracy,
        "long_factor_count":long_factor_count,
        "long_field_accuracy":long_field_accuracy,
        "long_field_count":long_field_count,
        "long_candidate_accuracy":long_candidate_accuracy,
        "long_candidate_count":long_candidate_count,
        "long_descriptor_accuracy":long_descriptor_accuracy,
        "long_descriptor_count":long_descriptor_count,
    }


def runtime_view_metrics(
    *,
    records: list[dict[str,Any]],
    rows: list[dict[str,Any]],
) -> dict[str,Any]:
    relevance,rel_count=record_subset(
        records=records,rows=rows,
        predicate=lambda row,_: row.get("scenario_family")=="runtime_view_relevance_flip",
    )
    reliability,quality_count=record_subset(
        records=records,rows=rows,
        predicate=lambda row,_: row.get("scenario_family")=="runtime_view_reliability_reversal",
    )
    unavailable_rows=[
        row for row in rows
        if any(not bool(v.get("available",True)) for v in row.get("additional_views") or [])
    ]
    unavailable_ids={str(x["id"]) for x in unavailable_rows}
    unavailable_values=[
        bool(rec["public_judgment_correct"])
        for rec in records
        if str(rec["id"]) in unavailable_ids
        and rec.get("public_judgment_correct") is not None
    ]
    return {
        "relevance_flip_accuracy":relevance,
        "relevance_flip_count":rel_count,
        "reliability_reversal_accuracy":reliability,
        "reliability_reversal_count":quality_count,
        "unavailable_view_accuracy":mean_bool(unavailable_values),
        "unavailable_view_count":len(unavailable_values),
    }


def _neutral_span(length: int) -> str:
    token="neutral"
    out=(token*((length+len(token)-1)//len(token)))[:length]
    return out


def semantic_evidence_counterfactual_metrics(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
    minimum_probability_drop: float = 0.10,
) -> dict[str,Any]:
    query_success=[]
    schema_success=[]
    for row in rows:
        targets=list(row.get("relation_sequence_target") or [])
        if not targets:
            continue
        original=to_device(
            compile_semantic_operator_batch(rows=[row],tokenizer=tokenizer),
            device,
        )
        output=system(task="semantic_operator",batch=original["batch"])
        logits=output["semantic_operator"]["relation_logits"][:,0,:].float()
        target=int(targets[0])
        base_prob=float(torch.softmax(logits,dim=-1)[0,target].item())

        q_spans=list(row.get("query_relation_evidence_char_spans") or [])
        if q_spans:
            changed=copy.deepcopy(row)
            span=changed["query_relation_evidence_char_spans"][0]
            start,end=int(span["start"]),int(span["end"])
            replacement=_neutral_span(end-start)
            query=str(changed["query"])
            changed["query"]=query[:start]+replacement+query[end:]
            span["text"]=replacement
            compiled=to_device(
                compile_semantic_operator_batch(rows=[changed],tokenizer=tokenizer),
                device,
            )
            altered=system(task="semantic_operator",batch=compiled["batch"])
            altered_prob=float(torch.softmax(
                altered["semantic_operator"]["relation_logits"][:,0,:].float(),
                dim=-1,
            )[0,target].item())
            query_success.append(base_prob-altered_prob>=minimum_probability_drop)

        s_spans=list(row.get("relation_schema_evidence_char_spans") or [])
        if s_spans:
            changed=copy.deepcopy(row)
            span=changed["relation_schema_evidence_char_spans"][0]
            index=int(span["candidate_index"])
            start,end=int(span["start"]),int(span["end"])
            replacement=_neutral_span(end-start)
            text=str(changed["relation_candidates"][index]["text"])
            changed["relation_candidates"][index]["text"]=(
                text[:start]+replacement+text[end:]
            )
            span["text"]=replacement
            compiled=to_device(
                compile_semantic_operator_batch(rows=[changed],tokenizer=tokenizer),
                device,
            )
            altered=system(task="semantic_operator",batch=compiled["batch"])
            altered_prob=float(torch.softmax(
                altered["semantic_operator"]["relation_logits"][:,0,:].float(),
                dim=-1,
            )[0,target].item())
            schema_success.append(base_prob-altered_prob>=minimum_probability_drop)
    return {
        "query_evidence_counterfactual_sensitivity":mean_bool(query_success),
        "query_evidence_counterfactual_count":len(query_success),
        "schema_evidence_counterfactual_sensitivity":mean_bool(schema_success),
        "schema_evidence_counterfactual_count":len(schema_success),
    }


def natural_candidate_permutation_delta(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    bank: Mapping[str,Any],
    tokenizer: Any,
    device: torch.device,
) -> tuple[float,int]:
    deltas=[]
    for row in rows:
        original=to_device(
            compile_natural_relation_batch(
                rows=[row],relation_bank=bank,tokenizer=tokenizer
            ),
            device,
        )
        out=system(task="natural_relation",batch=original["batch"])
        p=torch.softmax(
            out["semantic_operator"]["relation_logits"][:,0,:].float(),dim=-1
        )[0]
        original_keys=list(row["candidate_relation_keys"])
        by_key={key:float(p[i].item()) for i,key in enumerate(original_keys)}

        changed=copy.deepcopy(row)
        changed["candidate_relation_keys"]=list(reversed(original_keys))
        changed["target_candidate_index"]=changed["candidate_relation_keys"].index(
            str(changed["target_relation_key"])
        )
        perm=to_device(
            compile_natural_relation_batch(
                rows=[changed],relation_bank=bank,tokenizer=tokenizer
            ),
            device,
        )
        out2=system(task="natural_relation",batch=perm["batch"])
        p2=torch.softmax(
            out2["semantic_operator"]["relation_logits"][:,0,:].float(),dim=-1
        )[0]
        delta=max(
            abs(by_key[key]-float(p2[i].item()))
            for i,key in enumerate(changed["candidate_relation_keys"])
        )
        deltas.append(delta)
    return (max(deltas) if deltas else float("inf"),len(deltas))


def boundary_shift_delta(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
) -> tuple[float,int]:
    groups=defaultdict(dict)
    for row in rows:
        pair=row.get("boundary_shift_pair_id")
        variant=row.get("long_context_placement_variant")
        if pair and variant in {"boundary_early","boundary_late"}:
            groups[str(pair)][str(variant)]=row
    deltas=[]
    for variants in groups.values():
        if set(variants)!={"boundary_early","boundary_late"}:
            continue
        probs={}
        for variant,row in variants.items():
            compiled=to_device(
                compile_behavioral_batch(rows=[row],tokenizer=tokenizer),
                device,
            )
            out=system(task="full_envelope",batch=compiled["primary_batch"])
            probs[variant]=candidate_probabilities(out)[0]
        if probs["boundary_early"].shape!=probs["boundary_late"].shape:
            raise RuntimeError("boundary pair candidate geometry drift")
        deltas.append(float(
            (probs["boundary_early"]-probs["boundary_late"]).abs().max().item()
        ))
    return (max(deltas) if deltas else float("inf"),len(deltas))


def irrelevant_growth_delta(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
) -> tuple[float,int]:
    candidates=[x for x in rows if x.get("scenario_family")=="irrelevant_distractor"]
    deltas=[]
    for row in candidates:
        original=to_device(
            compile_behavioral_batch(rows=[row],tokenizer=tokenizer),
            device,
        )
        p=candidate_probabilities(
            system(task="full_envelope",batch=original["primary_batch"])
        )[0]
        changed=copy.deepcopy(row)
        template_index=(
            int(changed["irrelevant_field_indices"][0])
            if changed.get("irrelevant_field_indices")
            else len(changed["fields"])-1
        )
        template=copy.deepcopy(changed["fields"][template_index])
        for i in range(8):
            item=copy.deepcopy(template)
            item["text"]=(
                str(item["text"])
                + f" Extra held-out irrelevant background item {i} adds no evidence."
            )
            changed["irrelevant_field_indices"].append(len(changed["fields"]))
            changed["fields"].append(item)
        changed["runtime_field_count"]=len(changed["fields"])
        changed["field_permutation"]=list(range(len(changed["fields"])))
        compiled=to_device(
            compile_behavioral_batch(rows=[changed],tokenizer=tokenizer),
            device,
        )
        p2=candidate_probabilities(
            system(task="full_envelope",batch=compiled["primary_batch"])
        )[0]
        deltas.append(float((p-p2).abs().max().item()))
    return (max(deltas) if deltas else float("inf"),len(deltas))


def paraphrase_consistency(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
) -> tuple[float,int]:
    values=[]
    for row in rows:
        alternate=row.get("final_paraphrase_query")
        if not isinstance(alternate,str) or not alternate.strip():
            continue
        compiled=to_device(
            compile_behavioral_batch(rows=[row],tokenizer=tokenizer),
            device,
        )
        normal=system(task="full_envelope",batch=compiled["primary_batch"])
        normal_pred=int(
            normal["public_judgment"]["candidate_logits"].argmax(dim=-1)[0].item()
        )
        changed=copy.deepcopy(row)
        changed["query"]=alternate
        compiled2=to_device(
            compile_behavioral_batch(rows=[changed],tokenizer=tokenizer),
            device,
        )
        alt=system(task="full_envelope",batch=compiled2["primary_batch"])
        alt_pred=int(
            alt["public_judgment"]["candidate_logits"].argmax(dim=-1)[0].item()
        )
        target=int(row["public_target_index"])
        values.append(normal_pred==target and alt_pred==target)
    return mean_bool(values),len(values)


def candidate_context_swap_metric(
    *,
    records: list[dict[str,Any]],
    rows: list[dict[str,Any]],
) -> tuple[float,int]:
    rec_by_id={str(x["id"]):x for x in records}
    groups=defaultdict(list)
    for row in rows:
        pair=row.get("candidate_context_swap_pair_id")
        if pair:
            rec=rec_by_id.get(str(row["id"]))
            if rec is not None and rec.get("public_judgment_correct") is not None:
                groups[str(pair)].append(bool(rec["public_judgment_correct"]))
    values=[len(v)>=2 and all(v) for v in groups.values()]
    return mean_bool(values),len(values)


def evaluate_gate_registry(
    *,
    registry: Mapping[str,Any],
    final_contract: Mapping[str,Any],
    metrics: Mapping[str,Any],
    proof_contract: Mapping[str,Any],
    static_receipt: Mapping[str,Any],
    source_revision: str,
) -> tuple[dict[str,Any],list[str]]:
    if static_receipt.get("status")!="PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1":
        raise SystemExit("static proof receipt not PASS")
    if static_receipt.get("source_revision")!=source_revision:
        raise SystemExit("FINAL candidate/static-proof source revision drift")
    obligations={
        str(x["id"]):x for x in proof_contract.get("obligations") or []
    }
    failures=[]
    results={}
    for section,mapping in registry["sections"].items():
        section_result={}
        for gate,spec in mapping.items():
            threshold=final_contract[section][gate]
            kind=spec["kind"]
            passed=False
            observed=None
            coverage=None
            if kind=="metric":
                try:
                    observed=nested_get(metrics,spec["path"])
                except KeyError:
                    failures.append(f"{section}.{gate}: metric missing {spec['path']}")
                    section_result[gate]={"pass":False,"reason":"metric_missing"}
                    continue
                coverage_path=spec.get("coverage_path")
                if coverage_path:
                    try:
                        coverage=nested_get(metrics,coverage_path)
                    except KeyError:
                        coverage=0
                    if float(coverage)<=0.0:
                        failures.append(
                            f"{section}.{gate}: empty empirical coverage"
                        )
                comparison=spec.get("comparison")
                if comparison=="positive":
                    passed=float(observed)>0.0
                elif gate.endswith("_min") or gate.endswith("_floor"):
                    passed=float(observed)>=float(threshold)
                elif gate.endswith("_max"):
                    passed=float(observed)<=float(threshold)
                else:
                    passed=bool(observed)==bool(threshold)
                if coverage_path and float(coverage)<=0.0:
                    passed=False
            elif kind=="static_proof":
                ids=list(spec.get("obligation_ids") or [])
                if not ids:
                    raise SystemExit(f"{section}.{gate}: empty static proof mapping")
                passed=True
                for oid in ids:
                    obligation=obligations.get(str(oid))
                    if obligation is None or obligation.get("kind")!="STATIC_REQUIRED":
                        passed=False
                        break
                observed={"obligation_ids":ids}
            elif kind=="declaration":
                observed=threshold
                passed=threshold==spec.get("expected")
            else:
                raise SystemExit(f"unknown FINAL gate mapping kind: {kind}")
            if not passed:
                failures.append(
                    f"{section}.{gate}: observed={observed!r} threshold={threshold!r}"
                )
            section_result[gate]={
                "pass":bool(passed),
                "observed":observed,
                "threshold":threshold,
                "coverage":coverage,
                "kind":kind,
            }
        results[section]=section_result
    return results,failures


def main() -> None:
    p=argparse.ArgumentParser(
        description=(
            "Sealed successor FINAL-v2 evaluator for N0FullEnvelopeTrainableSystemV1. "
            "It performs no checkpoint selection, training, repair, or rerun."
        )
    )
    p.add_argument("--execute-final-evaluation",action="store_true")
    p.add_argument("--final-opening-authorization",required=True)
    p.add_argument("--freeze-receipt",required=True)
    p.add_argument("--opening-authorizer",required=True)
    p.add_argument("--package-config",required=True)
    p.add_argument("--package-manifest",required=True)
    p.add_argument("--package-audit",required=True)
    p.add_argument("--final-contract",required=True)
    p.add_argument("--evaluator-contract",required=True)
    p.add_argument("--gate-registry",required=True)
    p.add_argument("--proof-contract",required=True)
    p.add_argument("--static-proof-receipt",required=True)
    p.add_argument("--topology-config",required=True)
    p.add_argument("--semantic-config",required=True)
    p.add_argument("--semantic-checkpoint",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--candidate-system",required=True)
    p.add_argument("--candidate-receipt",required=True)
    p.add_argument("--dev-selection-receipt",required=True)
    p.add_argument("--dev-evaluation-receipt",required=True)
    p.add_argument("--synthetic-final-rows",required=True)
    p.add_argument("--semantic-final-rows",required=True)
    p.add_argument("--runtime-view-final-rows",required=True)
    p.add_argument("--long-context-final-rows",required=True)
    p.add_argument("--fewrel-final-rows",required=True)
    p.add_argument("--fewrel-final-bank",required=True)
    p.add_argument("--fewrel-manifest",required=True)
    p.add_argument("--teacher-registry",required=True)
    p.add_argument("--teacher-audit",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--teacher-batch-size",type=int,default=8)
    args=p.parse_args()

    if not args.execute_final_evaluation:
        raise SystemExit(
            "FINAL remains sealed: --execute-final-evaluation was not supplied"
        )
    if args.teacher_batch_size<=0:
        raise SystemExit("teacher batch size must be positive")
    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite FINAL-v2 result")

    this_source=Path(__file__).resolve()
    freeze=read_json(args.freeze_receipt)
    if freeze.get("status")!="FROZEN_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_BEFORE_GRADIENT":
        raise SystemExit("FINAL-v2 package freeze receipt not valid")
    if freeze.get("results_observed") is not False:
        raise SystemExit("freeze receipt says FINAL results were already observed")
    frozen_inputs={
        "package_config":args.package_config,
        "evaluator_contract":args.evaluator_contract,
        "evaluator_implementation":this_source,
        "gate_registry":args.gate_registry,
        "opening_authorizer":args.opening_authorizer,
        "final_contract":args.final_contract,
        "synthetic_final_rows":args.synthetic_final_rows,
        "semantic_final_rows":args.semantic_final_rows,
        "runtime_view_final_rows":args.runtime_view_final_rows,
        "long_context_final_rows":args.long_context_final_rows,
        "package_manifest":args.package_manifest,
        "fewrel_final_rows":args.fewrel_final_rows,
        "fewrel_final_bank":args.fewrel_final_bank,
        "fewrel_manifest":args.fewrel_manifest,
        "audit":args.package_audit,
    }
    frozen_hashes=dict(freeze.get("hashes") or {})
    for name,path in frozen_inputs.items():
        observed=sha256_file(path)
        expected=frozen_hashes.get(name)
        if observed!=expected:
            raise SystemExit(
                "frozen FINAL package artifact hash drift: "
                f"{name} expected={expected!r} observed={observed!r}"
            )

    opening=read_json(args.final_opening_authorization)
    selection=read_json(args.dev_selection_receipt)
    dev_selection=read_json(args.dev_evaluation_receipt)
    if opening.get("schema")!="alice.eipm.n0.full-envelope-final-opening-authorization.v1":
        raise SystemExit("FINAL opening authorization schema drift")
    if opening.get("final_opening_authorized") is not True:
        raise SystemExit("FINAL opening is not authorized")
    if opening.get("dev_selection_pass") is not True:
        raise SystemExit("FINAL opening requires passed DEV selection")
    if opening.get("automatic_repair_or_rerun") is not False:
        raise SystemExit("FINAL opening may not authorize automatic repair/rerun")
    if opening.get("threshold_changes_after_results") is not False:
        raise SystemExit("FINAL thresholds may not change after results")
    if opening.get("freeze_receipt_sha256")!=sha256_file(args.freeze_receipt):
        raise SystemExit("FINAL opening/freeze lineage drift")
    if opening.get("opening_authorizer_sha256")!=frozen_hashes.get(
        "opening_authorizer"
    ):
        raise SystemExit("opening authorizer/freeze hash drift")
    if opening.get("candidate_checkpoint_sha256")!=sha256_file(args.candidate_system):
        raise SystemExit("FINAL opening/candidate checkpoint drift")
    if opening.get("selection_receipt_sha256")!=sha256_file(args.dev_selection_receipt):
        raise SystemExit("opening/DEV selection receipt drift")
    if opening.get("dev_receipt_sha256")!=sha256_file(args.dev_evaluation_receipt):
        raise SystemExit("opening/DEV evaluation receipt drift")
    if opening.get("candidate_checkpoint_receipt_sha256")!=sha256_file(
        args.candidate_receipt
    ):
        raise SystemExit("opening/candidate checkpoint receipt drift")
    if opening.get("automatic_checkpoint_selection") is not False:
        raise SystemExit("FINAL opening may not perform automatic checkpoint selection")
    if opening.get("checkpoint_selection_surface")!="DEV_ONLY":
        raise SystemExit("FINAL opening checkpoint selection surface drift")
    if opening.get("selected_stage")!="J3_full_public_n0_coadaptation":
        raise SystemExit("FINAL opening selected stage drift")
    if opening.get("final_results_observed") is not False:
        raise SystemExit("FINAL opening receipt already observed FINAL")

    proof_contract_path=require_canonical_source_file(
        args.proof_contract,
        "configs/eipm/n0/n0_v02_full_envelope_proof_obligations_v1.json",
        label="FINAL proof contract",
    )
    final_contract=read_json(args.final_contract)
    evaluator_contract=read_json(args.evaluator_contract)
    registry=read_json(args.gate_registry)
    proof_contract=read_json(proof_contract_path)
    static_receipt=read_json(args.static_proof_receipt)
    package_manifest=read_json(args.package_manifest)
    package_audit=read_json(args.package_audit)
    candidate=read_json(args.candidate_receipt)
    tracked_status=subprocess.check_output(["git","status","--porcelain"],text=True)
    if tracked_status.strip():
        raise SystemExit("FINAL evaluator requires a clean exact-source worktree")
    current_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if freeze.get("source_revision")!=current_revision:
        raise SystemExit("FINAL freeze source revision drift")
    if candidate.get("source_revision")!=current_revision:
        raise SystemExit(
            "FINAL evaluator source revision does not match candidate"
        )

    if opening.get("source_revision")!=current_revision:
        raise SystemExit("FINAL opening source revision drift")
    if selection.get("source_revision")!=current_revision:
        raise SystemExit("FINAL DEV selection source revision drift")
    if dev_selection.get("source_revision")!=current_revision:
        raise SystemExit("FINAL DEV evaluation source revision drift")

    repo_root=repository_root()
    if static_receipt.get("proof_contract_sha256")!=sha256_file(
        proof_contract_path
    ):
        raise SystemExit("static proof contract hash drift")
    if static_receipt.get("supersession_sha256")!=sha256_file(
        repo_root/"configs/eipm/n0/n0_v02_full_envelope_supersession_map_v1.json"
    ):
        raise SystemExit("static proof supersession hash drift")
    if static_receipt.get("retrospective_sha256")!=sha256_file(
        repo_root/"configs/eipm/n0/n0_v02_full_envelope_retrospective_audit_v1.json"
    ):
        raise SystemExit("static proof retrospective hash drift")

    if static_receipt.get("static_suite_executed") is not True:
        raise SystemExit("FINAL static proof suite was not executed")
    if (
        static_receipt.get("static_suite_pass") is not True
        or int(static_receipt.get("static_suite_exit_code",-1))!=0
    ):
        raise SystemExit("FINAL static proof suite did not pass")
    expected_static_files=[
        str(value) for value in proof_contract.get("static_test_files", [])
    ]
    if list(static_receipt.get("executed_static_test_files") or [])!=expected_static_files:
        raise SystemExit("FINAL static proof suite file coverage drift")

    if final_contract.get("schema")!="alice.eipm.n0.full-envelope-final-validation-contract.v2":
        raise SystemExit("FINAL contract schema drift")
    if evaluator_contract.get("registered_system")!="N0FullEnvelopeTrainableSystemV1":
        raise SystemExit("FINAL evaluator registered-system drift")
    if registry.get("schema")!="alice.eipm.n0.full-envelope-final-v2-gate-registry.v1":
        raise SystemExit("FINAL gate registry schema drift")
    if package_audit.get("status")!="PASS_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_AUDIT_V1":
        raise SystemExit("FINAL package audit not PASS")
    if package_manifest.get("results_observed") is not False:
        raise SystemExit("FINAL package manifest already observed results")
    if selection.get("schema")!="alice.eipm.n0.full-envelope-dev-checkpoint-selection.v1":
        raise SystemExit("FINAL DEV selection receipt schema drift")
    if selection.get("status")!="SELECTED_FIRST_PASSING_N0_DEV_CHECKPOINT":
        raise SystemExit("FINAL DEV selection receipt status drift")
    if selection.get("stage")!="J3_full_public_n0_coadaptation":
        raise SystemExit("FINAL DEV selection stage drift")
    if selection.get("checkpoint_selection_surface")!="DEV_ONLY":
        raise SystemExit("FINAL DEV selection surface drift")
    if selection.get("first_passing_checkpoint") is not True:
        raise SystemExit("FINAL DEV selection is not first passing checkpoint")
    if selection.get("selected_checkpoint_receipt_sha256")!=sha256_file(
        args.candidate_receipt
    ):
        raise SystemExit("FINAL DEV selection/candidate receipt drift")
    if selection.get("selected_system_sha256")!=sha256_file(
        args.candidate_system
    ):
        raise SystemExit("FINAL DEV selection/candidate system drift")
    if selection.get("selected_dev_receipt_sha256")!=sha256_file(
        args.dev_evaluation_receipt
    ):
        raise SystemExit("FINAL DEV selection/evaluation receipt drift")
    selector=Path(__file__).resolve().with_name(
        "select_n0_v02_full_envelope_dev_checkpoint_v1.py"
    )
    if selection.get("selection_authorizer_sha256")!=sha256_file(selector):
        raise SystemExit("FINAL DEV selection authorizer hash drift")
    if dev_selection.get("schema")!="alice.eipm.n0.full-envelope-dev-evaluation.v1":
        raise SystemExit("FINAL DEV selection receipt schema drift")
    if dev_selection.get("status")!="PASS_DEV_STAGE_GATE":
        raise SystemExit("FINAL opening DEV stage gate not passed")
    if dev_selection.get("stage")!="J3_full_public_n0_coadaptation":
        raise SystemExit("FINAL DEV selection stage drift")
    if dev_selection.get("checkpoint_selection_surface")!="DEV_ONLY":
        raise SystemExit("FINAL DEV selection surface drift")
    if dev_selection.get("stage_gate_pass") is not True:
        raise SystemExit("FINAL opening DEV stage gate not passed")
    if dev_selection.get("stage_gate_coverage_complete") is not True:
        raise SystemExit("FINAL opening DEV gate coverage incomplete")
    if dev_selection.get("registry_matches_declared_stage_gates") is not True:
        raise SystemExit("FINAL opening DEV gate mapping incomplete")
    if dev_selection.get("final_results_observed") is not False:
        raise SystemExit("FINAL DEV selection receipt observed FINAL")
    if dev_selection.get("final_opening_authorized") is not False:
        raise SystemExit("DEV selection receipt may not itself authorize FINAL")
    if dev_selection.get("candidate_checkpoint_receipt_sha256")!=sha256_file(
        args.candidate_receipt
    ):
        raise SystemExit("opening/candidate checkpoint receipt drift")
    if dev_selection.get("candidate_system_sha256")!=sha256_file(
        args.candidate_system
    ):
        raise SystemExit("DEV-selected candidate system drift")
    if dev_selection.get("training_authorization_sha256")!=candidate.get(
        "training_authorization_sha256"
    ):
        raise SystemExit("FINAL DEV/training authorization lineage drift")
    if candidate.get("stage")!="J3_full_public_n0_coadaptation":
        raise SystemExit("FINAL may evaluate only selected J3 candidate")
    if candidate.get("final_results_observed") is not False:
        raise SystemExit("candidate receipt already observed FINAL")
    if candidate.get("private_identity_data") is not False:
        raise SystemExit("private identity data forbidden in N0 FINAL")
    if candidate.get("full_system_sha256")!=sha256_file(args.candidate_system):
        raise SystemExit("candidate full-system hash drift")
    if candidate.get("registered_topology_sha256")!=sha256_file(args.topology_config):
        raise SystemExit("candidate topology lineage drift")
    if candidate.get("semantic_initialization_sha256")!=sha256_file(args.semantic_checkpoint):
        raise SystemExit("candidate semantic initialization lineage drift")

    synthetic=read_jsonl(args.synthetic_final_rows)
    semantic_rows=read_jsonl(args.semantic_final_rows)
    runtime_rows=read_jsonl(args.runtime_view_final_rows)
    long_rows=read_jsonl(args.long_context_final_rows)
    natural_rows=read_jsonl(args.fewrel_final_rows)
    natural_bank=read_json(args.fewrel_final_bank)
    for label,rows in (
        ("synthetic",synthetic),("semantic",semantic_rows),
        ("runtime",runtime_rows),("long",long_rows),("natural",natural_rows),
    ):
        if any(row.get("split")!="final" for row in rows):
            raise SystemExit(f"{label} artifact contains non-FINAL row")
        if any(row.get("training_authorized") is not False for row in rows):
            raise SystemExit(f"{label} FINAL row training-authorized")
        if any(row.get("model_selection_authorized") is not False for row in rows):
            raise SystemExit(f"{label} FINAL row model-selection-authorized")

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("safetensors required for FINAL-v2 evaluation") from exc

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system,_=load_registered_full_envelope_system(
        topology_path=args.topology_config,
        semantic_config_path=args.semantic_config,
        semantic_checkpoint_path=args.semantic_checkpoint,
        device=device,
        runtime_profile=None,
    )
    state=load_file(str(args.candidate_system),device="cpu")
    system.load_state_dict(state,strict=True)
    system.eval()
    tokenizer=load_tokenizer(args.tokenizer_dir)
    if candidate.get("tokenizer_sha256")!=sha256_file(
        Path(args.tokenizer_dir)/"tokenizer.json"
    ):
        raise SystemExit("candidate/tokenizer lineage drift")

    repo_root=Path(__file__).resolve().parents[3]
    curriculum_paths,teacher_report=verify_teacher_registry(
        repo_root,args.teacher_registry,args.teacher_audit
    )
    teacher_dev=CurriculumDataset(curriculum_paths,"dev")
    teacher_loader=DataLoader(
        teacher_dev,
        batch_size=args.teacher_batch_size,
        shuffle=False,
        collate_fn=TeacherMultitaskCollator(tokenizer,256),
        num_workers=0,
    )
    core_rows=read_fixed_eval_rows(
        repo_root/"evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl"
    )
    voice_rows=read_fixed_eval_rows(
        repo_root/"evaluation/eipm/n0/n0_v02_voice_readiness_base_v0.1.jsonl"
    )
    novel_rows=read_fixed_eval_rows(
        repo_root/"evaluation/eipm/n0/n0_v02_novel_cross_competency_base_v0.1.jsonl"
    )

    with torch.inference_mode():
        semantic=semantic_lane_metrics(
            system=system,rows=semantic_rows,tokenizer=tokenizer,device=device
        )
        natural=natural_lane_metrics(
            system=system,rows=natural_rows,bank=natural_bank,
            tokenizer=tokenizer,device=device
        )
        all_fabric_rows=synthetic+runtime_rows+long_rows
        fabric=fabric_lane_metrics(
            system=system,rows=all_fabric_rows,tokenizer=tokenizer,
            device=device,stage="J3_full_public_n0_coadaptation"
        )
        regression_teacher,regression_teacher_loss=teacher_top1_rate(
            system=system,loader=teacher_loader,device=device
        )
        core_fixed=fixed_preference_top1(
            system=system,rows=core_rows,tokenizer=tokenizer,device=device
        )
        voice_fixed=fixed_preference_top1(
            system=system,rows=voice_rows,tokenizer=tokenizer,device=device
        )
        novel_cross=fixed_preference_top1(
            system=system,rows=novel_rows,tokenizer=tokenizer,device=device
        )
        semantic_final=final_semantic_derived_metrics(
            rows=semantic_rows,semantic_metrics=semantic
        )
        evidence_cf=semantic_evidence_counterfactual_metrics(
            system=system,rows=semantic_rows,tokenizer=tokenizer,device=device
        )
        semantic_final.update(evidence_cf)
        ctx_accuracy,ctx_count=candidate_context_swap_metric(
            records=list(fabric["records"]),rows=synthetic
        )
        semantic_final["candidate_context_swap_accuracy"]=ctx_accuracy
        semantic_final["candidate_context_swap_count"]=ctx_count
        extrapolation=extrapolation_metrics(
            system=system,tokenizer=tokenizer,device=device,
            behavioral_rows=synthetic,long_rows=long_rows,fabric=fabric
        )
        runtime_view=runtime_view_metrics(
            records=list(fabric["records"]),rows=runtime_rows
        )
        natural_delta,natural_delta_count=natural_candidate_permutation_delta(
            system=system,rows=natural_rows,bank=natural_bank,
            tokenizer=tokenizer,device=device
        )
        boundary_delta,boundary_count=boundary_shift_delta(
            system=system,rows=long_rows,tokenizer=tokenizer,device=device
        )
        growth_delta,growth_count=irrelevant_growth_delta(
            system=system,rows=synthetic,tokenizer=tokenizer,device=device
        )
        para,para_count=paraphrase_consistency(
            system=system,rows=synthetic,tokenizer=tokenizer,device=device
        )

    metrics={
        "semantic":semantic,
        "semantic_final":semantic_final,
        "natural":natural,
        "fabric":fabric,
        "runtime_view":runtime_view,
        "extrapolation":extrapolation,
        "invariance":{
            "natural_relation_candidate_permutation_max_delta":natural_delta,
            "natural_relation_candidate_permutation_count":natural_delta_count,
            "native_window_boundary_shift_max_delta":boundary_delta,
            "native_window_boundary_shift_count":boundary_count,
            "irrelevant_context_growth_max_delta":growth_delta,
            "irrelevant_context_growth_count":growth_count,
            "paraphrase_consistency":para,
            "paraphrase_count":para_count,
        },
        "regression":{
            "teacher_dev_top1":float(regression_teacher),
            "teacher_dev_loss":float(regression_teacher_loss),
            "teacher_dev_rows":len(teacher_dev),
            "core_fixed_top1":float(core_fixed),
            "core_fixed_rows":len(core_rows),
            "voice_fixed_top1":float(voice_fixed),
            "voice_fixed_rows":len(voice_rows),
            "novel_cross_top1":float(novel_cross),
            "novel_cross_rows":len(novel_rows),
            "teacher_registered_rows":int(teacher_report["registered_rows"]),
        },
    }
    gate_results,failures=evaluate_gate_registry(
        registry=registry,
        final_contract=final_contract,
        metrics=metrics,
        proof_contract=proof_contract,
        static_receipt=static_receipt,
        source_revision=str(candidate["source_revision"]),
    )
    passed=not failures
    result={
        "schema":"alice.eipm.n0.full-envelope-final-v2-result.v1",
        "status":PASS_STATUS if passed else FAIL_STATUS,
        "evaluation_status":COMPLETE_STATUS,
        "source_revision":candidate["source_revision"],
        "candidate_system_sha256":sha256_file(args.candidate_system),
        "candidate_receipt_sha256":sha256_file(args.candidate_receipt),
        "final_opening_authorization_sha256":sha256_file(
            args.final_opening_authorization
        ),
        "dev_selection_receipt_sha256":sha256_file(args.dev_selection_receipt),
        "dev_evaluation_receipt_sha256":sha256_file(args.dev_evaluation_receipt),
        "static_proof_receipt_sha256":sha256_file(args.static_proof_receipt),
        "proof_contract_sha256":sha256_file(proof_contract_path),
        "opening_authorizer_sha256":sha256_file(args.opening_authorizer),
        "training_authorization_sha256":candidate.get(
            "training_authorization_sha256"
        ),
        "closure_authority_chain_complete":True,
        "freeze_receipt_sha256":sha256_file(args.freeze_receipt),
        "final_contract_sha256":sha256_file(args.final_contract),
        "evaluator_contract_sha256":sha256_file(args.evaluator_contract),
        "gate_registry_sha256":sha256_file(args.gate_registry),
        "evaluator_implementation_sha256":sha256_file(this_source),
        "package_manifest_sha256":sha256_file(args.package_manifest),
        "package_audit_sha256":sha256_file(args.package_audit),
        "gate_results":gate_results,
        "failures":failures,
        "metrics":metrics,
        "all_gate_sections_pass":passed,
        "checkpoint_selection_performed":False,
        "automatic_repair_or_rerun":False,
        "threshold_changes_after_results":False,
        "final_results_observed":True,
        "training_authorized":False,
        "model_selection_authorized":False,
        "private_identity_data":False,
        "n0_complete":bool(passed),
        "n1_authorized":bool(passed),
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__=="__main__":
    main()
