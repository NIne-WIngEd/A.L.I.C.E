#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
    compile_behavioral_batch,
)
from alice_personality.n0.full_envelope_dev_gate_v1 import (
    evaluate_stage_gate_registry,
)
from alice_personality.n0.full_envelope_dev_metrics_v1 import (
    boolean_rate,
    decisive_removal_success,
    endpoint_pair_correct,
    irrelevant_removal_invariance_success,
    latent_noncollapse_success,
    mean,
    public_judgment_correct,
    semantic_batch_record,
    support_edge_f1,
)
from alice_personality.n0.full_envelope_joint_step_v1 import (
    governed_judgment_replay_loss,
)
from alice_personality.n0.full_envelope_runtime_factory_v1 import (
    load_registered_full_envelope_system,
)
from alice_personality.n0.full_envelope_stage_policy_v1 import (
    J1,J2,J3,apply_stage_trainability,resolve_stage_policy,
)
from alice_personality.n0.full_envelope_training_objective_v1 import (
    behavioral_supervision,
    semantic_operator_supervision,
)
from alice_personality.n0.natural_relation_batch_v1 import (
    compile_natural_relation_batch,
    natural_relation_semantic_loss,
)
from alice_personality.n0.semantic_operator_batch_v1 import (
    compile_semantic_operator_batch,
)
from alice_personality.n0.v02_training import (
    TeacherMultitaskCollator,
    sha256_file,
    verify_teacher_registry,
)


STAGES=(J1,J2,J3)


def read_json(path: str | Path) -> dict[str,Any]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def read_dev_rows(path: str | Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected=[]
    for row in rows:
        split=str(row.get("split",""))
        if split=="final" or row.get("final_validation_only") is True:
            raise SystemExit("FINAL rows are forbidden in successor DEV evaluator")
        if split!="dev":
            continue
        if row.get("private_identity_data") is not False:
            raise SystemExit("private identity data forbidden in N0 DEV")
        if row.get("model_selection_authorized") not in (None,True):
            raise SystemExit(f"DEV model-selection authority drift: {row.get('id')}")
        selected.append(row)
    if not selected:
        raise SystemExit(f"DEV lane is empty: {path}")
    return selected


def read_fixed_eval_rows(path: Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"fixed regression suite empty: {path}")
    for row in rows:
        if row.get("eval_only") is not True:
            raise SystemExit(f"fixed regression row lacks eval_only: {row.get('id')}")
        if row.get("training_authorized") is not False:
            raise SystemExit(f"fixed regression row training-authorized: {row.get('id')}")
    return rows


def to_device(value: Any,device: torch.device) -> Any:
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):
        return [to_device(v,device) for v in value]
    if isinstance(value,tuple):
        return tuple(to_device(v,device) for v in value)
    return value


def teacher_top1_rate(
    *,
    system: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float,float]:
    correct=0
    total=0
    losses=[]
    for batch in loader:
        batch=to_device(batch,device)
        loss,_=governed_judgment_replay_loss(system=system,batch=batch)
        losses.append(float(loss.detach().cpu()))
        outputs=system(task="teacher",batch=batch)
        scores=outputs["scores"]
        offset=0
        for size,preferred in zip(batch["group_sizes"],batch["preferred_masks"]):
            local=scores[offset:offset+int(size)]
            prediction=int(local.argmax().item())
            mask=preferred.to(device=local.device,dtype=torch.bool)
            correct+=int(bool(mask[prediction]))
            total+=1
            offset+=int(size)
    if total<=0:
        raise RuntimeError("teacher DEV produced no groups")
    return correct/total,mean(losses)


def fixed_preference_top1(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
    max_length: int=512,
) -> dict[str,float]:
    correct=0
    total=0
    paraphrase_correct=0
    paraphrase_total=0
    for row in rows:
        candidates=[str(x) for x in row["candidates"]]
        preferred={int(x) for x in row["preferred_indices"]}
        prompts=[("prompt",str(row["prompt"]))]
        if str(row.get("prompt_paraphrase","")).strip():
            prompts.append(("paraphrase",str(row["prompt_paraphrase"])))
        for kind,prompt in prompts:
            encoded=tokenizer(
                [prompt]*len(candidates),
                candidates,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            scores=system.semantic_model.score_candidates(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
            ok=int(int(scores.argmax().item()) in preferred)
            correct+=ok
            total+=1
            if kind=="paraphrase":
                paraphrase_correct+=ok
                paraphrase_total+=1
    return {
        "top1":correct/max(total,1),
        "paraphrase_top1":paraphrase_correct/max(paraphrase_total,1),
        "evaluated_prompt_variants":float(total),
    }


def _tensor_bool(value: torch.Tensor) -> bool:
    if value.numel()!=1:
        raise ValueError("expected one-row DEV tensor")
    return bool(value.reshape(-1)[0].item())


def _tensor_bool_values(value: torch.Tensor) -> list[bool]:
    return [bool(x) for x in value.detach().cpu().reshape(-1).tolist()]


def semantic_lane_metrics(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
) -> dict[str,Any]:
    loss_totals=defaultdict(list)
    records=[]
    global_bank=defaultdict(list)
    step_bank=defaultdict(list)
    relation_margin_values=[]
    factor_margin_values=[]
    query_evidence=[]
    relation_evidence=[]
    factor_evidence=[]
    step_factor_evidence=[]

    for row in rows:
        compiled=compile_semantic_operator_batch(rows=[row],tokenizer=tokenizer)
        compiled=to_device(compiled,device)
        outputs=system(task="semantic_operator",batch=compiled["batch"])
        semantic=outputs["semantic_operator"]
        targets=compiled["operator_targets"]
        losses=semantic_operator_supervision(outputs=outputs,targets=targets)
        for name,value in losses.items():
            if isinstance(value,torch.Tensor) and value.ndim==0:
                loss_totals[name].append(float(value.detach().cpu()))

        measured=semantic_batch_record(semantic=semantic,targets=targets)
        global_correct={
            name:_tensor_bool(value)
            for name,value in measured["global_factor_correct"].items()
        }
        step_correct={
            name:_tensor_bool(value)
            for name,value in measured["step_factor_exact"].items()
        }
        for name,value in global_correct.items():
            global_bank[name].append(value)
        if bool(targets["step_factor_mask"].any()):
            for name,value in step_correct.items():
                step_bank[name].append(value)

        relation_margin_values.extend(
            _tensor_bool_values(measured["relation_margin_success"])
        )
        factor_margin_values.extend(
            _tensor_bool_values(measured["factor_margin_success"])
        )
        query_evidence.append(float(measured["query_evidence_f1"]))
        relation_evidence.append(float(measured["relation_schema_evidence_f1"]))
        factor_evidence.append(float(measured["factor_schema_evidence_f1"]))
        step_factor_evidence.append(
            float(measured["step_factor_schema_evidence_f1"])
        )
        records.append({
            "id":str(row.get("id")),
            "intervention":str(row.get("intervention","")),
            "counterfactual_pair_id":row.get("counterfactual_pair_id"),
            "long_context_surface":row.get("long_context_surface"),
            "relation_exact":_tensor_bool(measured["relation_exact"]),
            "event_exact":_tensor_bool(measured["event_exact"]),
            "applicability_correct":_tensor_bool(
                measured["applicability_correct"]
            ),
            "uncertainty_correct":_tensor_bool(
                measured["uncertainty_correct"]
            ),
            "global_factor_correct":global_correct,
            "step_factor_exact":step_correct,
            "query_evidence_f1":float(measured["query_evidence_f1"]),
            "relation_schema_evidence_f1":float(
                measured["relation_schema_evidence_f1"]
            ),
            "factor_schema_evidence_f1":float(
                measured["factor_schema_evidence_f1"]
            ),
            "step_factor_schema_evidence_f1":float(
                measured["step_factor_schema_evidence_f1"]
            ),
        })

    nonunknown=[x for x in records if x["intervention"]!="unknown_defer"]
    relation_exact=boolean_rate([x["relation_exact"] for x in nonunknown])

    all_factor_rates=[
        boolean_rate(values)
        for values in global_bank.values()
    ] + [
        boolean_rate(values)
        for values in step_bank.values()
    ]
    dynamic_factor_macro=mean(all_factor_rates)

    pairs=defaultdict(list)
    for record in records:
        pair=record.get("counterfactual_pair_id")
        if pair:
            role_ok=bool(record["global_factor_correct"].get("role",False))
            pairs[str(pair)].append(
                bool(record["relation_exact"])
                and bool(record["event_exact"])
                and role_ok
            )
    pair_success=[
        len(values)>=2 and all(values)
        for values in pairs.values()
    ]

    ordered=[
        x for x in records
        if x["intervention"] in {
            "ordered_composition","reverse_ordered_composition"
        }
    ]
    ordered_success=[
        bool(x["relation_exact"])
        and bool(x["event_exact"])
        and bool(x["global_factor_correct"].get("traversal",False))
        for x in ordered
    ]

    unknown=[x for x in records if x["intervention"]=="unknown_defer"]
    unknown_success=[
        bool(x["relation_exact"])
        and bool(x["event_exact"])
        and bool(x["global_factor_correct"].get("control",False))
        and bool(x["applicability_correct"])
        and bool(x["uncertainty_correct"])
        for x in unknown
    ]

    mixed_direction=[
        x for x in records
        if x["intervention"]=="mixed_direction_composition"
    ]
    mixed_direction_success=[
        bool(x["relation_exact"])
        and bool(x["event_exact"])
        and bool(x["step_factor_exact"].get("direction",False))
        for x in mixed_direction
    ]

    modifier_names=(
        "reliability_modifier","recency_modifier",
        "temporal_constraint_modifier","provenance_constraint_modifier",
    )
    mixed_modifier=[
        x for x in records
        if x["intervention"]=="mixed_step_modifier_composition"
    ]
    mixed_modifier_success=[
        bool(x["relation_exact"])
        and bool(x["event_exact"])
        and all(bool(x["step_factor_exact"].get(name,False)) for name in modifier_names)
        for x in mixed_modifier
    ]

    long_relation_rows=[
        x for x in records
        if x.get("long_context_surface")=="relation_schema"
    ]
    long_factor_rows=[
        x for x in records
        if x.get("long_context_surface")=="factor_schema"
    ]
    long_relation_factor_values=[]
    if long_relation_rows:
        long_relation_factor_values.append(
            boolean_rate([bool(x["relation_exact"]) for x in long_relation_rows])
        )
    if long_factor_rows:
        long_relation_factor_values.append(
            boolean_rate([
                bool(x["relation_exact"])
                and bool(x["global_factor_correct"].get("direction",False))
                for x in long_factor_rows
            ])
        )

    return {
        "rows":len(records),
        "losses":{
            name:mean(values)
            for name,values in sorted(loss_totals.items())
        },
        "relation_sequence_exact":relation_exact,
        "dynamic_factor_macro_accuracy":dynamic_factor_macro,
        "global_factor_bank_accuracy":{
            name:boolean_rate(values)
            for name,values in sorted(global_bank.items())
        },
        "step_factor_bank_sequence_exact":{
            name:boolean_rate(values)
            for name,values in sorted(step_bank.items())
        },
        "source_target_pair_completion":boolean_rate(pair_success),
        "ordered_relation_sequence_exact":boolean_rate(ordered_success),
        "unknown_defer_accuracy":boolean_rate(unknown_success),
        "mixed_step_direction_sequence_exact":boolean_rate(
            mixed_direction_success
        ),
        "mixed_step_modifier_sequence_exact":boolean_rate(
            mixed_modifier_success
        ),
        "relation_counterfactual_margin_success":boolean_rate(
            relation_margin_values
        ),
        "factor_counterfactual_margin_success":boolean_rate(
            factor_margin_values
        ),
        "query_token_grounding_f1":mean(query_evidence),
        "relation_schema_token_grounding_f1":mean(relation_evidence),
        "factor_schema_token_grounding_f1":mean(factor_evidence),
        "step_factor_schema_token_grounding_f1":mean(step_factor_evidence),
        "bidirectional_relation_token_grounding_min_f1":min(
            mean(query_evidence),mean(relation_evidence)
        ),
        "factor_token_grounding_min_f1":min(
            mean(factor_evidence),mean(step_factor_evidence)
        ),
        "long_relation_factor_semantics_min":(
            min(long_relation_factor_values)
            if long_relation_factor_values else relation_exact
        ),
        "records":records,
    }


def natural_lane_metrics(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    bank: Mapping[str,Any],
    tokenizer: Any,
    device: torch.device,
) -> dict[str,float]:
    losses=[]
    correct=0
    total=0
    for row in rows:
        compiled=compile_natural_relation_batch(
            rows=[row],relation_bank=bank,tokenizer=tokenizer
        )
        compiled=to_device(compiled,device)
        outputs=system(task="natural_relation",batch=compiled["batch"])
        loss=natural_relation_semantic_loss(
            outputs=outputs,
            target_relation_index=compiled["target_relation_index"],
        )
        losses.append(float(loss.detach().cpu()))
        logits=outputs["semantic_operator"]["relation_logits"][:,0,:]
        target=compiled["target_relation_index"]
        correct+=int((logits.argmax(dim=-1)==target).sum().item())
        total+=int(target.numel())
    return {"loss":mean(losses),"top1":correct/max(total,1),"rows":float(total)}


def _scalar_bool(value: torch.Tensor) -> bool:
    flat=value.detach().bool().reshape(-1)
    if flat.numel()!=1:
        raise ValueError("expected one-row DEV boolean")
    return bool(flat[0].item())


def _candidate_subset_batch_accuracy(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
) -> float:
    by_count={}
    for row in rows:
        count=int(row.get("runtime_candidate_answer_count",0))
        if count>0 and count not in by_count:
            by_count[count]=row
    selected=[by_count[key] for key in sorted(by_count)]
    if len(selected)<2:
        raise ValueError(
            "per-example candidate-subset DEV gate requires at least two candidate counts"
        )
    compiled=compile_behavioral_batch(rows=selected,tokenizer=tokenizer)
    compiled=to_device(compiled,device)
    outputs=system(task="full_envelope",batch=compiled["primary_batch"])
    judgment=outputs["public_judgment"]
    correct=public_judgment_correct(
        candidate_logits=judgment["candidate_logits"],
        target_index=compiled["behavioral_targets"]["public_target_index"],
        candidate_valid_mask=judgment["candidate_valid_mask"],
    )
    return boolean_rate([bool(x) for x in correct.detach().cpu().tolist()])


def fabric_lane_metrics(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
    stage: str,
) -> dict[str,Any]:
    policy=resolve_stage_policy(stage)
    totals=defaultdict(list)
    records=[]
    for row in rows:
        compiled=compile_behavioral_batch(rows=[row],tokenizer=tokenizer)
        compiled=to_device(compiled,device)
        targets=compiled["behavioral_targets"]
        primary=system(task="full_envelope",batch=compiled["primary_batch"])
        decisive=irrelevant=permuted=None
        causal_active="multi_view_causal_preservation" in policy.active_macro_families
        judgment_active="latent_judgment_and_noncollapse" in policy.active_macro_families
        if causal_active:
            decisive=system(
                task="full_envelope",batch=compiled["decisive_ablated_batch"]
            )
            irrelevant=system(
                task="full_envelope",batch=compiled["irrelevant_removed_batch"]
            )
            permuted=system(
                task="full_envelope",batch=compiled["permuted_batch"]
            )
        losses=behavioral_supervision(
            primary_outputs=primary,
            decisive_ablated_outputs=decisive,
            irrelevant_removed_outputs=irrelevant,
            permuted_outputs=permuted,
            targets=targets,
            active_families=policy.active_macro_families,
        )
        for name,value in losses.items():
            if isinstance(value,torch.Tensor) and value.ndim==0:
                totals[name].append(float(value.detach().cpu()))

        binder=primary["binder"]
        executor=primary["executor"]
        support_f1=support_edge_f1(
            edge_support_weight=binder["edge_support_weight"],
            support_target=targets["support_target"],
            valid_mask=targets["support_valid_mask"].bool(),
        )
        endpoint_values=endpoint_pair_correct(
            source_weight=executor["source_support_weight"],
            target_weight=executor["target_support_weight"],
            source_target=targets["source_target_index"],
            target_target=targets["target_target_index"],
            active_mask=targets["endpoint_active_mask"].bool(),
        )
        endpoint_ok=(
            _scalar_bool(endpoint_values)
            if endpoint_values.numel()
            else True
        )
        no_support=not bool(targets["support_target"].bool().any().item())
        null_exact=(
            float(binder["edge_support_weight"].abs().max().item())==0.0
            and float(binder["support_available"].abs().max().item())==0.0
        ) if no_support else None
        no_support_confidence=(
            float(executor["execution_confidence"].abs().max().item())
            if no_support else None
        )

        public_ok=None
        decisive_ok=None
        irrelevant_ok=None
        noncollapse_ok=None
        if judgment_active:
            judgment=primary["public_judgment"]
            public_ok=_scalar_bool(public_judgment_correct(
                candidate_logits=judgment["candidate_logits"],
                target_index=targets["public_target_index"],
                candidate_valid_mask=judgment["candidate_valid_mask"],
            ))
            noncollapse_ok=_scalar_bool(latent_noncollapse_success(
                latent_slots=primary["latent"]["latent_slots"],
            ))
        if causal_active:
            assert decisive is not None and irrelevant is not None
            judgment=primary["public_judgment"]
            decisive_values=decisive_removal_success(
                normal_logits=judgment["candidate_logits"],
                ablated_logits=decisive["public_judgment"]["candidate_logits"],
                target_index=targets["public_target_index"],
                candidate_valid_mask=judgment["candidate_valid_mask"],
                active_mask=targets["decisive_view_active_mask"].bool(),
            )
            if decisive_values.numel():
                decisive_ok=_scalar_bool(decisive_values)
            irrelevant_values=irrelevant_removal_invariance_success(
                normal_logits=judgment["candidate_logits"],
                removed_logits=irrelevant["public_judgment"]["candidate_logits"],
                candidate_valid_mask=judgment["candidate_valid_mask"],
                active_mask=targets["irrelevant_view_active_mask"].bool(),
            )
            if irrelevant_values.numel():
                irrelevant_ok=_scalar_bool(irrelevant_values)

        role=str((row.get("factor_target_keys") or {}).get("role",""))
        expected_readout=None
        endpoint=row.get("endpoint_target") or {}
        if bool(endpoint.get("active")):
            if role=="ROLE_SOURCE":
                expected_readout=int(endpoint["source_field"])
            elif role=="ROLE_TARGET":
                expected_readout=int(endpoint["target_field"])
        executor_readout_ok=None
        if expected_readout is not None:
            executor_readout_ok=(
                int(executor["relational_probability"].argmax(dim=-1)[0].item())
                == expected_readout
            )

        structural_ok=(support_f1>=0.95 and endpoint_ok)
        records.append({
            "id":str(row.get("id")),
            "scenario_family":str(row.get("scenario_family","")),
            "long_context_surface":row.get("long_context_surface"),
            "long_context_placement_variant":row.get(
                "long_context_placement_variant"
            ),
            "candidate_cardinality_extrapolation":bool(
                row.get("candidate_cardinality_extrapolation",False)
            ),
            "composition_extrapolation":bool(
                row.get("composition_extrapolation",False)
            ),
            "support_edge_f1":support_f1,
            "endpoint_pair_correct":endpoint_ok,
            "structural_success":structural_ok,
            "null_support_exact":null_exact,
            "no_support_execution_confidence":no_support_confidence,
            "public_judgment_correct":public_ok,
            "decisive_removal_success":decisive_ok,
            "irrelevant_removal_invariance":irrelevant_ok,
            "latent_noncollapse_success":noncollapse_ok,
            "executor_readout_correct":executor_readout_ok,
        })

    def rate(key: str, *, where=lambda _: True) -> float:
        values=[
            bool(item[key])
            for item in records
            if where(item) and item.get(key) is not None
        ]
        return boolean_rate(values)

    def surface_rates(key: str) -> dict[str,float]:
        surfaces=sorted({
            str(item["long_context_surface"])
            for item in records
            if item.get("long_context_surface")
        })
        return {
            surface:rate(
                key,
                where=lambda item,surface=surface: (
                    item.get("long_context_surface")==surface
                ),
            )
            for surface in surfaces
        }

    null_records=[
        item for item in records if item["null_support_exact"] is not None
    ]
    confidence_values=[
        float(item["no_support_execution_confidence"])
        for item in records
        if item["no_support_execution_confidence"] is not None
    ]
    ordered_families={
        "ordered_composition","reverse_traversal",
        "mixed_direction_composition","causal_chain",
    }
    scenario_names=sorted({
        str(item["scenario_family"]) for item in records
    })
    result={
        "rows":len(records),
        "losses":{
            name:mean(values)
            for name,values in sorted(totals.items())
        },
        "support_edge_f1":mean([
            float(item["support_edge_f1"]) for item in records
        ]),
        "endpoint_pair_accuracy":rate("endpoint_pair_correct"),
        "null_support_exact_rate":(
            boolean_rate([
                bool(item["null_support_exact"]) for item in null_records
            ])
            if null_records else 1.0
        ),
        "no_support_execution_confidence_max":(
            max(confidence_values) if confidence_values else 0.0
        ),
        "surface_structural_accuracy":surface_rates("structural_success"),
        "ordered_executor_endpoint_accuracy":rate(
            "executor_readout_correct",
            where=lambda item: item["scenario_family"] in ordered_families,
        ),
        "records":records,
    }
    if judgment_active:
        result.update({
            "public_judgment_top1":rate("public_judgment_correct"),
            "scenario_public_accuracy":{
                scenario:rate(
                    "public_judgment_correct",
                    where=lambda item,scenario=scenario: (
                        item["scenario_family"]==scenario
                    ),
                )
                for scenario in scenario_names
            },
            "surface_public_accuracy":surface_rates(
                "public_judgment_correct"
            ),
            "candidate_cardinality_extrapolation_accuracy":rate(
                "public_judgment_correct",
                where=lambda item: item[
                    "candidate_cardinality_extrapolation"
                ],
            ),
            "composition_extrapolation_accuracy":rate(
                "public_judgment_correct",
                where=lambda item: item["composition_extrapolation"],
            ),
            "latent_noncollapse_success_rate":rate(
                "latent_noncollapse_success"
            ),
        })
    if causal_active:
        distant_surfaces={"field_text","additional_view_source"}
        result.update({
            "decisive_source_removal_success":rate(
                "decisive_removal_success"
            ),
            "irrelevant_source_removal_invariance":rate(
                "irrelevant_removal_invariance"
            ),
            "distant_decisive_removal_success":rate(
                "decisive_removal_success",
                where=lambda item: (
                    item.get("long_context_surface") in distant_surfaces
                    and item.get("long_context_placement_variant")
                    in {"tail","boundary_late"}
                ),
            ),
        })
    return result


def main() -> None:
    p=argparse.ArgumentParser(
        description=(
            "DEV-only successor evaluator for N0FullEnvelopeTrainableSystemV1. "
            "FINAL rows/results are forbidden and this script cannot open FINAL."
        )
    )
    p.add_argument("--stage",choices=STAGES,required=True)
    p.add_argument("--topology-config",required=True)
    p.add_argument("--semantic-config",required=True)
    p.add_argument("--semantic-checkpoint",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--candidate-system",required=True)
    p.add_argument("--candidate-receipt",required=True)
    p.add_argument("--mixture-manifest",required=True)
    p.add_argument("--mixture-audit",required=True)
    p.add_argument("--teacher-registry",required=True)
    p.add_argument("--teacher-audit",required=True)
    p.add_argument("--semantic-rows",required=True)
    p.add_argument("--semantic-long-rows",required=True)
    p.add_argument("--behavioral-rows",required=True)
    p.add_argument("--runtime-view-rows",required=True)
    p.add_argument("--long-context-rows",required=True)
    p.add_argument("--natural-rows",required=True)
    p.add_argument("--natural-bank",required=True)
    p.add_argument("--dev-contract",required=True)
    p.add_argument("--gate-registry",required=True)
    p.add_argument("--proof-contract",required=True)
    p.add_argument("--static-proof-receipt",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--teacher-batch-size",type=int,default=8)
    args=p.parse_args()

    if args.teacher_batch_size<=0:
        raise SystemExit("teacher DEV batch size must be positive")
    contract=read_json(args.dev_contract)
    if contract.get("schema")!="alice.eipm.n0.full-envelope-dev-validation-contract.v1":
        raise SystemExit("DEV validation contract schema drift")
    if contract.get("checkpoint_selection_surface")!="DEV_ONLY":
        raise SystemExit("DEV evaluator contract lost DEV-only authority")
    if contract.get("final_rows_allowed") is not False:
        raise SystemExit("DEV contract unexpectedly allows FINAL rows")
    if contract.get("final_results_allowed") is not False:
        raise SystemExit("DEV contract unexpectedly allows FINAL results")

    candidate=read_json(args.candidate_receipt)
    if candidate.get("stage")!=args.stage:
        raise SystemExit("candidate checkpoint stage drift")
    if candidate.get("final_results_observed") is not False:
        raise SystemExit("candidate checkpoint has observed FINAL")
    if candidate.get("private_identity_data") is not False:
        raise SystemExit("candidate checkpoint contains private identity data")
    if candidate.get("full_system_sha256")!=sha256_file(args.candidate_system):
        raise SystemExit("candidate full-system hash drift")
    if candidate.get("registered_topology_sha256")!=sha256_file(
        args.topology_config
    ):
        raise SystemExit("candidate topology lineage drift")
    if candidate.get("semantic_initialization_sha256")!=sha256_file(
        args.semantic_checkpoint
    ):
        raise SystemExit("candidate semantic initialization lineage drift")
    if candidate.get("static_proof_receipt_sha256")!=sha256_file(
        args.static_proof_receipt
    ):
        raise SystemExit("candidate static-proof lineage drift")

    static_receipt=read_json(args.static_proof_receipt)
    if static_receipt.get("status")!="PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1":
        raise SystemExit("static proof receipt not PASS")
    if static_receipt.get("source_revision")!=candidate.get("source_revision"):
        raise SystemExit("candidate/static-proof source revision drift")

    mixture=read_json(args.mixture_manifest)
    mixture_audit=read_json(args.mixture_audit)
    if mixture_audit.get("status")!="PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1":
        raise SystemExit("full public-mixture audit not PASS")
    if mixture.get("final_results_observed") is not False:
        raise SystemExit("mixture observed FINAL")
    if candidate.get("full_public_mixture_manifest_sha256")!=sha256_file(
        args.mixture_manifest
    ):
        raise SystemExit("candidate/mixture manifest lineage drift")
    if candidate.get("full_public_mixture_audit_sha256")!=sha256_file(
        args.mixture_audit
    ):
        raise SystemExit("candidate/mixture audit lineage drift")

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("safetensors required for DEV evaluation") from exc

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
    apply_stage_trainability(system,stage=args.stage)
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

    semantic=read_dev_rows(args.semantic_rows)
    semantic_long=read_dev_rows(args.semantic_long_rows)
    behavioral=read_dev_rows(args.behavioral_rows)
    runtime_view=read_dev_rows(args.runtime_view_rows)
    long_context=read_dev_rows(args.long_context_rows)
    natural=read_dev_rows(args.natural_rows)
    natural_bank=read_json(args.natural_bank)

    fixed=contract["fixed_regression_suites"]
    core_rows=read_fixed_eval_rows(repo_root/str(fixed["core_fixed"]))
    voice_rows=read_fixed_eval_rows(repo_root/str(fixed["voice_fixed"]))
    novel_rows=read_fixed_eval_rows(repo_root/str(fixed["novel_cross"]))

    with torch.inference_mode():
        teacher_top1,teacher_loss=teacher_top1_rate(
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
        semantic_metrics=semantic_lane_metrics(
            system=system,rows=semantic,tokenizer=tokenizer,device=device
        )
        semantic_long_metrics=semantic_lane_metrics(
            system=system,rows=semantic_long,tokenizer=tokenizer,device=device
        )
        natural_metrics=natural_lane_metrics(
            system=system,rows=natural,bank=natural_bank,
            tokenizer=tokenizer,device=device
        )
        fabric_metrics={}
        if args.stage in {J2,J3}:
            fabric_metrics["full_envelope_behavioral"]=fabric_lane_metrics(
                system=system,rows=behavioral,tokenizer=tokenizer,
                device=device,stage=args.stage
            )
            fabric_metrics["long_context_supplement"]=fabric_lane_metrics(
                system=system,rows=long_context,tokenizer=tokenizer,
                device=device,stage=args.stage
            )
        if args.stage==J3:
            fabric_metrics["runtime_view_supplement"]=fabric_lane_metrics(
                system=system,rows=runtime_view,tokenizer=tokenizer,
                device=device,stage=args.stage
            )
            fabric_metrics["full_envelope_behavioral"][
                "per_example_candidate_subset_batch_accuracy"
            ]=_candidate_subset_batch_accuracy(
                system=system,
                rows=behavioral,
                tokenizer=tokenizer,
                device=device,
            )

    metrics={
        "historical_regression":{
            "teacher_dev_top1":teacher_top1,
            "teacher_dev_loss":teacher_loss,
            "core_fixed_top1":core_fixed["top1"],
            "voice_fixed_top1":voice_fixed["top1"],
            "novel_cross_top1":novel_cross["top1"],
            "core_fixed":core_fixed,
            "voice_fixed":voice_fixed,
            "novel_cross":novel_cross,
        },
        "semantic_operator":semantic_metrics,
        "semantic_operator_long_context":semantic_long_metrics,
        "natural_relation":natural_metrics,
        "full_fabric":fabric_metrics,
    }

    registry=read_json(args.gate_registry)
    proof_contract=read_json(args.proof_contract)
    plan=read_json(contract["stage_gate_source"])
    declared=list(plan["stage_gates"][
        "J1" if args.stage==J1 else "J2" if args.stage==J2 else "J3"
    ])
    stage_registry=registry["stages"].get(args.stage) or {}
    registered=list((stage_registry.get("gates") or {}).keys())
    registry_matches_declared=set(registered)==set(declared)
    if bool(stage_registry.get("mapping_complete")) and not registry_matches_declared:
        raise SystemExit(
            "DEV gate registry claims completeness but does not match declared stage gates"
        )

    gate_result=evaluate_stage_gate_registry(
        registry=registry,
        stage=args.stage,
        metrics=metrics,
        proof_contract=proof_contract,
        static_receipt=static_receipt,
    )
    if not registry_matches_declared:
        gate_result["stage_gate_coverage_complete"]=False
        gate_result["stage_gate_pass"]=False
        gate_result["mapping_errors"].append(
            "registered gate set does not match joint training plan"
        )

    result={
        "schema":"alice.eipm.n0.full-envelope-dev-evaluation.v1",
        "status":(
            "PASS_DEV_STAGE_GATE"
            if gate_result["stage_gate_pass"]
            else "DEV_METRICS_RECORDED_STAGE_BLOCKED"
        ),
        "stage":args.stage,
        "source_revision":candidate.get("source_revision"),
        "checkpoint_selection_surface":"DEV_ONLY",
        "candidate_checkpoint_receipt_sha256":sha256_file(args.candidate_receipt),
        "candidate_system_sha256":sha256_file(args.candidate_system),
        "static_proof_receipt_sha256":sha256_file(args.static_proof_receipt),
        "teacher_registered_rows":int(teacher_report["registered_rows"]),
        "metrics":metrics,
        "declared_stage_gates":declared,
        "registered_stage_gates":registered,
        "registry_matches_declared_stage_gates":registry_matches_declared,
        **gate_result,
        "checkpoint_selection_performed":False,
        "automatic_stage_transition":False,
        "final_validation_only":False,
        "final_results_observed":False,
        "final_opening_authorized":False,
        "final_checkpoint_selection_allowed":False,
        "private_identity_data":False,
        "n0_complete":False,
    }
    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite DEV evaluation")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
