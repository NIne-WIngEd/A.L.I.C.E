from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from alice_personality.n0.semantic_operator_evidence_targets_v1 import (
    compile_operator_evidence_targets,
)


EVENT_INDEX={"CONTINUE":0,"STOP":1,"UNKNOWN":2}
PROGRAM_GLOBAL_BANKS=frozenset({"role","traversal","control"})


def _tokenize(tokenizer: Any, texts: Sequence[str]) -> tuple[Tensor,Tensor]:
    if not texts:
        raise ValueError("semantic-operator batch text list cannot be empty")
    encoded=tokenizer(
        list(texts),
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return (
        torch.as_tensor(encoded["input_ids"],dtype=torch.long),
        torch.as_tensor(encoded["attention_mask"],dtype=torch.bool),
    )


def _canonical_relations(
    rows: Sequence[Mapping[str,Any]],
) -> tuple[list[dict[str,Any]],dict[str,int]]:
    ordered=[]
    by_key={}
    for row in rows:
        for raw in row["relation_candidates"]:
            item=dict(raw)
            key=str(item["key"])
            prior=by_key.get(key)
            if prior is None:
                by_key[key]=item
                ordered.append(item)
            elif prior!=item:
                raise ValueError(
                    f"inconsistent semantic relation definition for {key!r}"
                )
    if not ordered:
        raise ValueError("semantic-operator batch requires relation candidates")
    return ordered,{str(item["key"]):i for i,item in enumerate(ordered)}


def _factor_schema(
    rows: Sequence[Mapping[str,Any]],
) -> tuple[dict[str,list[dict[str,Any]]],dict[str,dict[str,int]],dict[str,list[str]]]:
    first={
        str(name):[dict(item) for item in bank]
        for name,bank in rows[0]["factor_schemas"].items()
    }
    if not first:
        raise ValueError("semantic-operator rows require factor schemas")
    for row in rows[1:]:
        current={
            str(name):[dict(item) for item in bank]
            for name,bank in row["factor_schemas"].items()
        }
        if current!=first:
            raise ValueError(
                "factor schema banks must be identical inside one semantic-operator batch"
            )
    key_index={}
    opcodes={}
    for name,bank in first.items():
        keys=[str(item["key"]) for item in bank]
        if len(keys)!=len(set(keys)):
            raise ValueError(f"duplicate factor key in {name!r}")
        key_index[name]={key:i for i,key in enumerate(keys)}
        values=[item.get("opcode") for item in bank]
        non_null=[value is not None for value in values]
        if any(non_null) and not all(non_null):
            raise ValueError(
                f"factor bank {name!r} mixes executable and semantic-only candidates"
            )
        if all(non_null):
            opcodes[name]=[str(value) for value in values]
    return first,key_index,opcodes


def _copy_prefix(target: Tensor, source: Tensor) -> None:
    slices=tuple(slice(0,size) for size in source.shape)
    target[slices]=source.to(dtype=target.dtype)


def compile_semantic_operator_batch(
    *,
    rows: Sequence[Mapping[str,Any]],
    tokenizer: Any,
    evidence_max_length: int | None = None,
) -> dict[str,Any]:
    """Compile public semantic-operator rows into the registered J1 task.

    The compiler owns only semantic/operator labels. It never manufactures
    fields, graph edges, support labels, answer candidates, fusion views, or
    public-judgment targets. Relation candidates are unioned dynamically across
    the batch and masked per example; runtime candidate cardinality is therefore
    an operating axis rather than a learned identity axis or fixed ceiling.
    """
    if not rows:
        raise ValueError("semantic-operator batch cannot be empty")
    for row in rows:
        if row.get("private_identity_data") is not False:
            raise ValueError("private identity data forbidden in public N0")
        if row.get("final_validation_only") is True:
            raise ValueError("FINAL rows forbidden in optimizer-facing compiler")
        if "uncertainty_target" not in row:
            raise ValueError(
                "semantic-operator row must own explicit uncertainty_target"
            )
        if not 0.0 <= float(row["uncertainty_target"]) <= 1.0:
            raise ValueError("uncertainty_target must stay inside [0,1]")

    batch_size=len(rows)
    relation_items,relation_index=_canonical_relations(rows)
    factor_banks,factor_key_index,factor_opcodes=_factor_schema(rows)

    query_ids,query_mask=_tokenize(
        tokenizer,[str(row["query"]) for row in rows]
    )
    relation_ids,relation_mask=_tokenize(
        tokenizer,[str(item["text"]) for item in relation_items]
    )

    relation_candidate_mask=torch.zeros(
        batch_size,len(relation_items),dtype=torch.bool
    )
    for b,row in enumerate(rows):
        for item in row["relation_candidates"]:
            relation_candidate_mask[b,relation_index[str(item["key"])]]=True
        if not bool(relation_candidate_mask[b].any()):
            raise ValueError("every semantic row requires an active relation")

    # Semantic-operator J1 does not own the downstream type-execution task.
    # Use one explicit generic argument type so DynamicRelationSchema retains
    # its complete structural contract without inventing query-specific type
    # supervision. Type transfer is trained/audited in the full-envelope lane.
    relation_domain=torch.ones(len(relation_items),1,dtype=torch.bool)
    relation_range=torch.ones_like(relation_domain)
    relation_symmetric=torch.tensor(
        [bool(item.get("symmetric",False)) for item in relation_items],
        dtype=torch.bool,
    )

    factor_input_ids={}
    factor_attention_mask={}
    factor_candidate_masks={}
    for name,bank in factor_banks.items():
        ids,mask=_tokenize(tokenizer,[str(item["text"]) for item in bank])
        factor_input_ids[name]=ids
        factor_attention_mask[name]=mask
        factor_candidate_masks[name]=torch.ones(
            batch_size,len(bank),dtype=torch.bool
        )

    max_slots=max(int(row["runtime_operator_slots"]) for row in rows)
    if max_slots<=0:
        raise ValueError("semantic-operator rows require runtime slots")
    relation_targets=torch.zeros(batch_size,max_slots,dtype=torch.long)
    relation_plurality_target_distribution=torch.zeros(
        batch_size,max_slots,len(relation_items),dtype=torch.float32
    )
    relation_plurality_mask=torch.zeros(
        batch_size,max_slots,dtype=torch.bool
    )
    relation_counter=torch.full(
        (batch_size,max_slots),-1,dtype=torch.long
    )
    relation_step_mask=torch.zeros(
        batch_size,max_slots,dtype=torch.bool
    )
    event_targets=torch.zeros(batch_size,max_slots,dtype=torch.long)
    event_mask=torch.zeros(batch_size,max_slots,dtype=torch.bool)
    step_factor_mask=torch.zeros(
        batch_size,max_slots,dtype=torch.bool
    )

    factor_targets={
        name:torch.zeros(batch_size,dtype=torch.long)
        for name in factor_banks
    }
    factor_counter={
        name:torch.full((batch_size,),-1,dtype=torch.long)
        for name in factor_banks
    }
    step_factor_targets={
        name:torch.zeros(batch_size,max_slots,dtype=torch.long)
        for name in factor_banks
    }
    applicability=torch.zeros(batch_size,dtype=torch.float32)
    uncertainty=torch.zeros(batch_size,dtype=torch.float32)

    query_tokens=query_ids.size(1)
    relation_tokens=relation_ids.size(1)
    query_evidence_target=torch.zeros(
        batch_size,max_slots,len(relation_items),query_tokens
    )
    query_evidence_valid=torch.zeros_like(
        query_evidence_target,dtype=torch.bool
    )
    relation_evidence_target=torch.zeros(
        batch_size,max_slots,len(relation_items),relation_tokens
    )
    relation_evidence_valid=torch.zeros_like(
        relation_evidence_target,dtype=torch.bool
    )

    factor_evidence_target={}
    factor_evidence_valid={}
    step_factor_evidence_target={}
    step_factor_evidence_valid={}
    for name,ids in factor_input_ids.items():
        candidates,tokens=ids.shape
        factor_evidence_target[name]=torch.zeros(
            batch_size,candidates,tokens
        )
        factor_evidence_valid[name]=torch.zeros(
            batch_size,candidates,tokens,dtype=torch.bool
        )
        step_factor_evidence_target[name]=torch.zeros(
            batch_size,max_slots,candidates,tokens
        )
        step_factor_evidence_valid[name]=torch.zeros(
            batch_size,max_slots,candidates,tokens,dtype=torch.bool
        )

    for b,row in enumerate(rows):
        local_relations=list(row["relation_candidates"])
        local_keys=[str(item["key"]) for item in local_relations]
        local_targets=[int(value) for value in row["relation_sequence_target"]]
        relation_steps=len(local_targets)
        if len(local_targets)>max_slots:
            raise ValueError("relation program exceeds compiled slot axis")
        for step,local_index in enumerate(local_targets):
            if not 0 <= local_index < len(local_keys):
                raise ValueError("local relation target outside candidate bank")
            relation_targets[b,step]=relation_index[local_keys[local_index]]
            relation_step_mask[b,step]=True
            step_factor_mask[b,step]=True

        plurality_local=[
            int(value)
            for value in (row.get("relation_plurality_target_indices") or [])
        ]
        if plurality_local:
            if not relation_steps:
                raise ValueError(
                    "plurality supervision requires an executable relation step"
                )
            if len(plurality_local)<2 or len(plurality_local)!=len(set(plurality_local)):
                raise ValueError(
                    "plurality supervision requires at least two unique candidates"
                )
            if any(value<0 or value>=len(local_keys) for value in plurality_local):
                raise ValueError("plurality target outside local relation bank")
            representative=int(local_targets[0])
            if representative not in plurality_local:
                raise ValueError(
                    "representative relation target must belong to plurality set"
                )
            mass=1.0/float(len(plurality_local))
            for local_index in plurality_local:
                relation_plurality_target_distribution[
                    b,0,relation_index[local_keys[local_index]]
                ]=mass
            relation_plurality_mask[b,0]=True

        explicit_relation_counter=list(
            row.get("counterfactual_relation_sequence_target") or []
        )
        if explicit_relation_counter:
            if len(explicit_relation_counter)!=len(local_targets):
                raise ValueError("counterfactual relation/program length drift")
            for step,local_index in enumerate(explicit_relation_counter):
                if int(local_index)>=0:
                    relation_counter[b,step]=relation_index[
                        local_keys[int(local_index)]
                    ]

        events=list(row["event_sequence_target"])
        if len(events)!=int(row["runtime_operator_slots"]):
            raise ValueError("event/runtime slot length drift")
        for step,event in enumerate(events):
            if str(event) not in EVENT_INDEX:
                raise ValueError(f"unknown operator event target: {event!r}")
            event_targets[b,step]=EVENT_INDEX[str(event)]
            event_mask[b,step]=True

        row_factor_targets=dict(row["factor_targets"])
        if set(row_factor_targets)!=set(factor_banks):
            raise ValueError("factor target bank mismatch")
        for name,local_index_raw in row_factor_targets.items():
            local_index=int(local_index_raw)
            if not 0 <= local_index < len(factor_banks[name]):
                raise ValueError(f"factor target outside bank {name!r}")
            factor_targets[name][b]=local_index

        explicit_factor_counter=dict(
            row.get("counterfactual_factor_targets") or {}
        )
        unknown_counter=set(explicit_factor_counter)-set(factor_banks)
        if unknown_counter:
            raise ValueError(
                "counterfactual factor target refers to unknown banks: "
                +repr(sorted(unknown_counter))
            )
        for name,value in explicit_factor_counter.items():
            if value is None or int(value)<0:
                continue
            index=int(value)
            if not 0 <= index < len(factor_banks[name]):
                raise ValueError(
                    f"counterfactual factor target outside bank {name!r}"
                )
            if index!=int(factor_targets[name][b]):
                factor_counter[name][b]=index

        local_step=dict(row.get("step_factor_targets") or {})
        for name in factor_banks:
            if name in local_step:
                values=[int(value) for value in local_step[name]]
                if len(values)!=relation_steps:
                    raise ValueError(
                        f"step-local factor/program length drift: {name!r}"
                    )
            elif name in PROGRAM_GLOBAL_BANKS:
                values=[int(row_factor_targets[name])]*relation_steps
            else:
                # A new executable step-local bank cannot silently inherit a
                # global label. Its builder must declare step supervision.
                if relation_steps:
                    raise ValueError(
                        f"missing explicit step targets for step-local factor bank {name!r}"
                    )
                values=[]
            for step,value in enumerate(values):
                if not 0 <= value < len(factor_banks[name]):
                    raise ValueError(
                        f"step factor target outside bank {name!r}"
                    )
                step_factor_targets[name][b,step]=value

        applicability[b]=float(row["applicability_target"])
        uncertainty[b]=float(row["uncertainty_target"])

        evidence=compile_operator_evidence_targets(
            row,
            tokenizer,
            max_length=evidence_max_length,
        )
        q_local=evidence["relation_query_evidence_target"]
        q_valid=evidence["relation_query_evidence_valid_mask"]
        r_local=evidence["relation_schema_evidence_target"]
        r_valid=evidence["relation_schema_evidence_valid_mask"]
        for local_index,key in enumerate(local_keys):
            global_index=relation_index[key]
            q_width=q_local.size(-1)
            r_width=r_local.size(-1)
            query_evidence_target[
                b,:q_local.size(0),global_index,:q_width
            ]=q_local[:,local_index]
            query_evidence_valid[
                b,:q_valid.size(0),global_index,:q_width
            ]=q_valid[:,local_index]
            relation_evidence_target[
                b,:r_local.size(0),global_index,:r_width
            ]=r_local[:,local_index]
            relation_evidence_valid[
                b,:r_valid.size(0),global_index,:r_width
            ]=r_valid[:,local_index]

        for name in factor_banks:
            local_target=evidence["factor_schema_evidence_target"][name]
            local_valid=evidence["factor_schema_evidence_valid_mask"][name]
            factor_evidence_target[name][
                b,:,:local_target.size(-1)
            ]=local_target
            factor_evidence_valid[name][
                b,:,:local_valid.size(-1)
            ]=local_valid

            if name in evidence["step_factor_schema_evidence_target"]:
                local_step_target=evidence[
                    "step_factor_schema_evidence_target"
                ][name]
                local_step_valid=evidence[
                    "step_factor_schema_evidence_valid_mask"
                ][name]
                step_factor_evidence_target[name][
                    b,:local_step_target.size(0),:,:local_step_target.size(-1)
                ]=local_step_target
                step_factor_evidence_valid[name][
                    b,:local_step_valid.size(0),:,:local_step_valid.size(-1)
                ]=local_step_valid

    batch={
        "query_input_ids":query_ids,
        "query_attention_mask":query_mask,
        "relation_input_ids":relation_ids,
        "relation_attention_mask":relation_mask,
        "relation_domain_type_mask":relation_domain,
        "relation_range_type_mask":relation_range,
        "relation_symmetric":relation_symmetric,
        "relation_candidate_mask":relation_candidate_mask,
        "factor_input_ids":factor_input_ids,
        "factor_attention_mask":factor_attention_mask,
        "factor_candidate_masks":factor_candidate_masks,
        "factor_opcodes":factor_opcodes,
        "max_reasoning_steps":max_slots,
    }
    targets={
        "relation_targets":relation_targets,
        "relation_plurality_target_distribution":relation_plurality_target_distribution,
        "relation_plurality_mask":relation_plurality_mask,
        "counterfactual_relation_targets":relation_counter,
        "relation_step_mask":relation_step_mask,
        "factor_targets":factor_targets,
        "counterfactual_factor_targets":factor_counter,
        "step_factor_targets":step_factor_targets,
        "step_factor_mask":step_factor_mask,
        "event_targets":event_targets,
        "event_mask":event_mask,
        "applicability_target":applicability,
        "query_evidence_target":query_evidence_target,
        "query_evidence_valid_mask":query_evidence_valid,
        "relation_schema_evidence_target":relation_evidence_target,
        "relation_schema_evidence_valid_mask":relation_evidence_valid,
        "factor_schema_evidence_target":factor_evidence_target,
        "factor_schema_evidence_valid_mask":factor_evidence_valid,
        "step_factor_schema_evidence_target":step_factor_evidence_target,
        "step_factor_schema_evidence_valid_mask":step_factor_evidence_valid,
        "uncertainty_target":uncertainty,
    }
    return {
        "batch":batch,
        "operator_targets":targets,
        "metadata":{
            "batch_size":batch_size,
            "runtime_relation_count":len(relation_items),
            "runtime_factor_banks":len(factor_banks),
            "max_reasoning_steps":max_slots,
            "private_identity_data":False,
            "fabricated_downstream_labels":False,
            "relation_count_is_capability_ceiling":False,
            "factor_count_is_capability_ceiling":False,
            "reasoning_step_count_is_capability_ceiling":False,
            "plurality_supervision_rows":int(
                relation_plurality_mask.any(dim=1).sum().item()
            ),
        },
    }
