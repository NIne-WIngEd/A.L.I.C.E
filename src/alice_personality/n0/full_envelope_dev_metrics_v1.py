from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def _mean_bool(value: Tensor) -> float:
    if value.numel() == 0:
        return 1.0
    return float(value.float().mean().item())


def relation_program_exact(
    *,
    relation_logits: Tensor,
    relation_step_mass: Tensor,
    relation_targets: Tensor,
    relation_step_mask: Tensor,
    active_mass_threshold: float = 0.50,
) -> Tensor:
    if relation_logits.ndim != 3:
        raise ValueError("relation_logits must be [B,S,R]")
    if relation_step_mass.shape != relation_logits.shape[:2]:
        raise ValueError("relation_step_mass geometry drift")
    if relation_targets.shape != relation_logits.shape[:2]:
        raise ValueError("relation target geometry drift")
    if relation_step_mask.shape != relation_logits.shape[:2]:
        raise ValueError("relation mask geometry drift")
    pred=relation_logits.argmax(dim=-1)
    pred_active=relation_step_mass.ge(float(active_mass_threshold))
    exact_relation=torch.where(
        relation_step_mask.bool(),
        pred.eq(relation_targets.long()),
        torch.ones_like(relation_step_mask,dtype=torch.bool),
    )
    return pred_active.eq(relation_step_mask.bool()).logical_and(
        exact_relation
    ).all(dim=-1)


def event_program_exact(
    *,
    event_distribution: Tensor,
    event_targets: Tensor,
    event_mask: Tensor,
) -> Tensor:
    if event_distribution.ndim != 3 or event_distribution.size(-1)!=3:
        raise ValueError("event_distribution must be [B,S,3]")
    if event_targets.shape != event_distribution.shape[:2]:
        raise ValueError("event target geometry drift")
    if event_mask.shape != event_distribution.shape[:2]:
        raise ValueError("event mask geometry drift")
    pred=event_distribution.argmax(dim=-1)
    exact=torch.where(
        event_mask.bool(),
        pred.eq(event_targets.long()),
        torch.ones_like(event_mask,dtype=torch.bool),
    )
    return exact.all(dim=-1)


def global_factor_correct(
    *,
    factor_logits: Mapping[str,Tensor],
    factor_targets: Mapping[str,Tensor],
) -> dict[str,Tensor]:
    if set(factor_logits)!=set(factor_targets):
        raise ValueError("global factor bank names drift")
    result={}
    for name in sorted(factor_logits):
        logits=factor_logits[name]
        target=factor_targets[name]
        if logits.ndim!=2 or target.shape!=(logits.size(0),):
            raise ValueError(f"global factor geometry drift: {name}")
        result[name]=logits.argmax(dim=-1).eq(target.long())
    return result


def step_factor_exact(
    *,
    step_factor_logits: Mapping[str,Tensor],
    step_factor_targets: Mapping[str,Tensor],
    step_factor_mask: Tensor,
) -> dict[str,Tensor]:
    if set(step_factor_logits)!=set(step_factor_targets):
        raise ValueError("step factor bank names drift")
    if step_factor_mask.ndim!=2:
        raise ValueError("step factor mask must be [B,S]")
    result={}
    for name in sorted(step_factor_logits):
        logits=step_factor_logits[name]
        target=step_factor_targets[name]
        if logits.ndim!=3 or target.shape!=logits.shape[:2]:
            raise ValueError(f"step factor geometry drift: {name}")
        if step_factor_mask.shape!=logits.shape[:2]:
            raise ValueError(f"step factor mask geometry drift: {name}")
        correct=torch.where(
            step_factor_mask.bool(),
            logits.argmax(dim=-1).eq(target.long()),
            torch.ones_like(step_factor_mask,dtype=torch.bool),
        )
        result[name]=correct.all(dim=-1)
    return result


def applicability_correct(
    *,
    applicability: Tensor,
    target: Tensor,
) -> Tensor:
    if applicability.shape!=target.shape:
        raise ValueError("applicability geometry drift")
    return applicability.ge(0.5).eq(target.float().ge(0.5))


def uncertainty_correct(
    *,
    uncertainty: Tensor,
    target: Tensor,
) -> Tensor:
    if uncertainty.shape!=target.shape:
        raise ValueError("uncertainty geometry drift")
    return uncertainty.ge(0.5).eq(target.float().ge(0.5))


def margin_success(
    *,
    logits: Tensor,
    targets: Tensor,
    counterfactual_targets: Tensor,
    valid_mask: Tensor,
    margin: float = 0.20,
) -> Tensor:
    if logits.ndim!=targets.ndim+1:
        raise ValueError("causal margin rank drift")
    if targets.shape!=counterfactual_targets.shape:
        raise ValueError("causal target geometry drift")
    if valid_mask.shape!=targets.shape:
        raise ValueError("causal valid-mask geometry drift")
    valid=valid_mask.bool() & counterfactual_targets.ge(0)
    if not bool(valid.any()):
        return torch.empty(0,dtype=torch.bool,device=logits.device)
    correct=logits.gather(-1,targets.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
    counter=logits.gather(
        -1,counterfactual_targets.long().clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    return (correct-counter).masked_select(valid).ge(float(margin))


def token_evidence_f1(
    *,
    predicted: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    threshold: float = 0.50,
) -> float:
    if predicted.shape!=target.shape or valid_mask.shape!=target.shape:
        raise ValueError("token evidence geometry drift")
    valid=valid_mask.bool()
    if not bool(valid.any()):
        return 1.0
    pred=predicted.ge(float(threshold)) & valid
    gold=target.float().ge(0.5) & valid
    tp=int((pred & gold).sum().item())
    fp=int((pred & ~gold & valid).sum().item())
    fn=int((~pred & gold & valid).sum().item())
    if tp==0 and fp==0 and fn==0:
        return 1.0
    precision=tp/max(tp+fp,1)
    recall=tp/max(tp+fn,1)
    return float(2.0*precision*recall/max(precision+recall,1.0e-12))


def mapping_token_evidence_f1(
    *,
    predicted: Mapping[str,Tensor],
    target: Mapping[str,Tensor],
    valid_mask: Mapping[str,Tensor],
    threshold: float = 0.50,
) -> float:
    if set(predicted)!=set(target) or set(predicted)!=set(valid_mask):
        raise ValueError("token-evidence bank names drift")
    values=[
        token_evidence_f1(
            predicted=predicted[name],
            target=target[name],
            valid_mask=valid_mask[name],
            threshold=threshold,
        )
        for name in sorted(predicted)
    ]
    if not values:
        raise ValueError("token-evidence mapping cannot be empty")
    return float(sum(values)/len(values))


def semantic_batch_record(
    *,
    semantic: Mapping[str,object],
    targets: Mapping[str,object],
) -> dict[str,object]:
    operator=semantic["operator"]
    relation_exact=relation_program_exact(
        relation_logits=semantic["relation_logits"],
        relation_step_mass=operator.relation_step_mass,
        relation_targets=targets["relation_targets"],
        relation_step_mask=targets["relation_step_mask"],
    )
    event_exact=event_program_exact(
        event_distribution=operator.event_distribution,
        event_targets=targets["event_targets"],
        event_mask=targets["event_mask"],
    )
    global_correct=global_factor_correct(
        factor_logits=semantic["factor_logits"],
        factor_targets=targets["factor_targets"],
    )
    step_correct=step_factor_exact(
        step_factor_logits=semantic["step_factor_logits"],
        step_factor_targets=targets["step_factor_targets"],
        step_factor_mask=targets["step_factor_mask"],
    )
    relation_margin=margin_success(
        logits=semantic["relation_logits"],
        targets=targets["relation_targets"],
        counterfactual_targets=targets["counterfactual_relation_targets"],
        valid_mask=targets["relation_step_mask"],
    )
    factor_margin_values=[]
    for name in sorted(semantic["factor_logits"]):
        counter=targets["counterfactual_factor_targets"][name]
        valid=counter.ge(0)
        value=margin_success(
            logits=semantic["factor_logits"][name],
            targets=targets["factor_targets"][name],
            counterfactual_targets=counter,
            valid_mask=valid,
        )
        if value.numel():
            factor_margin_values.append(value)
    factor_margin=(
        torch.cat(factor_margin_values)
        if factor_margin_values
        else torch.empty(0,dtype=torch.bool,device=relation_exact.device)
    )
    return {
        "relation_exact":relation_exact,
        "event_exact":event_exact,
        "global_factor_correct":global_correct,
        "step_factor_exact":step_correct,
        "applicability_correct":applicability_correct(
            applicability=operator.applicability,
            target=targets["applicability_target"],
        ),
        "uncertainty_correct":uncertainty_correct(
            uncertainty=operator.uncertainty,
            target=targets["uncertainty_target"],
        ),
        "relation_margin_success":relation_margin,
        "factor_margin_success":factor_margin,
        "query_evidence_f1":token_evidence_f1(
            predicted=semantic["relation_query_evidence"],
            target=targets["query_evidence_target"],
            valid_mask=targets["query_evidence_valid_mask"],
        ),
        "relation_schema_evidence_f1":token_evidence_f1(
            predicted=semantic["relation_schema_evidence"],
            target=targets["relation_schema_evidence_target"],
            valid_mask=targets["relation_schema_evidence_valid_mask"],
        ),
        "factor_schema_evidence_f1":mapping_token_evidence_f1(
            predicted=semantic["factor_schema_evidence"],
            target=targets["factor_schema_evidence_target"],
            valid_mask=targets["factor_schema_evidence_valid_mask"],
        ),
        "step_factor_schema_evidence_f1":mapping_token_evidence_f1(
            predicted=semantic["step_factor_schema_evidence"],
            target=targets["step_factor_schema_evidence_target"],
            valid_mask=targets["step_factor_schema_evidence_valid_mask"],
        ),
    }


def boolean_rate(values: list[bool]) -> float:
    if not values:
        return 1.0
    return float(sum(int(x) for x in values)/len(values))


def mean(values: list[float]) -> float:
    if not values:
        return 1.0
    return float(sum(float(x) for x in values)/len(values))
