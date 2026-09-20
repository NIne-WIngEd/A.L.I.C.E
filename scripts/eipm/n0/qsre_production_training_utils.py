from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from alice_personality.n0.qsre_production_core import (
    CONTROL_RELATIONAL,
    QSREDynamicRelationSchema,
    QSREProductionConfig,
    QSREProductionOperatorState,
)


PATH_FAMILIES = {"path_role", "ordered_path", "three_hop", "path_latest"}


def load_plan(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "alice.eipm.n0.qsre-production-training-plan.v1":
        raise RuntimeError("production training-plan schema drift")
    return data


def config_from_plan(plan: dict) -> QSREProductionConfig:
    return QSREProductionConfig(**plan["model"])


def family_metrics(families: list[str], success: torch.Tensor) -> dict[str, float]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for family, ok in zip(families, success.tolist()):
        totals[str(family)][0] += int(bool(ok))
        totals[str(family)][1] += 1
    return {
        family: passed / count
        for family, (passed, count) in sorted(totals.items())
    }


def oracle_operator_from_targets(
    split: dict,
    indices: torch.Tensor,
    *,
    schema_relation_state: torch.Tensor,
    config: QSREProductionConfig,
    device: torch.device,
) -> QSREProductionOperatorState:
    relation_target = split["relation_target"][indices].to(device)
    relation_mask = split["relation_target_mask"][indices].to(device)
    batch, steps = relation_target.shape
    relation_count = schema_relation_state.size(0)

    relation_distribution = torch.zeros(
        batch,
        steps,
        relation_count,
        device=device,
    )
    if bool(relation_mask.any()):
        b_index, s_index = relation_mask.nonzero(as_tuple=True)
        relation_distribution[
            b_index,
            s_index,
            relation_target[b_index, s_index],
        ] = 1.0
    relation_step_mass = relation_mask.to(torch.float32)

    role_target = split["role_target"][indices].to(device)
    role_distribution = F.one_hot(
        role_target,
        num_classes=config.role_count,
    ).float()

    traversal_target = split["traversal_target"][indices].to(device)
    traversal_distribution = F.one_hot(
        traversal_target,
        num_classes=config.traversal_count,
    ).float()

    direction_target = split["direction_target"][indices].to(device)
    direction_distribution = F.one_hot(
        direction_target,
        num_classes=config.direction_count,
    ).float()

    modifier_weight = split["modifier_target"][indices].to(device).float()
    applicability = split["applicability_target"][indices].to(device).float()
    control_target = split["control_target"][indices].to(device)
    control_distribution = F.one_hot(
        control_target,
        num_classes=config.control_count,
    ).float()

    # Production P1 sees a nonzero semantic continuous context. It is the
    # average current relation-schema state for relational rows. P2 is trained
    # to align its continuous state to this same target.
    relation_context = torch.einsum(
        "bsr,rd->bsd",
        relation_distribution,
        schema_relation_state,
    )
    denom = relation_step_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
    continuous = (
        relation_context
        * relation_step_mass.unsqueeze(-1)
    ).sum(dim=1) / denom
    continuous = torch.where(
        relation_step_mass.any(dim=1, keepdim=True),
        continuous,
        torch.zeros_like(continuous),
    )

    operator = QSREProductionOperatorState(
        relation_distribution=relation_distribution,
        relation_step_mass=relation_step_mass,
        stop_probability=torch.where(
            relation_mask,
            torch.zeros_like(relation_step_mass),
            torch.ones_like(relation_step_mass),
        ),
        unknown_probability=torch.zeros_like(relation_step_mass),
        role_distribution=role_distribution,
        traversal_distribution=traversal_distribution,
        direction_distribution=direction_distribution,
        modifier_weight=modifier_weight,
        applicability=applicability,
        control_distribution=control_distribution,
        continuous_state=continuous,
        uncertainty=torch.zeros(batch, device=device),
    )
    operator.validate(
        relation_count=relation_count,
        model_dim=config.model_dim,
    )
    return operator


def batch_indices(
    count: int,
    *,
    batch_size: int,
    seed: int,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(count, generator=generator)
    batches = [
        order[start : start + batch_size]
        for start in range(0, count, batch_size)
    ]
    if device is not None:
        batches = [batch.to(device) for batch in batches]
    return batches


def normalized_distribution(probability: torch.Tensor) -> torch.Tensor:
    total = probability.sum(dim=-1, keepdim=True)
    return torch.where(
        total > 0,
        probability / total.clamp_min(1.0e-12),
        torch.zeros_like(probability),
    )


def target_distribution_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target_mass = target.sum(dim=-1)
    relational = target_mass > 0
    normalized = normalized_distribution(probability)
    per = torch.zeros_like(target_mass)
    if bool(relational.any()):
        t = target[relational]
        p = normalized[relational].clamp_min(1.0e-8)
        per[relational] = -(t * p.log()).sum(dim=-1)
    if bool((~relational).any()):
        per[~relational] = probability[~relational].sum(dim=-1)
    return per.mean()


def downstream_success(
    probability: torch.Tensor,
    target: torch.Tensor,
    control_target: torch.Tensor,
    *,
    plural_l1_threshold: float = 0.10,
    nonrelational_mass_threshold: float = 0.01,
) -> dict[str, torch.Tensor]:
    normalized = normalized_distribution(probability)
    target_mass = target.sum(dim=-1)
    relational = control_target.eq(CONTROL_RELATIONAL) & target_mass.gt(0)
    target_count = target.gt(0).sum(dim=-1)
    single = relational & target_count.eq(1)
    plural = relational & target_count.gt(1)
    nonrel = ~relational

    success = torch.zeros(
        probability.size(0),
        dtype=torch.bool,
        device=probability.device,
    )
    top1_correct = torch.zeros_like(success)
    if bool(single.any()):
        pred = normalized[single].argmax(dim=-1)
        gold = target[single].argmax(dim=-1)
        correct = pred.eq(gold)
        success[single] = correct
        top1_correct[single] = correct

    plural_l1 = torch.zeros(
        probability.size(0),
        dtype=probability.dtype,
        device=probability.device,
    )
    if bool(plural.any()):
        l1 = (normalized[plural] - target[plural]).abs().sum(dim=-1)
        plural_l1[plural] = l1
        success[plural] = l1.le(plural_l1_threshold)

    if bool(nonrel.any()):
        mass = probability[nonrel].sum(dim=-1)
        success[nonrel] = mass.le(nonrelational_mass_threshold)

    return {
        "success": success,
        "single_mask": single,
        "single_top1_correct": top1_correct,
        "plural_mask": plural,
        "plural_l1": plural_l1,
        "nonrel_mask": nonrel,
    }


def summarize_downstream(
    *,
    probability: torch.Tensor,
    target: torch.Tensor,
    control_target: torch.Tensor,
    families: list[str],
    open_schema: torch.Tensor,
    causal_groups: list[str],
) -> dict[str, object]:
    parts = downstream_success(
        probability,
        target,
        control_target,
    )
    success = parts["success"]
    family = family_metrics(families, success)
    single = parts["single_mask"]
    plural = parts["plural_mask"]
    open_mask = open_schema.bool()

    single_acc = (
        float(parts["single_top1_correct"][single].float().mean().item())
        if bool(single.any())
        else 1.0
    )
    plural_l1 = (
        float(parts["plural_l1"][plural].mean().item())
        if bool(plural.any())
        else 0.0
    )
    open_success = (
        float(success[open_mask].float().mean().item())
        if bool(open_mask.any())
        else 1.0
    )
    path_values = [
        value
        for name, value in family.items()
        if name in PATH_FAMILIES
    ]
    path_min = min(path_values) if path_values else 1.0

    # Outside-support causal pairs must be exactly invariant to distractor
    # metadata changes. Compare full probability tensors within the group.
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (group, fam) in enumerate(zip(causal_groups, families)):
        if fam == "outside_support":
            groups[str(group)].append(index)
    invariance = 0.0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        base = probability[indices[0]]
        for index in indices[1:]:
            invariance = max(
                invariance,
                float((base - probability[index]).abs().max().item()),
            )

    return {
        "row_success_accuracy": float(success.float().mean().item()),
        "single_target_top1_accuracy": single_acc,
        "plural_l1": plural_l1,
        "family_success": family,
        "family_min_success": min(family.values()) if family else 1.0,
        "open_schema_success": open_success,
        "path_family_min_success": path_min,
        "outside_support_invariance_max_delta": invariance,
        "row_success_tensor": success,
    }


def relation_program_exact(
    operator: QSREProductionOperatorState,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    active_mass_threshold: float = 0.50,
) -> torch.Tensor:
    pred_relation = operator.relation_distribution.argmax(dim=-1)
    pred_active = operator.relation_step_mass.ge(active_mass_threshold)
    pred_steps = pred_relation.size(1)
    target_steps = target.size(1)
    if pred_steps < target_steps:
        raise ValueError("operator emitted fewer relation steps than supervised target")
    if pred_steps > target_steps:
        pad = pred_steps - target_steps
        target = F.pad(target, (0, pad), value=0)
        target_mask = F.pad(target_mask, (0, pad), value=False)
    mask_exact = pred_active.eq(target_mask)
    relation_exact = torch.where(
        target_mask,
        pred_relation.eq(target),
        torch.ones_like(target_mask),
    )
    return (mask_exact & relation_exact).all(dim=-1)


def modifier_exact(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return predicted.ge(0.5).eq(target.ge(0.5)).all(dim=-1)


def operator_metrics(
    *,
    operator: QSREProductionOperatorState,
    split: dict,
    indices: torch.Tensor,
) -> dict[str, object]:
    device = operator.continuous_state.device
    relation_target = split["relation_target"][indices].to(device)
    relation_mask = split["relation_target_mask"][indices].to(device)
    role_target = split["role_target"][indices].to(device)
    traversal_target = split["traversal_target"][indices].to(device)
    direction_target = split["direction_target"][indices].to(device)
    modifier_target = split["modifier_target"][indices].to(device)
    control_target = split["control_target"][indices].to(device)
    termination_target = split["termination_target"][indices].to(device)
    open_schema = split["open_schema"][indices].to(device)

    relation_exact = relation_program_exact(
        operator,
        relation_target,
        relation_mask,
    )
    relational = control_target.eq(CONTROL_RELATIONAL)
    role_correct = operator.role_distribution.argmax(dim=-1).eq(role_target)
    traversal_correct = operator.traversal_distribution.argmax(dim=-1).eq(
        traversal_target
    )
    direction_correct = operator.direction_distribution.argmax(dim=-1).eq(
        direction_target
    )
    modifier_correct = modifier_exact(operator.modifier_weight, modifier_target)
    control_correct = operator.control_distribution.argmax(dim=-1).eq(
        control_target
    )
    lengths = relation_mask.long().sum(dim=-1)
    terminal_index = lengths.clamp(max=operator.stop_probability.size(1) - 1)
    row_index = torch.arange(lengths.size(0), device=device)
    terminal_unknown = operator.unknown_probability[row_index, terminal_index]
    terminal_stop = operator.stop_probability[row_index, terminal_index]
    termination_pred = terminal_unknown.gt(terminal_stop).long()
    termination_correct = termination_pred.eq(termination_target)
    unknown_rows = termination_target.eq(1)

    return {
        "relation_sequence_exact_accuracy": float(
            relation_exact.float().mean().item()
        ),
        "open_schema_relation_exact_accuracy": (
            float(relation_exact[open_schema].float().mean().item())
            if bool(open_schema.any())
            else 1.0
        ),
        "role_accuracy_relational": (
            float(role_correct[relational].float().mean().item())
            if bool(relational.any())
            else 1.0
        ),
        "traversal_accuracy_relational": (
            float(traversal_correct[relational].float().mean().item())
            if bool(relational.any())
            else 1.0
        ),
        "direction_accuracy_relational": (
            float(direction_correct[relational].float().mean().item())
            if bool(relational.any())
            else 1.0
        ),
        "modifier_exact_accuracy_relational": (
            float(modifier_correct[relational].float().mean().item())
            if bool(relational.any())
            else 1.0
        ),
        "control_accuracy": float(control_correct.float().mean().item()),
        "termination_accuracy": float(termination_correct.float().mean().item()),
        "unknown_termination_accuracy": (
            float(termination_correct[unknown_rows].float().mean().item())
            if bool(unknown_rows.any())
            else 1.0
        ),
        "relation_exact_tensor": relation_exact,
        "role_correct_tensor": role_correct,
        "traversal_correct_tensor": traversal_correct,
        "direction_correct_tensor": direction_correct,
        "modifier_correct_tensor": modifier_correct,
        "control_correct_tensor": control_correct,
        "termination_correct_tensor": termination_correct,
    }


def paired_view_consistency(
    *,
    first: QSREProductionOperatorState,
    second: QSREProductionOperatorState,
) -> float:
    relation_same = first.relation_distribution.argmax(dim=-1).eq(
        second.relation_distribution.argmax(dim=-1)
    ).all(dim=-1)
    active_same = first.relation_step_mass.ge(0.5).eq(
        second.relation_step_mass.ge(0.5)
    ).all(dim=-1)
    role_same = first.role_distribution.argmax(dim=-1).eq(
        second.role_distribution.argmax(dim=-1)
    )
    traversal_same = first.traversal_distribution.argmax(dim=-1).eq(
        second.traversal_distribution.argmax(dim=-1)
    )
    direction_same = first.direction_distribution.argmax(dim=-1).eq(
        second.direction_distribution.argmax(dim=-1)
    )
    modifiers_same = first.modifier_weight.ge(0.5).eq(
        second.modifier_weight.ge(0.5)
    ).all(dim=-1)
    control_same = first.control_distribution.argmax(dim=-1).eq(
        second.control_distribution.argmax(dim=-1)
    )
    same = (
        relation_same
        & active_same
        & role_same
        & traversal_same
        & direction_same
        & modifiers_same
        & control_same
    )
    return float(same.float().mean().item())


def support_metrics(
    *,
    predicted: torch.Tensor,
    oracle: torch.Tensor,
    type_compatible: torch.Tensor,
) -> dict[str, float]:
    pred = predicted.gt(1.0e-8)
    gold = oracle.gt(0)
    tp = (pred & gold).sum().item()
    fp = (pred & ~gold).sum().item()
    fn = (~pred & gold).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
    exact = pred.eq(gold).all(dim=-1).float().mean().item()
    violation = (pred & ~type_compatible).sum().item() / max(pred.sum().item(), 1)
    return {
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "exact_set_accuracy": float(exact),
        "type_violation_rate": float(violation),
    }



def focus_metrics(
    *,
    predicted: torch.Tensor,
    oracle: torch.Tensor,
    traversal_target: torch.Tensor,
) -> dict[str, float]:
    if predicted.shape != oracle.shape:
        raise ValueError("focus prediction/oracle shape drift")
    path = traversal_target.eq(1)
    if not bool(path.any()):
        return {
            "path_focus_top1_accuracy": 1.0,
            "path_focus_exact_set_accuracy": 1.0,
        }
    pred_mask = predicted[path].gt(1.0e-8)
    gold_mask = oracle[path].gt(0)
    top1 = predicted[path].argmax(dim=-1).eq(
        oracle[path].argmax(dim=-1)
    ).float().mean().item()
    exact = pred_mask.eq(gold_mask).all(dim=-1).float().mean().item()
    return {
        "path_focus_top1_accuracy": float(top1),
        "path_focus_exact_set_accuracy": float(exact),
    }
