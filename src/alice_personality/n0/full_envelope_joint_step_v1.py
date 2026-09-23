from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.natural_relation_batch_v1 import (
    natural_relation_semantic_loss,
)
from alice_personality.n0.ranker import listwise_preference_loss
from alice_personality.n0.v02_objectives import (
    multi_positive_contrastive_loss,
    principle_alignment_loss,
)


def _scalar_loss(value: Any, *, label: str) -> Tensor:
    if isinstance(value, Mapping):
        loss=value.get("loss")
    else:
        loss=getattr(value,"loss",None)
    if not isinstance(loss,Tensor) or loss.ndim!=0:
        raise ValueError(f"{label} task must expose one scalar loss")
    if not bool(torch.isfinite(loss)):
        raise ValueError(f"{label} loss is non-finite")
    return loss


def broad_semantic_replay_loss(
    *,
    system: nn.Module,
    batch: Mapping[str,Any],
) -> Tensor:
    """Execute the governed public MLM replay lane through the shared system."""
    return _scalar_loss(
        system(task="mlm",batch=batch),
        label="broad semantic replay",
    )


def governed_judgment_replay_loss(
    *,
    system: nn.Module,
    batch: Mapping[str,Any],
    temperature: float = 0.05,
) -> tuple[Tensor,dict[str,Tensor]]:
    """Execute all three public teacher replay objectives through one backbone.

    The successor treats preference, rationale alignment, and reusable-principle
    contrastive learning as equal components inside the single governed-
    judgment replay macro family. Their relative weights are therefore fixed
    before results rather than searched after DEV/FINAL outcomes.
    """
    outputs=system(task="teacher",batch=batch)
    if not isinstance(outputs,Mapping):
        raise ValueError("teacher replay task must return a mapping")
    required={"scores","semantic","rationale","alignment_logits"}
    missing=required-set(outputs)
    if missing:
        raise ValueError(
            "teacher replay outputs missing required fields: "
            +repr(sorted(missing))
        )
    scores=outputs["scores"]
    semantic=outputs["semantic"]
    rationale=outputs["rationale"]
    alignment_logits=outputs["alignment_logits"]
    if not all(isinstance(x,Tensor) for x in (
        scores,semantic,rationale,alignment_logits
    )):
        raise ValueError("teacher replay outputs must be tensors")

    group_sizes=[int(x) for x in batch["group_sizes"]]
    preferred_masks=list(batch["preferred_masks"])
    if len(group_sizes)!=len(preferred_masks):
        raise ValueError("teacher group/preferred-mask count drift")
    if sum(group_sizes)!=int(scores.numel()):
        raise ValueError("teacher candidate/group geometry drift")

    preference=listwise_preference_loss(
        scores,
        group_sizes,
        preferred_masks,
    )

    labels=[]
    positive_indices=[]
    offset=0
    for size,preferred in zip(group_sizes,preferred_masks):
        preferred=preferred.to(
            device=semantic.device,
            dtype=torch.bool,
        )
        if preferred.shape!=(size,) or not bool(preferred.any()):
            raise ValueError("teacher group requires a valid preferred mask")
        labels.append(
            preferred.to(
                device=alignment_logits.device,
                dtype=alignment_logits.dtype,
            )
        )
        first=int(torch.nonzero(preferred,as_tuple=False)[0].item())
        positive_indices.append(offset+first)
        offset+=size
    labels_tensor=torch.cat(labels,dim=0)
    alignment=principle_alignment_loss(
        alignment_logits,
        labels_tensor,
    )

    selected=semantic[
        torch.tensor(
            positive_indices,
            device=semantic.device,
            dtype=torch.long,
        )
    ]
    principle_tags=[str(x) for x in batch["principle_tags"]]
    contrastive=multi_positive_contrastive_loss(
        selected,
        rationale,
        principle_tags,
        temperature=temperature,
    )
    components={
        "candidate_preference":preference,
        "principle_rationale_alignment":alignment,
        "semantic_contrastive":contrastive,
    }
    for name,value in components.items():
        if value.ndim!=0 or not bool(torch.isfinite(value)):
            raise ValueError(f"teacher replay component is non-finite: {name}")
    return torch.stack(list(components.values())).mean(),components


def execute_full_envelope_joint_step(
    *,
    system: nn.Module,
    objective: nn.Module,
    mlm_batch: Mapping[str,Any],
    teacher_batch: Mapping[str,Any],
    semantic_operator_compiled: Mapping[str,Any],
    full_fabric_compiled: Mapping[str,Any],
    natural_relation_compiled: Mapping[str,Any],
    update_ema: bool = False,
) -> dict[str,Any]:
    """Execute one differentiable all-lane public N0 successor step.

    This function is deliberately optimizer-agnostic. It establishes the one
    registered causal path that a future authorized trainer must use. Every
    macro-family input comes from its real task/loss path; no proxy loss is
    synthesized from another lane just to make the integrated objective run.
    """
    broad_loss=broad_semantic_replay_loss(
        system=system,
        batch=mlm_batch,
    )
    teacher_loss,teacher_components=governed_judgment_replay_loss(
        system=system,
        batch=teacher_batch,
    )

    semantic_batch=semantic_operator_compiled["batch"]
    operator_targets=semantic_operator_compiled["operator_targets"]
    semantic_outputs=system(
        task="semantic_operator",
        batch=semantic_batch,
    )

    primary=system(
        task="full_envelope",
        batch=full_fabric_compiled["primary_batch"],
    )
    decisive=system(
        task="full_envelope",
        batch=full_fabric_compiled["decisive_ablated_batch"],
    )
    irrelevant=system(
        task="full_envelope",
        batch=full_fabric_compiled["irrelevant_removed_batch"],
    )
    permuted=system(
        task="full_envelope",
        batch=full_fabric_compiled["permuted_batch"],
    )

    natural_outputs=system(
        task="natural_relation",
        batch=natural_relation_compiled["batch"],
    )
    natural_loss=natural_relation_semantic_loss(
        outputs=natural_outputs,
        target_relation_index=natural_relation_compiled[
            "target_relation_index"
        ],
    )

    result=objective(
        primary_outputs=primary,
        decisive_ablated_outputs=decisive,
        irrelevant_removed_outputs=irrelevant,
        permuted_outputs=permuted,
        operator_targets=operator_targets,
        behavioral_targets=full_fabric_compiled[
            "behavioral_targets"
        ],
        broad_semantic_replay_loss=broad_loss,
        governed_judgment_replay_loss=teacher_loss,
        natural_relation_loss=natural_loss,
        semantic_operator_outputs=semantic_outputs,
        update_ema=bool(update_ema),
    )
    return {
        **result,
        "lane_losses":{
            "broad_semantic_replay":broad_loss,
            "governed_judgment_replay":teacher_loss,
            "natural_relation_semantics":natural_loss,
        },
        "governed_judgment_replay_components":teacher_components,
        "all_public_training_lanes_executed":True,
        "placeholder_losses_used":False,
        "private_identity_data":False,
    }
