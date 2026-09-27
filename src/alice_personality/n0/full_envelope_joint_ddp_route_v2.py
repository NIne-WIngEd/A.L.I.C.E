"""One distributed forward for one intact N0 joint training microbatch.

The registered system and objective remain separately addressable so existing
system/objective checkpoint files retain their original unprefixed key space.
DDP sees only the differentiable scalar loss. Detached reporting is carried on
the local module, never returned as an additional DDP graph root.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.full_envelope_joint_step_v1 import (
    execute_full_envelope_joint_step,
)
from alice_personality.n0.full_envelope_training_objective_v1 import (
    FullEnvelopeJointTrainingObjectiveV1,
)


class FullEnvelopeJointDDPRouteV2(nn.Module):
    def __init__(
        self,
        system: nn.Module,
        objective: FullEnvelopeJointTrainingObjectiveV1,
    ) -> None:
        super().__init__()
        if any(True for _ in objective.parameters()):
            raise ValueError("the registered joint objective must have no optimizer parameters")
        self.system=system
        self.objective=objective
        self._observation: dict[str,Any] | None=None

    def take_observation(self) -> dict[str,Any]:
        if self._observation is None:
            raise RuntimeError("no completed joint forward to observe")
        observation=self._observation
        self._observation=None
        return observation

    def forward(
        self,
        *,
        mlm_batch: Mapping[str,Any],
        teacher_batch: Mapping[str,Any],
        semantic_operator_compiled: Mapping[str,Any],
        full_fabric_compiled: Mapping[str,Any] | None,
        natural_relation_compiled: Mapping[str,Any],
        stage: str,
    ) -> Tensor:
        if self._observation is not None:
            raise RuntimeError("previous joint observation must be consumed before next forward")
        result=execute_full_envelope_joint_step(
            system=self.system,
            objective=self.objective,
            mlm_batch=mlm_batch,
            teacher_batch=teacher_batch,
            semantic_operator_compiled=semantic_operator_compiled,
            full_fabric_compiled=full_fabric_compiled,
            natural_relation_compiled=natural_relation_compiled,
            update_ema=False,
            stage=stage,
        )
        loss=result["loss"]
        if not isinstance(loss,Tensor) or loss.ndim!=0 or not bool(torch.isfinite(loss.detach())):
            raise RuntimeError("whole-step DDP forward requires a finite scalar loss")
        active=tuple(result["active_families"])
        self._observation={
            "active_families":active,
            "balanced":{
                f"raw/{name}":result["balanced"][f"raw/{name}"].detach()
                for name in active
            },
            "all_active_stage_lanes_executed":result["all_active_stage_lanes_executed"],
            "all_public_training_lanes_executed":result["all_public_training_lanes_executed"],
            "placeholder_losses_used":result["placeholder_losses_used"],
            "requires_full_fabric_primary":result["requires_full_fabric_primary"],
            "requires_full_fabric_counterfactuals":result["requires_full_fabric_counterfactuals"],
        }
        return loss
