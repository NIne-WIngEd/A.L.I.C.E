from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MacroFamilyLossBalancerConfig:
    ema_decay: float = 0.98
    minimum_scale: float = 1.0e-4
    maximum_inverse_scale: float = 100.0

    def validate(self) -> None:
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0,1)")
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        if self.maximum_inverse_scale <= 0.0:
            raise ValueError("maximum_inverse_scale must be positive")


class MacroFamilyLossBalancer(nn.Module):
    """Balance heterogeneous N0 objectives without learning task authority.

    Each capability family gets one explicit governance weight. Within a family,
    raw losses are divided by a detached EMA of their observed magnitude so a
    numerically large objective cannot silently dominate the architecture.

    The EMA buffers are optimization mechanics only. They are not learned model
    parameters and are never fitted to TEST/final-validation results.
    """

    def __init__(
        self,
        family_weights: Mapping[str, float],
        config: MacroFamilyLossBalancerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MacroFamilyLossBalancerConfig()
        self.config.validate()
        if not family_weights:
            raise ValueError("at least one capability family is required")
        ordered = OrderedDict((str(k), float(v)) for k, v in family_weights.items())
        if any(v <= 0.0 for v in ordered.values()):
            raise ValueError("family weights must be positive")
        total = sum(ordered.values())
        self.family_names = tuple(ordered)
        self.family_weights = {
            name: value / total for name, value in ordered.items()
        }
        self.register_buffer(
            "_ema_scale",
            torch.ones(len(self.family_names), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "_seen",
            torch.zeros(len(self.family_names), dtype=torch.bool),
            persistent=True,
        )

    def forward(
        self,
        losses: Mapping[str, Mapping[str, Tensor]],
        *,
        update_ema: bool,
    ) -> dict[str, Tensor]:
        if set(losses) != set(self.family_names):
            raise ValueError(
                "loss families must exactly match precommitted family names"
            )
        normalized_family: dict[str, Tensor] = {}
        raw_family: dict[str, Tensor] = {}

        for index, name in enumerate(self.family_names):
            objectives = losses[name]
            if not objectives:
                raise ValueError(f"family {name!r} has no objectives")
            values = []
            for objective_name, value in objectives.items():
                if value.ndim != 0:
                    raise ValueError(
                        f"{name}/{objective_name} must be a scalar tensor"
                    )
                if not bool(torch.isfinite(value.detach())):
                    raise ValueError(
                        f"{name}/{objective_name} is non-finite"
                    )
                values.append(value)
            family_raw = torch.stack(values).mean()
            raw_family[name] = family_raw

            detached = family_raw.detach().abs().float().clamp_min(
                self.config.minimum_scale
            )
            if update_ema:
                if not bool(self._seen[index]):
                    self._ema_scale[index].copy_(detached)
                    self._seen[index] = True
                else:
                    self._ema_scale[index].mul_(self.config.ema_decay).add_(
                        detached * (1.0 - self.config.ema_decay)
                    )
            scale = self._ema_scale[index].clamp_min(
                self.config.minimum_scale
            )
            inverse = (1.0 / scale).clamp_max(
                self.config.maximum_inverse_scale
            )
            normalized_family[name] = family_raw * inverse.to(
                family_raw.dtype
            )

        total = None
        for name in self.family_names:
            term = normalized_family[name] * self.family_weights[name]
            total = term if total is None else total + term

        assert total is not None
        result: dict[str, Tensor] = {
            "loss": total,
            "ema_scale": self._ema_scale.detach().clone(),
        }
        for name in self.family_names:
            result[f"raw/{name}"] = raw_family[name]
            result[f"normalized/{name}"] = normalized_family[name]
        return result

    def parameter_report(self) -> dict[str, int | bool]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "learned_family_weight_parameters": 0,
            "test_adaptive_weights": False,
            "ema_buffers_are_gradient_free": True,
        }
