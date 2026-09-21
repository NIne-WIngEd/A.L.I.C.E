from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DynamicLatentPoolConfig:
    semantic_dim: int = 640
    model_dim: int = 640
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class DynamicCompetitiveLatentPoolV3(nn.Module):
    """Runtime-slot/runtime-view competitive latent pool.

    Slot count is a runtime axis. Slots are initialized from a shared continuous
    coordinate function instead of a learned per-slot embedding table. Runtime
    view count is also parameter-independent. Evidence items compete for slots,
    preserving the v0.2 anti-collapse lesson without baking a fixed slot/view
    cardinality into the checkpoint.
    """

    def __init__(self, config: DynamicLatentPoolConfig | None = None) -> None:
        super().__init__()
        self.config = config or DynamicLatentPoolConfig()
        self.config.validate()
        d = self.config.model_dim

        self.source_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.context_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.query_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.descriptor_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.channel_embedding = nn.Parameter(torch.empty(2, d))

        self.slot_coordinate = nn.Sequential(
            nn.Linear(6, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.slot_state = nn.GRUCell(2 * d, d)
        self.item_key = nn.Linear(d, d, bias=False)
        self.slot_query = nn.Linear(d, d, bias=False)
        self.slot_value = nn.Linear(d, d, bias=False)
        self.pooled_query = nn.Linear(d, d, bias=False)
        self.output_norm = nn.LayerNorm(d)

        nn.init.normal_(self.channel_embedding, std=0.02)

    @staticmethod
    def _slot_coordinates(
        slot_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if slot_count == 1:
            x = torch.zeros(1, device=device, dtype=dtype)
        else:
            x = torch.linspace(-1.0, 1.0, slot_count, device=device, dtype=dtype)
        return torch.stack(
            [
                x,
                x.square(),
                x.pow(3),
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.sin(2.0 * math.pi * x),
            ],
            dim=-1,
        )

    @staticmethod
    def _masked_normalize(weight: Tensor, mask: Tensor, dim: int) -> Tensor:
        value = weight * mask.to(weight.dtype)
        return value / value.sum(dim=dim, keepdim=True).clamp_min(1.0e-12)

    def forward(
        self,
        *,
        source_view_summaries: Tensor,
        contextualized_view_summaries: Tensor,
        view_descriptor_states: Tensor,
        view_available: Tensor,
        query_state: Tensor,
        view_reliability: Tensor,
        slot_count: int,
        refinement_steps: int,
    ) -> dict[str, Tensor]:
        if source_view_summaries.ndim != 3:
            raise ValueError("source_view_summaries must be [B,V,D]")
        batch, views, semantic_width = source_view_summaries.shape
        if semantic_width != self.config.semantic_dim:
            raise ValueError("source semantic width drift")
        if contextualized_view_summaries.shape != (
            batch,
            views,
            self.config.semantic_dim,
        ):
            raise ValueError("contextualized view shape drift")
        if view_descriptor_states.shape != (
            batch,
            views,
            self.config.semantic_dim,
        ):
            raise ValueError("view descriptor shape drift")
        if view_available.shape != (batch, views) or view_available.dtype != torch.bool:
            raise ValueError("view_available must be bool [B,V]")
        if bool((view_available.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one available view")
        if query_state.shape != (batch, self.config.semantic_dim):
            raise ValueError("query_state shape drift")
        if view_reliability.shape != (batch, views):
            raise ValueError("view_reliability must be [B,V]")
        if refinement_steps <= 0:
            raise ValueError("refinement_steps must be positive")

        source = self.source_projection(source_view_summaries.float())
        context = self.context_projection(contextualized_view_summaries.float())
        descriptor = self.descriptor_projection(view_descriptor_states.float())
        query = self.query_projection(query_state.float())

        source_item = source + descriptor + self.channel_embedding[0]
        context_item = context + descriptor + self.channel_embedding[1]
        item = torch.stack([source_item, context_item], dim=2)
        item = item.reshape(batch, views * 2, self.config.model_dim)
        item_mask = (
            view_available[:, :, None]
            .expand(batch, views, 2)
            .reshape(batch, views * 2)
        )
        reliability = (
            view_reliability[:, :, None]
            .expand(batch, views, 2)
            .reshape(batch, views * 2)
            .float()
        )

        coordinate = self._slot_coordinates(
            int(slot_count),
            device=item.device,
            dtype=item.dtype,
        )
        slot = (
            query[:, None, :]
            + self.slot_coordinate(coordinate)[None, :, :]
        )
        slot = self.output_norm(slot)

        last_attention = torch.zeros(
            batch,
            slot_count,
            views * 2,
            device=item.device,
            dtype=item.dtype,
        )

        for _ in range(refinement_steps):
            score = torch.einsum(
                "bsd,bid->bsi",
                self.slot_query(slot),
                self.item_key(item),
            ) / math.sqrt(float(self.config.model_dim))
            score = score + torch.log(
                reliability[:, None, :].clamp_min(1.0e-3)
            )
            score = score.masked_fill(~item_mask[:, None, :], -1.0e4)

            # First normalize over slots: every evidence item must choose among
            # the available latent explanations. This is the competition term.
            ownership = torch.softmax(score, dim=1)
            ownership = ownership * item_mask[:, None, :].to(ownership.dtype)

            # Then normalize within each slot to create a stable read distribution.
            attention = self._masked_normalize(
                ownership,
                item_mask[:, None, :],
                dim=-1,
            )
            last_attention = attention
            message = torch.einsum(
                "bsi,bid->bsd",
                attention,
                self.slot_value(item),
            )
            q = query[:, None, :].expand(batch, slot_count, -1)
            slot = self.slot_state(
                torch.cat([message, q], dim=-1).reshape(
                    batch * slot_count, -1
                ),
                slot.reshape(batch * slot_count, -1),
            ).reshape(batch, slot_count, -1)
            slot = self.output_norm(slot)

        pooled_score = torch.einsum(
            "bsd,bd->bs",
            slot,
            self.pooled_query(query),
        )
        pooled_weight = torch.softmax(pooled_score, dim=-1)
        pooled_state = torch.einsum("bs,bsd->bd", pooled_weight, slot)

        channel_attention_mass = last_attention.reshape(
            batch, slot_count, views, 2
        ).sum(dim=2)
        view_attention_mass = last_attention.reshape(
            batch, slot_count, views, 2
        ).sum(dim=3)

        return {
            "latent_slots": slot,
            "pooled_state": pooled_state,
            "pooled_weight": pooled_weight,
            "item_attention": last_attention,
            "view_attention_mass": view_attention_mass,
            "channel_attention_mass": channel_attention_mass,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "slot_identity_parameters": 0,
            "view_identity_parameters": 0,
            "slot_count_dependent_parameters": 0,
            "view_count_dependent_parameters": 0,
            "refinement_step_dependent_parameters": 0,
            "competitive_item_ownership": True,
            "source_and_contextualized_channels": True,
            "runtime_slot_count": True,
            "runtime_view_count": True,
            "slot_count_ceiling": None,
            "view_count_ceiling": None,
            "refinement_step_ceiling": None,
        }
