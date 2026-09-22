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
    slot_chunk_size: int = 32
    item_chunk_size: int = 64
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("slot_chunk_size", self.slot_chunk_size),
            ("item_chunk_size", self.item_chunk_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class DynamicCompetitiveLatentPoolV3(nn.Module):
    """Runtime-slot/runtime-view competitive latent pool.

    Slots and views are runtime axes. Competitive assignment is computed with
    exact streaming normalization so the full slot x item score matrix is not
    materialized. View reliability scales item contribution *after* slot
    competition; unlike the older formulation it therefore cannot cancel as a
    softmax-constant term.
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
        values: list[float] = []
        for index in range(slot_count):
            n = index + 1
            inverse = 0.0
            scale = 0.5
            while n:
                inverse += scale * (n & 1)
                n >>= 1
                scale *= 0.5
            values.append(2.0 * inverse - 1.0)
        x = torch.tensor(values, device=device, dtype=dtype)
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

    def _item_slot_normalizers(
        self,
        *,
        slot_key: Tensor,
        item_key: Tensor,
        item_mask: Tensor,
    ) -> list[tuple[int, int, Tensor, Tensor]]:
        """Exact per-item softmax normalizers over runtime slots.

        Returns one differentiable max/denominator pair per item chunk.
        """
        batch, slots, width = slot_key.shape
        items = item_key.size(1)
        scale = 1.0 / math.sqrt(float(width))
        negative = torch.finfo(slot_key.dtype).min
        result: list[tuple[int, int, Tensor, Tensor]] = []

        for i0 in range(0, items, self.config.item_chunk_size):
            i1 = min(i0 + self.config.item_chunk_size, items)
            key_chunk = item_key[:, i0:i1, :]
            valid = item_mask[:, i0:i1]
            running_max = torch.full(
                (batch, i1 - i0),
                negative,
                device=slot_key.device,
                dtype=slot_key.dtype,
            )
            running_denominator = torch.zeros_like(running_max)

            for s0 in range(0, slots, self.config.slot_chunk_size):
                s1 = min(s0 + self.config.slot_chunk_size, slots)
                score = torch.einsum(
                    "bsd,bid->bsi",
                    slot_key[:, s0:s1, :],
                    key_chunk,
                ) * scale
                score = score.masked_fill(
                    ~valid[:, None, :],
                    negative,
                )
                chunk_max = score.max(dim=1).values
                new_max = torch.maximum(running_max, chunk_max)
                old_scale = torch.exp(running_max - new_max)
                exponent = torch.exp(
                    score - new_max[:, None, :]
                )
                exponent = exponent * valid[:, None, :].to(exponent.dtype)
                running_denominator = (
                    running_denominator * old_scale
                    + exponent.sum(dim=1)
                )
                running_max = new_max

            result.append(
                (i0, i1, running_max, running_denominator)
            )
        return result

    def _competitive_message(
        self,
        *,
        slot: Tensor,
        item: Tensor,
        item_mask: Tensor,
        item_reliability: Tensor,
        return_attention_diagnostics: bool,
        views: int,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch, slots, width = slot.shape
        items = item.size(1)
        slot_key = self.slot_query(slot)
        item_key = self.item_key(item)
        item_value = self.slot_value(item)
        normalizers = self._item_slot_normalizers(
            slot_key=slot_key,
            item_key=item_key,
            item_mask=item_mask,
        )
        scale = 1.0 / math.sqrt(float(width))

        messages: list[Tensor] = []
        diagnostic_chunks: list[Tensor] = []

        for s0 in range(0, slots, self.config.slot_chunk_size):
            s1 = min(s0 + self.config.slot_chunk_size, slots)
            slot_key_chunk = slot_key[:, s0:s1, :]
            denominator = torch.zeros(
                batch,
                s1 - s0,
                device=slot.device,
                dtype=slot.dtype,
            )
            numerator = torch.zeros(
                batch,
                s1 - s0,
                width,
                device=slot.device,
                dtype=slot.dtype,
            )
            ownership_chunks: list[Tensor] = []

            for i0, i1, item_max, item_denominator in normalizers:
                valid = item_mask[:, i0:i1]
                score = torch.einsum(
                    "bsd,bid->bsi",
                    slot_key_chunk,
                    item_key[:, i0:i1, :],
                ) * scale
                # An unavailable item has a zero softmax denominator from pass
                # one. Neutralize both score and max before exponentiation so
                # masking cannot become inf * 0 -> NaN.
                safe_score = score.masked_fill(
                    ~valid[:, None, :],
                    0.0,
                )
                safe_item_max = torch.where(
                    valid,
                    item_max,
                    torch.zeros_like(item_max),
                )
                ownership = torch.exp(
                    safe_score - safe_item_max[:, None, :]
                ) / item_denominator[:, None, :].clamp_min(1.0e-12)
                ownership = (
                    ownership
                    * valid[:, None, :].to(ownership.dtype)
                    * item_reliability[:, None, i0:i1].to(ownership.dtype)
                )
                denominator = denominator + ownership.sum(dim=-1)
                numerator = numerator + torch.einsum(
                    "bsi,bid->bsd",
                    ownership,
                    item_value[:, i0:i1, :],
                )
                if return_attention_diagnostics:
                    ownership_chunks.append(ownership)

            message = numerator / denominator.unsqueeze(-1).clamp_min(1.0e-12)
            messages.append(message)

            if return_attention_diagnostics:
                weighted_ownership = torch.cat(ownership_chunks, dim=-1)
                attention = weighted_ownership / denominator.unsqueeze(
                    -1
                ).clamp_min(1.0e-12)
                diagnostic_chunks.append(attention)

        message = torch.cat(messages, dim=1)
        if not return_attention_diagnostics:
            return message, None, None

        attention = torch.cat(diagnostic_chunks, dim=1)
        if attention.shape != (batch, slots, items):
            raise RuntimeError("latent attention diagnostic geometry drift")
        reshaped = attention.reshape(batch, slots, views, 2)
        view_attention_mass = reshaped.sum(dim=3)
        channel_attention_mass = reshaped.sum(dim=2)
        return message, view_attention_mass, channel_attention_mass

    def forward(
        self,
        *,
        source_view_summaries: Tensor,
        contextualized_view_summaries: Tensor,
        view_descriptor_states: Tensor,
        view_available: Tensor,
        query_state: Tensor,
        view_reliability: Tensor,
        view_activity: Tensor | None = None,
        slot_count: int,
        refinement_steps: int,
        return_attention_diagnostics: bool = False,
    ) -> dict[str, Tensor | None]:
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
        if not bool(torch.isfinite(view_reliability).all()):
            raise ValueError("view_reliability contains non-finite values")
        if view_activity is None:
            view_activity = view_available.to(view_reliability.dtype)
        if view_activity.shape != (batch, views):
            raise ValueError("view_activity must be [B,V]")
        if not bool(torch.isfinite(view_activity).all()):
            raise ValueError("view_activity contains non-finite values")
        if bool(
            ((view_activity < 0.0) | (view_activity > 1.0)).any()
        ):
            raise ValueError("view_activity must stay inside [0,1]")
        view_activity = (
            view_activity.float()
            * view_available.to(view_activity.dtype)
        )
        if bool(
            (
                (view_reliability < 0.0)
                | (view_reliability > 1.0)
            ).any()
        ):
            raise ValueError("view_reliability must stay inside [0,1]")
        effective_view_strength = (
            view_reliability.float() * view_activity.float()
        )
        if bool(
            (
                view_available
                & effective_view_strength.le(0.0)
            ).all(dim=-1).any()
        ):
            raise ValueError(
                "every example requires positive reliability/activity on at least one available view"
            )
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
        item_reliability = (
            effective_view_strength[:, :, None]
            .expand(batch, views, 2)
            .reshape(batch, views * 2)
            .float()
        )
        item_reliability = (
            item_reliability
            * item_mask.to(item_reliability.dtype)
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

        view_attention_mass: Tensor | None = None
        channel_attention_mass: Tensor | None = None
        for step in range(refinement_steps):
            need_diagnostics = (
                return_attention_diagnostics
                and step == refinement_steps - 1
            )
            message, step_view_mass, step_channel_mass = self._competitive_message(
                slot=slot,
                item=item,
                item_mask=item_mask,
                item_reliability=item_reliability,
                return_attention_diagnostics=need_diagnostics,
                views=views,
            )
            q = query[:, None, :].expand(batch, slot_count, -1)
            slot = self.slot_state(
                torch.cat([message, q], dim=-1).reshape(
                    batch * slot_count, -1
                ),
                slot.reshape(batch * slot_count, -1),
            ).reshape(batch, slot_count, -1)
            slot = self.output_norm(slot)
            if need_diagnostics:
                view_attention_mass = step_view_mass
                channel_attention_mass = step_channel_mass

        pooled_score = torch.einsum(
            "bsd,bd->bs",
            slot,
            self.pooled_query(query),
        )
        pooled_weight = torch.softmax(pooled_score, dim=-1)
        pooled_state = torch.einsum("bs,bsd->bd", pooled_weight, slot)

        return {
            "latent_slots": slot,
            "pooled_state": pooled_state,
            "pooled_weight": pooled_weight,
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
            "view_reliability_causally_weights_item_contribution": True,
            "query_conditioned_view_activity_supported": True,
            "view_activity_causally_weights_item_contribution": True,
            "reliability_softmax_constant_cancellation": False,
            "full_slot_item_score_matrix_materialized": False,
            "attention_diagnostics_optional": True,
            "runtime_slot_count": True,
            "slot_seed_coordinates_count_stable": True,
            "runtime_view_count": True,
            "slot_chunk_is_operating_point": True,
            "item_chunk_is_operating_point": True,
            "slot_count_ceiling": None,
            "view_count_ceiling": None,
            "refinement_step_ceiling": None,
        }
