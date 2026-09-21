from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DynamicCrossContextFusionConfig:
    semantic_dim: int = 640
    model_dim: int = 640
    num_attention_heads: int = 10
    recurrent_refinement_steps: int = 2
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("num_attention_heads", self.num_attention_heads),
            ("recurrent_refinement_steps", self.recurrent_refinement_steps),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class DynamicCrossContextFusionV3(nn.Module):
    """Runtime-view fusion with exact source anchors and no view-ID semantics.

    View meaning is supplied by runtime descriptor states. The same refinement,
    attention and routing weights are shared over every view, so adding a new
    runtime view does not create or require a new learned view identity.
    """

    def __init__(self, config: DynamicCrossContextFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DynamicCrossContextFusionConfig()
        self.config.validate()
        d = self.config.model_dim

        self.source_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.descriptor_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.query_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.reliability_projection = nn.Linear(1, d, bias=False)

        self.self_refine = nn.GRUCell(2 * d, d)
        self.cross_attention = nn.MultiheadAttention(
            d,
            self.config.num_attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d)
        self.route_score = nn.Sequential(
            nn.Linear(4 * d + 2, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.fused_projection = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(
        self,
        *,
        source_view_summaries: Tensor,
        view_descriptor_states: Tensor,
        view_available: Tensor,
        query_state: Tensor,
        view_reliability: Tensor,
        refinement_steps: int | None = None,
    ) -> dict[str, Tensor]:
        if source_view_summaries.ndim != 3:
            raise ValueError("source_view_summaries must be [B,V,D]")
        batch, views, semantic_width = source_view_summaries.shape
        if semantic_width != self.config.semantic_dim:
            raise ValueError("source view semantic width drift")
        if views <= 0:
            raise ValueError("at least one runtime view is required")
        if view_descriptor_states.shape != (batch, views, self.config.semantic_dim):
            raise ValueError("view_descriptor_states shape drift")
        if view_available.shape != (batch, views) or view_available.dtype != torch.bool:
            raise ValueError("view_available must be bool [B,V]")
        if bool((view_available.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one available view")
        if query_state.shape != (batch, self.config.semantic_dim):
            raise ValueError("query_state shape drift")
        if view_reliability.shape != (batch, views):
            raise ValueError("view_reliability must be [B,V]")

        steps = (
            self.config.recurrent_refinement_steps
            if refinement_steps is None
            else int(refinement_steps)
        )
        if steps <= 0:
            raise ValueError("refinement_steps must be positive")

        # Exact source channel is retained separately and returned unchanged.
        source_anchor = source_view_summaries
        source = self.source_projection(source_view_summaries.float())
        descriptor = self.descriptor_projection(view_descriptor_states.float())
        query = self.query_projection(query_state.float())
        reliability = self.reliability_projection(
            view_reliability.float().unsqueeze(-1)
        )
        state = source + descriptor + reliability
        state = state * view_available.unsqueeze(-1).to(state.dtype)

        for _ in range(steps):
            q = query[:, None, :].expand(batch, views, -1)
            updated = self.self_refine(
                torch.cat([state, q], dim=-1).reshape(batch * views, -1),
                state.reshape(batch * views, -1),
            ).reshape(batch, views, -1)
            state = torch.where(
                view_available.unsqueeze(-1),
                updated,
                state,
            )

            attended, _ = self.cross_attention(
                state,
                state,
                state,
                key_padding_mask=~view_available,
                need_weights=False,
            )
            state = self.cross_norm(state + attended)
            state = state * view_available.unsqueeze(-1).to(state.dtype)

        q = query[:, None, :].expand(batch, views, -1)
        source_projected = source
        route_scalar = torch.stack(
            [
                view_reliability.float(),
                view_available.to(view_reliability.dtype),
            ],
            dim=-1,
        )
        route_logit = self.route_score(
            torch.cat(
                [
                    state,
                    source_projected,
                    descriptor,
                    q,
                    route_scalar,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        route_logit = route_logit.masked_fill(~view_available, -1.0e4)
        view_weight = torch.softmax(route_logit, dim=-1)
        view_weight = view_weight * view_available.to(view_weight.dtype)
        view_weight = view_weight / view_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)

        contextualized_summary = state
        weighted_context = torch.einsum(
            "bv,bvd->bd",
            view_weight,
            contextualized_summary,
        )
        weighted_source = torch.einsum(
            "bv,bvd->bd",
            view_weight,
            source_projected,
        )
        fused_state = self.fused_projection(
            torch.cat([weighted_context, weighted_source], dim=-1)
        )

        return {
            "source_view_summaries": source_anchor,
            "contextualized_view_summaries": contextualized_summary,
            "view_weight": view_weight,
            "route_logit": route_logit,
            "fused_state": fused_state,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "view_identity_parameters": 0,
            "view_count_dependent_parameters": 0,
            "refinement_step_dependent_parameters": 0,
            "runtime_view_descriptors": True,
            "exact_source_anchor_retained": True,
            "shared_cross_view_attention": True,
            "view_count_ceiling": None,
            "refinement_step_ceiling": None,
        }
