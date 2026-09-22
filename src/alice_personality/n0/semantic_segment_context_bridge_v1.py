from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from alice_personality.n0.chunked_set_attention_v1 import (
    ChunkedSetAttentionConfig,
    ChunkedSetTransformerEncoder,
)
from alice_personality.n0.numeric_contracts import require_finite


@dataclass(frozen=True)
class SemanticSegmentContextBridgeConfig:
    semantic_dim: int = 640
    num_hidden_states: int = 17
    num_attention_heads: int = 10
    num_layers: int = 2
    metadata_dim: int = 3
    query_chunk_segments: int = 32
    key_chunk_segments: int = 64
    dropout: float = 0.0
    initial_context_scale: float = 1.0e-3

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("num_attention_heads", self.num_attention_heads),
            ("num_layers", self.num_layers),
            ("metadata_dim", self.metadata_dim),
            ("query_chunk_segments", self.query_chunk_segments),
            ("key_chunk_segments", self.key_chunk_segments),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.semantic_dim % self.num_attention_heads:
            raise ValueError("semantic_dim must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 <= self.initial_context_scale <= 0.1:
            raise ValueError("initial_context_scale must be in [0,0.1]")


class SemanticSegmentContextBridgeV1(nn.Module):
    """Hierarchical cross-window semantic bridge for virtualized context.

    Native backbone windows remain local operating units. This module summarizes
    every encoded segment using all hidden layers, injects continuous absolute
    position metadata, performs exact memory-bounded global segment attention,
    and feeds the resulting global context back into each segment's token
    states before unique-token stitching.

    The bridge does not claim dense token-by-token equivalence with an
    arbitrarily long native transformer. It removes the stronger defect in the
    old virtualizer: independently encoded windows no longer remain
    semantically isolated from one another.
    """

    def __init__(
        self,
        config: SemanticSegmentContextBridgeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SemanticSegmentContextBridgeConfig()
        self.config.validate()
        d = self.config.semantic_dim

        self.layer_gate = nn.Sequential(
            nn.Linear(d + 1, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.metadata_projection = nn.Sequential(
            nn.Linear(self.config.metadata_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.input_norm = nn.LayerNorm(d)
        self.segment_encoder = ChunkedSetTransformerEncoder(
            ChunkedSetAttentionConfig(
                model_dim=d,
                num_attention_heads=self.config.num_attention_heads,
                query_chunk_fields=self.config.query_chunk_segments,
                key_chunk_fields=self.config.key_chunk_segments,
                feedforward_multiplier=4,
                dropout=self.config.dropout,
            ),
            num_layers=self.config.num_layers,
        )
        self.context_projection = nn.Linear(d, d, bias=False)
        self.injection_gate = nn.Sequential(
            nn.Linear(d + 1, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.context_scale = nn.Parameter(
            torch.tensor(float(self.config.initial_context_scale))
        )

    def _segment_summary(
        self,
        segment_hidden_states: Tensor,
        segment_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # states [B,G,L,W,D], mask [B,G,W]
        if segment_hidden_states.ndim != 5:
            raise ValueError("segment_hidden_states must be [B,G,L,W,D]")
        batch, segments, layers, window, width = segment_hidden_states.shape
        if layers != self.config.num_hidden_states:
            raise ValueError("segment hidden-state depth drift")
        if width != self.config.semantic_dim:
            raise ValueError("segment semantic width drift")
        if segment_attention_mask.shape != (batch, segments, window):
            raise ValueError("segment_attention_mask geometry drift")
        if segment_attention_mask.dtype != torch.bool:
            raise ValueError("segment_attention_mask must be bool")

        weight = segment_attention_mask[:, :, None, :, None].to(
            segment_hidden_states.dtype
        )
        per_layer = (
            (segment_hidden_states * weight).sum(dim=3)
            / weight.sum(dim=3).clamp_min(1.0)
        ).float()
        position = torch.linspace(
            -1.0,
            1.0,
            layers,
            device=per_layer.device,
            dtype=per_layer.dtype,
        ).view(1, 1, layers, 1).expand(batch, segments, layers, 1)
        layer_logit = self.layer_gate(
            torch.cat([per_layer, position], dim=-1)
        ).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        summary = torch.einsum("bgl,bgld->bgd", layer_weight, per_layer)
        return summary, layer_weight

    def forward(
        self,
        *,
        segment_hidden_states: Tensor,
        segment_attention_mask: Tensor,
        segment_valid_mask: Tensor,
        segment_metadata: Tensor,
    ) -> dict[str, Tensor]:
        batch, segments, layers, window, width = segment_hidden_states.shape
        if segment_valid_mask.shape != (batch, segments):
            raise ValueError("segment_valid_mask must be [B,G]")
        if segment_valid_mask.dtype != torch.bool:
            raise ValueError("segment_valid_mask must be bool")
        if bool((segment_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one valid segment")
        if segment_metadata.shape != (
            batch,
            segments,
            self.config.metadata_dim,
        ):
            raise ValueError("segment_metadata geometry drift")
        require_finite("segment_hidden_states", segment_hidden_states)
        require_finite("segment_metadata", segment_metadata)

        valid_token = segment_attention_mask.any(dim=-1)
        if bool((segment_valid_mask & ~valid_token).any()):
            raise ValueError("valid segment has no attended tokens")

        summary, layer_weight = self._segment_summary(
            segment_hidden_states,
            segment_attention_mask,
        )
        state = self.input_norm(
            summary + self.metadata_projection(segment_metadata.float())
        )
        state = state * segment_valid_mask.unsqueeze(-1).to(state.dtype)
        contextual = self.segment_encoder(state, segment_valid_mask)
        contextual = contextual * segment_valid_mask.unsqueeze(-1).to(
            contextual.dtype
        )

        layer_position = torch.linspace(
            -1.0,
            1.0,
            layers,
            device=contextual.device,
            dtype=contextual.dtype,
        ).view(1, 1, layers, 1).expand(batch, segments, layers, 1)
        context_by_layer = contextual[:, :, None, :].expand(
            batch, segments, layers, width
        )
        injection_logit = self.injection_gate(
            torch.cat([context_by_layer, layer_position], dim=-1)
        ).squeeze(-1)
        injection_weight = torch.sigmoid(injection_logit)
        context_delta = self.context_projection(contextual)
        scale = torch.tanh(self.context_scale)
        contextualized = (
            segment_hidden_states
            + scale
            * injection_weight[:, :, :, None, None]
            * context_delta[:, :, None, None, :]
        )
        contextualized = contextualized * segment_valid_mask[
            :, :, None, None, None
        ].to(contextualized.dtype)

        return {
            "contextualized_segment_hidden_states": contextualized,
            "segment_context_state": contextual,
            "segment_layer_weight": layer_weight,
            "context_injection_weight": injection_weight,
            "context_scale": scale,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "segment_identity_parameters": 0,
            "segment_count_dependent_parameters": 0,
            "continuous_absolute_position_metadata": True,
            "all_hidden_layers_used": True,
            "cross_window_semantic_interaction": True,
            "global_segment_attention_memory_bounded": True,
            "full_segment_pair_matrix_materialized": False,
            "native_window_is_operating_point": True,
            "segment_count_ceiling": None,
            "product_context_token_ceiling": None,
            "dense_unbounded_token_attention_equivalence_claimed": False,
        }
