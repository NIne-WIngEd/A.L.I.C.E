from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EvidenceViewAdapterConfig:
    """Query-conditioned evidence view over a frozen structured-state parent.

    Parameter counts are configuration facts, not capability ceilings. The
    adapter exists so relation learning can specialize without overwriting the
    ratified structured representation.

    ``max_fields`` is a legacy/checkpoint operating-shape hint only. Runtime
    validation does not impose it as a field-count ceiling.
    """

    semantic_size: int = 640
    adapter_size: int = 384
    num_layers: int = 2
    num_heads: int = 6
    feedforward_size: int = 1536
    dropout: float = 0.0
    max_fields: int = 64
    base_prior_scale: float = 0.5

    def validate(self) -> None:
        if self.semantic_size < 1 or self.adapter_size < 1:
            raise ValueError("semantic_size and adapter_size must be positive")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if self.adapter_size % self.num_heads != 0:
            raise ValueError("adapter_size must be divisible by num_heads")
        if self.feedforward_size < self.adapter_size:
            raise ValueError("feedforward_size must be at least adapter_size")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_fields < 1:
            raise ValueError("legacy max_fields operating-shape hint must be positive")
        if self.base_prior_scale < 0.0:
            raise ValueError("base_prior_scale must be non-negative")


class EvidenceViewAdapter(nn.Module):
    """Trainable evidence-specific view that leaves the parent view immutable.

    ``parent_field_states`` and ``parent_field_weights`` come from the ratified
    StructuredStateEncoder. This module never mutates that parent. It learns a
    query-conditioned, permutation-equivariant evidence view used only by the
    relation graph/fusion path.
    """

    def __init__(self, config: EvidenceViewAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or EvidenceViewAdapterConfig()
        self.config.validate()

        d = self.config.adapter_size
        self.input_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.query_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.input_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_size,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.output_norm = nn.LayerNorm(d)
        self.output_projection = nn.Linear(d, self.config.semantic_size, bias=False)
        self.prior_head = nn.Linear(d, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def _validate_inputs(
        self,
        *,
        parent_field_states: torch.Tensor,
        valid_mask: torch.Tensor,
        query_semantic: torch.Tensor,
        parent_field_weights: torch.Tensor | None,
    ) -> tuple[int, int]:
        if parent_field_states.ndim != 3:
            raise ValueError("parent_field_states must have shape [batch, fields, semantic_size]")
        batch, fields, semantic = parent_field_states.shape
        if semantic != self.config.semantic_size:
            raise ValueError(
                f"semantic width mismatch: expected {self.config.semantic_size}, observed {semantic}"
            )
        if fields < 1:
            raise ValueError("evidence view requires at least one field slot")
        if valid_mask.shape != (batch, fields) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [batch, fields]")
        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError("every evidence view requires at least one valid field")
        if query_semantic.shape != (batch, self.config.semantic_size):
            raise ValueError("query_semantic must have shape [batch, semantic_size]")
        if not torch.isfinite(parent_field_states).all() or not torch.isfinite(query_semantic).all():
            raise ValueError("evidence view inputs must be finite")
        if parent_field_weights is not None:
            if parent_field_weights.shape != (batch, fields):
                raise ValueError("parent_field_weights must have shape [batch, fields]")
            if not torch.isfinite(parent_field_weights).all() or (parent_field_weights < 0).any():
                raise ValueError("parent_field_weights must be finite and non-negative")
        return batch, fields

    def forward(
        self,
        *,
        parent_field_states: torch.Tensor,
        valid_mask: torch.Tensor,
        query_semantic: torch.Tensor,
        parent_field_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _batch, _fields = self._validate_inputs(
            parent_field_states=parent_field_states,
            valid_mask=valid_mask,
            query_semantic=query_semantic,
            parent_field_weights=parent_field_weights,
        )

        x = self.input_projection(parent_field_states)
        x = x + self.query_projection(query_semantic).unsqueeze(1)
        x = self.input_norm(x)
        x = self.encoder(x, src_key_padding_mask=~valid_mask)
        x = self.output_norm(x)

        delta = self.output_projection(x)
        scale = torch.tanh(self.residual_scale)
        field_states = parent_field_states + scale * delta
        field_states = torch.where(valid_mask.unsqueeze(-1), field_states, parent_field_states)

        logits = self.prior_head(x).squeeze(-1)
        if parent_field_weights is not None and self.config.base_prior_scale > 0.0:
            logits = logits + self.config.base_prior_scale * parent_field_weights.clamp_min(1e-6).log()
        logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        field_weights = torch.softmax(logits, dim=-1)
        pooled_state = torch.einsum("bf,bfd->bd", field_weights, field_states)

        return {
            "field_states": field_states,
            "field_weights": field_weights,
            "pooled_state": pooled_state,
            "valid_mask": valid_mask,
            "residual_scale": scale,
        }

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "semantic_size": self.config.semantic_size,
            "adapter_size": self.config.adapter_size,
            "layers": self.config.num_layers,
            "heads": self.config.num_heads,
            "feedforward_size": self.config.feedforward_size,
            "position_embeddings": 0,
            "private_identity_parameters": 0,
            "field_count_limit": None,
            "legacy_max_fields_operating_shape_hint": self.config.max_fields,
            "hard_parameter_ceiling": None,
        }
