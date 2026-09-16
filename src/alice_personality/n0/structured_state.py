from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class StructuredStateConfig:
    """Identity-neutral typed-state branch for N0.

    The branch intentionally carries no positional embeddings. Fields whose
    order is not semantically meaningful therefore remain permutation
    equivariant, while the pooled state is permutation invariant.

    ``max_fields`` is retained only as a legacy/checkpoint operating-shape hint.
    It is not a capability ceiling and is not enforced by the encoder. Successor
    data paths may supply as many fields as memory/compute permit without an
    architecture redesign.
    """

    semantic_size: int = 640
    state_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    feedforward_size: int = 512
    num_field_types: int = 64
    num_provenance_classes: int = 16
    num_relation_roles: int = 64
    num_temporal_scopes: int = 16
    dropout: float = 0.0
    max_fields: int = 64

    def validate(self) -> None:
        if self.semantic_size < 1 or self.state_size < 1:
            raise ValueError("semantic_size and state_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.num_heads < 1 or self.state_size % self.num_heads != 0:
            raise ValueError("state_size must be divisible by num_heads")
        if self.feedforward_size < self.state_size:
            raise ValueError("feedforward_size must be >= state_size")
        if self.max_fields < 1:
            raise ValueError("legacy max_fields operating-shape hint must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class StructuredStateEncoder(nn.Module):
    """Encode typed ACFP-like state without private identity supervision.

    Inputs are pre-indexed, provenance-safe structural features plus semantic
    field vectors produced by the ratified N0 semantic substrate. This module
    does not tokenize private text and does not contain identity concepts.
    """

    def __init__(self, config: StructuredStateConfig | None = None) -> None:
        super().__init__()
        self.config = config or StructuredStateConfig()
        self.config.validate()

        d = self.config.state_size
        self.semantic_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.field_type_embedding = nn.Embedding(self.config.num_field_types, d, padding_idx=0)
        self.provenance_embedding = nn.Embedding(
            self.config.num_provenance_classes, d, padding_idx=0
        )
        self.relation_role_embedding = nn.Embedding(
            self.config.num_relation_roles, d, padding_idx=0
        )
        self.temporal_scope_embedding = nn.Embedding(
            self.config.num_temporal_scopes, d, padding_idx=0
        )
        self.missing_embedding = nn.Embedding(2, d)
        self.confidence_projection = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
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

        self.pool_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.pool_query, mean=0.0, std=d ** -0.5)

        self.field_output_projection = nn.Linear(d, self.config.semantic_size)
        self.pooled_output_projection = nn.Linear(d, self.config.semantic_size)

    def _validate_batch(
        self,
        semantic_values: torch.Tensor,
        field_type_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
        relation_role_ids: torch.Tensor,
        temporal_scope_ids: torch.Tensor,
        confidence: torch.Tensor,
        missing_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if semantic_values.ndim != 3:
            raise ValueError("semantic_values must have shape [batch, fields, semantic_size]")
        batch, fields, semantic = semantic_values.shape
        if semantic != self.config.semantic_size:
            raise ValueError(
                f"semantic width mismatch: expected {self.config.semantic_size}, observed {semantic}"
            )
        if fields < 1:
            raise ValueError("structured state requires at least one field slot")

        expected = (batch, fields)
        for name, value in (
            ("field_type_ids", field_type_ids),
            ("provenance_ids", provenance_ids),
            ("relation_role_ids", relation_role_ids),
            ("temporal_scope_ids", temporal_scope_ids),
            ("missing_mask", missing_mask),
            ("valid_mask", valid_mask),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, observed {tuple(value.shape)}")
        if tuple(confidence.shape) != (batch, fields, 1):
            raise ValueError(
                f"confidence must have shape {(batch, fields, 1)}, observed {tuple(confidence.shape)}"
            )
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool")
        if missing_mask.dtype != torch.bool:
            raise ValueError("missing_mask must be bool")
        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError("every example must contain at least one valid structured field")
        if not torch.isfinite(confidence).all():
            raise ValueError("confidence contains non-finite values")

    def forward(
        self,
        *,
        semantic_values: torch.Tensor,
        field_type_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
        relation_role_ids: torch.Tensor,
        temporal_scope_ids: torch.Tensor,
        confidence: torch.Tensor,
        missing_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_batch(
            semantic_values,
            field_type_ids,
            provenance_ids,
            relation_role_ids,
            temporal_scope_ids,
            confidence,
            missing_mask,
            valid_mask,
        )

        confidence = confidence.to(dtype=semantic_values.dtype).clamp(0.0, 1.0)
        x = self.semantic_projection(semantic_values)
        x = x + self.field_type_embedding(field_type_ids)
        x = x + self.provenance_embedding(provenance_ids)
        x = x + self.relation_role_embedding(relation_role_ids)
        x = x + self.temporal_scope_embedding(temporal_scope_ids)
        x = x + self.missing_embedding(missing_mask.long())
        x = x + self.confidence_projection(confidence)
        x = self.input_norm(x)

        x = self.encoder(x, src_key_padding_mask=~valid_mask)
        x = self.output_norm(x)

        pool_scores = torch.einsum("bfd,d->bf", x, self.pool_query) / math.sqrt(
            self.config.state_size
        )
        pool_scores = pool_scores.masked_fill(~valid_mask, torch.finfo(pool_scores.dtype).min)
        field_weights = torch.softmax(pool_scores, dim=-1)
        pooled_state = torch.einsum("bf,bfd->bd", field_weights, x)

        return {
            "field_states": self.field_output_projection(x),
            "pooled_state": self.pooled_output_projection(pooled_state),
            "field_weights": field_weights,
            "valid_mask": valid_mask,
        }

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "semantic_size": self.config.semantic_size,
            "state_size": self.config.state_size,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "position_embeddings": 0,
            "private_identity_parameters": 0,
            "field_count_limit": None,
            "legacy_max_fields_operating_shape_hint": self.config.max_fields,
            "hard_parameter_ceiling": None,
        }
