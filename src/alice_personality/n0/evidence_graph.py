from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from torch import nn


class EvidenceRelationType(IntEnum):
    """Stable public relation ids for the N0 evidence graph.

    Direction follows Memory Core: ``source -> target``. ``CORRECTS`` and
    ``SUPERSEDES`` therefore point from the replacement record to the
    historical record. ``CONFLICTS_WITH`` is semantically symmetric even when
    storage keeps one canonical edge.
    """

    PAD = 0
    SUPPORTS = 1
    CORRECTS = 2
    SUPERSEDES = 3
    CONFLICTS_WITH = 4
    DERIVED_FROM = 5
    CAUSES = 6
    TEMPORAL_SUCCESSOR = 7


_RELATION_NAME_TO_ID = {
    "supports": EvidenceRelationType.SUPPORTS,
    "corrects": EvidenceRelationType.CORRECTS,
    "supersedes": EvidenceRelationType.SUPERSEDES,
    "conflicts_with": EvidenceRelationType.CONFLICTS_WITH,
    "derived_from": EvidenceRelationType.DERIVED_FROM,
    "causes": EvidenceRelationType.CAUSES,
    "temporal_successor": EvidenceRelationType.TEMPORAL_SUCCESSOR,
}


def relation_type_id(name: str) -> int:
    """Resolve a governed public relation name to its stable integer id."""
    try:
        return int(_RELATION_NAME_TO_ID[name.strip().lower()])
    except KeyError as exc:
        raise ValueError(f"unsupported evidence relation type: {name!r}") from exc


@dataclass(frozen=True)
class EvidenceGraphConfig:
    """Compact relation sidecar over contextualized structured-state fields."""

    semantic_size: int = 640
    graph_size: int = 256
    num_relation_types: int = 64
    graph_layers: int = 1
    max_fields: int = 64
    max_edges: int = 256
    base_weight_scale: float = 0.5

    def validate(self) -> None:
        if self.semantic_size < 1 or self.graph_size < 1:
            raise ValueError("semantic_size and graph_size must be positive")
        if self.graph_layers < 1:
            raise ValueError("graph_layers must be positive")
        if self.max_fields < 1 or self.max_edges < 1:
            raise ValueError("max_fields and max_edges must be positive")
        if self.num_relation_types <= int(EvidenceRelationType.TEMPORAL_SUCCESSOR):
            raise ValueError("num_relation_types is too small for the stable relation vocabulary")
        if self.base_weight_scale < 0.0:
            raise ValueError("base_weight_scale must be non-negative")


class _DirectedRelationLayer(nn.Module):
    def __init__(self, config: EvidenceGraphConfig) -> None:
        super().__init__()
        d = config.graph_size
        self.relation_embedding = nn.Embedding(config.num_relation_types, d, padding_idx=0)
        self.source_projection = nn.Linear(d, d, bias=False)
        self.edge_confidence_projection = nn.Linear(1, d, bias=False)
        self.message_norm = nn.LayerNorm(d)
        self.update_projection = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.output_norm = nn.LayerNorm(d)

        status = torch.zeros(config.num_relation_types)
        status[int(EvidenceRelationType.SUPPORTS)] = 0.20
        status[int(EvidenceRelationType.CORRECTS)] = -1.00
        status[int(EvidenceRelationType.SUPERSEDES)] = -1.00
        status[int(EvidenceRelationType.CONFLICTS_WITH)] = -0.35
        self.register_buffer("relation_status_prior", status, persistent=True)

    def _messages(
        self,
        node_state: torch.Tensor,
        relation: torch.Tensor,
        confidence: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        message = self.source_projection(node_state)
        message = message + relation + self.edge_confidence_projection(confidence)
        message = torch.nn.functional.silu(self.message_norm(message))
        message = message * confidence
        return message * active.to(message.dtype).unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_confidence: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, fields, width = x.shape
        device = x.device
        dtype = x.dtype

        source = edge_index[..., 0]
        target = edge_index[..., 1]
        batch_index = torch.arange(batch, device=device).unsqueeze(1).expand_as(source)
        safe_source = source.clamp(0, fields - 1)
        safe_target = target.clamp(0, fields - 1)
        endpoint_valid = (
            valid_mask[batch_index, safe_source]
            & valid_mask[batch_index, safe_target]
        )
        active = edge_valid_mask & endpoint_valid

        flat_x = x.reshape(batch * fields, width)
        offsets = (torch.arange(batch, device=device) * fields).unsqueeze(1)
        source_flat = (safe_source + offsets).reshape(-1)
        target_flat = (safe_target + offsets).reshape(-1)
        active_flat = active.reshape(-1)

        relation = self.relation_embedding(edge_type_ids).reshape(-1, width)
        confidence = edge_confidence.to(dtype=dtype).clamp(0.0, 1.0).reshape(-1, 1)
        message = self._messages(
            flat_x[source_flat],
            relation,
            confidence,
            active_flat,
        )

        aggregate = torch.zeros_like(flat_x)
        aggregate.index_add_(0, target_flat, message)
        degree = torch.zeros(batch * fields, 1, device=device, dtype=dtype)
        degree.index_add_(0, target_flat, active_flat.to(dtype).unsqueeze(-1))

        conflict = active & (edge_type_ids == int(EvidenceRelationType.CONFLICTS_WITH))
        conflict_flat = conflict.reshape(-1)
        reverse_message = self._messages(
            flat_x[target_flat],
            relation,
            confidence,
            conflict_flat,
        )
        aggregate.index_add_(0, source_flat, reverse_message)
        degree.index_add_(0, source_flat, conflict_flat.to(dtype).unsqueeze(-1))

        aggregate = aggregate / degree.clamp_min(1.0)
        aggregate = aggregate.reshape(batch, fields, width)
        has_message = degree.reshape(batch, fields, 1) > 0
        candidate = self.output_norm(
            x + self.update_projection(torch.cat([x, aggregate], dim=-1))
        )
        x = torch.where(has_message, candidate, x)

        status_bias = torch.zeros(batch * fields, device=device, dtype=dtype)
        edge_prior = self.relation_status_prior[edge_type_ids].to(dtype).reshape(-1)
        weighted_prior = edge_prior * confidence.reshape(-1) * active_flat.to(dtype)
        status_bias.index_add_(0, target_flat, weighted_prior)
        status_bias.index_add_(
            0,
            source_flat,
            weighted_prior * conflict_flat.to(dtype),
        )

        return (
            x,
            status_bias.reshape(batch, fields),
            aggregate.norm(dim=-1),
        )


class EvidenceGraphEncoder(nn.Module):
    """Relation-aware public evidence sidecar for N0.

    ``field_states`` are contextualized 640d outputs from the separately
    ratified ``StructuredStateEncoder``. Keeping this module separate preserves
    the structured-state pilot's exact parameter lineage and lets graph/evidence
    learning be authorized or rejected independently.
    """

    def __init__(self, config: EvidenceGraphConfig | None = None) -> None:
        super().__init__()
        self.config = config or EvidenceGraphConfig()
        self.config.validate()

        d = self.config.graph_size
        self.input_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.input_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(
            _DirectedRelationLayer(self.config) for _ in range(self.config.graph_layers)
        )
        self.pool_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.pool_query, mean=0.0, std=d ** -0.5)
        self.query_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.field_output_projection = nn.Linear(d, self.config.semantic_size)
        self.pooled_output_projection = nn.Linear(d, self.config.semantic_size)

    def _validate_inputs(
        self,
        *,
        field_states: torch.Tensor,
        valid_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_confidence: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        query_semantic: torch.Tensor | None,
        base_field_weights: torch.Tensor | None,
    ) -> tuple[int, int]:
        if field_states.ndim != 3:
            raise ValueError("field_states must have shape [batch, fields, semantic_size]")
        batch, fields, semantic = field_states.shape
        if semantic != self.config.semantic_size:
            raise ValueError(
                f"semantic width mismatch: expected {self.config.semantic_size}, observed {semantic}"
            )
        if fields > self.config.max_fields:
            raise ValueError(
                f"field count {fields} exceeds configured max_fields={self.config.max_fields}"
            )
        if valid_mask.shape != (batch, fields) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [batch, fields]")
        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError("every graph example must contain at least one valid field")

        if edge_index.ndim != 3 or edge_index.shape[0] != batch or edge_index.shape[2] != 2:
            raise ValueError("edge_index must have shape [batch, edges, 2]")
        edges = edge_index.shape[1]
        if edges > self.config.max_edges:
            raise ValueError(f"edge count {edges} exceeds configured max_edges={self.config.max_edges}")
        if edge_type_ids.shape != (batch, edges):
            raise ValueError("edge_type_ids must have shape [batch, edges]")
        if edge_confidence.shape != (batch, edges, 1):
            raise ValueError("edge_confidence must have shape [batch, edges, 1]")
        if edge_valid_mask.shape != (batch, edges) or edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool with shape [batch, edges]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must be an integer tensor")
        if edge_type_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_type_ids must be an integer tensor")
        if not torch.isfinite(edge_confidence).all():
            raise ValueError("edge_confidence contains non-finite values")
        if edge_type_ids.numel() and (
            edge_type_ids.min().item() < 0
            or edge_type_ids.max().item() >= self.config.num_relation_types
        ):
            raise ValueError("edge_type_ids contains an unsupported relation id")
        if edge_valid_mask.any():
            active_index = edge_index[edge_valid_mask]
            if active_index.min().item() < 0 or active_index.max().item() >= fields:
                raise ValueError("active edge_index contains a field index outside the valid range")
            if edge_type_ids[edge_valid_mask].min().item() <= 0:
                raise ValueError("active edges may not use the PAD relation id")

        if query_semantic is not None:
            if query_semantic.shape != (batch, self.config.semantic_size):
                raise ValueError("query_semantic must have shape [batch, semantic_size]")
            if not torch.isfinite(query_semantic).all():
                raise ValueError("query_semantic contains non-finite values")
        if base_field_weights is not None:
            if base_field_weights.shape != (batch, fields):
                raise ValueError("base_field_weights must have shape [batch, fields]")
            if not torch.isfinite(base_field_weights).all() or (base_field_weights < 0).any():
                raise ValueError("base_field_weights must be finite and non-negative")
        return batch, fields

    def forward(
        self,
        *,
        field_states: torch.Tensor,
        valid_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_confidence: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        query_semantic: torch.Tensor | None = None,
        base_field_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, fields = self._validate_inputs(
            field_states=field_states,
            valid_mask=valid_mask,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            query_semantic=query_semantic,
            base_field_weights=base_field_weights,
        )

        x = self.input_norm(self.input_projection(field_states))
        status_bias = torch.zeros(batch, fields, device=x.device, dtype=x.dtype)
        relation_update_norm = torch.zeros_like(status_bias)
        for layer_index, layer in enumerate(self.layers):
            x, layer_status, layer_norm = layer(
                x,
                edge_index=edge_index,
                edge_type_ids=edge_type_ids,
                edge_confidence=edge_confidence,
                edge_valid_mask=edge_valid_mask,
                valid_mask=valid_mask,
            )
            if layer_index == 0:
                status_bias = layer_status
            relation_update_norm = relation_update_norm + layer_norm

        query = self.pool_query.unsqueeze(0).expand(batch, -1)
        if query_semantic is not None:
            query = query + self.query_projection(query_semantic)
        query = torch.nn.functional.normalize(query, dim=-1)

        pool_scores = torch.einsum("bfd,bd->bf", x, query) / math.sqrt(self.config.graph_size)
        pool_scores = pool_scores + status_bias
        if base_field_weights is not None and self.config.base_weight_scale > 0.0:
            base_prior = base_field_weights.to(x.dtype).clamp_min(1e-6).log()
            pool_scores = pool_scores + self.config.base_weight_scale * base_prior
        pool_scores = pool_scores.masked_fill(~valid_mask, torch.finfo(pool_scores.dtype).min)
        field_weights = torch.softmax(pool_scores, dim=-1)
        pooled = torch.einsum("bf,bfd->bd", field_weights, x)

        return {
            "field_states": self.field_output_projection(x),
            "pooled_state": self.pooled_output_projection(pooled),
            "field_weights": field_weights,
            "relation_status_bias": status_bias,
            "relation_update_norm": relation_update_norm,
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
            "graph_size": self.config.graph_size,
            "graph_layers": self.config.graph_layers,
            "relation_types": self.config.num_relation_types,
            "position_embeddings": 0,
            "private_identity_parameters": 0,
            "semantic_core_parameter_growth": 0,
        }
