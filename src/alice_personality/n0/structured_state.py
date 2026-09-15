from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from torch import nn


class EvidenceRelationType(IntEnum):
    """Stable public relation ids used by the N0 structured evidence graph.

    Direction follows the memory layer convention: ``source -> target``.
    In particular, ``CORRECTS`` and ``SUPERSEDES`` point from the replacement
    record to the historical record. ``CONFLICTS_WITH`` is symmetric even if
    storage keeps only one canonical edge.
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
class StructuredStateConfig:
    """Identity-neutral typed-state and evidence-graph branch for N0.

    The branch intentionally carries no positional embeddings. Node order is
    therefore non-semantic. Explicit sparse edges carry relation structure,
    while the pooled state remains invariant to a consistent permutation of
    nodes and edge indices.
    """

    semantic_size: int = 640
    state_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    feedforward_size: int = 512
    num_field_types: int = 64
    num_provenance_classes: int = 16
    num_relation_roles: int = 64
    num_relation_types: int = 64
    num_temporal_scopes: int = 16
    graph_layers: int = 1
    dropout: float = 0.0
    max_fields: int = 64
    max_edges: int = 256

    def validate(self) -> None:
        if self.semantic_size < 1 or self.state_size < 1:
            raise ValueError("semantic_size and state_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.graph_layers < 0:
            raise ValueError("graph_layers must be non-negative")
        if self.num_heads < 1 or self.state_size % self.num_heads != 0:
            raise ValueError("state_size must be divisible by num_heads")
        if self.feedforward_size < self.state_size:
            raise ValueError("feedforward_size must be >= state_size")
        if self.max_fields < 1 or self.max_edges < 1:
            raise ValueError("max_fields and max_edges must be positive")
        if self.num_relation_types <= int(EvidenceRelationType.TEMPORAL_SUCCESSOR):
            raise ValueError("num_relation_types is too small for the stable relation vocabulary")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _EvidenceGraphLayer(nn.Module):
    """Sparse directed message passing without a graph-library dependency."""

    def __init__(self, config: StructuredStateConfig) -> None:
        super().__init__()
        d = config.state_size
        self.config = config
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

        # Deterministic lifecycle semantics. These are not learned identity
        # preferences. They only control whether an incoming relation should
        # make a node more or less eligible for evidence pooling.
        status = torch.zeros(config.num_relation_types)
        status[int(EvidenceRelationType.SUPPORTS)] = 0.20
        status[int(EvidenceRelationType.CORRECTS)] = -1.00
        status[int(EvidenceRelationType.SUPERSEDES)] = -1.00
        status[int(EvidenceRelationType.CONFLICTS_WITH)] = -0.35
        self.register_buffer("relation_status_prior", status, persistent=True)

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
        message = self.source_projection(flat_x[source_flat])
        message = message + relation + self.edge_confidence_projection(confidence)
        message = torch.nn.functional.silu(self.message_norm(message))
        message = message * confidence
        message = message * active_flat.to(dtype).unsqueeze(-1)

        aggregate = torch.zeros_like(flat_x)
        aggregate.index_add_(0, target_flat, message)
        degree = torch.zeros(batch * fields, 1, device=device, dtype=dtype)
        degree.index_add_(0, target_flat, active_flat.to(dtype).unsqueeze(-1))

        # Stored conflict edges are canonicalized to one direction, but the
        # semantic relation is symmetric. Mirror only conflict messages.
        conflict = active & (edge_type_ids == int(EvidenceRelationType.CONFLICTS_WITH))
        conflict_flat = conflict.reshape(-1)
        reverse_message = message * conflict_flat.to(dtype).unsqueeze(-1)
        aggregate.index_add_(0, source_flat, reverse_message)
        degree.index_add_(0, source_flat, conflict_flat.to(dtype).unsqueeze(-1))

        aggregate = aggregate / degree.clamp_min(1.0)
        aggregate = aggregate.reshape(batch, fields, width)
        updated = self.update_projection(torch.cat([x, aggregate], dim=-1))
        x = self.output_norm(x + updated)

        status_bias = torch.zeros(batch * fields, device=device, dtype=dtype)
        edge_prior = self.relation_status_prior[edge_type_ids].to(dtype).reshape(-1)
        weighted_prior = edge_prior * confidence.reshape(-1) * active_flat.to(dtype)
        status_bias.index_add_(0, target_flat, weighted_prior)

        conflict_prior = weighted_prior * conflict_flat.to(dtype)
        status_bias.index_add_(0, source_flat, conflict_prior)
        status_bias = status_bias.reshape(batch, fields)

        aggregate_norm = aggregate.norm(dim=-1)
        return x, status_bias, aggregate_norm


class StructuredStateEncoder(nn.Module):
    """Encode typed public state plus an explicit sparse evidence graph.

    Inputs are pre-indexed, provenance-safe structural features plus semantic
    field vectors produced by the ratified N0 semantic substrate. The graph is
    optional for backwards compatibility. When supplied, edges are directed,
    typed, confidence-weighted, and permutation equivariant.

    ``query_semantic`` is also optional. Supplying it turns pooling into a
    query-conditioned evidence read rather than a single static summary.
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
        self.graph_layers = nn.ModuleList(
            _EvidenceGraphLayer(self.config) for _ in range(self.config.graph_layers)
        )

        self.pool_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.pool_query, mean=0.0, std=d ** -0.5)
        self.query_projection = nn.Linear(self.config.semantic_size, d, bias=False)

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
    ) -> tuple[int, int]:
        if semantic_values.ndim != 3:
            raise ValueError("semantic_values must have shape [batch, fields, semantic_size]")
        batch, fields, semantic = semantic_values.shape
        if semantic != self.config.semantic_size:
            raise ValueError(
                f"semantic width mismatch: expected {self.config.semantic_size}, observed {semantic}"
            )
        if fields > self.config.max_fields:
            raise ValueError(
                f"field count {fields} exceeds configured max_fields={self.config.max_fields}"
            )

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
        return batch, fields

    def _validate_graph(
        self,
        *,
        batch: int,
        fields: int,
        edge_index: torch.Tensor | None,
        edge_type_ids: torch.Tensor | None,
        edge_confidence: torch.Tensor | None,
        edge_valid_mask: torch.Tensor | None,
    ) -> bool:
        supplied = tuple(
            value is not None
            for value in (edge_index, edge_type_ids, edge_confidence, edge_valid_mask)
        )
        if not any(supplied):
            return False
        if not all(supplied):
            raise ValueError(
                "edge_index, edge_type_ids, edge_confidence, and edge_valid_mask must be supplied together"
            )
        assert edge_index is not None
        assert edge_type_ids is not None
        assert edge_confidence is not None
        assert edge_valid_mask is not None

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
        if not torch.isfinite(edge_confidence).all():
            raise ValueError("edge_confidence contains non-finite values")
        if edge_type_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_type_ids must be an integer tensor")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must be an integer tensor")
        if edge_valid_mask.any():
            active_index = edge_index[edge_valid_mask]
            if active_index.min().item() < 0 or active_index.max().item() >= fields:
                raise ValueError("active edge_index contains a field index outside the valid range")
            active_types = edge_type_ids[edge_valid_mask]
            if active_types.min().item() <= 0 or active_types.max().item() >= self.config.num_relation_types:
                raise ValueError("active edge_type_ids contains an unsupported relation id")
        return True

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
        edge_index: torch.Tensor | None = None,
        edge_type_ids: torch.Tensor | None = None,
        edge_confidence: torch.Tensor | None = None,
        edge_valid_mask: torch.Tensor | None = None,
        query_semantic: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, fields = self._validate_batch(
            semantic_values,
            field_type_ids,
            provenance_ids,
            relation_role_ids,
            temporal_scope_ids,
            confidence,
            missing_mask,
            valid_mask,
        )
        has_graph = self._validate_graph(
            batch=batch,
            fields=fields,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
        )
        if query_semantic is not None:
            if query_semantic.shape != (batch, self.config.semantic_size):
                raise ValueError(
                    "query_semantic must have shape "
                    f"{(batch, self.config.semantic_size)}, observed {tuple(query_semantic.shape)}"
                )
            if not torch.isfinite(query_semantic).all():
                raise ValueError("query_semantic contains non-finite values")

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

        status_bias = torch.zeros(batch, fields, device=x.device, dtype=x.dtype)
        relation_update_norm = torch.zeros_like(status_bias)
        if has_graph and self.graph_layers:
            assert edge_index is not None
            assert edge_type_ids is not None
            assert edge_confidence is not None
            assert edge_valid_mask is not None
            for graph_layer in self.graph_layers:
                x, layer_status, layer_norm = graph_layer(
                    x,
                    edge_index=edge_index,
                    edge_type_ids=edge_type_ids,
                    edge_confidence=edge_confidence,
                    edge_valid_mask=edge_valid_mask,
                    valid_mask=valid_mask,
                )
                status_bias = status_bias + layer_status
                relation_update_norm = relation_update_norm + layer_norm

        query = self.pool_query.unsqueeze(0).expand(batch, -1)
        if query_semantic is not None:
            query = query + self.query_projection(query_semantic)
        query = torch.nn.functional.normalize(query, dim=-1)

        pool_scores = torch.einsum("bfd,bd->bf", x, query) / math.sqrt(self.config.state_size)
        pool_scores = pool_scores + status_bias
        pool_scores = pool_scores.masked_fill(~valid_mask, torch.finfo(pool_scores.dtype).min)
        field_weights = torch.softmax(pool_scores, dim=-1)
        pooled_state = torch.einsum("bf,bfd->bd", field_weights, x)

        return {
            "field_states": self.field_output_projection(x),
            "pooled_state": self.pooled_output_projection(pooled_state),
            "field_weights": field_weights,
            "valid_mask": valid_mask,
            "relation_status_bias": status_bias,
            "relation_update_norm": relation_update_norm,
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
            "graph_layers": self.config.graph_layers,
            "relation_types": self.config.num_relation_types,
            "position_embeddings": 0,
            "private_identity_parameters": 0,
        }
