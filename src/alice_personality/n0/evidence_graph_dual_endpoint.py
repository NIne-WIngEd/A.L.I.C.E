from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .evidence_graph import (
    EvidenceGraphConfig,
    EvidenceGraphEncoder,
    EvidenceRelationType,
)


class DualEndpointEvidenceGraphEncoder(EvidenceGraphEncoder):
    """Evidence graph read that gives directed relations distinct source/target roles.

    The v0.2 graph read learned a query-conditioned bias for the relation target.
    That is sufficient for many evidence tasks but leaves source-role queries
    implicit. Root-cause, latest-state, supported-claim, correction-source, and
    other direction-sensitive reads need the source endpoint to carry its own
    learned query-conditioned status.

    Existing v0.2 graph checkpoints remain loadable with ``strict=False``. The
    only new parameters are ``source_relation_pool_mlp`` and its final layer is
    zero-initialized so a loaded parent initially preserves the old behavior.
    """

    def __init__(self, config: EvidenceGraphConfig | None = None) -> None:
        super().__init__(config)
        d = self.config.graph_size
        self.source_relation_pool_mlp = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        final = self.source_relation_pool_mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _query_conditioned_relation_bias(
        self,
        *,
        x: torch.Tensor,
        query: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_confidence: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, fields, _width = x.shape
        source = edge_index[..., 0].clamp(0, fields - 1)
        target = edge_index[..., 1].clamp(0, fields - 1)
        batch_index = torch.arange(batch, device=x.device).unsqueeze(1).expand_as(source)
        active = (
            edge_valid_mask
            & valid_mask[batch_index, source]
            & valid_mask[batch_index, target]
        )

        source_state = x[batch_index, source]
        target_state = x[batch_index, target]
        relation = self.pool_relation_embedding(edge_type_ids)
        query_state = query.unsqueeze(1).expand(-1, source.size(1), -1)
        confidence = edge_confidence.to(x.dtype).clamp(0.0, 1.0).squeeze(-1)

        directed_features = torch.cat(
            [query_state, relation, source_state, target_state], dim=-1
        )
        source_bias = self.source_relation_pool_mlp(directed_features).squeeze(-1)
        target_bias = self.directed_relation_pool_mlp(directed_features).squeeze(-1)

        symmetric_features = torch.cat(
            [
                query_state,
                relation,
                source_state + target_state,
                (source_state - target_state).abs(),
            ],
            dim=-1,
        )
        conflict_bias = self.conflict_pool_mlp(symmetric_features).squeeze(-1)
        conflict = active & (
            edge_type_ids == int(EvidenceRelationType.CONFLICTS_WITH)
        )
        directed = active & ~conflict

        scale = confidence * directed.to(x.dtype)
        source_bias = source_bias * scale
        target_bias = target_bias * scale
        conflict_bias = (
            conflict_bias
            * confidence
            * conflict.to(x.dtype)
        )

        flat_bias = torch.zeros(batch * fields, device=x.device, dtype=x.dtype)
        offsets = (torch.arange(batch, device=x.device) * fields).unsqueeze(1)
        source_flat = (source + offsets).reshape(-1)
        target_flat = (target + offsets).reshape(-1)
        flat_bias.index_add_(0, source_flat, source_bias.reshape(-1))
        flat_bias.index_add_(0, target_flat, target_bias.reshape(-1))

        flat_conflict = conflict_bias.reshape(-1)
        flat_bias.index_add_(0, source_flat, flat_conflict)
        flat_bias.index_add_(0, target_flat, flat_conflict)
        return flat_bias.reshape(batch, fields)

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report["total_parameters"] = sum(p.numel() for p in self.parameters())
        report["trainable_parameters"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        report["relation_pooling"] = (
            "query_conditioned_distinct_source_and_target_roles_conflict_symmetric"
        )
        report["hard_parameter_ceiling"] = None
        return report
