from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .evidence_graph import EvidenceGraphConfig, EvidenceRelationType
from .evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder


class SemanticRoleResidualEvidenceGraphEncoder(DualEndpointEvidenceGraphEncoder):
    """Dual-endpoint evidence graph with a separate relation-semantic role residual.

    The parent dual-endpoint read already knows how to distinguish source and
    target endpoints when query wording explicitly names an endpoint role.
    This residual is intentionally separate: it learns whether natural query
    semantics such as current, previous, cause, effect, supported claim, or
    derivation basis should prefer the source or target endpoint for a given
    relation type.

    The residual is zero at initialization, so loading an existing
    DualEndpointEvidenceGraphEncoder checkpoint preserves parent behavior
    exactly until the new semantic-role parameters are trained.
    """

    def __init__(self, config: EvidenceGraphConfig | None = None) -> None:
        super().__init__(config)
        d = self.config.graph_size

        self.semantic_role_query_adapter = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
        )
        self.semantic_role_delta_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

        # Exact parent parity at initialization. Earlier layers can learn once
        # the final layer moves away from zero.
        final = self.semantic_role_delta_mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _semantic_role_residual_bias(
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

        relation = self.pool_relation_embedding(edge_type_ids)
        semantic_query = self.semantic_role_query_adapter(query).unsqueeze(1)
        semantic_query = semantic_query.expand(-1, source.size(1), -1)
        features = torch.cat([semantic_query, relation], dim=-1)
        delta = self.semantic_role_delta_mlp(features).squeeze(-1)

        conflict = active & (
            edge_type_ids == int(EvidenceRelationType.CONFLICTS_WITH)
        )
        directed = active & ~conflict
        confidence = edge_confidence.to(x.dtype).clamp(0.0, 1.0).squeeze(-1)
        delta = delta * confidence * directed.to(x.dtype)

        # Positive delta means "prefer the relation source"; negative means
        # "prefer the relation target". This complementary update makes the
        # learned role explicit and prevents two unrelated endpoint heads from
        # independently drifting.
        flat_bias = torch.zeros(batch * fields, device=x.device, dtype=x.dtype)
        offsets = (torch.arange(batch, device=x.device) * fields).unsqueeze(1)
        source_flat = (source + offsets).reshape(-1)
        target_flat = (target + offsets).reshape(-1)
        flat_bias.index_add_(0, source_flat, delta.reshape(-1))
        flat_bias.index_add_(0, target_flat, -delta.reshape(-1))
        return flat_bias.reshape(batch, fields)

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
        parent_bias = super()._query_conditioned_relation_bias(
            x=x,
            query=query,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
        )
        semantic_bias = self._semantic_role_residual_bias(
            x=x,
            query=query,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
        )
        return parent_bias + semantic_bias

    def semantic_role_parameter_names(self) -> list[str]:
        prefixes = (
            "semantic_role_query_adapter.",
            "semantic_role_delta_mlp.",
        )
        return [
            name
            for name, _parameter in self.named_parameters()
            if name.startswith(prefixes)
        ]

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        semantic_role_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(
                ("semantic_role_query_adapter.", "semantic_role_delta_mlp.")
            )
        )
        report["total_parameters"] = sum(p.numel() for p in self.parameters())
        report["trainable_parameters"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        report["semantic_role_residual_parameters"] = semantic_role_parameters
        report["semantic_role_residual"] = (
            "signed_source_target_delta_from_query_semantics_and_relation_type"
        )
        report["parent_endpoint_read_can_remain_frozen"] = True
        report["hard_parameter_ceiling"] = None
        return report
