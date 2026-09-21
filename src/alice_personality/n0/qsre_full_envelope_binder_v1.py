from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.chunked_late_interaction import (
    chunked_batched_bidirectional_late_max,
)

from alice_personality.n0.qsre_production_binder_v2 import masked_sparsemax
from alice_personality.n0.qsre_production_core import (
    CONTROL_RELATIONAL,
    QSREProductionOperatorState,
)


@dataclass(frozen=True)
class FullEnvelopeBinderConfig:
    semantic_dim: int = 640
    model_dim: int = 640
    num_hidden_states: int = 17
    edge_metadata_dim: int = 4
    interaction_chunk_tokens: int = 128

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("edge_metadata_dim", self.edge_metadata_dim),
            ("interaction_chunk_tokens", self.interaction_chunk_tokens),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")


class FullEnvelopeQSREBinderV1(nn.Module):
    """First exact sparse boundary for the full-envelope N0 successor.

    Unlike Binder v2, token relevance reads every semantic layer rather than
    only the final layer. Relation and type semantics arrive from runtime
    schema state. Exact zero/one/many structural support is still delayed until
    this module.
    """

    def __init__(self, config: FullEnvelopeBinderConfig | None = None) -> None:
        super().__init__()
        self.config = config or FullEnvelopeBinderConfig()
        self.config.validate()
        d = self.config.model_dim

        self.query_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.field_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.field_state_projection = nn.Linear(d, d, bias=False)
        self.relation_projection = nn.Linear(d, d, bias=False)
        self.operator_projection = nn.Linear(d, d, bias=False)
        self.layer_logits = nn.Parameter(torch.zeros(self.config.num_hidden_states))
        self.edge_score = nn.Sequential(
            nn.Linear(5 * d + 7, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, 1),
        )
        self.focus_score = nn.Sequential(
            nn.Linear(3 * d + 1, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

    def _token_late(
        self,
        *,
        query: Tensor,
        query_mask: Tensor,
        field: Tensor,
        field_mask: Tensor,
    ) -> Tensor:
        # query [B,L,T,D], field [B,F,L,S,D] -> [B,F]
        q = F.normalize(self.query_projection(query.float()), dim=-1)
        f = F.normalize(self.field_projection(field.float()), dim=-1)
        q_to_f, f_to_q = chunked_batched_bidirectional_late_max(
            query=q,
            query_mask=query_mask,
            items=f,
            item_mask=field_mask,
            chunk_tokens=self.config.interaction_chunk_tokens,
        )

        qmask = query_mask[:, None, None, :].expand_as(q_to_f)
        fmask = field_mask[:, :, None, :].expand_as(f_to_q)
        q_score = (
            q_to_f.masked_fill(~qmask, 0.0).sum(dim=-1)
            / qmask.sum(dim=-1).clamp_min(1).to(q_to_f.dtype)
        )
        f_score = (
            f_to_q.masked_fill(~fmask, 0.0).sum(dim=-1)
            / fmask.sum(dim=-1).clamp_min(1).to(f_to_q.dtype)
        )
        per_layer = 0.5 * (q_score + f_score)
        layer_weight = torch.softmax(self.layer_logits, dim=0)
        return torch.einsum("l,bfl->bf", layer_weight, per_layer)

    def _query_summary(self, query: Tensor, mask: Tensor) -> Tensor:
        projected = self.query_projection(query.float())
        token_weight = mask[:, None, :, None].to(projected.dtype)
        layer_summary = (
            (projected * token_weight).sum(dim=2)
            / token_weight.sum(dim=2).clamp_min(1.0)
        )
        return torch.einsum(
            "l,bld->bd",
            torch.softmax(self.layer_logits, dim=0),
            layer_summary,
        )

    @staticmethod
    def _type_compatibility(
        *,
        relation_domain_type_mask: Tensor,
        relation_range_type_mask: Tensor,
        edge_relation_index: Tensor,
        edge_index: Tensor,
        field_type_index: Tensor,
        edge_valid_mask: Tensor,
    ) -> Tensor:
        if relation_domain_type_mask.ndim != 3:
            raise ValueError("relation domain mask must be [B,R,K]")
        if relation_range_type_mask.shape != relation_domain_type_mask.shape:
            raise ValueError("domain/range type-mask drift")
        batch, relations, type_count = relation_domain_type_mask.shape
        if field_type_index.ndim != 2 or field_type_index.size(0) != batch:
            raise ValueError("field_type_index must be [B,F]")
        if bool((field_type_index < 0).any()) or bool((field_type_index >= type_count).any()):
            raise ValueError("field type index outside runtime type schema")
        source_index = edge_index[..., 0].clamp(min=0, max=field_type_index.size(1) - 1)
        target_index = edge_index[..., 1].clamp(min=0, max=field_type_index.size(1) - 1)
        relation_index = edge_relation_index.clamp(min=0, max=relations - 1)
        b = torch.arange(batch, device=edge_index.device)[:, None].expand_as(edge_relation_index)
        source_type = field_type_index.gather(1, source_index)
        target_type = field_type_index.gather(1, target_index)
        domain = relation_domain_type_mask[b, relation_index, source_type]
        range_ok = relation_range_type_mask[b, relation_index, target_type]
        return edge_valid_mask & domain & range_ok

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        field_hidden_states: Tensor,
        field_token_mask: Tensor,
        field_state: Tensor,
        field_valid_mask: Tensor,
        field_type_index: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        edge_valid_mask: Tensor,
        edge_reliability: Tensor,
        edge_recency: Tensor,
        relation_domain_type_mask: Tensor,
        relation_range_type_mask: Tensor,
        relation_schema_state: Tensor,
        operator: QSREProductionOperatorState,
    ) -> dict[str, Tensor]:
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        batch, layers, query_tokens, width = query_hidden_states.shape
        if layers != self.config.num_hidden_states or width != self.config.semantic_dim:
            raise ValueError("query semantic geometry drift")
        if query_token_mask.shape != (batch, query_tokens) or query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool [B,T]")
        if field_hidden_states.ndim != 5:
            raise ValueError("field_hidden_states must be [B,F,L,T,D]")
        if field_hidden_states.size(0) != batch or field_hidden_states.size(2) != layers:
            raise ValueError("field semantic depth drift")
        fields = field_hidden_states.size(1)
        field_tokens = field_hidden_states.size(3)
        if field_hidden_states.size(-1) != self.config.semantic_dim:
            raise ValueError("field semantic width drift")
        if field_token_mask.shape != (batch, fields, field_tokens):
            raise ValueError("field_token_mask shape drift")
        if field_state.shape != (batch, fields, self.config.model_dim):
            raise ValueError("field_state must match model_dim")
        if field_valid_mask.shape != (batch, fields) or field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool [B,F]")
        if relation_schema_state.ndim != 3 or relation_schema_state.size(0) != batch:
            raise ValueError("relation_schema_state must be [B,R,D]")
        relation_count = relation_schema_state.size(1)
        if relation_schema_state.size(-1) != self.config.model_dim:
            raise ValueError("relation schema state width drift")
        operator.validate(relation_count=relation_count, model_dim=self.config.model_dim)

        if edge_index.ndim != 3 or edge_index.size(0) != batch or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [B,E,2]")
        edges = edge_index.size(1)
        for name, value in (
            ("edge_relation_index", edge_relation_index),
            ("edge_valid_mask", edge_valid_mask),
            ("edge_reliability", edge_reliability),
            ("edge_recency", edge_recency),
        ):
            if value.shape != (batch, edges):
                raise ValueError(f"{name} shape drift")
        if edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool")

        field_late = self._token_late(
            query=query_hidden_states,
            query_mask=query_token_mask,
            field=field_hidden_states,
            field_mask=field_token_mask,
        )
        query_summary = self._query_summary(query_hidden_states, query_token_mask)
        field_projected = self.field_state_projection(field_state.float())
        q_field = query_summary[:, None, :].expand(batch, fields, -1)
        op_field = self.operator_projection(operator.continuous_state)[:, None, :].expand(batch, fields, -1)
        focus_feature = torch.cat(
            [q_field, field_projected, op_field, field_late.unsqueeze(-1)],
            dim=-1,
        )
        focus_logits = self.focus_score(focus_feature).squeeze(-1)
        focus_logits = focus_logits.masked_fill(~field_valid_mask, -1.0e4)
        focus_field_weight = torch.softmax(focus_logits, dim=-1)
        focus_field_weight = focus_field_weight * field_valid_mask.to(focus_field_weight.dtype)
        focus_field_weight = focus_field_weight / focus_field_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)

        source_index = edge_index[..., 0].clamp(min=0, max=fields - 1)
        target_index = edge_index[..., 1].clamp(min=0, max=fields - 1)
        b = torch.arange(batch, device=edge_index.device)[:, None].expand(batch, edges)
        source_state = field_projected[b, source_index]
        target_state = field_projected[b, target_index]
        source_late = field_late.gather(1, source_index)
        target_late = field_late.gather(1, target_index)
        endpoint_late = torch.maximum(source_late, target_late)

        relation_index = edge_relation_index.clamp(min=0, max=max(relation_count - 1, 0))
        relation_state = self.relation_projection(relation_schema_state[b, relation_index])

        relation_mass = torch.einsum(
            "bsr,bs->br",
            operator.relation_distribution,
            operator.relation_step_mass,
        )
        relation_mass = relation_mass / operator.relation_step_mass.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        edge_relation_mass = relation_mass.gather(1, relation_index)

        q = query_summary[:, None, :].expand(batch, edges, -1)
        op = self.operator_projection(operator.continuous_state)[:, None, :].expand(batch, edges, -1)
        scalar = torch.stack(
            [
                edge_relation_mass,
                edge_reliability,
                edge_recency,
                operator.applicability[:, None].expand(batch, edges),
                source_late,
                target_late,
                endpoint_late,
            ],
            dim=-1,
        )
        feature = torch.cat(
            [q, source_state, target_state, relation_state, op, scalar],
            dim=-1,
        )
        learned_score = self.edge_score(feature).squeeze(-1)
        semantic_bonus = F.cosine_similarity(q, relation_state, dim=-1)
        support_logits = (
            learned_score
            + semantic_bonus
            + 1.5 * endpoint_late
            + torch.log(edge_relation_mass.clamp_min(1.0e-6))
        )

        type_compatible = self._type_compatibility(
            relation_domain_type_mask=relation_domain_type_mask,
            relation_range_type_mask=relation_range_type_mask,
            edge_relation_index=edge_relation_index,
            edge_index=edge_index,
            field_type_index=field_type_index,
            edge_valid_mask=edge_valid_mask,
        )
        sparse = masked_sparsemax(support_logits, type_compatible, dim=-1)
        sparse_max = sparse.max(dim=-1, keepdim=True).values
        sparse = torch.where(
            sparse_max > 0,
            sparse / sparse_max.clamp_min(1.0e-12),
            torch.zeros_like(sparse),
        )

        activation = operator.applicability.clamp(0.0, 1.0)
        activation = activation * operator.control_distribution[:, CONTROL_RELATIONAL]
        known_mass = (
            1.0 - operator.unknown_probability.sum(dim=1).clamp(max=1.0)
        ).clamp(min=0.0, max=1.0)
        edge_support_weight = sparse * activation[:, None] * known_mass[:, None]

        field_support_weight = torch.zeros(
            batch, fields, dtype=edge_support_weight.dtype, device=edge_support_weight.device
        )
        field_support_weight.scatter_add_(1, source_index, edge_support_weight)
        field_support_weight.scatter_add_(1, target_index, edge_support_weight)
        field_support_weight = field_support_weight.clamp(max=1.0)

        return {
            "support_logits": support_logits,
            "edge_support_weight": edge_support_weight,
            "field_support_weight": field_support_weight,
            "type_compatible": type_compatible,
            "relation_mass": relation_mass,
            "field_query_relevance": field_late,
            "focus_logits": focus_logits,
            "focus_field_weight": focus_field_weight,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_count_dependent_parameters": 0,
            "factor_count_dependent_parameters": 0,
            "type_vocab_dependent_parameters": 0,
            "fixed_top_k": False,
            "exact_zero_sparse_support": True,
            "multilayer_query_field_interaction": True,
            "final_layer_only_query": False,
            "runtime_relation_schema": True,
            "runtime_type_schema": True,
            "token_interaction_chunk_is_operating_point": True,
            "query_token_count_ceiling": None,
            "field_token_count_ceiling": None,
            "field_count_ceiling": None,
            "edge_count_ceiling": None,
            "support_count_ceiling": None,
            "relation_count_ceiling": None,
            "type_vocab_ceiling": None,
        }
