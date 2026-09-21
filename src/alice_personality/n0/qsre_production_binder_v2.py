from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.qsre_mechanics import masked_sparsemax
from alice_personality.n0.qsre_production_core import (
    CONTROL_RELATIONAL,
    QSREDynamicRelationSchema,
    QSREProductionConfig,
    QSREProductionOperatorState,
)


class QSREProductionBinderV2(nn.Module):
    """Open-schema support/focus binder.

    Query and field text share one semantic metric. Relation semantics are owned
    by the operator and enter binding only through the operator's dynamic
    relation mass plus exact runtime type constraints. The binder therefore
    cannot memorize P1 relation-state vectors as a second hidden ontology.
    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        self.semantic_norm = nn.LayerNorm(config.semantic_dim)
        self.semantic_projection = nn.Linear(
            config.semantic_dim,
            d,
            bias=False,
        )
        self.operator_projection = nn.Linear(d, d, bias=False)

        self.focus_score = nn.Sequential(
            nn.Linear(2 * d + 1, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(4 * d + 7, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, 1),
        )

        if config.semantic_dim == config.model_dim:
            nn.init.eye_(self.semantic_projection.weight)
        else:
            nn.init.orthogonal_(self.semantic_projection.weight)
        nn.init.xavier_uniform_(self.operator_projection.weight)
        for head in (self.focus_score, self.edge_score):
            final = head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _project(self, states: Tensor) -> Tensor:
        return self.semantic_projection(self.semantic_norm(states))

    @staticmethod
    def _masked_token_mean(
        token_state: Tensor,
        token_mask: Tensor,
        *,
        dim: int,
    ) -> Tensor:
        weight = token_mask.to(token_state.dtype).unsqueeze(-1)
        return (token_state * weight).sum(dim=dim) / weight.sum(
            dim=dim
        ).clamp_min(1.0)

    def _project_query_tokens(
        self,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
    ) -> Tensor:
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        batch, _, tokens, width = query_hidden_states.shape
        if width != self.config.semantic_dim:
            raise ValueError("query semantic width drift")
        if query_token_mask.shape != (batch, tokens):
            raise ValueError("query_token_mask shape drift")
        if query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool")
        return self._project(query_hidden_states[:, -1])

    def _project_field_tokens(
        self,
        field_token_states: Tensor,
        field_token_mask: Tensor,
    ) -> Tensor:
        if field_token_states.ndim != 4:
            raise ValueError("field_token_states must be [B,F,T,D]")
        batch, fields, tokens, width = field_token_states.shape
        if width != self.config.semantic_dim:
            raise ValueError("field token width drift")
        if field_token_mask.shape != (batch, fields, tokens):
            raise ValueError("field_token_mask shape drift")
        if field_token_mask.dtype != torch.bool:
            raise ValueError("field_token_mask must be bool")
        return self._project(field_token_states)

    def _late_endpoint_score(
        self,
        *,
        query_token: Tensor,
        query_mask: Tensor,
        endpoint_token: Tensor,
        endpoint_mask: Tensor,
    ) -> Tensor:
        # query_token [B,Tq,D], endpoint_token [B,E,Tf,D]
        q = F.normalize(query_token, dim=-1)
        e = F.normalize(endpoint_token, dim=-1)
        similarity = torch.einsum("bqd,besd->beqs", q, e)
        valid = (
            query_mask[:, None, :, None]
            & endpoint_mask[:, :, None, :]
        )
        similarity = similarity.masked_fill(~valid, -1.0e4)

        q_to_e = similarity.max(dim=-1).values
        q_weight = query_mask[:, None, :].to(q_to_e.dtype)
        q_score = (q_to_e * q_weight).sum(dim=-1) / q_weight.sum(
            dim=-1
        ).clamp_min(1.0)

        e_to_q = similarity.max(dim=-2).values
        e_weight = endpoint_mask.to(e_to_q.dtype)
        e_score = (e_to_q * e_weight).sum(dim=-1) / e_weight.sum(
            dim=-1
        ).clamp_min(1.0)

        endpoint_available = endpoint_mask.any(dim=-1)
        score = 0.5 * (q_score + e_score)
        return torch.where(
            endpoint_available,
            score,
            torch.full_like(score, -1.0),
        )

    def _type_compatibility(
        self,
        *,
        schema: QSREDynamicRelationSchema,
        edge_relation_index: Tensor,
        edge_index: Tensor,
        field_type_id: Tensor,
        edge_valid_mask: Tensor,
    ) -> Tensor:
        batch, edges = edge_relation_index.shape
        fields = field_type_id.size(1)
        type_vocab = schema.domain_type_mask.size(1)

        if bool(edge_valid_mask.any()):
            rel = edge_relation_index[edge_valid_mask]
            if int(rel.min()) < 0 or int(rel.max()) >= schema.token_states.size(0):
                raise ValueError("edge relation index outside runtime schema")
            valid_edges = edge_index[edge_valid_mask]
            if int(valid_edges.min()) < 0 or int(valid_edges.max()) >= fields:
                raise ValueError("edge endpoint outside field range")
            valid_types = field_type_id[field_type_id >= 0]
            if valid_types.numel() and int(valid_types.max()) >= type_vocab:
                raise ValueError("field type outside runtime type vocabulary")

        source_index = edge_index[..., 0].clamp(
            min=0,
            max=max(fields - 1, 0),
        )
        target_index = edge_index[..., 1].clamp(
            min=0,
            max=max(fields - 1, 0),
        )
        batch_index = torch.arange(
            batch,
            device=edge_index.device,
        )[:, None].expand(batch, edges)
        source_type = field_type_id[
            batch_index, source_index
        ].clamp(min=0, max=max(type_vocab - 1, 0))
        target_type = field_type_id[
            batch_index, target_index
        ].clamp(min=0, max=max(type_vocab - 1, 0))

        rel_index = edge_relation_index.clamp(
            min=0,
            max=schema.token_states.size(0) - 1,
        )
        direct = (
            schema.domain_type_mask[rel_index, source_type]
            & schema.range_type_mask[rel_index, target_type]
        )
        symmetric = schema.symmetric[rel_index]
        reverse = (
            schema.domain_type_mask[rel_index, target_type]
            & schema.range_type_mask[rel_index, source_type]
        )
        return edge_valid_mask & (direct | (symmetric & reverse))

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        field_token_states: Tensor,
        field_token_mask: Tensor,
        field_type_id: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        edge_valid_mask: Tensor,
        edge_reliability: Tensor,
        edge_recency: Tensor,
        schema: QSREDynamicRelationSchema,
        schema_relation_state: Tensor,
        operator: QSREProductionOperatorState,
    ) -> dict[str, Tensor]:
        # P1 schema relation states remain an executor interface but are not
        # relation-matching authority for the binder.
        del schema_relation_state

        relation_count = schema.token_states.size(0)
        operator.validate(
            relation_count=relation_count,
            model_dim=self.config.model_dim,
        )
        if edge_index.ndim != 3 or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [B,E,2]")
        batch, edges, _ = edge_index.shape
        if edge_relation_index.shape != (batch, edges):
            raise ValueError("edge_relation_index shape drift")
        if edge_valid_mask.shape != (batch, edges):
            raise ValueError("edge_valid_mask shape drift")
        if edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool")
        if edge_reliability.shape != (batch, edges):
            raise ValueError("edge_reliability shape drift")
        if edge_recency.shape != (batch, edges):
            raise ValueError("edge_recency shape drift")

        query_token = self._project_query_tokens(
            query_hidden_states,
            query_token_mask,
        )
        field_token = self._project_field_tokens(
            field_token_states,
            field_token_mask,
        )
        fields = field_token.size(1)
        if field_type_id.shape != (batch, fields):
            raise ValueError("field_type_id shape drift")

        query_summary = self._masked_token_mean(
            query_token,
            query_token_mask,
            dim=1,
        )
        field_summary = self._masked_token_mean(
            field_token,
            field_token_mask,
            dim=2,
        )

        # Focus is query-to-field semantics only. It has no oracle or relation-ID
        # input, so path origin prediction remains independently testable.
        field_late = self._late_endpoint_score(
            query_token=query_token,
            query_mask=query_token_mask,
            endpoint_token=field_token,
            endpoint_mask=field_token_mask,
        )
        q_field = query_summary[:, None, :].expand(
            batch,
            fields,
            -1,
        )
        focus_feature = torch.cat(
            [q_field, field_summary, field_late.unsqueeze(-1)],
            dim=-1,
        )
        focus_logits = (
            field_late
            + self.focus_score(focus_feature).squeeze(-1)
        )
        field_valid = field_token_mask.any(dim=-1)
        focus_field_weight = masked_sparsemax(
            focus_logits,
            field_valid,
            dim=-1,
        )
        focus_max = focus_field_weight.max(
            dim=-1,
            keepdim=True,
        ).values
        focus_field_weight = torch.where(
            focus_max > 0,
            focus_field_weight / focus_max.clamp_min(1.0e-12),
            torch.zeros_like(focus_field_weight),
        )

        source_index = edge_index[..., 0].clamp(
            min=0,
            max=max(fields - 1, 0),
        )
        target_index = edge_index[..., 1].clamp(
            min=0,
            max=max(fields - 1, 0),
        )
        batch_index = torch.arange(
            batch,
            device=edge_index.device,
        )[:, None].expand(batch, edges)

        source_state = field_summary[batch_index, source_index]
        target_state = field_summary[batch_index, target_index]
        source_token = field_token[batch_index, source_index]
        target_token = field_token[batch_index, target_index]
        source_token_mask = field_token_mask[
            batch_index, source_index
        ]
        target_token_mask = field_token_mask[
            batch_index, target_index
        ]

        source_late = self._late_endpoint_score(
            query_token=query_token,
            query_mask=query_token_mask,
            endpoint_token=source_token,
            endpoint_mask=source_token_mask,
        )
        target_late = self._late_endpoint_score(
            query_token=query_token,
            query_mask=query_token_mask,
            endpoint_token=target_token,
            endpoint_mask=target_token_mask,
        )
        endpoint_late = torch.maximum(
            source_late,
            target_late,
        )

        rel_index = edge_relation_index.clamp(
            min=0,
            max=relation_count - 1,
        )
        relation_mass = torch.einsum(
            "bsr,bs->br",
            operator.relation_distribution,
            operator.relation_step_mass,
        )
        relation_mass = relation_mass / operator.relation_step_mass.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        edge_relation_mass = relation_mass.gather(1, rel_index)

        q = query_summary[:, None, :].expand(
            batch,
            edges,
            -1,
        )
        op = self.operator_projection(
            operator.continuous_state
        )[:, None, :].expand(batch, edges, -1)
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
            [
                q,
                source_state,
                target_state,
                op,
                scalar,
            ],
            dim=-1,
        )
        learned_score = self.edge_score(feature).squeeze(-1)

        support_logits = (
            learned_score
            + 1.5 * endpoint_late
            + torch.log(edge_relation_mass.clamp_min(1.0e-6))
        )

        type_compatible = self._type_compatibility(
            schema=schema,
            edge_relation_index=edge_relation_index,
            edge_index=edge_index,
            field_type_id=field_type_id,
            edge_valid_mask=edge_valid_mask,
        )
        valid = edge_valid_mask & type_compatible
        sparse = masked_sparsemax(
            support_logits,
            valid,
            dim=-1,
        )
        sparse_max = sparse.max(
            dim=-1,
            keepdim=True,
        ).values
        sparse = torch.where(
            sparse_max > 0,
            sparse / sparse_max.clamp_min(1.0e-12),
            torch.zeros_like(sparse),
        )

        activation = torch.clamp(
            (operator.applicability - 0.25) / 0.50,
            min=0.0,
            max=1.0,
        )
        activation = (
            activation
            * operator.control_distribution[:, CONTROL_RELATIONAL]
        )
        known_mass = (
            1.0
            - operator.unknown_probability.sum(dim=1).clamp(max=1.0)
        ).clamp(min=0.0, max=1.0)
        edge_support_weight = (
            sparse
            * activation[:, None]
            * known_mass[:, None]
        )

        field_support_weight = torch.zeros(
            batch,
            fields,
            dtype=edge_support_weight.dtype,
            device=edge_support_weight.device,
        )
        field_support_weight.scatter_add_(
            1,
            source_index,
            edge_support_weight,
        )
        field_support_weight.scatter_add_(
            1,
            target_index,
            edge_support_weight,
        )
        field_support_weight = field_support_weight.clamp(max=1.0)

        return {
            "support_logits": support_logits,
            "edge_support_weight": edge_support_weight,
            "field_support_weight": field_support_weight,
            "type_compatible": type_compatible,
            "relation_mass": relation_mass,
            "source_late_interaction": source_late,
            "target_late_interaction": target_late,
            "endpoint_late_interaction": endpoint_late,
            "field_query_relevance": field_late,
            "focus_logits": focus_logits,
            "focus_field_weight": focus_field_weight,
        }

    def parameter_report(self) -> dict[str, int | bool | None]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_count_dependent_parameters": 0,
            "fixed_top_k": False,
            "exact_zero_sparse_support": True,
            "support_strength_not_simplex_cardinality_limited": True,
            "schema_type_compatibility": True,
            "relation_semantics_owned_by_operator": True,
            "p1_schema_relation_state_not_match_authority": True,
            "shared_query_field_metric": True,
            "symmetric_query_field_late_interaction": True,
            "runtime_support_score_is_supervised_score": True,
            "field_count_ceiling": None,
            "edge_count_ceiling": None,
            "support_count_ceiling": None,
            "focus_field_count_ceiling": None,
            "focus_is_predicted_without_oracle": True,
        }
