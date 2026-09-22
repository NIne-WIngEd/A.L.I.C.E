from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.chunked_late_interaction import (
    chunked_batched_bidirectional_late_max,
)

from alice_personality.n0.numeric_contracts import (
    exact_masked_softmax,
    require_finite,
    require_unit_interval,
)

from alice_personality.n0.full_envelope_structural_types import (
    CONTROL_RELATIONAL,
    MOD_RECENCY,
    MOD_RELIABILITY,
    FullEnvelopeOperatorState,
    masked_sparsemax,
    runtime_edge_type_compatibility,
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
        self.query_layer_gate = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1, bias=False),
        )
        self.interaction_layer_gate = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
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
        self.null_support_score = nn.Sequential(
            nn.Linear(2 * d + 3, d),
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
        layers = per_layer.size(-1)
        position = torch.linspace(
            -1.0,
            1.0,
            layers,
            device=per_layer.device,
            dtype=per_layer.dtype,
        ).view(1, 1, layers).expand_as(per_layer)
        layer_feature = torch.stack(
            [per_layer, q_score - f_score, position],
            dim=-1,
        )
        layer_logit = self.interaction_layer_gate(layer_feature).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        return torch.einsum("bfl,bfl->bf", layer_weight, per_layer)

    def _query_summary(self, query: Tensor, mask: Tensor) -> Tensor:
        projected = self.query_projection(query.float())
        token_weight = mask[:, None, :, None].to(projected.dtype)
        layer_summary = (
            (projected * token_weight).sum(dim=2)
            / token_weight.sum(dim=2).clamp_min(1.0)
        )
        layer_logit = self.query_layer_gate(
            torch.tanh(layer_summary)
        ).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        return torch.einsum(
            "bl,bld->bd",
            layer_weight,
            layer_summary,
        )

    @staticmethod
    def _type_compatibility(
        *,
        relation_domain_type_mask: Tensor,
        relation_range_type_mask: Tensor,
        relation_symmetric: Tensor,
        edge_relation_index: Tensor,
        edge_index: Tensor,
        field_type_index: Tensor,
        edge_valid_mask: Tensor,
    ) -> Tensor:
        return runtime_edge_type_compatibility(
            relation_domain_type_mask=relation_domain_type_mask,
            relation_range_type_mask=relation_range_type_mask,
            relation_symmetric=relation_symmetric,
            edge_relation_index=edge_relation_index,
            edge_index=edge_index,
            field_type_index=field_type_index,
            edge_valid_mask=edge_valid_mask,
        )

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
        relation_symmetric: Tensor,
        relation_schema_state: Tensor,
        operator: FullEnvelopeOperatorState,
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
        if field_token_mask.dtype != torch.bool:
            raise ValueError("field_token_mask must be bool")
        if field_state.shape != (batch, fields, self.config.model_dim):
            raise ValueError("field_state must match model_dim")
        if field_valid_mask.shape != (batch, fields) or field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool [B,F]")
        if bool((field_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one valid field")
        if relation_schema_state.ndim != 3 or relation_schema_state.size(0) != batch:
            raise ValueError("relation_schema_state must be [B,R,D]")
        relation_count = relation_schema_state.size(1)
        if relation_count <= 0:
            raise ValueError("runtime relation schema is empty")
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
        if (
            relation_domain_type_mask.dtype != torch.bool
            or relation_range_type_mask.dtype != torch.bool
        ):
            raise ValueError("relation domain/range type masks must be bool")
        if bool(edge_valid_mask.any()):
            valid_endpoint = edge_index[edge_valid_mask]
            if int(valid_endpoint.min()) < 0 or int(valid_endpoint.max()) >= fields:
                raise ValueError("valid edge endpoint outside runtime field set")
            valid_relation = edge_relation_index[edge_valid_mask]
            if int(valid_relation.min()) < 0 or int(valid_relation.max()) >= relation_count:
                raise ValueError("valid edge relation outside runtime relation schema")
            source_index_valid = edge_index[...,0].clamp(min=0,max=fields-1)
            target_index_valid = edge_index[...,1].clamp(min=0,max=fields-1)
            endpoint_field_valid = (
                field_valid_mask.gather(1,source_index_valid)
                & field_valid_mask.gather(1,target_index_valid)
            )
            if bool((edge_valid_mask & ~endpoint_field_valid).any()):
                raise ValueError("valid edge references padded invalid field")
        require_unit_interval("edge_reliability", edge_reliability)
        require_unit_interval("edge_recency", edge_recency)
        require_finite("field_state", field_state)
        require_finite("relation_schema_state", relation_schema_state)

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
        focus_field_weight = exact_masked_softmax(
            focus_logits,
            field_valid_mask,
            dim=-1,
        )

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
        reliability_weight = operator.modifier_weight[:, MOD_RELIABILITY][:, None]
        recency_weight = operator.modifier_weight[:, MOD_RECENCY][:, None]
        effective_reliability = (
            0.5 + reliability_weight * (edge_reliability - 0.5)
        )
        effective_recency = (
            0.5 + recency_weight * (edge_recency - 0.5)
        )
        scalar = torch.stack(
            [
                edge_relation_mass,
                effective_reliability,
                effective_recency,
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
            relation_symmetric=relation_symmetric,
            edge_relation_index=edge_relation_index,
            edge_index=edge_index,
            field_type_index=field_type_index,
            edge_valid_mask=edge_valid_mask,
        )
        active_relation_edge = edge_relation_mass > 0.0
        support_compatible = type_compatible & active_relation_edge

        null_feature = torch.cat(
            [
                query_summary,
                self.operator_projection(operator.continuous_state),
                operator.applicability[:, None],
                operator.uncertainty[:, None],
                operator.control_distribution[:, CONTROL_RELATIONAL][:, None],
            ],
            dim=-1,
        )
        null_support_logit = self.null_support_score(null_feature).squeeze(-1)
        augmented_logits = torch.cat(
            [support_logits, null_support_logit[:, None]],
            dim=-1,
        )
        augmented_mask = torch.cat(
            [
                support_compatible,
                torch.ones(
                    batch,
                    1,
                    device=type_compatible.device,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )
        augmented_sparse = masked_sparsemax(
            augmented_logits,
            augmented_mask,
            dim=-1,
        )
        sparse = augmented_sparse[:, :edges]
        null_support_mass = augmented_sparse[:, edges]
        support_available = (1.0 - null_support_mass).clamp(0.0, 1.0)

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
            "null_support_logit": null_support_logit,
            "null_support_mass": null_support_mass,
            "support_available": support_available,
            "edge_support_weight": edge_support_weight,
            "field_support_weight": field_support_weight,
            "type_compatible": type_compatible,
            "support_compatible": support_compatible,
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
            "modifier_off_neutralizes_explicit_reliability_recency_features": True,
            "explicit_null_support_option": True,
            "zero_support_possible_with_compatible_edges": True,
            "multilayer_query_field_interaction": True,
            "content_conditioned_query_layer_read": True,
            "field_conditioned_interaction_layer_read": True,
            "global_static_layer_mixture": False,
            "final_layer_only_query": False,
            "runtime_relation_schema": True,
            "inactive_runtime_relation_edges_exactly_excluded": True,
            "runtime_type_schema": True,
            "padded_field_type_minus_one_supported": True,
            "runtime_relation_symmetry": True,
            "token_interaction_chunk_is_operating_point": True,
            "query_token_count_ceiling": None,
            "field_token_count_ceiling": None,
            "field_count_ceiling": None,
            "edge_count_ceiling": None,
            "support_count_ceiling": None,
            "relation_count_ceiling": None,
            "type_vocab_ceiling": None,
        }
