from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.qsre_mechanics import masked_sparsemax


ROLE_SOURCE = 0
ROLE_TARGET = 1
ROLE_SYMMETRIC = 2
ROLE_NONE = 3

TRAVERSAL_LOCAL = 0
TRAVERSAL_PATH = 1
TRAVERSAL_AGGREGATE = 2

MOD_RELIABILITY = 0
MOD_RECENCY = 1
MOD_TEMPORAL_CONSTRAINT = 2
MOD_PROVENANCE_CONSTRAINT = 3

CONTROL_FALLBACK = 0
CONTROL_RELATIONAL = 1
CONTROL_DEFER = 2


@dataclass(frozen=True)
class QSREProductionConfig:
    semantic_dim: int = 640
    model_dim: int = 512
    num_hidden_states: int = 17
    num_attention_heads: int = 8
    operator_refinement_layers: int = 2
    field_state_dim: int = 640
    field_metadata_dim: int = 3
    role_count: int = 4
    traversal_count: int = 3
    modifier_count: int = 4
    control_count: int = 3
    dropout: float = 0.05

    def validate(self) -> None:
        values = {
            "semantic_dim": self.semantic_dim,
            "model_dim": self.model_dim,
            "num_hidden_states": self.num_hidden_states,
            "num_attention_heads": self.num_attention_heads,
            "operator_refinement_layers": self.operator_refinement_layers,
            "field_state_dim": self.field_state_dim,
            "field_metadata_dim": self.field_metadata_dim,
            "role_count": self.role_count,
            "traversal_count": self.traversal_count,
            "modifier_count": self.modifier_count,
            "control_count": self.control_count,
        }
        for name, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.role_count != 4:
            raise ValueError("production structural role count must be 4")
        if self.traversal_count != 3:
            raise ValueError("production traversal factor count must be 3")
        if self.modifier_count < 4:
            raise ValueError("production modifier bank requires at least 4 factors")
        if self.control_count != 3:
            raise ValueError("production control count must be 3")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


@dataclass(frozen=True)
class QSREDynamicRelationSchema:
    """Runtime relation schema.

    R (relation count), S (schema token count), and K (type vocabulary size)
    are runtime axes. None of them creates trainable relation-class parameters.
    """

    token_states: Tensor
    token_mask: Tensor
    domain_type_mask: Tensor
    range_type_mask: Tensor
    symmetric: Tensor

    def validate(
        self,
        *,
        num_hidden_states: int,
        semantic_dim: int,
    ) -> dict[str, int]:
        if self.token_states.ndim != 4:
            raise ValueError("schema token_states must be [R,L,S,D]")
        relations, layers, schema_tokens, width = self.token_states.shape
        if relations <= 0:
            raise ValueError("schema requires at least one relation")
        if layers != num_hidden_states:
            raise ValueError("schema hidden-state depth drift")
        if width != semantic_dim:
            raise ValueError("schema semantic width drift")
        if self.token_mask.shape != (relations, schema_tokens):
            raise ValueError("schema token_mask must be [R,S]")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("schema token_mask must be bool")
        if bool((self.token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every relation schema entry requires content tokens")
        if self.domain_type_mask.ndim != 2:
            raise ValueError("domain_type_mask must be [R,K]")
        if self.range_type_mask.shape != self.domain_type_mask.shape:
            raise ValueError("domain/range type mask shape drift")
        if self.domain_type_mask.size(0) != relations:
            raise ValueError("schema type relation axis drift")
        if self.domain_type_mask.dtype != torch.bool:
            raise ValueError("domain_type_mask must be bool")
        if self.range_type_mask.dtype != torch.bool:
            raise ValueError("range_type_mask must be bool")
        if self.symmetric.shape != (relations,) or self.symmetric.dtype != torch.bool:
            raise ValueError("symmetric must be bool [R]")
        if self.domain_type_mask.size(1) <= 0:
            raise ValueError("schema type vocabulary must be non-empty")
        return {
            "relations": relations,
            "layers": layers,
            "schema_tokens": schema_tokens,
            "type_vocab": self.domain_type_mask.size(1),
        }


@dataclass(frozen=True)
class QSREProductionOperatorState:
    relation_distribution: Tensor
    relation_step_mass: Tensor
    stop_probability: Tensor
    unknown_probability: Tensor
    role_distribution: Tensor
    traversal_distribution: Tensor
    modifier_weight: Tensor
    applicability: Tensor
    control_distribution: Tensor
    continuous_state: Tensor
    uncertainty: Tensor

    def validate(self, *, relation_count: int, model_dim: int) -> dict[str, int]:
        if self.relation_distribution.ndim != 3:
            raise ValueError("relation_distribution must be [B,S,R]")
        batch, steps, relations = self.relation_distribution.shape
        if relations != relation_count:
            raise ValueError("operator/schema relation cardinality drift")
        if self.relation_step_mass.shape != (batch, steps):
            raise ValueError("relation_step_mass must be [B,S]")
        if self.stop_probability.shape != (batch, steps):
            raise ValueError("stop_probability must be [B,S]")
        if self.unknown_probability.shape != (batch, steps):
            raise ValueError("unknown_probability must be [B,S]")
        if self.role_distribution.shape != (batch, 4):
            raise ValueError("role_distribution must be [B,4]")
        if self.traversal_distribution.shape != (batch, 3):
            raise ValueError("traversal_distribution must be [B,3]")
        if self.modifier_weight.ndim != 2 or self.modifier_weight.size(0) != batch:
            raise ValueError("modifier_weight must be [B,M]")
        if self.applicability.shape != (batch,):
            raise ValueError("applicability must be [B]")
        if self.control_distribution.shape != (batch, 3):
            raise ValueError("control_distribution must be [B,3]")
        if self.continuous_state.shape != (batch, model_dim):
            raise ValueError("continuous_state shape drift")
        if self.uncertainty.shape != (batch,):
            raise ValueError("uncertainty must be [B]")
        for name, tensor in (
            ("relation_distribution", self.relation_distribution),
            ("relation_step_mass", self.relation_step_mass),
            ("stop_probability", self.stop_probability),
            ("unknown_probability", self.unknown_probability),
            ("role_distribution", self.role_distribution),
            ("traversal_distribution", self.traversal_distribution),
            ("modifier_weight", self.modifier_weight),
            ("applicability", self.applicability),
            ("control_distribution", self.control_distribution),
            ("continuous_state", self.continuous_state),
            ("uncertainty", self.uncertainty),
        ):
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains nonfinite values")
        return {"batch": batch, "steps": steps, "relations": relations}


def _masked_mean(value: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    numerator = (value * weight).sum(dim=dim)
    denominator = weight.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    safe = logits.masked_fill(~mask, -1.0e30)
    probability = torch.softmax(safe, dim=dim) * mask.to(logits.dtype)
    denominator = probability.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        probability / denominator.clamp_min(1.0e-12),
        torch.zeros_like(probability),
    )


class QSREProductionOperatorInducer(nn.Module):
    """Runtime-dynamic, schema-grounded operator inducer.

    Relation semantics come entirely from runtime schema states. The module owns
    no parameter whose shape depends on relation count or reasoning-step count.
    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        self.query_norm = nn.LayerNorm(config.semantic_dim)
        self.query_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.schema_norm = nn.LayerNorm(config.semantic_dim)
        self.schema_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.layer_embedding = nn.Embedding(config.num_hidden_states, d)

        self.start_state = nn.Parameter(torch.empty(d))
        self.query_cross_attention = nn.MultiheadAttention(
            d,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.query_cross_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_attention_heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.initial_refiner = nn.TransformerEncoder(
            layer,
            num_layers=config.operator_refinement_layers,
        )

        self.step_query = nn.Linear(d, d, bias=False)
        self.step_transition = nn.GRUCell(2 * d, d)
        self.stop_head = nn.Linear(d, 1)
        self.unknown_head = nn.Linear(d, 1)
        self.relation_logit_scale = nn.Parameter(torch.tensor(2.0))

        self.role_head = nn.Linear(d, config.role_count)
        self.traversal_head = nn.Linear(d, config.traversal_count)
        self.modifier_head = nn.Linear(d, config.modifier_count)
        self.applicability_head = nn.Linear(d, 1)
        self.control_head = nn.Linear(d, config.control_count)
        self.continuous_projection = nn.Linear(d, d)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.start_state, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.step_query.weight)
        for head in (
            self.stop_head,
            self.unknown_head,
            self.role_head,
            self.traversal_head,
            self.modifier_head,
            self.applicability_head,
            self.control_head,
            self.continuous_projection,
        ):
            if hasattr(head, "weight"):
                nn.init.xavier_uniform_(head.weight)
            if getattr(head, "bias", None) is not None:
                nn.init.zeros_(head.bias)

    def project_schema(
        self,
        schema: QSREDynamicRelationSchema,
    ) -> tuple[Tensor, Tensor]:
        schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        token = self.schema_projection(self.schema_norm(schema.token_states))
        layer_ids = torch.arange(
            self.config.num_hidden_states,
            device=token.device,
        )
        token = token + self.layer_embedding(layer_ids).view(
            1,
            self.config.num_hidden_states,
            1,
            self.config.model_dim,
        )
        mask = schema.token_mask[:, None, :, None].to(token.dtype)
        per_layer = (token * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        summary = per_layer.mean(dim=1)
        return token, summary

    def project_query(self, query_hidden_states: Tensor) -> Tensor:
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        batch, layers, tokens, width = query_hidden_states.shape
        if layers != self.config.num_hidden_states:
            raise ValueError("query hidden-state depth drift")
        if width != self.config.semantic_dim:
            raise ValueError("query semantic width drift")
        projected = self.query_projection(self.query_norm(query_hidden_states))
        layer_ids = torch.arange(layers, device=projected.device)
        projected = projected + self.layer_embedding(layer_ids).view(
            1, layers, 1, self.config.model_dim
        )
        return projected

    def _initial_state(
        self,
        query_memory: Tensor,
        flat_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = query_memory.size(0)
        start = self.start_state.view(1, 1, -1).expand(batch, 1, -1)
        attended, attention = self.query_cross_attention(
            start,
            query_memory,
            query_memory,
            key_padding_mask=~flat_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        state = self.query_cross_norm(start + attended)
        state = self.initial_refiner(state).squeeze(1)
        return state, attention.squeeze(1)

    def _relation_logits(
        self,
        *,
        state: Tensor,
        query_projected: Tensor,
        query_attention: Tensor,
        schema_projected: Tensor,
        schema_summary: Tensor,
        schema_token_mask: Tensor,
    ) -> Tensor:
        batch, layers, query_tokens, _ = query_projected.shape
        relations, schema_layers, schema_tokens, _ = schema_projected.shape
        if schema_layers != layers:
            raise ValueError("query/schema layer mismatch")

        query_normalized = F.normalize(query_projected, dim=-1)
        schema_normalized = F.normalize(schema_projected, dim=-1)

        similarity = torch.einsum(
            "bltd,rlsd->brlts",
            query_normalized,
            schema_normalized,
        )
        schema_valid = schema_token_mask[None, :, None, None, :].expand(
            batch,
            relations,
            layers,
            query_tokens,
            schema_tokens,
        )
        similarity = similarity.masked_fill(~schema_valid, -1.0e4)
        token_match = similarity.max(dim=-1).values

        token_weight = query_attention.reshape(batch, layers, query_tokens)
        grounded = torch.einsum("blt,brlt->br", token_weight, token_match)

        state_score = torch.einsum(
            "bd,rd->br",
            F.normalize(self.step_query(state), dim=-1),
            F.normalize(schema_summary, dim=-1),
        )
        scale = self.relation_logit_scale.exp().clamp(max=100.0)
        return scale * (grounded + state_score)

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        schema: QSREDynamicRelationSchema,
        max_steps: int,
    ) -> dict[str, Tensor | QSREProductionOperatorState]:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if query_token_mask.ndim != 2 or query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool [B,T]")

        query_projected = self.project_query(query_hidden_states)
        batch, layers, tokens, _ = query_projected.shape
        if query_token_mask.shape != (batch, tokens):
            raise ValueError("query_token_mask shape drift")
        if bool((query_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every query requires at least one valid token")

        schema_projected, schema_summary = self.project_schema(schema)
        relations = schema_projected.size(0)

        memory = query_projected.reshape(batch, layers * tokens, -1)
        flat_valid = (
            query_token_mask[:, None, :]
            .expand(batch, layers, tokens)
            .reshape(batch, layers * tokens)
        )
        state, initial_attention = self._initial_state(memory, flat_valid)

        relation_steps: list[Tensor] = []
        relation_mass_steps: list[Tensor] = []
        stop_steps: list[Tensor] = []
        unknown_steps: list[Tensor] = []
        state_steps: list[Tensor] = []
        attention_steps: list[Tensor] = []

        survival = torch.ones(batch, device=state.device, dtype=state.dtype)

        for _ in range(max_steps):
            attended, attention = self.query_cross_attention(
                state.unsqueeze(1),
                memory,
                memory,
                key_padding_mask=~flat_valid,
                need_weights=True,
                average_attn_weights=True,
            )
            step_state = self.query_cross_norm(state + attended.squeeze(1))
            relation_logits = self._relation_logits(
                state=step_state,
                query_projected=query_projected,
                query_attention=attention.squeeze(1),
                schema_projected=schema_projected,
                schema_summary=schema_summary,
                schema_token_mask=schema.token_mask,
            )
            stop_logit = self.stop_head(step_state)
            unknown_logit = self.unknown_head(step_state)
            all_logits = torch.cat([relation_logits, stop_logit, unknown_logit], dim=-1)
            all_mask = torch.ones_like(all_logits, dtype=torch.bool)
            sparse = masked_sparsemax(all_logits, all_mask, dim=-1)

            relation_raw = sparse[:, :relations]
            stop_probability = sparse[:, relations]
            unknown_probability = sparse[:, relations + 1]
            relation_mass = relation_raw.sum(dim=-1)
            relation_distribution = torch.where(
                relation_mass[:, None] > 0,
                relation_raw / relation_mass[:, None].clamp_min(1.0e-12),
                torch.zeros_like(relation_raw),
            )

            effective_mass = survival * relation_mass
            relation_steps.append(relation_distribution)
            relation_mass_steps.append(effective_mass)
            stop_steps.append(stop_probability)
            unknown_steps.append(unknown_probability)
            state_steps.append(step_state)
            attention_steps.append(
                attention.squeeze(1).reshape(batch, layers, tokens)
            )

            expected_relation = torch.einsum(
                "br,rd->bd",
                relation_distribution,
                schema_summary,
            )
            query_summary = attended.squeeze(1)
            state = self.step_transition(
                torch.cat([expected_relation, query_summary], dim=-1),
                step_state,
            )
            survival = survival * (1.0 - stop_probability)

        relation_distribution = torch.stack(relation_steps, dim=1)
        relation_step_mass = torch.stack(relation_mass_steps, dim=1)
        stop_probability = torch.stack(stop_steps, dim=1)
        unknown_probability = torch.stack(unknown_steps, dim=1)
        step_state_history = torch.stack(state_steps, dim=1)
        step_attention = torch.stack(attention_steps, dim=1)

        role_distribution = torch.softmax(self.role_head(state), dim=-1)
        traversal_distribution = torch.softmax(self.traversal_head(state), dim=-1)
        modifier_weight = torch.sigmoid(self.modifier_head(state))
        applicability = torch.sigmoid(self.applicability_head(state)).squeeze(-1)
        control_distribution = torch.softmax(self.control_head(state), dim=-1)
        continuous_state = self.continuous_projection(state)

        relation_entropy = -(
            relation_distribution.clamp_min(1.0e-12)
            * relation_distribution.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        relation_entropy = (
            relation_entropy * relation_step_mass
        ).sum(dim=-1) / relation_step_mass.sum(dim=-1).clamp_min(1.0e-6)
        role_entropy = -(
            role_distribution.clamp_min(1.0e-12)
            * role_distribution.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        traversal_entropy = -(
            traversal_distribution.clamp_min(1.0e-12)
            * traversal_distribution.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        uncertainty = torch.sigmoid(
            relation_entropy + 0.5 * role_entropy + 0.5 * traversal_entropy
            + unknown_probability.max(dim=1).values
        )

        operator = QSREProductionOperatorState(
            relation_distribution=relation_distribution,
            relation_step_mass=relation_step_mass,
            stop_probability=stop_probability,
            unknown_probability=unknown_probability,
            role_distribution=role_distribution,
            traversal_distribution=traversal_distribution,
            modifier_weight=modifier_weight,
            applicability=applicability,
            control_distribution=control_distribution,
            continuous_state=continuous_state,
            uncertainty=uncertainty,
        )
        operator.validate(relation_count=relations, model_dim=self.config.model_dim)

        return {
            "operator": operator,
            "schema_relation_state": schema_summary,
            "schema_token_state": schema_projected,
            "initial_query_attention": initial_attention.reshape(batch, layers, tokens),
            "step_query_attention": step_attention,
            "step_state_history": step_state_history,
        }

    def parameter_report(self) -> dict[str, int | bool]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "relation_count_dependent_parameters": 0,
            "hop_count_dependent_parameters": 0,
            "runtime_dynamic_relation_schema": True,
            "shared_iterative_relation_cell": True,
            "early_relation_argmax": False,
            "unknown_separate_from_stop": True,
            "continuous_operator_state": True,
        }


class QSREProductionBinder(nn.Module):
    """Token-level, type-aware adaptive structural support."""

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        self.query_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.field_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.edge_score = nn.Sequential(
            nn.Linear(4 * d + 4, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, 1),
        )

    def _field_summary(
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
        projected = self.field_projection(field_token_states)
        mask = field_token_mask.unsqueeze(-1).to(projected.dtype)
        return (projected * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

    def _query_summary(
        self,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
    ) -> Tensor:
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        final = query_hidden_states[:, -1]
        projected = self.query_projection(final)
        mask = query_token_mask.unsqueeze(-1).to(projected.dtype)
        return (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

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

        source_index = edge_index[..., 0].clamp(min=0, max=max(fields - 1, 0))
        target_index = edge_index[..., 1].clamp(min=0, max=max(fields - 1, 0))
        batch_index = torch.arange(batch, device=edge_index.device)[:, None].expand(batch, edges)
        source_type = field_type_id[batch_index, source_index].clamp(min=0, max=max(type_vocab - 1, 0))
        target_type = field_type_id[batch_index, target_index].clamp(min=0, max=max(type_vocab - 1, 0))

        rel_index = edge_relation_index.clamp(min=0, max=schema.token_states.size(0) - 1)
        domain = schema.domain_type_mask[rel_index, source_type]
        range_ok = schema.range_type_mask[rel_index, target_type]
        direct = domain & range_ok

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
        if edge_valid_mask.shape != (batch, edges) or edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool [B,E]")
        if edge_reliability.shape != (batch, edges):
            raise ValueError("edge_reliability shape drift")
        if edge_recency.shape != (batch, edges):
            raise ValueError("edge_recency shape drift")

        query_summary = self._query_summary(query_hidden_states, query_token_mask)
        field_summary = self._field_summary(field_token_states, field_token_mask)
        fields = field_summary.size(1)
        if field_type_id.shape != (batch, fields):
            raise ValueError("field_type_id shape drift")

        source_index = edge_index[..., 0].clamp(min=0, max=max(fields - 1, 0))
        target_index = edge_index[..., 1].clamp(min=0, max=max(fields - 1, 0))
        batch_index = torch.arange(batch, device=edge_index.device)[:, None].expand(batch, edges)
        source_state = field_summary[batch_index, source_index]
        target_state = field_summary[batch_index, target_index]

        rel_index = edge_relation_index.clamp(min=0, max=relation_count - 1)
        relation_state = schema_relation_state[rel_index]

        relation_mass = torch.einsum(
            "bsr,bs->br",
            operator.relation_distribution,
            operator.relation_step_mass,
        )
        relation_mass = relation_mass / operator.relation_step_mass.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        edge_relation_mass = relation_mass.gather(1, rel_index)

        q = query_summary[:, None, :].expand(batch, edges, -1)
        feature = torch.cat(
            [
                q,
                source_state,
                target_state,
                relation_state,
                edge_relation_mass.unsqueeze(-1),
                edge_reliability.unsqueeze(-1),
                edge_recency.unsqueeze(-1),
                operator.applicability[:, None, None].expand(batch, edges, 1),
            ],
            dim=-1,
        )
        learned_score = self.edge_score(feature).squeeze(-1)

        semantic_bonus = F.cosine_similarity(
            q,
            relation_state,
            dim=-1,
        )
        support_logits = (
            learned_score
            + semantic_bonus
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
        sparse = masked_sparsemax(support_logits, valid, dim=-1)

        # Exact-zero relational activation below the fallback/defer boundary.
        activation = torch.clamp(
            (operator.applicability - 0.25) / 0.50,
            min=0.0,
            max=1.0,
        )
        known_mass = (1.0 - operator.unknown_probability.max(dim=1).values).clamp(
            min=0.0,
            max=1.0,
        )
        edge_support_weight = sparse * activation[:, None] * known_mass[:, None]

        field_support_weight = torch.zeros(
            batch,
            fields,
            dtype=edge_support_weight.dtype,
            device=edge_support_weight.device,
        )
        source_scatter = source_index
        target_scatter = target_index
        field_support_weight.scatter_add_(1, source_scatter, edge_support_weight)
        field_support_weight.scatter_add_(1, target_scatter, edge_support_weight)
        field_support_weight = field_support_weight.clamp(max=1.0)

        return {
            "support_logits": support_logits,
            "edge_support_weight": edge_support_weight,
            "field_support_weight": field_support_weight,
            "type_compatible": type_compatible,
            "relation_mass": relation_mass,
        }

    def parameter_report(self) -> dict[str, int | bool]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_count_dependent_parameters": 0,
            "fixed_top_k": False,
            "exact_zero_sparse_support": True,
            "schema_type_compatibility": True,
            "runtime_support_score_is_supervised_score": True,
        }


class QSREProductionExecutor(nn.Module):
    """Schema-conditioned shared iterative node/edge executor."""

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        self.node_projection = nn.Linear(config.field_state_dim, d)
        self.field_metadata_projection = nn.Linear(config.field_metadata_dim, d)
        self.relation_projection = nn.Linear(d, d, bias=False)
        self.operator_projection = nn.Linear(d, d, bias=False)

        self.role_embedding = nn.Embedding(config.role_count, d)
        self.traversal_embedding = nn.Embedding(config.traversal_count, d)
        self.source_position = nn.Parameter(torch.empty(d))
        self.target_position = nn.Parameter(torch.empty(d))
        self.modifier_projection = nn.Linear(config.modifier_count, d, bias=False)

        # Shared across every runtime relation step.
        self.edge_update = nn.Sequential(
            nn.Linear(8 * d + 4, 2 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.source_message = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.target_message = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.node_update = nn.GRUCell(2 * d, d)
        self.readout = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.summary_projection = nn.Linear(2 * d, d)

        nn.init.normal_(self.source_position, std=0.02)
        nn.init.normal_(self.target_position, std=0.02)

    def _expected_embedding(self, probability: Tensor, embedding: nn.Embedding) -> Tensor:
        return probability @ embedding.weight

    def forward(
        self,
        *,
        field_state: Tensor,
        field_metadata: Tensor,
        field_valid_mask: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        edge_valid_mask: Tensor,
        edge_support_weight: Tensor,
        edge_reliability: Tensor,
        edge_recency: Tensor,
        edge_temporal_match: Optional[Tensor],
        edge_provenance_match: Optional[Tensor],
        schema_relation_state: Tensor,
        operator: QSREProductionOperatorState,
        focus_field_weight: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if field_state.ndim != 3:
            raise ValueError("field_state must be [B,F,D]")
        batch, fields, width = field_state.shape
        if width != self.config.field_state_dim:
            raise ValueError("field_state width drift")
        if field_metadata.shape != (batch, fields, self.config.field_metadata_dim):
            raise ValueError("field_metadata shape drift")
        if field_valid_mask.shape != (batch, fields) or field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool [B,F]")
        if edge_index.ndim != 3 or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [B,E,2]")
        edges = edge_index.size(1)
        expected_edge_shape = (batch, edges)
        for name, value in (
            ("edge_relation_index", edge_relation_index),
            ("edge_valid_mask", edge_valid_mask),
            ("edge_support_weight", edge_support_weight),
            ("edge_reliability", edge_reliability),
            ("edge_recency", edge_recency),
        ):
            if value.shape != expected_edge_shape:
                raise ValueError(f"{name} shape drift")
        if edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool")
        relation_count = schema_relation_state.size(0)
        operator.validate(relation_count=relation_count, model_dim=self.config.model_dim)
        if edge_temporal_match is None:
            edge_temporal_match = torch.ones_like(edge_support_weight)
        if edge_provenance_match is None:
            edge_provenance_match = torch.ones_like(edge_support_weight)
        if edge_temporal_match.shape != expected_edge_shape:
            raise ValueError("edge_temporal_match shape drift")
        if edge_provenance_match.shape != expected_edge_shape:
            raise ValueError("edge_provenance_match shape drift")

        if focus_field_weight is None:
            focus_field_weight = torch.zeros(
                batch, fields, dtype=field_state.dtype, device=field_state.device
            )
        if focus_field_weight.shape != (batch, fields):
            raise ValueError("focus_field_weight shape drift")

        node = (
            self.node_projection(field_state)
            + self.field_metadata_projection(field_metadata)
        )
        node = node * field_valid_mask.unsqueeze(-1).to(node.dtype)

        role_state = self._expected_embedding(operator.role_distribution, self.role_embedding)
        traversal_state = self._expected_embedding(
            operator.traversal_distribution,
            self.traversal_embedding,
        )
        modifier_state = self.modifier_projection(operator.modifier_weight)
        operator_state = self.operator_projection(operator.continuous_state)

        source_index = edge_index[..., 0].clamp(min=0, max=max(fields - 1, 0))
        target_index = edge_index[..., 1].clamp(min=0, max=max(fields - 1, 0))
        batch_index = torch.arange(batch, device=edge_index.device)[:, None].expand(batch, edges)
        rel_index = edge_relation_index.clamp(min=0, max=max(relation_count - 1, 0))
        edge_relation_state = self.relation_projection(schema_relation_state[rel_index])

        frontier = focus_field_weight.clamp(min=0.0, max=1.0)
        if bool((frontier.sum(dim=-1) == 0).any()):
            # Non-path rows do not require focus. For path rows with missing focus,
            # execution stays total and yields no path support instead of crashing.
            pass

        path_probability = operator.traversal_distribution[:, TRAVERSAL_PATH]
        last_edge_state = torch.zeros(
            batch,
            edges,
            self.config.model_dim,
            dtype=node.dtype,
            device=node.device,
        )

        for step in range(operator.relation_distribution.size(1)):
            relation_distribution = operator.relation_distribution[:, step]
            step_mass = operator.relation_step_mass[:, step]
            edge_relation_mass = relation_distribution.gather(1, rel_index)

            source_frontier = frontier.gather(1, source_index)
            path_gate = (
                path_probability[:, None] * source_frontier
                + (1.0 - path_probability[:, None])
            )

            reliability_multiplier = (
                1.0
                + operator.modifier_weight[:, MOD_RELIABILITY][:, None]
                * (edge_reliability - 0.5)
            ).clamp_min(0.0)
            recency_multiplier = (
                1.0
                + operator.modifier_weight[:, MOD_RECENCY][:, None]
                * (edge_recency - 0.5)
            ).clamp_min(0.0)
            temporal_multiplier = (
                1.0
                - operator.modifier_weight[:, MOD_TEMPORAL_CONSTRAINT][:, None]
                * (1.0 - edge_temporal_match)
            ).clamp_min(0.0)
            provenance_multiplier = (
                1.0
                - operator.modifier_weight[:, MOD_PROVENANCE_CONSTRAINT][:, None]
                * (1.0 - edge_provenance_match)
            ).clamp_min(0.0)

            gate = (
                edge_support_weight
                * edge_valid_mask.to(edge_support_weight.dtype)
                * edge_relation_mass
                * step_mass[:, None]
                * path_gate
                * reliability_multiplier
                * recency_multiplier
                * temporal_multiplier
                * provenance_multiplier
            )

            source_node = node[batch_index, source_index]
            target_node = node[batch_index, target_index]
            q_edge = operator_state[:, None, :].expand(batch, edges, -1)
            role_edge = role_state[:, None, :].expand(batch, edges, -1)
            traversal_edge = traversal_state[:, None, :].expand(batch, edges, -1)
            modifier_edge = modifier_state[:, None, :].expand(batch, edges, -1)
            source_pos = self.source_position.view(1, 1, -1).expand(batch, edges, -1)
            target_pos = self.target_position.view(1, 1, -1).expand(batch, edges, -1)

            scalar = torch.stack(
                [
                    edge_reliability,
                    edge_recency,
                    edge_temporal_match,
                    edge_provenance_match,
                ],
                dim=-1,
            )
            edge_input = torch.cat(
                [
                    source_node,
                    target_node,
                    edge_relation_state,
                    q_edge,
                    role_edge,
                    traversal_edge,
                    modifier_edge,
                    0.5 * (source_pos + target_pos),
                    scalar,
                ],
                dim=-1,
            )
            edge_state = self.edge_update(edge_input)
            last_edge_state = edge_state

            message_input = torch.cat([edge_state, q_edge], dim=-1)
            source_message = self.source_message(message_input) * gate.unsqueeze(-1)
            target_message = self.target_message(message_input) * gate.unsqueeze(-1)

            aggregate = torch.zeros_like(node)
            source_scatter = source_index.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
            target_scatter = target_index.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
            aggregate.scatter_add_(1, source_scatter, source_message)
            aggregate.scatter_add_(1, target_scatter, target_message)

            active_count = torch.zeros(
                batch,
                fields,
                dtype=node.dtype,
                device=node.device,
            )
            active_count.scatter_add_(1, source_index, gate)
            active_count.scatter_add_(1, target_index, gate)
            active_node = active_count > 0

            q_node = (operator_state + role_state + traversal_state + modifier_state)[
                :, None, :
            ].expand(batch, fields, -1)
            updated = self.node_update(
                torch.cat([aggregate, q_node], dim=-1).reshape(batch * fields, -1),
                node.reshape(batch * fields, -1),
            ).reshape(batch, fields, -1)
            node = torch.where(
                (active_node & field_valid_mask).unsqueeze(-1),
                updated,
                node,
            )

            next_frontier = torch.zeros_like(frontier)
            next_frontier.scatter_add_(1, target_index, gate)
            next_frontier = next_frontier.clamp(max=1.0)
            frontier = (
                path_probability[:, None] * next_frontier
                + (1.0 - path_probability[:, None]) * frontier
            )

        support_field_weight = torch.zeros(
            batch,
            fields,
            dtype=edge_support_weight.dtype,
            device=edge_support_weight.device,
        )
        support_field_weight.scatter_add_(1, source_index, edge_support_weight)
        support_field_weight.scatter_add_(1, target_index, edge_support_weight)
        support_field_weight = support_field_weight.clamp(max=1.0)

        role_query = role_state[:, None, :].expand(batch, fields, -1)
        operator_query = operator_state[:, None, :].expand(batch, fields, -1)
        readout_input = torch.cat(
            [node, role_query, operator_query, node * role_query],
            dim=-1,
        )
        relational_logit = self.readout(readout_input).squeeze(-1)

        # Path execution reads the reached frontier. Other traversal modes read
        # only the support-local field domain.
        path_mask = frontier > 0
        support_mask = (support_field_weight > 0) & field_valid_mask
        path_rows = path_probability >= 0.5
        readout_mask = torch.where(path_rows[:, None], path_mask, support_mask)
        relational_probability = _masked_softmax(
            relational_logit,
            readout_mask,
            dim=-1,
        )
        relational_probability = (
            relational_probability * operator.applicability[:, None]
        )

        summary_weight = relational_probability / relational_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        node_summary = torch.einsum("bf,bfd->bd", summary_weight, node)
        relational_summary = self.summary_projection(
            torch.cat([node_summary, operator_state], dim=-1)
        )

        has_support = readout_mask.any(dim=-1)
        return {
            "node_state": node,
            "last_edge_state": last_edge_state,
            "path_frontier": frontier,
            "support_field_weight": support_field_weight,
            "readout_mask": readout_mask,
            "relational_logit": relational_logit,
            "relational_probability": relational_probability,
            "relational_summary": relational_summary,
            "has_support": has_support,
        }

    def parameter_report(self) -> dict[str, int | bool]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_count_dependent_parameters": 0,
            "hop_count_dependent_parameters": 0,
            "learned_relation_id_embedding_authority": False,
            "dynamic_relation_semantic_state": True,
            "shared_iterative_execution_cell": True,
            "continuous_operator_state_consumed": True,
            "support_local_readout": True,
        }


class QSREProductionCore(nn.Module):
    """Production QSRE core interface.

    Causal stages may train the three submodules independently, but the interface
    is fixed here before those stages begin.
    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        self.config = config
        self.operator = QSREProductionOperatorInducer(config)
        self.binder = QSREProductionBinder(config)
        self.executor = QSREProductionExecutor(config)

    def parameter_report(self) -> dict[str, object]:
        return {
            "operator": self.operator.parameter_report(),
            "binder": self.binder.parameter_report(),
            "executor": self.executor.parameter_report(),
            "production_interfaces_fixed_before_causal_training": True,
            "private_identity_parameters": 0,
        }
