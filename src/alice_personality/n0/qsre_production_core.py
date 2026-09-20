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

DIRECTION_FORWARD = 0
DIRECTION_REVERSE = 1
DIRECTION_BIDIRECTIONAL = 2

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
    direction_count: int = 3
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
            "direction_count": self.direction_count,
            "modifier_count": self.modifier_count,
            "control_count": self.control_count,
        }
        for name, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.role_count < 4:
            raise ValueError("production structural role bank requires at least 4 seed roles")
        if self.traversal_count < 3:
            raise ValueError("production traversal bank requires at least 3 seed factors")
        if self.direction_count < 3:
            raise ValueError("production direction bank requires at least 3 seed factors")
        if self.modifier_count < 4:
            raise ValueError("production modifier bank requires at least 4 seed factors")
        if self.control_count < 3:
            raise ValueError("production control bank requires at least 3 seed states")
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
    direction_distribution: Tensor
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
        if self.role_distribution.ndim != 2 or self.role_distribution.size(0) != batch or self.role_distribution.size(1) < 4:
            raise ValueError("role_distribution must be [B,R_role] with at least 4 seed roles")
        if self.traversal_distribution.ndim != 2 or self.traversal_distribution.size(0) != batch or self.traversal_distribution.size(1) < 3:
            raise ValueError("traversal_distribution must be [B,R_traversal] with at least 3 seed factors")
        if self.direction_distribution.ndim != 2 or self.direction_distribution.size(0) != batch or self.direction_distribution.size(1) < 3:
            raise ValueError("direction_distribution must be [B,R_direction] with at least 3 seed factors")
        if self.modifier_weight.ndim != 2 or self.modifier_weight.size(0) != batch:
            raise ValueError("modifier_weight must be [B,M]")
        if self.applicability.shape != (batch,):
            raise ValueError("applicability must be [B]")
        if self.control_distribution.ndim != 2 or self.control_distribution.size(0) != batch or self.control_distribution.size(1) < 3:
            raise ValueError("control_distribution must be [B,R_control] with at least 3 seed states")
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
            ("direction_distribution", self.direction_distribution),
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


class QSREProductionSchemaEncoder(nn.Module):
    """Shared dynamic-schema encoder used by operator, binder, and executor.

    This module has no relation-cardinality-dependent parameters. It is trained
    with the first executor stage, then frozen for operator/binder causal stages.
    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim
        self.schema_norm = nn.LayerNorm(config.semantic_dim)
        self.schema_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.layer_embedding = nn.Embedding(config.num_hidden_states, d)
        self.pool_query = nn.Parameter(torch.empty(d))
        self.layer_pool_query = nn.Parameter(torch.empty(d))
        # Start as a geometry-preserving semantic adapter instead of a random
        # schema ontology. P1 may learn deviations, but open-schema semantics
        # begin from the ratified backbone geometry. Token and layer pooling
        # both begin uniform, then may learn content-dependent emphasis.
        nn.init.orthogonal_(self.schema_projection.weight)
        nn.init.zeros_(self.layer_embedding.weight)
        nn.init.zeros_(self.pool_query)
        nn.init.zeros_(self.layer_pool_query)

    def forward(
        self,
        schema: QSREDynamicRelationSchema,
    ) -> dict[str, Tensor]:
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

        score = torch.einsum(
            "rlsd,d->rls",
            torch.tanh(token),
            self.pool_query,
        )
        valid = schema.token_mask[:, None, :].expand(
            token.size(0),
            token.size(1),
            token.size(2),
        )
        weight = _masked_softmax(score, valid, dim=-1)
        facet = torch.einsum("rls,rlsd->rld", weight, token)
        layer_score = torch.einsum(
            "rld,d->rl",
            torch.tanh(facet),
            self.layer_pool_query,
        )
        layer_weight = torch.softmax(layer_score, dim=-1)
        summary = torch.einsum("rl,rld->rd", layer_weight, facet)

        return {
            "schema_token_state": token,
            "schema_facet_state": facet,
            "schema_relation_state": summary,
            "schema_token_attention": weight,
            "schema_layer_attention": layer_weight,
        }

    def parameter_report(self) -> dict[str, int | bool]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_count_dependent_parameters": 0,
            "runtime_dynamic_relation_schema": True,
            "token_level_schema_state_retained": True,
            "content_dependent_layer_pooling": True,
            "uniform_layer_pooling_at_initialization": True,
            "shared_across_operator_binder_executor": True,
            "schema_geometry_preserving_initialization": True,
            "current_relation_count_is_not_parameter_topology": True,
        }


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
        self.direction_head = nn.Linear(d, config.direction_count)
        self.modifier_head = nn.Linear(d, config.modifier_count)
        self.applicability_head = nn.Linear(d, 1)
        self.control_head = nn.Linear(d, config.control_count)
        self.continuous_projection = nn.Linear(d, d)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.query_projection.weight)
        nn.init.zeros_(self.layer_embedding.weight)
        nn.init.normal_(self.start_state, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.step_query.weight)
        for head in (
            self.stop_head,
            self.unknown_head,
            self.role_head,
            self.traversal_head,
            self.direction_head,
            self.modifier_head,
            self.applicability_head,
            self.control_head,
            self.continuous_projection,
        ):
            if hasattr(head, "weight"):
                nn.init.xavier_uniform_(head.weight)
            if getattr(head, "bias", None) is not None:
                nn.init.zeros_(head.bias)

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
        schema_token_state: Tensor,
        schema_relation_state: Tensor,
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

        schema_info = schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        relations = schema_info["relations"]
        expected_token_shape = (
            relations,
            self.config.num_hidden_states,
            schema_info["schema_tokens"],
            self.config.model_dim,
        )
        if tuple(schema_token_state.shape) != expected_token_shape:
            raise ValueError(
                "encoded schema token-state shape drift: "
                f"{tuple(schema_token_state.shape)} != {expected_token_shape}"
            )
        if schema_relation_state.shape != (
            relations,
            self.config.model_dim,
        ):
            raise ValueError("encoded schema relation-state shape drift")
        schema_projected = schema_token_state
        schema_summary = schema_relation_state

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
            candidate_step_state = self.query_cross_norm(
                state + attended.squeeze(1)
            )
            # Rows that already terminated retain their exact prior semantic
            # state even when the external compute budget continues for other
            # rows in the batch.
            step_state = (
                survival[:, None] * candidate_step_state
                + (1.0 - survival[:, None]) * state
            )
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
            effective_stop = survival * stop_probability
            effective_unknown = survival * unknown_probability
            relation_steps.append(relation_distribution)
            relation_mass_steps.append(effective_mass)
            stop_steps.append(effective_stop)
            unknown_steps.append(effective_unknown)
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
            next_state = self.step_transition(
                torch.cat([expected_relation, query_summary], dim=-1),
                step_state,
            )
            # STOP and UNKNOWN both terminate relational continuation. Relation
            # mass is therefore the differentiable continuation probability.
            # Once a row terminates, later externally-budgeted iterations cannot
            # mutate its semantic operator state.
            continuation = relation_mass.clamp(min=0.0, max=1.0)
            state = (
                continuation[:, None] * next_state
                + (1.0 - continuation[:, None]) * step_state
            )
            survival = survival * continuation

        relation_distribution = torch.stack(relation_steps, dim=1)
        relation_step_mass = torch.stack(relation_mass_steps, dim=1)
        stop_probability = torch.stack(stop_steps, dim=1)
        unknown_probability = torch.stack(unknown_steps, dim=1)
        step_state_history = torch.stack(state_steps, dim=1)
        step_attention = torch.stack(attention_steps, dim=1)

        role_distribution = torch.softmax(self.role_head(state), dim=-1)
        traversal_distribution = torch.softmax(self.traversal_head(state), dim=-1)
        direction_distribution = torch.softmax(self.direction_head(state), dim=-1)
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
        relation_entropy = relation_entropy / max(
            float(torch.log(torch.tensor(float(relations))).item()),
            1.0e-6,
        )
        role_entropy = -(
            role_distribution.clamp_min(1.0e-12)
            * role_distribution.clamp_min(1.0e-12).log()
        ).sum(dim=-1) / max(
            float(torch.log(torch.tensor(float(self.config.role_count))).item()),
            1.0e-6,
        )
        traversal_entropy = -(
            traversal_distribution.clamp_min(1.0e-12)
            * traversal_distribution.clamp_min(1.0e-12).log()
        ).sum(dim=-1) / max(
            float(torch.log(torch.tensor(float(self.config.traversal_count))).item()),
            1.0e-6,
        )
        direction_entropy = -(
            direction_distribution.clamp_min(1.0e-12)
            * direction_distribution.clamp_min(1.0e-12).log()
        ).sum(dim=-1) / max(
            float(torch.log(torch.tensor(float(self.config.direction_count))).item()),
            1.0e-6,
        )
        unknown_event_mass = unknown_probability.sum(dim=1).clamp(
            min=0.0,
            max=1.0,
        )
        # A deterministic known operator now reports uncertainty near zero
        # instead of the old sigmoid baseline of 0.5. Multiple independent
        # uncertainty sources compose as a differentiable probabilistic union.
        uncertainty_components = torch.stack(
            [
                relation_entropy.clamp(0.0, 1.0),
                role_entropy.clamp(0.0, 1.0),
                traversal_entropy.clamp(0.0, 1.0),
                direction_entropy.clamp(0.0, 1.0),
                unknown_event_mass,
            ],
            dim=-1,
        )
        uncertainty = 1.0 - torch.prod(
            1.0 - uncertainty_components,
            dim=-1,
        )

        operator = QSREProductionOperatorState(
            relation_distribution=relation_distribution,
            relation_step_mass=relation_step_mass,
            stop_probability=stop_probability,
            unknown_probability=unknown_probability,
            role_distribution=role_distribution,
            traversal_distribution=traversal_distribution,
            direction_distribution=direction_distribution,
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
            "termination_events_survival_weighted": True,
            "uncertainty_zero_based_and_normalized": True,
            "continuous_operator_state": True,
            "role_count_ceiling": None,
            "traversal_count_ceiling": None,
            "direction_count_ceiling": None,
            "control_count_ceiling": None,
            "seed_factor_counts_are_checkpoint_topology_not_product_limits": True,
        }


class QSREProductionBinder(nn.Module):
    """Token-level, type-aware adaptive structural support.

    The runtime support score is the supervised score. Endpoint relevance uses
    late token interaction instead of relying only on pooled field states.
    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        self.query_norm = nn.LayerNorm(config.semantic_dim)
        self.field_norm = nn.LayerNorm(config.semantic_dim)
        self.query_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.field_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.operator_projection = nn.Linear(d, d, bias=False)
        self.focus_score = nn.Sequential(
            nn.Linear(2 * d + 1, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(5 * d + 7, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, 1),
        )
        nn.init.orthogonal_(self.query_projection.weight)
        nn.init.orthogonal_(self.field_projection.weight)
        for head in (self.focus_score, self.edge_score):
            final = head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

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
        projected = self.field_projection(self.field_norm(field_token_states))
        return projected

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
        # Binding is entity/evidence relevance rather than relation decoding.
        # Use token states directly, not a pooled query. The final frozen
        # semantic layer is the field-token-aligned semantic surface; relation
        # depth variation remains the operator inducer's responsibility.
        final = query_hidden_states[:, -1]
        return self.query_projection(self.query_norm(final))

    @staticmethod
    def _masked_token_mean(
        token_state: Tensor,
        token_mask: Tensor,
        *,
        dim: int,
    ) -> Tensor:
        weight = token_mask.to(token_state.dtype).unsqueeze(-1)
        return (token_state * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)

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
        token_match = similarity.max(dim=-1).values
        q_weight = query_mask[:, None, :].to(token_match.dtype)
        score = (token_match * q_weight).sum(dim=-1) / q_weight.sum(
            dim=-1
        ).clamp_min(1.0)
        endpoint_available = endpoint_mask.any(dim=-1)
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

        field_late = self._late_endpoint_score(
            query_token=query_token,
            query_mask=query_token_mask,
            endpoint_token=field_token,
            endpoint_mask=field_token_mask,
        )
        q_field = query_summary[:, None, :].expand(batch, fields, -1)
        focus_feature = torch.cat(
            [q_field, field_summary, field_late.unsqueeze(-1)],
            dim=-1,
        )
        focus_logits = field_late + self.focus_score(focus_feature).squeeze(-1)
        field_valid = field_token_mask.any(dim=-1)
        focus_field_weight = masked_sparsemax(
            focus_logits,
            field_valid,
            dim=-1,
        )
        focus_max = focus_field_weight.max(dim=-1, keepdim=True).values
        focus_field_weight = torch.where(
            focus_max > 0,
            focus_field_weight / focus_max.clamp_min(1.0e-12),
            torch.zeros_like(focus_field_weight),
        )

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

        source_index = edge_index[..., 0].clamp(min=0, max=max(fields - 1, 0))
        target_index = edge_index[..., 1].clamp(min=0, max=max(fields - 1, 0))
        batch_index = torch.arange(batch, device=edge_index.device)[:, None].expand(batch, edges)
        source_state = field_summary[batch_index, source_index]
        target_state = field_summary[batch_index, target_index]
        source_token = field_token[batch_index, source_index]
        target_token = field_token[batch_index, target_index]
        source_token_mask = field_token_mask[batch_index, source_index]
        target_token_mask = field_token_mask[batch_index, target_index]

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
        endpoint_late = torch.maximum(source_late, target_late)

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
        operator_state = self.operator_projection(operator.continuous_state)
        op = operator_state[:, None, :].expand(batch, edges, -1)
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
                relation_state,
                op,
                scalar,
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
        sparse = masked_sparsemax(support_logits, valid, dim=-1)
        # Sparsemax discovers a variable-cardinality support set, but its
        # simplex normalization would otherwise make each valid support weaker
        # merely because a query needs more edges. Re-normalize by the strongest
        # retained support so path length and plural evidence do not create a
        # hidden execution-capacity penalty.
        sparse_max = sparse.max(dim=-1, keepdim=True).values
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
            1.0 - operator.unknown_probability.sum(dim=1).clamp(max=1.0)
        ).clamp(min=0.0, max=1.0)
        edge_support_weight = sparse * activation[:, None] * known_mass[:, None]

        field_support_weight = torch.zeros(
            batch,
            fields,
            dtype=edge_support_weight.dtype,
            device=edge_support_weight.device,
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
            "runtime_support_score_is_supervised_score": True,
            "token_level_late_interaction": True,
            "pooled_only_binding": False,
            "field_count_ceiling": None,
            "edge_count_ceiling": None,
            "support_count_ceiling": None,
            "focus_field_count_ceiling": None,
            "focus_is_predicted_from_query_field_late_interaction": True,
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
        self.direction_embedding = nn.Embedding(config.direction_count, d)
        self.source_position = nn.Parameter(torch.empty(d))
        self.target_position = nn.Parameter(torch.empty(d))
        self.modifier_projection = nn.Linear(config.modifier_count, d, bias=False)

        # Shared across every runtime relation step.
        self.edge_update = nn.Sequential(
            nn.Linear(9 * d + 4, 2 * d),
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
        direction_state = self._expected_embedding(
            operator.direction_distribution,
            self.direction_embedding,
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
            target_frontier = frontier.gather(1, target_index)
            forward_probability = operator.direction_distribution[:, DIRECTION_FORWARD][:, None]
            reverse_probability = operator.direction_distribution[:, DIRECTION_REVERSE][:, None]
            bidirectional_probability = operator.direction_distribution[:, DIRECTION_BIDIRECTIONAL][:, None]

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

            common_gate = (
                edge_support_weight
                * edge_valid_mask.to(edge_support_weight.dtype)
                * edge_relation_mass
                * step_mass[:, None]
                * reliability_multiplier
                * recency_multiplier
                * temporal_multiplier
                * provenance_multiplier
            )
            forward_gate = common_gate * source_frontier * forward_probability
            reverse_gate = common_gate * target_frontier * reverse_probability
            bidirectional_forward_gate = (
                common_gate * source_frontier * bidirectional_probability
            )
            bidirectional_reverse_gate = (
                common_gate * target_frontier * bidirectional_probability
            )
            path_directional_gate = (
                forward_gate
                + reverse_gate
                + bidirectional_forward_gate
                + bidirectional_reverse_gate
            ).clamp(max=1.0)
            gate = (
                path_probability[:, None] * path_directional_gate
                + (1.0 - path_probability[:, None]) * common_gate
            )

            source_node = node[batch_index, source_index]
            target_node = node[batch_index, target_index]
            q_edge = operator_state[:, None, :].expand(batch, edges, -1)
            role_edge = role_state[:, None, :].expand(batch, edges, -1)
            traversal_edge = traversal_state[:, None, :].expand(batch, edges, -1)
            direction_edge = direction_state[:, None, :].expand(batch, edges, -1)
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
                    direction_edge,
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

            q_node = (
                operator_state
                + role_state
                + traversal_state
                + direction_state
                + modifier_state
            )[:, None, :].expand(batch, fields, -1)
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
            next_frontier.scatter_add_(
                1,
                target_index,
                forward_gate + bidirectional_forward_gate,
            )
            next_frontier.scatter_add_(
                1,
                source_index,
                reverse_gate + bidirectional_reverse_gate,
            )
            next_frontier = next_frontier.clamp(max=1.0)
            frontier = (
                path_probability[:, None] * next_frontier
                + (1.0 - path_probability[:, None]) * frontier
            )

        source_support_weight = torch.zeros(
            batch,
            fields,
            dtype=edge_support_weight.dtype,
            device=edge_support_weight.device,
        )
        target_support_weight = torch.zeros_like(source_support_weight)
        source_support_weight.scatter_add_(1, source_index, edge_support_weight)
        target_support_weight.scatter_add_(1, target_index, edge_support_weight)
        source_support_weight = source_support_weight.clamp(max=1.0)
        target_support_weight = target_support_weight.clamp(max=1.0)
        support_field_weight = (
            source_support_weight + target_support_weight
        ).clamp(max=1.0)

        role_query = role_state[:, None, :].expand(batch, fields, -1)
        operator_query = operator_state[:, None, :].expand(batch, fields, -1)
        readout_input = torch.cat(
            [node, role_query, operator_query, node * role_query],
            dim=-1,
        )
        relational_logit = self.readout(readout_input).squeeze(-1)

        # Argument roles are structural, not merely opaque learned labels.
        # For local/aggregate execution SOURCE and TARGET select oriented
        # endpoint support directly. For a path they select the path origin or
        # reached frontier. SYMMETRIC preserves both endpoints. This makes role
        # reversal systematic across relation families and path lengths.
        source_role = operator.role_distribution[:, ROLE_SOURCE][:, None]
        target_role = operator.role_distribution[:, ROLE_TARGET][:, None]
        symmetric_role = operator.role_distribution[:, ROLE_SYMMETRIC][:, None]
        none_role = operator.role_distribution[:, ROLE_NONE][:, None]

        local_union = support_field_weight
        local_role_weight = (
            source_role * source_support_weight
            + target_role * target_support_weight
            + symmetric_role * local_union
            + none_role * local_union
        ).clamp(min=0.0, max=1.0)

        path_origin = focus_field_weight.clamp(min=0.0, max=1.0)
        path_reached = frontier.clamp(min=0.0, max=1.0)
        path_union = (path_origin + path_reached).clamp(max=1.0)
        forward = operator.direction_distribution[:, DIRECTION_FORWARD][:, None]
        reverse = operator.direction_distribution[:, DIRECTION_REVERSE][:, None]
        bidirectional = operator.direction_distribution[:, DIRECTION_BIDIRECTIONAL][:, None]
        semantic_source = (
            forward * path_origin
            + reverse * path_reached
            + bidirectional * path_union
        ).clamp(max=1.0)
        semantic_target = (
            forward * path_reached
            + reverse * path_origin
            + bidirectional * path_union
        ).clamp(max=1.0)
        path_role_weight = (
            source_role * semantic_source
            + target_role * semantic_target
            + symmetric_role * path_union
            + none_role * path_union
        ).clamp(min=0.0, max=1.0)

        path_rows = path_probability >= 0.5
        structural_role_weight = torch.where(
            path_rows[:, None],
            path_role_weight,
            local_role_weight,
        )
        readout_mask = (structural_role_weight > 0) & field_valid_mask
        structural_logit = relational_logit + torch.log(
            structural_role_weight.clamp_min(1.0e-8)
        )
        relational_probability = _masked_softmax(
            structural_logit,
            readout_mask,
            dim=-1,
        )
        # Preserve calibrated fail-closed mass. Normalizing inside the support
        # set must not erase uncertainty, UNKNOWN, or non-relational control.
        relational_control = operator.control_distribution[:, CONTROL_RELATIONAL]
        known_probability = (
            1.0 - operator.unknown_probability.sum(dim=1).clamp(max=1.0)
        ).clamp(min=0.0, max=1.0)
        program_presence = operator.relation_step_mass.sum(dim=1).clamp(
            min=0.0,
            max=1.0,
        )
        execution_confidence = (
            operator.applicability
            * relational_control
            * known_probability
            * program_presence
        ).clamp(min=0.0, max=1.0)
        relational_probability = (
            relational_probability * execution_confidence[:, None]
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
            "source_support_weight": source_support_weight,
            "target_support_weight": target_support_weight,
            "structural_role_weight": structural_role_weight,
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
            "structural_argument_role_readout": True,
            "path_source_target_role_systematicity": True,
            "forward_reverse_bidirectional_path_execution": True,
            "calibrated_execution_confidence_preserved": True,
            "role_count_ceiling": None,
            "traversal_count_ceiling": None,
            "direction_count_ceiling": None,
            "modifier_count_ceiling": None,
        }


class QSREProductionCore(nn.Module):
    """Production QSRE core interface.

    Causal stages may train the three submodules independently, but the interface
    is fixed here before those stages begin.
    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        self.config = config
        self.schema_encoder = QSREProductionSchemaEncoder(config)
        self.operator = QSREProductionOperatorInducer(config)
        self.binder = QSREProductionBinder(config)
        self.executor = QSREProductionExecutor(config)

    def encode_schema(
        self,
        schema: QSREDynamicRelationSchema,
    ) -> dict[str, Tensor]:
        return self.schema_encoder(schema)

    def induce_operator(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        schema: QSREDynamicRelationSchema,
        max_steps: int,
    ) -> dict[str, Tensor | QSREProductionOperatorState]:
        encoded = self.schema_encoder(schema)
        output = self.operator(
            query_hidden_states=query_hidden_states,
            query_token_mask=query_token_mask,
            schema=schema,
            schema_token_state=encoded["schema_token_state"],
            schema_relation_state=encoded["schema_relation_state"],
            max_steps=max_steps,
        )
        output["schema_facet_state"] = encoded["schema_facet_state"]
        output["schema_token_attention"] = encoded["schema_token_attention"]
        output["schema_layer_attention"] = encoded["schema_layer_attention"]
        return output

    def parameter_report(self) -> dict[str, object]:
        return {
            "schema_encoder": self.schema_encoder.parameter_report(),
            "operator": self.operator.parameter_report(),
            "binder": self.binder.parameter_report(),
            "executor": self.executor.parameter_report(),
            "production_interfaces_fixed_before_causal_training": True,
            "private_identity_parameters": 0,
        }
