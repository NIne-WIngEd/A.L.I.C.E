from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


EVENT_CONTINUE = 0
EVENT_STOP = 1
EVENT_UNKNOWN = 2


@dataclass(frozen=True)
class SemanticOperatorFoundationConfig:
    """Shared N0 semantic/operator mechanics over runtime-supplied schema text.

    Numeric widths/depths describe one checkpoint topology. They are not product
    capability ceilings. Relation count, factor-bank count/cardinality, runtime
    step count, schema token count and type vocabulary size are runtime axes.
    """

    semantic_dim: int = 640
    model_dim: int = 640
    num_hidden_states: int = 17
    num_attention_heads: int = 10
    interaction_layers: int = 2
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("num_attention_heads", self.num_attention_heads),
            ("interaction_layers", self.interaction_layers),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


@dataclass(frozen=True)
class DynamicSemanticSchema:
    """Runtime semantic candidate bank with no learned candidate identity axis."""

    token_states: Tensor
    token_mask: Tensor

    def validate(
        self,
        *,
        num_hidden_states: int,
        semantic_dim: int,
    ) -> dict[str, int]:
        if self.token_states.ndim != 4:
            raise ValueError("schema token_states must be [C,L,S,D]")
        candidates, layers, tokens, width = self.token_states.shape
        if candidates <= 0:
            raise ValueError("schema requires at least one candidate")
        if layers != int(num_hidden_states):
            raise ValueError("schema hidden-state depth drift")
        if width != int(semantic_dim):
            raise ValueError("schema semantic width drift")
        if self.token_mask.shape != (candidates, tokens):
            raise ValueError("schema token_mask must be [C,S]")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("schema token_mask must be bool")
        if bool((self.token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every schema candidate requires at least one content token")
        if not bool(torch.isfinite(self.token_states).all()):
            raise ValueError("schema token states contain non-finite values")
        return {
            "candidates": candidates,
            "layers": layers,
            "tokens": tokens,
        }


@dataclass(frozen=True)
class DynamicRelationSchema(DynamicSemanticSchema):
    """Runtime relation semantics plus exact structural type constraints.

    The boolean type masks are structural compatibility, not learned relation
    identity. K is runtime-variable and carries no trainable parameter axis.
    """

    domain_type_mask: Tensor
    range_type_mask: Tensor
    symmetric: Tensor

    def validate(
        self,
        *,
        num_hidden_states: int,
        semantic_dim: int,
    ) -> dict[str, int]:
        result = super().validate(
            num_hidden_states=num_hidden_states,
            semantic_dim=semantic_dim,
        )
        candidates = result["candidates"]
        if self.domain_type_mask.ndim != 2:
            raise ValueError("domain_type_mask must be [R,K]")
        if self.range_type_mask.shape != self.domain_type_mask.shape:
            raise ValueError("domain/range type-mask shape drift")
        if self.domain_type_mask.size(0) != candidates:
            raise ValueError("relation/type candidate axis drift")
        if self.domain_type_mask.size(1) <= 0:
            raise ValueError("runtime type vocabulary must be non-empty")
        if self.domain_type_mask.dtype != torch.bool:
            raise ValueError("domain_type_mask must be bool")
        if self.range_type_mask.dtype != torch.bool:
            raise ValueError("range_type_mask must be bool")
        if self.symmetric.shape != (candidates,) or self.symmetric.dtype != torch.bool:
            raise ValueError("symmetric must be bool [R]")
        result["type_vocab"] = int(self.domain_type_mask.size(1))
        return result


@dataclass(frozen=True)
class SemanticOperatorState:
    relation_distribution: Tensor
    relation_step_mass: Tensor
    event_distribution: Tensor
    stop_probability: Tensor
    unknown_probability: Tensor
    continuous_state: Tensor
    query_coverage: Tensor
    uncertainty: Tensor
    factor_distributions: Mapping[str, Tensor]

    def validate(
        self,
        *,
        relation_count: int,
        model_dim: int,
        query_tokens: int,
    ) -> dict[str, int]:
        if self.relation_distribution.ndim != 3:
            raise ValueError("relation_distribution must be [B,S,R]")
        batch, steps, relations = self.relation_distribution.shape
        if relations != relation_count:
            raise ValueError("relation cardinality drift")
        if self.relation_step_mass.shape != (batch, steps):
            raise ValueError("relation_step_mass shape drift")
        if self.event_distribution.shape != (batch, steps, 3):
            raise ValueError("event_distribution must be [B,S,3]")
        if self.stop_probability.shape != (batch, steps):
            raise ValueError("stop_probability shape drift")
        if self.unknown_probability.shape != (batch, steps):
            raise ValueError("unknown_probability shape drift")
        if self.continuous_state.shape != (batch, model_dim):
            raise ValueError("continuous_state shape drift")
        if self.query_coverage.shape != (batch, steps, query_tokens):
            raise ValueError("query_coverage shape drift")
        if self.uncertainty.shape != (batch,):
            raise ValueError("uncertainty shape drift")
        for name, value in self.factor_distributions.items():
            if value.ndim != 2 or value.size(0) != batch or value.size(1) <= 0:
                raise ValueError(f"factor distribution {name!r} must be [B,C]")
        return {"batch": batch, "steps": steps, "relations": relations}


class _SharedSchemaTokenInteraction(nn.Module):
    """One shared semantic metric for relations and every factor schema bank.

    There is no candidate-index embedding, relation table, factor-class table,
    or cardinality-dependent parameter. Candidate semantics enter only through
    runtime token states produced by the shared semantic backbone.
    """

    def __init__(self, config: SemanticOperatorFoundationConfig) -> None:
        super().__init__()
        d = config.model_dim
        self.config = config
        self.query_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.schema_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.summary_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.layer_gate = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.token_temperature = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
        masked = logits.masked_fill(~mask, -1.0e4)
        value = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
        return value / value.sum(dim=dim, keepdim=True).clamp_min(1.0e-12)

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        schema: DynamicSemanticSchema,
        query_remaining: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        batch, layers, query_tokens, width = query_hidden_states.shape
        if layers != self.config.num_hidden_states or width != self.config.semantic_dim:
            raise ValueError("query semantic geometry drift")
        if query_token_mask.shape != (batch, query_tokens) or query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool [B,T]")
        if bool((query_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every query requires at least one content token")
        info = schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )

        q = self.query_projection(query_hidden_states.float())
        s = self.schema_projection(schema.token_states.float())
        q = F.normalize(q, dim=-1)
        s = F.normalize(s, dim=-1)

        # [B,C,L,Tq,Ts]. Runtime C does not create parameters.
        similarity = torch.einsum("blqd,clsd->bclqs", q, s)
        temperature = self.token_temperature.abs().clamp_min(0.05)
        similarity = similarity / temperature

        q_valid = query_token_mask[:, None, None, :, None]
        s_valid = schema.token_mask[None, :, None, None, :]
        valid = q_valid & s_valid
        similarity = similarity.masked_fill(~valid, -1.0e4)

        q_to_s = similarity.max(dim=-1).values
        s_to_q = similarity.max(dim=-2).values

        if query_remaining is None:
            query_remaining = torch.ones(
                batch,
                query_tokens,
                device=q.device,
                dtype=q.dtype,
            )
        if query_remaining.shape != (batch, query_tokens):
            raise ValueError("query_remaining must be [B,T]")
        remaining = query_remaining.clamp(min=0.02, max=1.0)

        q_mask = query_token_mask[:, None, None, :].expand(
            batch, info["candidates"], layers, query_tokens
        )
        q_logits = 4.0 * q_to_s + torch.log(remaining[:, None, None, :])
        q_weight = self._masked_softmax(q_logits, q_mask, dim=-1)
        q_score = (q_weight * q_to_s).sum(dim=-1)

        s_mask = schema.token_mask[None, :, None, :].expand(
            batch, info["candidates"], layers, schema.token_mask.size(1)
        )
        s_weight = self._masked_softmax(4.0 * s_to_q, s_mask, dim=-1)
        s_score = (s_weight * s_to_q).sum(dim=-1)
        per_layer = 0.5 * (q_score + s_score)

        # Candidate-conditioned learned layer weighting. The gate sees only
        # shared semantic statistics and normalized layer position; there is no
        # relation/factor identity parameter.
        layer_position = torch.linspace(
            -1.0, 1.0, layers, device=per_layer.device, dtype=per_layer.dtype
        ).view(1, 1, layers).expand_as(per_layer)
        layer_features = torch.stack(
            [per_layer, q_score - s_score, layer_position],
            dim=-1,
        )
        layer_logits = self.layer_gate(layer_features).squeeze(-1)
        layer_weight = torch.softmax(layer_logits, dim=-1)
        score = (layer_weight * per_layer).sum(dim=-1)

        # Preserve token-local evidence for ordered coverage.
        query_evidence = torch.einsum("bcl,bclt->bct", layer_weight, q_weight)
        query_evidence = (
            query_evidence
            * query_token_mask[:, None, :].to(query_evidence.dtype)
        )

        raw_schema = schema.token_states.float()
        mask = schema.token_mask.to(raw_schema.dtype)[:, None, :, None]
        layer_schema_summary = (raw_schema * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        projected_summary = self.summary_projection(layer_schema_summary)
        schema_summary = torch.einsum(
            "bcl,cld->bcd",
            layer_weight,
            projected_summary,
        )

        remaining_support = (
            query_evidence * remaining[:, None, :]
        ).sum(dim=-1).clamp(min=0.02, max=1.0)

        return {
            "score": score,
            "per_layer_score": per_layer,
            "layer_weight": layer_weight,
            "query_evidence": query_evidence,
            "schema_summary": schema_summary,
            "remaining_support": remaining_support,
        }


class SchemaConditionedSemanticOperator(nn.Module):
    """Jointly trainable semantic/operator interface for N0.

    This module consumes hidden states from the shared semantic backbone without
    detaching them. During a future authorized training run, gradients from
    schema/operator objectives can therefore co-adapt the semantic
    representation and this shared runtime-schema operator.

    It performs no exact structural sparsification. Exact support remains the
    responsibility of Binder v2 or an evidence-supported successor.
    """

    def __init__(self, config: SemanticOperatorFoundationConfig | None = None) -> None:
        super().__init__()
        self.config = config or SemanticOperatorFoundationConfig()
        self.config.validate()
        d = self.config.model_dim

        self.matcher = _SharedSchemaTokenInteraction(self.config)
        self.query_state_projection = nn.Linear(self.config.semantic_dim, d)
        self.initial_state_norm = nn.LayerNorm(d)
        self.state_schema_score = nn.Linear(d, d, bias=False)
        self.transition = nn.GRUCell(2 * d, d)
        self.event_head = nn.Sequential(
            nn.Linear(d + 3, d),
            nn.SiLU(),
            nn.Linear(d, 3),
        )

    def _query_state(
        self,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
    ) -> Tensor:
        weight = query_token_mask[:, None, :, None].to(query_hidden_states.dtype)
        pooled = (query_hidden_states * weight).sum(dim=(1, 2))
        denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
        pooled = pooled / denom
        return self.initial_state_norm(self.query_state_projection(pooled.float()))

    @staticmethod
    def _entropy(probability: Tensor) -> Tensor:
        count = probability.size(-1)
        if count <= 1:
            return torch.zeros(probability.shape[:-1], device=probability.device, dtype=probability.dtype)
        entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
        return entropy / torch.log(
            torch.tensor(float(count), device=probability.device, dtype=probability.dtype)
        )

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        relation_schema: DynamicRelationSchema,
        factor_schemas: Mapping[str, DynamicSemanticSchema],
        max_steps: int,
    ) -> dict[str, Any]:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        relation_info = relation_schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        batch, _, query_tokens, _ = query_hidden_states.shape
        state = self._query_state(query_hidden_states, query_token_mask)
        coverage = torch.zeros(
            batch,
            query_tokens,
            device=query_hidden_states.device,
            dtype=query_hidden_states.dtype,
        )
        survival = torch.ones(batch, device=state.device, dtype=state.dtype)

        relation_distributions: list[Tensor] = []
        relation_masses: list[Tensor] = []
        events: list[Tensor] = []
        stops: list[Tensor] = []
        unknowns: list[Tensor] = []
        coverage_history: list[Tensor] = []
        relation_scores: list[Tensor] = []
        layer_weights: list[Tensor] = []

        for _ in range(max_steps):
            remaining = (1.0 - coverage.float()).clamp(min=0.02, max=1.0)
            matched = self.matcher(
                query_hidden_states=query_hidden_states,
                query_token_mask=query_token_mask,
                schema=relation_schema,
                query_remaining=remaining,
            )
            schema_summary = matched["schema_summary"]
            state_query = F.normalize(self.state_schema_score(state), dim=-1)
            state_score = torch.einsum(
                "bd,bcd->bc",
                state_query,
                F.normalize(schema_summary, dim=-1),
            )
            logits = (
                matched["score"]
                + 0.25 * torch.tanh(state_score)
                + torch.log(matched["remaining_support"].clamp_min(0.02))
            )
            distribution = torch.softmax(logits, dim=-1)

            if relation_info["candidates"] > 1:
                top2 = logits.topk(2, dim=-1).values
                best = top2[:, 0]
                margin = top2[:, 0] - top2[:, 1]
            else:
                best = logits[:, 0]
                margin = torch.ones_like(best)
            entropy = self._entropy(distribution)
            event_input = torch.cat(
                [state, best[:, None], margin[:, None], entropy[:, None]],
                dim=-1,
            )
            event_logits = self.event_head(event_input)
            event_probability = torch.softmax(event_logits, dim=-1)
            continue_probability = event_probability[:, EVENT_CONTINUE]
            stop_probability = event_probability[:, EVENT_STOP]
            unknown_probability = event_probability[:, EVENT_UNKNOWN]

            effective_mass = survival * continue_probability
            relation_distributions.append(distribution)
            relation_masses.append(effective_mass)
            events.append(event_probability)
            stops.append(survival * stop_probability)
            unknowns.append(survival * unknown_probability)
            relation_scores.append(logits)
            layer_weights.append(matched["layer_weight"])

            evidence = torch.einsum(
                "bc,bct->bt",
                distribution,
                matched["query_evidence"],
            )
            evidence = evidence / evidence.amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
            consumed = (continue_probability[:, None] * evidence).clamp(0.0, 1.0)
            coverage = 1.0 - ((1.0 - coverage.float()) * (1.0 - consumed))
            coverage = coverage.clamp(0.0, 1.0)
            coverage_history.append(coverage)

            expected_schema = torch.einsum(
                "bc,bcd->bd",
                distribution,
                schema_summary,
            )
            query_context = self._query_state(query_hidden_states, query_token_mask)
            next_state = self.transition(
                torch.cat([expected_schema, query_context], dim=-1),
                state,
            )
            state = (
                continue_probability[:, None] * next_state
                + (1.0 - continue_probability[:, None]) * state
            )
            survival = survival * continue_probability

        factor_distributions: dict[str, Tensor] = {}
        factor_scores: dict[str, Tensor] = {}
        factor_layer_weights: dict[str, Tensor] = {}
        for name, schema in factor_schemas.items():
            schema.validate(
                num_hidden_states=self.config.num_hidden_states,
                semantic_dim=self.config.semantic_dim,
            )
            matched = self.matcher(
                query_hidden_states=query_hidden_states,
                query_token_mask=query_token_mask,
                schema=schema,
            )
            factor_scores[str(name)] = matched["score"]
            factor_distributions[str(name)] = torch.softmax(matched["score"], dim=-1)
            factor_layer_weights[str(name)] = matched["layer_weight"]

        relation_distribution = torch.stack(relation_distributions, dim=1)
        relation_step_mass = torch.stack(relation_masses, dim=1)
        event_distribution = torch.stack(events, dim=1)
        stop_probability = torch.stack(stops, dim=1)
        unknown_probability = torch.stack(unknowns, dim=1)
        query_coverage = torch.stack(coverage_history, dim=1)

        relation_entropy = self._entropy(relation_distribution)
        unknown_mass = unknown_probability.sum(dim=1).clamp(0.0, 1.0)
        factor_uncertainty = []
        for probability in factor_distributions.values():
            factor_uncertainty.append(self._entropy(probability))
        if factor_uncertainty:
            factor_u = torch.stack(factor_uncertainty, dim=-1).mean(dim=-1)
        else:
            factor_u = torch.zeros_like(unknown_mass)
        relation_u = (
            relation_entropy
            * relation_step_mass
        ).sum(dim=1) / relation_step_mass.sum(dim=1).clamp_min(1.0e-6)
        uncertainty = 1.0 - (
            (1.0 - relation_u.clamp(0.0, 1.0))
            * (1.0 - factor_u.clamp(0.0, 1.0))
            * (1.0 - unknown_mass)
        )

        operator = SemanticOperatorState(
            relation_distribution=relation_distribution,
            relation_step_mass=relation_step_mass,
            event_distribution=event_distribution,
            stop_probability=stop_probability,
            unknown_probability=unknown_probability,
            continuous_state=state,
            query_coverage=query_coverage,
            uncertainty=uncertainty,
            factor_distributions=factor_distributions,
        )
        operator.validate(
            relation_count=relation_info["candidates"],
            model_dim=self.config.model_dim,
            query_tokens=query_tokens,
        )
        return {
            "operator": operator,
            "relation_logits": torch.stack(relation_scores, dim=1),
            "relation_layer_weights": torch.stack(layer_weights, dim=1),
            "factor_logits": factor_scores,
            "factor_layer_weights": factor_layer_weights,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "relation_identity_parameters": 0,
            "factor_identity_parameters": 0,
            "candidate_count_dependent_parameters": 0,
            "runtime_step_count_dependent_parameters": 0,
            "type_vocabulary_dependent_parameters": 0,
            "relation_count_ceiling": None,
            "factor_count_ceiling": None,
            "runtime_step_count_ceiling": None,
            "type_vocabulary_ceiling": None,
            "shared_query_schema_token_interaction": True,
            "candidate_conditioned_multilayer_read": True,
            "continuous_relation_hypotheses": True,
            "exact_structural_sparsity": False,
            "semantic_backbone_gradient_can_flow": True,
            "factor_scorer_shared_across_schema_banks": True,
        }
