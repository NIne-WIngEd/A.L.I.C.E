from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.chunked_late_interaction import (
    chunked_schema_bidirectional_late_max,
)
from alice_personality.n0.numeric_contracts import (
    exact_masked_logits,
    exact_masked_softmax,
)


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
    interaction_chunk_tokens: int = 128
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("interaction_chunk_tokens", self.interaction_chunk_tokens),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
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
    truncation_probability: Tensor
    continuous_state: Tensor
    applicability: Tensor
    query_coverage: Tensor
    uncertainty: Tensor
    factor_distributions: Mapping[str, Tensor]
    step_factor_distributions: Mapping[str, Tensor]

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
        if self.truncation_probability.shape != (batch,):
            raise ValueError("truncation_probability must be [B]")
        if self.continuous_state.shape != (batch, model_dim):
            raise ValueError("continuous_state shape drift")
        if self.applicability.shape != (batch,):
            raise ValueError("applicability shape drift")
        if self.query_coverage.shape != (batch, steps, query_tokens):
            raise ValueError("query_coverage shape drift")
        if self.uncertainty.shape != (batch,):
            raise ValueError("uncertainty shape drift")
        normalized_distributions = (
            ("relation_distribution", self.relation_distribution),
            ("event_distribution", self.event_distribution),
        )
        for name, tensor in normalized_distributions:
            if bool(((tensor < -1.0e-6) | (tensor > 1.0 + 1.0e-6)).any()):
                raise ValueError(f"{name} must stay inside [0,1]")
            total = tensor.sum(dim=-1)
            if not torch.allclose(
                total,
                torch.ones_like(total),
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                raise ValueError(f"{name} must sum to one on its candidate axis")

        for name, tensor in (
            ("relation_step_mass", self.relation_step_mass),
            ("stop_probability", self.stop_probability),
            ("unknown_probability", self.unknown_probability),
            ("truncation_probability", self.truncation_probability),
            ("applicability", self.applicability),
            ("uncertainty", self.uncertainty),
            ("query_coverage", self.query_coverage),
        ):
            if bool(((tensor < -1.0e-6) | (tensor > 1.0 + 1.0e-6)).any()):
                raise ValueError(f"{name} must stay inside [0,1]")

        event_mass = (
            self.relation_step_mass
            + self.stop_probability
            + self.unknown_probability
        )
        expected_mass = torch.cat(
            [
                torch.ones(
                    batch,
                    1,
                    device=event_mass.device,
                    dtype=event_mass.dtype,
                ),
                self.relation_step_mass[:, :-1],
            ],
            dim=1,
        )
        if not torch.allclose(
            event_mass,
            expected_mass,
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise ValueError(
                "relation/stop/unknown masses violate recurrent survival conservation"
            )
        if not torch.allclose(
            self.truncation_probability,
            self.relation_step_mass[:, -1],
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise ValueError(
                "truncation_probability must equal residual survival after final slot"
            )

        if set(self.factor_distributions) != set(self.step_factor_distributions):
            raise ValueError("global/step factor distribution names must match")
        for name, value in self.factor_distributions.items():
            if value.ndim != 2 or value.size(0) != batch or value.size(1) <= 0:
                raise ValueError(f"factor distribution {name!r} must be [B,C]")
            step_value = self.step_factor_distributions[name]
            if step_value.shape != (batch, steps, value.size(1)):
                raise ValueError(
                    f"step factor distribution {name!r} must be [B,S,C]"
                )
            for label, probability in (
                (f"factor distribution {name!r}", value),
                (f"step factor distribution {name!r}", step_value),
            ):
                if bool(
                    (
                        (probability < -1.0e-6)
                        | (probability > 1.0 + 1.0e-6)
                    ).any()
                ):
                    raise ValueError(f"{label} must stay inside [0,1]")
                total = probability.sum(dim=-1)
                if not torch.allclose(
                    total,
                    torch.ones_like(total),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                ):
                    raise ValueError(f"{label} must sum to one")
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
        return exact_masked_softmax(logits, mask, dim=dim)

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

        temperature = self.token_temperature.abs().clamp_min(0.05)
        q_to_s, s_to_q = chunked_schema_bidirectional_late_max(
            query=q,
            query_mask=query_token_mask,
            schema=s,
            schema_mask=schema.token_mask,
            chunk_tokens=self.config.interaction_chunk_tokens,
        )
        q_to_s = q_to_s / temperature
        s_to_q = s_to_q / temperature

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

        # Preserve both sides of the bidirectional token interaction.
        # Query evidence drives ordered coverage; schema evidence provides the
        # symmetric grounding target promised by the full-envelope curriculum.
        query_evidence = torch.einsum("bcl,bclt->bct", layer_weight, q_weight)
        query_evidence = (
            query_evidence
            * query_token_mask[:, None, :].to(query_evidence.dtype)
        )
        schema_evidence = torch.einsum(
            "bcl,bcls->bcs",
            layer_weight,
            s_weight,
        )
        schema_evidence = (
            schema_evidence
            * schema.token_mask[None, :, :].to(schema_evidence.dtype)
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
            "schema_evidence": schema_evidence,
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
        self.query_token_gate = nn.Linear(d, 1, bias=False)
        self.query_layer_gate = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1, bias=False),
        )
        self.initial_state_norm = nn.LayerNorm(d)
        self.state_schema_score = nn.Linear(d, d, bias=False)
        self.factor_context_query = nn.Linear(d, d, bias=False)
        self.factor_context_key = nn.Linear(d, d, bias=False)
        self.factor_context_value = nn.Linear(d, d, bias=False)
        self.factor_state_norm = nn.LayerNorm(d)
        self.transition = nn.GRUCell(3 * d, d)
        self.global_factor_transition = nn.GRUCell(d, d)
        self.event_head = nn.Sequential(
            nn.Linear(d + 3, d),
            nn.SiLU(),
            nn.Linear(d, 3),
        )
        self.applicability_head = nn.Linear(d, 1)

    def _query_state(
        self,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
    ) -> Tensor:
        projected = self.query_state_projection(query_hidden_states.float())
        token_logit = self.query_token_gate(torch.tanh(projected)).squeeze(-1)
        token_mask = query_token_mask[:, None, :].expand_as(token_logit)
        token_weight = exact_masked_softmax(
            token_logit,
            token_mask,
            dim=-1,
        )
        per_layer = torch.einsum(
            "blt,bltd->bld",
            token_weight,
            projected,
        )
        layer_logit = self.query_layer_gate(
            torch.tanh(per_layer)
        ).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        pooled = torch.einsum("bl,bld->bd", layer_weight, per_layer)
        return self.initial_state_norm(pooled)

    @staticmethod
    def _entropy(probability: Tensor) -> Tensor:
        count = probability.size(-1)
        if count <= 1:
            return torch.zeros(probability.shape[:-1], device=probability.device, dtype=probability.dtype)
        entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
        return entropy / torch.log(
            torch.tensor(float(count), device=probability.device, dtype=probability.dtype)
        )

    @staticmethod
    def _masked_entropy(probability: Tensor, mask: Tensor) -> Tensor:
        if probability.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError("probability/mask geometry drift")
        active = mask.sum(dim=-1)
        entropy = -(
            probability
            * probability.clamp_min(1.0e-12).log()
            * mask.to(probability.dtype)
        ).sum(dim=-1)
        denom = torch.log(
            active.clamp_min(2).to(probability.dtype)
        )
        normalized = entropy / denom
        return torch.where(
            active > 1,
            normalized,
            torch.zeros_like(normalized),
        )

    def _aggregate_factor_context(
        self,
        *,
        state: Tensor,
        factor_states: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Permutation-invariant context over a runtime number of factor banks."""
        if not factor_states:
            empty = torch.zeros_like(state)
            weight = torch.zeros(
                state.size(0),
                0,
                device=state.device,
                dtype=state.dtype,
            )
            return empty, weight
        bank_state = torch.stack(factor_states, dim=1)
        query = F.normalize(
            self.factor_context_query(state.float()),
            dim=-1,
        )
        key = F.normalize(
            self.factor_context_key(bank_state.float()),
            dim=-1,
        )
        logit = torch.einsum("bd,bfd->bf", query, key)
        weight = torch.softmax(logit, dim=-1)
        value = self.factor_context_value(bank_state.float())
        context = torch.einsum("bf,bfd->bd", weight, value)
        return context, weight

    @staticmethod
    def _masked_candidate_distribution(
        logits: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if logits.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError("candidate logits/mask geometry drift")
        if bool((mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one active candidate")
        masked_logits = exact_masked_logits(logits, mask)
        probability = exact_masked_softmax(
            logits,
            mask,
            dim=-1,
        )
        return masked_logits, probability

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        relation_schema: DynamicRelationSchema,
        factor_schemas: Mapping[str, DynamicSemanticSchema],
        max_steps: int,
        relation_candidate_mask: Tensor | None = None,
        factor_candidate_masks: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        relation_info = relation_schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        batch, _, query_tokens, _ = query_hidden_states.shape
        if relation_candidate_mask is None:
            relation_candidate_mask = torch.ones(
                batch,
                relation_info["candidates"],
                device=query_hidden_states.device,
                dtype=torch.bool,
            )
        if (
            relation_candidate_mask.shape
            != (batch, relation_info["candidates"])
            or relation_candidate_mask.dtype != torch.bool
        ):
            raise ValueError("relation_candidate_mask must be bool [B,R]")
        if bool((relation_candidate_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires an active relation candidate")

        factor_candidate_masks = (
            {} if factor_candidate_masks is None else dict(factor_candidate_masks)
        )
        unknown_factor_masks = set(factor_candidate_masks) - set(factor_schemas)
        if unknown_factor_masks:
            raise ValueError(
                "factor candidate mask supplied for unknown banks: "
                + repr(sorted(unknown_factor_masks))
            )
        resolved_factor_masks: dict[str, Tensor] = {}
        for name, schema in factor_schemas.items():
            info = schema.validate(
                num_hidden_states=self.config.num_hidden_states,
                semantic_dim=self.config.semantic_dim,
            )
            mask = factor_candidate_masks.get(str(name))
            if mask is None:
                mask = torch.ones(
                    batch,
                    info["candidates"],
                    device=query_hidden_states.device,
                    dtype=torch.bool,
                )
            if mask.shape != (batch, info["candidates"]) or mask.dtype != torch.bool:
                raise ValueError(
                    f"factor candidate mask {name!r} must be bool [B,C]"
                )
            if bool((mask.sum(dim=-1) == 0).any()):
                raise ValueError(
                    f"every example requires an active {name!r} candidate"
                )
            resolved_factor_masks[str(name)] = mask

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
        relation_schema_states: list[Tensor] = []
        relation_query_evidence: list[Tensor] = []
        relation_schema_evidence: list[Tensor] = []
        step_factor_scores: dict[str, list[Tensor]] = {
            str(name): [] for name in factor_schemas
        }
        step_factor_layer_weights: dict[str, list[Tensor]] = {
            str(name): [] for name in factor_schemas
        }
        step_factor_distributions_lists: dict[str, list[Tensor]] = {
            str(name): [] for name in factor_schemas
        }
        step_factor_schema_evidence_lists: dict[str, list[Tensor]] = {
            str(name): [] for name in factor_schemas
        }
        step_factor_context_states: list[Tensor] = []
        step_factor_bank_weights: list[Tensor] = []
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
            logits, distribution = self._masked_candidate_distribution(
                logits,
                relation_candidate_mask,
            )
            expected_schema = torch.einsum(
                "bc,bcd->bd",
                distribution,
                schema_summary,
            )

            # Factor meaning can change by relation step. This prevents a
            # multi-hop program from being forced to use one global direction
            # or one modifier setting for every edge in the sequence.
            factor_state_basis = state + expected_schema
            step_factor_expected_states: list[Tensor] = []
            for name, factor_schema in factor_schemas.items():
                step_matched = self.matcher(
                    query_hidden_states=query_hidden_states,
                    query_token_mask=query_token_mask,
                    schema=factor_schema,
                    query_remaining=remaining,
                )
                factor_state_query = F.normalize(
                    self.state_schema_score(factor_state_basis),
                    dim=-1,
                )
                factor_state_score = torch.einsum(
                    "bd,bcd->bc",
                    factor_state_query,
                    F.normalize(step_matched["schema_summary"], dim=-1),
                )
                step_logits = (
                    step_matched["score"]
                    + 0.25 * torch.tanh(factor_state_score)
                )
                name = str(name)
                step_logits, step_probability = self._masked_candidate_distribution(
                    step_logits,
                    resolved_factor_masks[name],
                )
                step_factor_scores[name].append(step_logits)
                step_factor_layer_weights[name].append(
                    step_matched["layer_weight"]
                )
                step_factor_distributions_lists[name].append(step_probability)
                step_factor_schema_evidence_lists[name].append(
                    step_matched["schema_evidence"]
                )
                step_factor_expected_states.append(
                    torch.einsum(
                        "bc,bcd->bd",
                        step_probability,
                        step_matched["schema_summary"],
                    )
                )

            step_factor_context, step_factor_bank_weight = self._aggregate_factor_context(
                state=factor_state_basis,
                factor_states=step_factor_expected_states,
            )
            step_factor_context_states.append(step_factor_context)
            step_factor_bank_weights.append(step_factor_bank_weight)
            factor_conditioned_state = self.factor_state_norm(
                state + step_factor_context
            )

            best = logits.max(dim=-1).values
            if relation_info["candidates"] > 1:
                top2 = logits.topk(2, dim=-1).values
                active_count = relation_candidate_mask.sum(dim=-1)
                margin = torch.where(
                    active_count > 1,
                    top2[:, 0] - top2[:, 1],
                    torch.ones_like(best),
                )
            else:
                margin = torch.ones_like(best)
            entropy = self._masked_entropy(
                distribution,
                relation_candidate_mask,
            )
            event_input = torch.cat(
                [
                    factor_conditioned_state,
                    best[:, None],
                    margin[:, None],
                    entropy[:, None],
                ],
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
            relation_schema_states.append(schema_summary)
            relation_query_evidence.append(matched["query_evidence"])
            relation_schema_evidence.append(matched["schema_evidence"])

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

            query_context = self._query_state(query_hidden_states, query_token_mask)
            next_state = self.transition(
                torch.cat(
                    [
                        expected_schema,
                        step_factor_context,
                        query_context,
                    ],
                    dim=-1,
                ),
                state,
            )
            state = (
                continue_probability[:, None] * next_state
                + (1.0 - continue_probability[:, None]) * state
            )
            survival = survival * continue_probability

        step_factor_distributions = {
            name: torch.stack(values, dim=1)
            for name, values in step_factor_distributions_lists.items()
        }
        factor_distributions: dict[str, Tensor] = {}
        factor_scores: dict[str, Tensor] = {}
        factor_layer_weights: dict[str, Tensor] = {}
        factor_schema_evidence: dict[str, Tensor] = {}
        global_factor_expected_states: list[Tensor] = []
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
            factor_state_query = F.normalize(self.state_schema_score(state), dim=-1)
            factor_state_score = torch.einsum(
                "bd,bcd->bc",
                factor_state_query,
                F.normalize(matched["schema_summary"], dim=-1),
            )
            factor_logits = matched["score"] + 0.25 * torch.tanh(factor_state_score)
            name = str(name)
            factor_logits, factor_probability = self._masked_candidate_distribution(
                factor_logits,
                resolved_factor_masks[name],
            )
            factor_scores[name] = factor_logits
            factor_distributions[name] = factor_probability
            factor_layer_weights[name] = matched["layer_weight"]
            factor_schema_evidence[name] = matched["schema_evidence"]
            global_factor_expected_states.append(
                torch.einsum(
                    "bc,bcd->bd",
                    factor_probability,
                    matched["schema_summary"],
                )
            )

        factor_context_state, factor_bank_weight = self._aggregate_factor_context(
            state=state,
            factor_states=global_factor_expected_states,
        )
        if global_factor_expected_states:
            state = self.global_factor_transition(
                factor_context_state,
                state,
            )
            state = self.factor_state_norm(state)

        relation_distribution = torch.stack(relation_distributions, dim=1)
        relation_step_mass = torch.stack(relation_masses, dim=1)
        event_distribution = torch.stack(events, dim=1)
        stop_probability = torch.stack(stops, dim=1)
        unknown_probability = torch.stack(unknowns, dim=1)
        query_coverage = torch.stack(coverage_history, dim=1)

        relation_mask_by_step = relation_candidate_mask[:, None, :].expand_as(
            relation_distribution
        )
        relation_entropy = self._masked_entropy(
            relation_distribution,
            relation_mask_by_step,
        )
        unknown_mass = unknown_probability.sum(dim=1).clamp(0.0, 1.0)
        factor_uncertainty = []
        for name, probability in factor_distributions.items():
            factor_uncertainty.append(
                self._masked_entropy(
                    probability,
                    resolved_factor_masks[name],
                )
            )
        if factor_uncertainty:
            global_factor_u = torch.stack(
                factor_uncertainty,
                dim=-1,
            ).mean(dim=-1)
        else:
            global_factor_u = torch.zeros_like(unknown_mass)

        step_factor_uncertainty = []
        for name, probability in step_factor_distributions.items():
            step_mask = resolved_factor_masks[name][:, None, :].expand_as(
                probability
            )
            entropy_by_step = self._masked_entropy(
                probability,
                step_mask,
            )
            weighted = (
                entropy_by_step
                * relation_step_mass
            ).sum(dim=1) / relation_step_mass.sum(
                dim=1
            ).clamp_min(1.0e-6)
            step_factor_uncertainty.append(weighted)
        if step_factor_uncertainty:
            step_factor_u = torch.stack(
                step_factor_uncertainty,
                dim=-1,
            ).mean(dim=-1)
            factor_u = 0.5 * (global_factor_u + step_factor_u)
        else:
            factor_u = global_factor_u
        relation_u = (
            relation_entropy
            * relation_step_mass
        ).sum(dim=1) / relation_step_mass.sum(dim=1).clamp_min(1.0e-6)
        truncation_probability = survival.clamp(0.0, 1.0)
        uncertainty = 1.0 - (
            (1.0 - relation_u.clamp(0.0, 1.0))
            * (1.0 - factor_u.clamp(0.0, 1.0))
            * (1.0 - unknown_mass)
            * (1.0 - truncation_probability)
        )

        applicability = torch.sigmoid(self.applicability_head(state)).squeeze(-1)
        operator = SemanticOperatorState(
            relation_distribution=relation_distribution,
            relation_step_mass=relation_step_mass,
            event_distribution=event_distribution,
            stop_probability=stop_probability,
            unknown_probability=unknown_probability,
            truncation_probability=truncation_probability,
            continuous_state=state,
            applicability=applicability,
            query_coverage=query_coverage,
            uncertainty=uncertainty,
            factor_distributions=factor_distributions,
            step_factor_distributions=step_factor_distributions,
        )
        operator.validate(
            relation_count=relation_info["candidates"],
            model_dim=self.config.model_dim,
            query_tokens=query_tokens,
        )

        relation_query_evidence_tensor = torch.stack(
            relation_query_evidence,
            dim=1,
        )
        relation_schema_evidence_tensor = torch.stack(
            relation_schema_evidence,
            dim=1,
        )
        relation_active = relation_candidate_mask[
            :, None, :, None
        ].to(relation_query_evidence_tensor.dtype)
        relation_query_evidence_tensor = (
            relation_query_evidence_tensor * relation_active
        )
        relation_schema_evidence_tensor = (
            relation_schema_evidence_tensor * relation_active
        )

        masked_factor_schema_evidence = {
            name: value
            * resolved_factor_masks[name][:, :, None].to(value.dtype)
            for name, value in factor_schema_evidence.items()
        }
        masked_step_factor_schema_evidence = {
            name: torch.stack(values, dim=1)
            * resolved_factor_masks[name][:, None, :, None].to(
                values[0].dtype
            )
            for name, values in step_factor_schema_evidence_lists.items()
        }

        return {
            "operator": operator,
            "relation_logits": torch.stack(relation_scores, dim=1),
            "relation_layer_weights": torch.stack(layer_weights, dim=1),
            "relation_schema_states": torch.stack(relation_schema_states, dim=1),
            "relation_query_evidence": relation_query_evidence_tensor,
            "relation_schema_evidence": relation_schema_evidence_tensor,
            "factor_logits": factor_scores,
            "factor_schema_evidence": masked_factor_schema_evidence,
            "factor_layer_weights": factor_layer_weights,
            "relation_candidate_mask": relation_candidate_mask,
            "factor_candidate_masks": resolved_factor_masks,
            "step_factor_logits": {
                name: torch.stack(values, dim=1)
                for name, values in step_factor_scores.items()
            },
            "step_factor_layer_weights": {
                name: torch.stack(values, dim=1)
                for name, values in step_factor_layer_weights.items()
            },
            "step_factor_schema_evidence": masked_step_factor_schema_evidence,
            "factor_context_state": factor_context_state,
            "factor_bank_weight": factor_bank_weight,
            "step_factor_context_states": torch.stack(
                step_factor_context_states,
                dim=1,
            ),
            "step_factor_bank_weights": (
                torch.stack(step_factor_bank_weights, dim=1)
                if factor_schemas
                else torch.zeros(
                    batch,
                    max_steps,
                    0,
                    device=query_hidden_states.device,
                    dtype=query_hidden_states.dtype,
                )
            ),
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "relation_identity_parameters": 0,
            "factor_identity_parameters": 0,
            "candidate_count_dependent_parameters": 0,
            "per_example_candidate_subset_supported": True,
            "masked_candidate_uncertainty_normalization": True,
            "runtime_step_count_dependent_parameters": 0,
            "type_vocabulary_dependent_parameters": 0,
            "relation_count_ceiling": None,
            "factor_count_ceiling": None,
            "runtime_step_count_ceiling": None,
            "type_vocabulary_ceiling": None,
            "shared_query_schema_token_interaction": True,
            "candidate_conditioned_multilayer_read": True,
            "content_conditioned_query_pooling": True,
            "continuous_relation_hypotheses": True,
            "runtime_step_truncation_exposed": True,
            "silent_program_truncation_forbidden": True,
            "operator_probability_contract_fail_closed": True,
            "recurrent_survival_mass_conservation_checked": True,
            "exact_structural_sparsity": False,
            "semantic_backbone_gradient_can_flow": True,
            "factor_scorer_shared_across_schema_banks": True,
            "runtime_factor_bank_set_aggregation": True,
            "semantic_factor_context_in_continuous_state": True,
            "semantic_factor_bank_count_ceiling": None,
            "step_conditioned_factor_semantics": True,
            "bidirectional_token_evidence_exposed": True,
            "relation_schema_token_evidence_exposed": True,
            "factor_schema_token_evidence_exposed": True,
            "step_factor_schema_token_evidence_exposed": True,
            "inactive_candidate_token_evidence_zeroed": True,
            "step_factor_uncertainty_supervised_in_state": True,
            "token_interaction_chunk_is_operating_point": True,
            "query_token_count_ceiling": None,
            "schema_token_count_ceiling": None,
        }
