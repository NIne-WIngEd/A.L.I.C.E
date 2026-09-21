from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.qsre_mechanics import masked_sparsemax
from alice_personality.n0.qsre_production_core import (
    QSREDynamicRelationSchema,
    QSREProductionConfig,
    QSREProductionOperatorState,
)


EVENT_CONTINUE = 0
EVENT_STOP = 1
EVENT_UNKNOWN = 2

FACTOR_ROLE = 0
FACTOR_TRAVERSAL = 1
FACTOR_DIRECTION = 2
FACTOR_MODIFIER = 3
FACTOR_CONTROL = 4
FACTOR_APPLICABILITY = 5
FACTOR_COUNT = 6


def _masked_mean(value: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    numerator = (value * weight).sum(dim=dim)
    denominator = weight.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class QSREProductionOperatorInducerV3(nn.Module):
    """Open-schema query-conditioned operator inducer.

    The design separates three concerns that must not compete in one classifier:

    1. relation-schema matching,
    2. program continuation / STOP / UNKNOWN,
    3. non-relational operator factors such as role, traversal and direction.

    Relation matching uses one shared semantic metric on both query and runtime
    schema text. No learned parameter is indexed by relation identity or relation
    count. The runtime schema can therefore grow without changing topology.

    STOP/UNKNOWN never compete with relation candidates. Their probabilities
    come from a separate 3-way event distribution and receive cardinality-
    invariant semantic-match confidence features. This prevents relation-logit
    scale or candidate count from suppressing fail-closed behavior.

    Factor-specific query slots keep role/traversal/direction/modifier/control\n    prediction from collapsing onto the terminal relation-decoder state.\n\n    Ordered relation programs additionally maintain a differentiable query-token\n    coverage state. Evidence strongly attended at one relation step is softly\n    consumed for the next step. The same recurrent cell is reused at every\n    step, so this introduces no hop-count parameter axis and does not assume a\n    fixed left-to-right language order.\n    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        # Shared query/schema semantic metric. This exact module is applied to
        # both query hidden states and raw runtime-schema hidden states.
        self.query_norm = nn.LayerNorm(config.semantic_dim)
        self.query_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.layer_embedding = nn.Embedding(config.num_hidden_states, d)

        # One program slot plus independent factor slots. All attend to the same
        # full hidden-state stack, but each factor receives its own latent query.
        self.program_query = nn.Parameter(torch.empty(d))
        self.factor_queries = nn.Parameter(torch.empty(FACTOR_COUNT, d))
        self.query_cross_attention = nn.MultiheadAttention(
            d,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_attention_heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_refiner = nn.TransformerEncoder(
            layer,
            num_layers=config.operator_refinement_layers,
        )

        # Shared recurrent program cell: no hop-count-dependent parameters.
        self.step_query = nn.Linear(d, d, bias=False)
        self.step_transition = nn.GRUCell(2 * d, d)

        # Relation distribution and event distribution are intentionally
        # decoupled. The relation scale cannot suppress STOP or UNKNOWN.
        self.relation_logit_scale = nn.Parameter(torch.tensor(2.0))
        self.continue_head = nn.Linear(d, 1)
        self.stop_head = nn.Linear(d, 1)
        self.unknown_head = nn.Linear(d, 1)

        # Confidence features are [best semantic match, best-vs-second margin].
        # They are relation-cardinality independent statistics, not relation IDs.
        self.event_match_projection = nn.Linear(2, 3, bias=False)

        self.role_head = nn.Linear(d, config.role_count)
        self.traversal_head = nn.Linear(d, config.traversal_count)
        self.direction_head = nn.Linear(d, config.direction_count)
        self.modifier_head = nn.Linear(d, config.modifier_count)
        self.control_head = nn.Linear(d, config.control_count)
        self.applicability_head = nn.Linear(d, 1)

        # Continuous state fuses the program state and all dedicated factor
        # states without compressing below the production model width.
        self.continuous_projection = nn.Linear(2 * d, d)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.config.semantic_dim == self.config.model_dim:
            nn.init.eye_(self.query_projection.weight)
        else:
            nn.init.orthogonal_(self.query_projection.weight)
        nn.init.zeros_(self.layer_embedding.weight)
        nn.init.normal_(self.program_query, mean=0.0, std=0.02)
        nn.init.normal_(self.factor_queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.step_query.weight)
        for head in (
            self.continue_head,
            self.stop_head,
            self.unknown_head,
            self.role_head,
            self.traversal_head,
            self.direction_head,
            self.modifier_head,
            self.control_head,
            self.applicability_head,
            self.continuous_projection,
        ):
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        # Sensible fail-closed initialization. Strong semantic match and margin
        # favor CONTINUE; weak match favors UNKNOWN. STOP is controlled by the
        # recurrent program state rather than candidate-set scale.
        with torch.no_grad():
            self.event_match_projection.weight.zero_()
            self.event_match_projection.weight[EVENT_CONTINUE] = torch.tensor(
                [1.0, 0.5]
            )
            self.event_match_projection.weight[EVENT_UNKNOWN] = torch.tensor(
                [-1.0, -0.5]
            )

    def project_semantic(self, states: Tensor) -> Tensor:
        if states.ndim != 4:
            raise ValueError("semantic states must be [B_or_R,L,T,D]")
        count, layers, tokens, width = states.shape
        if layers != self.config.num_hidden_states:
            raise ValueError("semantic hidden-state depth drift")
        if width != self.config.semantic_dim:
            raise ValueError("semantic width drift")
        projected = self.query_projection(self.query_norm(states))
        layer_ids = torch.arange(layers, device=states.device)
        return projected + self.layer_embedding(layer_ids).view(
            1, layers, 1, self.config.model_dim
        )

    def _initial_slots(
        self,
        query_memory: Tensor,
        flat_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = query_memory.size(0)
        seed = torch.cat(
            [self.program_query.unsqueeze(0), self.factor_queries],
            dim=0,
        ).unsqueeze(0).expand(batch, -1, -1)
        attended, attention = self.query_cross_attention(
            seed,
            query_memory,
            query_memory,
            key_padding_mask=~flat_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        slots = self.slot_norm(seed + attended)
        slots = self.slot_refiner(slots)
        program_state = slots[:, 0]
        factor_states = slots[:, 1:]
        program_attention = attention[:, 0]
        return program_state, factor_states, program_attention

    def _semantic_match(
        self,
        *,
        state: Tensor,
        query_projected: Tensor,
        query_attention: Tensor,
        query_token_mask: Tensor,
        schema_match_projected: Tensor,
        schema_match_summary: Tensor,\n        schema_token_mask: Tensor,\n        query_remaining: Tensor,\n    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, layers, query_tokens, _ = query_projected.shape
        relations, schema_layers, schema_tokens, _ = schema_match_projected.shape
        if schema_layers != layers:
            raise ValueError("query/schema layer mismatch")

        q = F.normalize(query_projected, dim=-1)
        s = F.normalize(schema_match_projected, dim=-1)
        similarity = torch.einsum("bltd,rlsd->brlts", q, s)

        schema_valid = schema_token_mask[None, :, None, None, :].expand(
            batch,
            relations,
            layers,
            query_tokens,
            schema_tokens,
        )
        query_valid = query_token_mask[:, None, None, :, None].expand(
            batch,
            relations,
            layers,
            query_tokens,
            schema_tokens,
        )
        valid = schema_valid & query_valid
        similarity = similarity.masked_fill(~valid, -1.0e4)\n\n        # Soft evidence consumption. A token heavily consumed by an earlier\n        # relation step remains available through a nonzero floor, but it no\n        # longer dominates every later step. This is position-agnostic and\n        # therefore does not impose a left-to-right parser.\n        remaining = query_remaining.clamp(min=0.05, max=1.0)\n        similarity = similarity + torch.log(remaining)[:, None, None, :, None]\n\n        # Query -> schema late interaction, weighted by current step attention.
        q_to_s = similarity.max(dim=-1).values
        q_weight = query_attention.reshape(batch, layers, query_tokens)\n        q_weight = q_weight * query_token_mask[:, None, :].to(q_weight.dtype)\n        q_weight = q_weight * remaining[:, None, :]\n        q_weight = q_weight / q_weight.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-12)
        q_to_s_score = torch.einsum("blt,brlt->br", q_weight, q_to_s)

        # Schema -> query late interaction keeps the relation description itself
        # from being reduced to whichever one query token happened to match.
        s_to_q = similarity.max(dim=-2).values
        s_valid = schema_token_mask[None, :, None, :].expand(
            batch, relations, layers, schema_tokens
        )
        s_to_q = s_to_q.masked_fill(~s_valid, 0.0)
        s_den = s_valid.to(s_to_q.dtype).sum(dim=(2, 3)).clamp_min(1.0)
        s_to_q_score = s_to_q.sum(dim=(2, 3)) / s_den

        grounded = 0.5 * (q_to_s_score + s_to_q_score)

        # Recurrent state provides ordered-program context, but it is only a
        # bounded residual on top of the shared text-to-schema match.
        state_score = torch.einsum(
            "bd,rd->br",
            F.normalize(self.step_query(state), dim=-1),
            F.normalize(schema_match_summary, dim=-1),
        )
        semantic_score = 0.8 * grounded + 0.2 * state_score
        scale = self.relation_logit_scale.exp().clamp(max=100.0)
        return scale * semantic_score, semantic_score, q_to_s

    def schema_identity_logits(self, schema: QSREDynamicRelationSchema) -> Tensor:
        """Schema-only self-calibration logits.

        Each runtime relation description is treated as a pseudo-query against
        all runtime relation descriptions. No DEV query/example labels enter.
        """
        info = schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        projected = self.project_semantic(schema.token_states)
        relation_count, layers, tokens, _ = projected.shape
        normalized = F.normalize(projected, dim=-1)
        similarity = torch.einsum(
            "altd,rlsd->arlts",
            normalized,
            normalized,
        )
        q_valid = schema.token_mask[:, None, None, :, None].expand(
            relation_count,
            relation_count,
            layers,
            tokens,
            tokens,
        )
        s_valid = schema.token_mask[None, :, None, None, :].expand(
            relation_count,
            relation_count,
            layers,
            tokens,
            tokens,
        )
        valid = q_valid & s_valid
        similarity = similarity.masked_fill(~valid, -1.0e4)

        q_to_s = similarity.max(dim=-1).values
        qmask = schema.token_mask[:, None, :, None].expand(
            relation_count, layers, tokens, 1
        ).squeeze(-1)
        qden = qmask.to(q_to_s.dtype).sum(dim=(1, 2)).clamp_min(1.0)
        qscore = (
            q_to_s
            * qmask[:, None, :, :].to(q_to_s.dtype)
        ).sum(dim=(2, 3)) / qden[:, None]

        s_to_q = similarity.max(dim=-2).values
        smask = schema.token_mask[None, :, None, :].expand(
            relation_count, relation_count, layers, tokens
        )
        sden = smask.to(s_to_q.dtype).sum(dim=(2, 3)).clamp_min(1.0)
        sscore = (
            s_to_q * smask.to(s_to_q.dtype)
        ).sum(dim=(2, 3)) / sden

        scale = self.relation_logit_scale.exp().clamp(max=100.0)
        return scale * 0.5 * (qscore + sscore)

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
        del schema_token_state  # Executor-space schema states are not matcher authority.
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        if query_token_mask.ndim != 2 or query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool [B,T]")

        query_projected = self.project_semantic(query_hidden_states)
        batch, layers, tokens, _ = query_projected.shape
        if query_token_mask.shape != (batch, tokens):
            raise ValueError("query token-mask shape drift")
        if bool((query_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every query requires at least one valid token")

        schema_info = schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        relations = schema_info["relations"]
        if schema_relation_state.shape != (relations, self.config.model_dim):
            raise ValueError("schema relation-state shape drift")
        schema_match_projected = self.project_semantic(schema.token_states)
        schema_match_mask = schema.token_mask[:, None, :, None].to(
            schema_match_projected.dtype
        )
        schema_match_per_layer = (
            schema_match_projected * schema_match_mask
        ).sum(dim=2) / schema_match_mask.sum(dim=2).clamp_min(1.0)
        schema_match_summary = schema_match_per_layer.mean(dim=1)

        memory = query_projected.reshape(batch, layers * tokens, -1)
        flat_valid = (
            query_token_mask[:, None, :]
            .expand(batch, layers, tokens)
            .reshape(batch, layers * tokens)
        )
        state, factor_states, initial_attention = self._initial_slots(
            memory,
            flat_valid,
        )

        relation_steps: list[Tensor] = []
        relation_logits_steps: list[Tensor] = []
        relation_mass_steps: list[Tensor] = []
        stop_steps: list[Tensor] = []
        unknown_steps: list[Tensor] = []
        event_steps: list[Tensor] = []
        event_logits_steps: list[Tensor] = []
        state_steps: list[Tensor] = []
        attention_steps: list[Tensor] = []
        semantic_score_steps: list[Tensor] = []\n        coverage_steps: list[Tensor] = []\n        remaining_steps: list[Tensor] = []\n\n        survival = torch.ones(batch, device=state.device, dtype=state.dtype)\n        query_coverage = torch.zeros(\n            batch, tokens, device=state.device, dtype=state.dtype\n        )\n\n        for _ in range(max_steps):\n            query_remaining = (1.0 - query_coverage).clamp(min=0.05, max=1.0)\n            remaining_flat = (\n                query_remaining[:, None, :]\n                .expand(batch, layers, tokens)\n                .reshape(batch, layers * tokens)\n            )\n            step_memory = memory * remaining_flat.unsqueeze(-1)\n            attended, attention = self.query_cross_attention(\n                state.unsqueeze(1),\n                step_memory,\n                step_memory,
                key_padding_mask=~flat_valid,
                need_weights=True,
                average_attn_weights=True,
            )
            candidate = self.slot_norm(state + attended.squeeze(1))
            step_state = (
                survival[:, None] * candidate
                + (1.0 - survival[:, None]) * state
            )
            relation_logits, semantic_score, query_relation_evidence = self._semantic_match(\n                state=step_state,
                query_projected=query_projected,
                query_attention=attention.squeeze(1),
                query_token_mask=query_token_mask,
                schema_match_projected=schema_match_projected,
                schema_match_summary=schema_match_summary,\n                schema_token_mask=schema.token_mask,\n                query_remaining=query_remaining,\n            )

            relation_mask = torch.ones_like(relation_logits, dtype=torch.bool)
            relation_distribution = masked_sparsemax(
                relation_logits,
                relation_mask,
                dim=-1,
            )
            relation_distribution = relation_distribution / relation_distribution.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-12)

            if relations > 1:
                top = semantic_score.topk(k=2, dim=-1).values
                best = top[:, 0]
                margin = top[:, 0] - top[:, 1]
            else:
                best = semantic_score[:, 0]
                margin = torch.ones_like(best)
            match_features = torch.stack([best, margin], dim=-1)

            base_event_logits = torch.cat(
                [
                    self.continue_head(step_state),
                    self.stop_head(step_state),
                    self.unknown_head(step_state),
                ],
                dim=-1,
            )
            event_logits = (
                base_event_logits
                + self.event_match_projection(match_features)
            )
            event_probability = torch.softmax(event_logits, dim=-1)
            continue_probability = event_probability[:, EVENT_CONTINUE]
            stop_probability = event_probability[:, EVENT_STOP]
            unknown_probability = event_probability[:, EVENT_UNKNOWN]

            effective_mass = survival * continue_probability
            effective_stop = survival * stop_probability
            effective_unknown = survival * unknown_probability

            relation_steps.append(relation_distribution)
            relation_logits_steps.append(relation_logits)
            relation_mass_steps.append(effective_mass)
            stop_steps.append(effective_stop)
            unknown_steps.append(effective_unknown)
            event_steps.append(event_probability)
            event_logits_steps.append(event_logits)
            state_steps.append(step_state)
            semantic_score_steps.append(semantic_score)\n            step_attention = attention.squeeze(1).reshape(batch, layers, tokens)\n            attention_steps.append(step_attention)\n\n            # Consume the token evidence used by this relation step. Attention\n            # is aggregated across semantic layers so all representations of a\n            # token position are jointly downweighted at the next step. The\n            # relation evidence term prevents generic high-attention tokens from\n            # being consumed solely because they are globally salient.\n            relation_evidence = torch.einsum(\n                "br,brlt->blt",\n                relation_distribution,\n                query_relation_evidence,\n            )\n            relation_evidence = torch.sigmoid(relation_evidence).mean(dim=1)\n            token_attention = step_attention.sum(dim=1)\n            consume_score = token_attention * relation_evidence\n            consume_score = consume_score / consume_score.amax(\n                dim=-1, keepdim=True\n            ).clamp_min(1.0e-12)\n            consume_score = (\n                consume_score\n                * continue_probability[:, None]\n                * query_token_mask.to(consume_score.dtype)\n            ).clamp(0.0, 1.0)\n            query_coverage = 1.0 - (\n                (1.0 - query_coverage) * (1.0 - consume_score)\n            )\n            query_coverage = query_coverage.clamp(0.0, 1.0)\n            coverage_steps.append(query_coverage)\n            remaining_steps.append((1.0 - query_coverage).clamp(0.05, 1.0))\n\n            expected_relation = torch.einsum(
                "br,rd->bd",
                relation_distribution,
                schema_match_summary,
            )
            next_state = self.step_transition(
                torch.cat([expected_relation, attended.squeeze(1)], dim=-1),
                step_state,
            )
            state = (
                continue_probability[:, None] * next_state
                + (1.0 - continue_probability[:, None]) * step_state
            )
            survival = survival * continue_probability

        relation_distribution = torch.stack(relation_steps, dim=1)
        relation_logits = torch.stack(relation_logits_steps, dim=1)
        relation_step_mass = torch.stack(relation_mass_steps, dim=1)
        stop_probability = torch.stack(stop_steps, dim=1)
        unknown_probability = torch.stack(unknown_steps, dim=1)
        event_distribution = torch.stack(event_steps, dim=1)
        event_logits = torch.stack(event_logits_steps, dim=1)
        step_state_history = torch.stack(state_steps, dim=1)
        step_attention = torch.stack(attention_steps, dim=1)
        semantic_scores = torch.stack(semantic_score_steps, dim=1)\n        step_query_coverage = torch.stack(coverage_steps, dim=1)\n        step_query_remaining = torch.stack(remaining_steps, dim=1)\n
        role_state = factor_states[:, FACTOR_ROLE]
        traversal_state = factor_states[:, FACTOR_TRAVERSAL]
        direction_state = factor_states[:, FACTOR_DIRECTION]
        modifier_state = factor_states[:, FACTOR_MODIFIER]
        control_state = factor_states[:, FACTOR_CONTROL]
        applicability_state = factor_states[:, FACTOR_APPLICABILITY]

        role_logits = self.role_head(role_state)
        traversal_logits = self.traversal_head(traversal_state)
        direction_logits = self.direction_head(direction_state)
        modifier_logits = self.modifier_head(modifier_state)
        control_logits = self.control_head(control_state)
        applicability_logit = self.applicability_head(
            applicability_state
        ).squeeze(-1)

        role_distribution = torch.softmax(role_logits, dim=-1)
        traversal_distribution = torch.softmax(
            traversal_logits,
            dim=-1,
        )
        direction_distribution = torch.softmax(
            direction_logits,
            dim=-1,
        )
        modifier_weight = torch.sigmoid(modifier_logits)
        control_distribution = torch.softmax(
            control_logits,
            dim=-1,
        )
        applicability = torch.sigmoid(applicability_logit)

        factor_summary = factor_states.mean(dim=1)
        continuous_state = self.continuous_projection(
            torch.cat([state, factor_summary], dim=-1)
        )

        relation_entropy = -(
            relation_distribution.clamp_min(1.0e-12)
            * relation_distribution.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        relation_entropy = (
            relation_entropy * relation_step_mass
        ).sum(dim=-1) / relation_step_mass.sum(dim=-1).clamp_min(1.0e-6)
        relation_entropy = relation_entropy / max(
            math.log(float(relations)),
            1.0e-6,
        )

        def normalized_entropy(probability: Tensor) -> Tensor:
            count = probability.size(-1)
            return -(
                probability.clamp_min(1.0e-12)
                * probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1) / max(math.log(float(count)), 1.0e-6)

        unknown_event_mass = unknown_probability.sum(dim=1).clamp(
            min=0.0,
            max=1.0,
        )
        uncertainty_components = torch.stack(
            [
                relation_entropy.clamp(0.0, 1.0),
                normalized_entropy(role_distribution).clamp(0.0, 1.0),
                normalized_entropy(traversal_distribution).clamp(0.0, 1.0),
                normalized_entropy(direction_distribution).clamp(0.0, 1.0),
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
        operator.validate(
            relation_count=relations,
            model_dim=self.config.model_dim,
        )

        return {
            "operator": operator,
            "relation_logits": relation_logits,
            "event_logits": event_logits,
            "event_distribution": event_distribution,
            "factor_logits": {
                "role": role_logits,
                "traversal": traversal_logits,
                "direction": direction_logits,
                "modifier": modifier_logits,
                "control": control_logits,
                "applicability": applicability_logit,
            },
            "semantic_relation_score": semantic_scores,
            "initial_query_attention": initial_attention.reshape(
                batch, layers, tokens
            ),
            "step_query_attention": step_attention,
            "step_state_history": step_state_history,\n            "step_query_coverage": step_query_coverage,\n            "step_query_remaining": step_query_remaining,\n            "factor_state": factor_states,
            "schema_relation_state": schema_relation_state,
            "schema_match_token_state": schema_match_projected,
            "schema_match_relation_state": schema_match_summary,
        }

    def parameter_report(self) -> dict[str, int | bool | None]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "relation_count_dependent_parameters": 0,
            "hop_count_dependent_parameters": 0,
            "runtime_dynamic_relation_schema": True,
            "shared_query_schema_metric": True,
            "symmetric_late_interaction": True,
            "p1_schema_encoder_is_not_relation_match_authority": True,
            "p1_schema_relation_state_is_interface_only_for_operator": True,
            "relation_selection_decoupled_from_stop_unknown": True,
            "dense_relation_logits_exposed_for_trainability": True,
            "dense_event_and_factor_logits_exposed_for_trainability": True,
            "cardinality_invariant_termination_event_head": True,
            "match_confidence_guides_unknown_rejection": True,
            "factor_specific_query_slots": True,
            "continuous_operator_state": True,
            "role_count_ceiling": None,
            "traversal_count_ceiling": None,
            "direction_count_ceiling": None,
            "control_count_ceiling": None,
            "relation_count_ceiling": None,
            "runtime_step_count_ceiling": None,
        }
