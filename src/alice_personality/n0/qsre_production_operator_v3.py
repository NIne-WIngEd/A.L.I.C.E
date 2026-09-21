from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.qsre_production_core import (
    QSREDynamicRelationSchema,
    QSREProductionConfig,
    QSREProductionOperatorState,
)
from alice_personality.n0.qsre_schema_matcher import QSRESchemaMatcher
from alice_personality.n0.qsre_semantic_authority import (
    combine_authority_components,
)


EVENT_CONTINUE = 0
EVENT_STOP = 1
EVENT_UNKNOWN = 2
EVENT_COUNT = 3


class QSREProductionOperatorInducerV3(nn.Module):
    """N0 operator using frozen semantic authority plus ordered evidence.

    Relation and factor identity scores are supplied by a separately qualified,
    zero-gradient semantic authority derived from the ratified N0 semantic
    model. The local schema matcher is parameter-free and only exposes
    candidate-conditioned token evidence and schema summaries.

    Production P2 may learn ordering, termination, applicability and continuous
    execution state, but it cannot rewrite relation or factor semantics.
    Relation hypotheses remain continuous until structural Binder v2.
    """

    def __init__(self, config: QSREProductionConfig) -> None:
        super().__init__()
        self.config = config
        d = int(config.model_dim)

        self.schema_matcher = QSRESchemaMatcher(
            semantic_dim=int(config.semantic_dim),
            model_dim=d,
            num_hidden_states=int(config.num_hidden_states),
        )

        self.program_query = nn.Parameter(torch.empty(1, 1, d))
        self.query_cross_attention = nn.MultiheadAttention(
            d,
            int(config.num_attention_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(d)
        self.step_query = nn.Linear(d, d, bias=False)
        self.step_transition = nn.GRUCell(2 * d, d)

        self.continue_head = nn.Linear(d, 1)
        self.stop_head = nn.Linear(d, 1)
        self.unknown_head = nn.Linear(d, 1)
        self.event_match_projection = nn.Linear(2, EVENT_COUNT)

        self.applicability_head = nn.Linear(2 * d, 1)
        self.continuous_projection = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        self._factor_cache: dict[str, Any] | None = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(
            self.program_query,
            mean=0.0,
            std=self.config.model_dim ** -0.5,
        )
        nn.init.zeros_(self.event_match_projection.weight)
        nn.init.zeros_(self.event_match_projection.bias)

    def project_semantic(self, states: Tensor) -> Tensor:
        return self.schema_matcher.project(states)

    def configure_factor_schema_cache(self, cache: dict[str, Any]) -> None:
        if cache.get("schema") != "alice.eipm.n0.qsre-frozen-factor-schema-cache.v3":
            raise ValueError("frozen factor-schema cache version drift")
        if cache.get("gradient") is not False or cache.get("optimizer") is not False:
            raise ValueError("factor-schema cache must be zero-gradient")
        if cache.get("private_identity_data") is not False:
            raise ValueError("private identity data entered factor schema cache")

        expected = {
            "role": int(self.config.role_count),
            "traversal": int(self.config.traversal_count),
            "direction": int(self.config.direction_count),
            "control": int(self.config.control_count),
        }
        for category, count in expected.items():
            row = cache.get(category)
            if not isinstance(row, dict):
                raise ValueError(f"missing factor schema category {category}")
            states = row.get("token_states")
            mask = row.get("token_mask")
            if not isinstance(states, Tensor) or states.ndim != 4:
                raise ValueError(f"{category}: factor token-state geometry drift")
            if states.size(0) != count:
                raise ValueError(
                    f"{category}: factor count {states.size(0)} != config {count}"
                )
            if not isinstance(mask, Tensor) or mask.shape != states.shape[:1] + states.shape[2:3]:
                raise ValueError(f"{category}: factor token-mask geometry drift")

        modifiers = cache.get("modifiers")
        if not isinstance(modifiers, dict):
            raise ValueError("missing modifier factor schema")
        modifier_states = modifiers.get("token_states")
        modifier_mask = modifiers.get("token_mask")
        if not isinstance(modifier_states, Tensor) or modifier_states.ndim != 5:
            raise ValueError("modifier factor token-state geometry drift")
        if modifier_states.size(0) != int(self.config.modifier_count):
            raise ValueError("modifier factor count drift")
        if modifier_states.size(1) != 2:
            raise ValueError("every modifier requires OFF/ON semantic states")
        if (
            not isinstance(modifier_mask, Tensor)
            or modifier_mask.shape
            != (
                modifier_states.size(0),
                2,
                modifier_states.size(3),
            )
        ):
            raise ValueError("modifier token-mask geometry drift")
        self._factor_cache = cache

    def _factor_row(
        self,
        category: str,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if self._factor_cache is None:
            raise RuntimeError(
                "closure factor schema cache must be configured before operator use"
            )
        row = self._factor_cache[category]
        return (
            row["token_states"].to(device=device, dtype=torch.float32),
            row["token_mask"].to(device=device).bool(),
        )

    def _initial_program_state(
        self,
        memory: Tensor,
        flat_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = memory.size(0)
        slot = self.program_query.expand(batch, -1, -1)
        attended, attention = self.query_cross_attention(
            slot,
            memory,
            memory,
            key_padding_mask=~flat_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        state = self.slot_norm(slot + attended).squeeze(1)
        return state, attention.squeeze(1)

    @staticmethod
    def _validate_authority_pair(
        authority: dict[str, Tensor],
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> None:
        for key in ("joint_preference", "semantic_projection"):
            value = authority.get(key)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} frozen semantic authority {key} shape drift"
                )
            if value.requires_grad:
                raise ValueError(
                    f"{name} frozen semantic authority unexpectedly requires gradient"
                )

    def _factor_match(
        self,
        *,
        query_projected: Tensor,
        query_token_mask: Tensor,
        category: str,
        authority: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        raw, mask = self._factor_row(
            category,
            device=query_projected.device,
        )
        matched = self.schema_matcher.match_projected(
            query_projected=query_projected,
            query_token_mask=query_token_mask,
            schema_projected=self.schema_matcher.project(raw),
            schema_token_mask=mask,
        )
        expected = (query_projected.size(0), raw.size(0))
        self._validate_authority_pair(
            authority,
            shape=expected,
            name=category,
        )
        matched["authority_score"] = combine_authority_components(
            joint_preference=authority["joint_preference"],
            semantic_projection=authority["semantic_projection"],
            token_evidence=matched["token_score"],
        )
        return matched

    def schema_identity_logits(
        self,
        schema: QSREDynamicRelationSchema,
    ) -> Tensor:
        schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        projected = self.schema_matcher.project(schema.token_states)
        return self.schema_matcher.match_projected(
            query_projected=projected,
            query_token_mask=schema.token_mask,
            schema_projected=projected,
            schema_token_mask=schema.token_mask,
        )["logits"]

    @staticmethod
    def _expected(
        probability: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        return torch.einsum("bc,cd->bd", probability, candidates)

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        schema: QSREDynamicRelationSchema,
        schema_token_state: Tensor,
        schema_relation_state: Tensor,
        relation_authority: dict[str, Tensor],
        factor_authority: dict[str, dict[str, Tensor]],
        max_steps: int,
    ) -> dict[str, Tensor | QSREProductionOperatorState]:
        del schema_token_state
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        if query_token_mask.ndim != 2 or query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool [B,T]")
        if self._factor_cache is None:
            raise RuntimeError("factor schema cache is not configured")

        batch, layers, tokens, semantic_dim = query_hidden_states.shape
        if layers != int(self.config.num_hidden_states):
            raise ValueError("query hidden-state depth drift")
        if semantic_dim != int(self.config.semantic_dim):
            raise ValueError("query semantic width drift")
        if query_token_mask.shape != (batch, tokens):
            raise ValueError("query token-mask shape drift")
        if bool((query_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every query requires at least one valid token")

        schema_info = schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        relations = int(schema_info["relations"])
        if schema_relation_state.shape != (relations, self.config.model_dim):
            raise ValueError("schema relation-state shape drift")
        self._validate_authority_pair(
            relation_authority,
            shape=(batch, relations),
            name="relation",
        )
        for category, count in (
            ("role", int(self.config.role_count)),
            ("traversal", int(self.config.traversal_count)),
            ("direction", int(self.config.direction_count)),
            ("control", int(self.config.control_count)),
        ):
            if category not in factor_authority:
                raise ValueError(f"missing frozen factor authority: {category}")
            self._validate_authority_pair(
                factor_authority[category],
                shape=(batch, count),
                name=category,
            )
        modifier_authority = factor_authority.get("modifiers")
        if not isinstance(modifier_authority, dict):
            raise ValueError("missing frozen modifier authority")
        for key in ("joint_preference", "semantic_projection"):
            value = modifier_authority.get(key)
            expected = (batch, int(self.config.modifier_count), 2)
            if not isinstance(value, Tensor) or tuple(value.shape) != expected:
                raise ValueError(f"modifier authority {key} shape drift")
            if value.requires_grad:
                raise ValueError("modifier frozen semantic authority requires gradient")

        query_projected = self.schema_matcher.project(query_hidden_states)
        relation_schema_projected = self.schema_matcher.project(
            schema.token_states
        )
        memory = query_projected.reshape(batch, layers * tokens, -1)
        flat_valid = (
            query_token_mask[:, None, :]
            .expand(batch, layers, tokens)
            .reshape(batch, layers * tokens)
        )
        state, initial_attention = self._initial_program_state(
            memory,
            flat_valid,
        )

        # Ordered-evidence state. It is token-position based and shared across
        # semantic layers; it is not a left-to-right pointer. A nonzero floor
        # lets later steps recover evidence if an earlier soft hypothesis was
        # wrong.
        query_coverage = torch.zeros(
            batch,
            tokens,
            device=query_hidden_states.device,
            dtype=query_projected.dtype,
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
        semantic_score_steps: list[Tensor] = []
        coverage_steps: list[Tensor] = []

        survival = torch.ones(
            batch,
            device=state.device,
            dtype=state.dtype,
        )

        for _ in range(max_steps):
            remaining = (1.0 - query_coverage).clamp(min=0.05, max=1.0)
            remaining_by_layer = (
                remaining[:, None, :]
                .expand(batch, layers, tokens)
                .reshape(batch, layers * tokens)
            )
            memory_step = memory * remaining_by_layer.unsqueeze(-1)

            attended, attention = self.query_cross_attention(
                state.unsqueeze(1),
                memory_step,
                memory_step,
                key_padding_mask=~flat_valid,
                need_weights=True,
                average_attn_weights=True,
            )
            candidate = self.slot_norm(state + attended.squeeze(1))
            step_state = (
                survival[:, None] * candidate
                + (1.0 - survival[:, None]) * state
            )
            query_prior = attention.squeeze(1).reshape(
                batch,
                layers,
                tokens,
            )

            matched = self.schema_matcher.match_projected(
                query_projected=query_projected,
                query_token_mask=query_token_mask,
                schema_projected=relation_schema_projected,
                schema_token_mask=schema.token_mask,
                query_remaining=remaining,
                query_prior=query_prior,
            )
            state_score = torch.einsum(
                "bd,rd->br",
                F.normalize(self.step_query(step_state), dim=-1),
                F.normalize(matched["schema_summary"], dim=-1),
            )
            # Frozen semantic authority owns relation identity. The
            # parameter-free token component is recomputed after every coverage
            # update. Remaining evidence and recurrent state may order a
            # program, but cannot learn a replacement relation ontology.
            authority_score = combine_authority_components(
                joint_preference=relation_authority["joint_preference"],
                semantic_projection=relation_authority["semantic_projection"],
                token_evidence=matched["token_score"],
            )
            semantic_score = authority_score
            relation_logits = (
                authority_score
                + torch.log(matched["remaining_support"].clamp_min(0.05))
                + 0.25 * torch.tanh(state_score)
            )
            relation_distribution = torch.softmax(
                relation_logits,
                dim=-1,
            )

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
            semantic_score_steps.append(semantic_score)
            attention_steps.append(query_prior)

            relation_evidence = torch.einsum(
                "br,brlt->blt",
                relation_distribution,
                matched["query_evidence"],
            )
            evidence_by_token = relation_evidence.sum(dim=1)
            evidence_by_token = (
                evidence_by_token
                / evidence_by_token.amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
            )
            evidence_by_token = (
                evidence_by_token
                * query_token_mask.to(evidence_by_token.dtype)
            )
            consumed = (
                continue_probability[:, None]
                * evidence_by_token
            ).clamp(0.0, 1.0)
            query_coverage = 1.0 - (
                (1.0 - query_coverage) * (1.0 - consumed)
            )
            query_coverage = query_coverage.clamp(0.0, 1.0)
            coverage_steps.append(query_coverage)

            expected_relation = torch.einsum(
                "br,rd->bd",
                relation_distribution,
                matched["schema_summary"],
            )
            next_state = self.step_transition(
                torch.cat(
                    [expected_relation, attended.squeeze(1)],
                    dim=-1,
                ),
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
        semantic_scores = torch.stack(semantic_score_steps, dim=1)
        query_coverage_history = torch.stack(coverage_steps, dim=1)

        role_match = self._factor_match(
            query_projected=query_projected,
            query_token_mask=query_token_mask,
            category="role",
            authority=factor_authority["role"],
        )
        traversal_match = self._factor_match(
            query_projected=query_projected,
            query_token_mask=query_token_mask,
            category="traversal",
            authority=factor_authority["traversal"],
        )
        direction_match = self._factor_match(
            query_projected=query_projected,
            query_token_mask=query_token_mask,
            category="direction",
            authority=factor_authority["direction"],
        )
        control_match = self._factor_match(
            query_projected=query_projected,
            query_token_mask=query_token_mask,
            category="control",
            authority=factor_authority["control"],
        )

        role_logits = role_match["authority_score"]
        traversal_logits = traversal_match["authority_score"]
        direction_logits = direction_match["authority_score"]
        control_logits = control_match["authority_score"]

        role_distribution = torch.softmax(role_logits, dim=-1)
        traversal_distribution = torch.softmax(
            traversal_logits,
            dim=-1,
        )
        direction_distribution = torch.softmax(
            direction_logits,
            dim=-1,
        )
        control_distribution = torch.softmax(
            control_logits,
            dim=-1,
        )

        modifier_raw, modifier_mask = self._factor_row(
            "modifiers",
            device=query_hidden_states.device,
        )
        modifier_count = int(modifier_raw.size(0))
        modifier_flat = modifier_raw.reshape(
            modifier_count * 2,
            *modifier_raw.shape[2:],
        )
        modifier_mask_flat = modifier_mask.reshape(
            modifier_count * 2,
            modifier_mask.size(-1),
        )
        modifier_match = self.schema_matcher.match_projected(
            query_projected=query_projected,
            query_token_mask=query_token_mask,
            schema_projected=self.schema_matcher.project(modifier_flat),
            schema_token_mask=modifier_mask_flat,
        )
        modifier_token_score = modifier_match["token_score"].reshape(
            batch,
            modifier_count,
            2,
        )
        modifier_pair_logits = combine_authority_components(
            joint_preference=modifier_authority["joint_preference"],
            semantic_projection=modifier_authority["semantic_projection"],
            token_evidence=modifier_token_score,
        )
        modifier_probability = torch.softmax(
            modifier_pair_logits,
            dim=-1,
        )
        modifier_weight = modifier_probability[..., 1]
        modifier_logits = (
            modifier_pair_logits[..., 1]
            - modifier_pair_logits[..., 0]
        )

        role_summary = self._expected(
            role_distribution,
            role_match["schema_summary"],
        )
        traversal_summary = self._expected(
            traversal_distribution,
            traversal_match["schema_summary"],
        )
        direction_summary = self._expected(
            direction_distribution,
            direction_match["schema_summary"],
        )
        control_summary = self._expected(
            control_distribution,
            control_match["schema_summary"],
        )
        modifier_summary_states = modifier_match[
            "schema_summary"
        ].reshape(modifier_count, 2, -1)
        modifier_expected = (
            modifier_probability.unsqueeze(-1)
            * modifier_summary_states.unsqueeze(0)
        ).sum(dim=2).mean(dim=1)
        factor_states = torch.stack(
            [
                role_summary,
                traversal_summary,
                direction_summary,
                control_summary,
                modifier_expected,
            ],
            dim=1,
        )
        factor_summary = factor_states.mean(dim=1)

        applicability_input = torch.cat(
            [state, factor_summary],
            dim=-1,
        )
        applicability_logit = self.applicability_head(
            applicability_input
        ).squeeze(-1)
        applicability = torch.sigmoid(applicability_logit)
        continuous_state = self.continuous_projection(
            applicability_input
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
                batch,
                layers,
                tokens,
            ),
            "step_query_attention": step_attention,
            "query_coverage": query_coverage_history,
            "step_state_history": step_state_history,
            "factor_state": factor_states,
            "schema_relation_state": schema_relation_state,
            "schema_match_relation_state": matched["schema_summary"],
        }

    def parameter_report(self) -> dict[str, int | bool | None]:
        matcher_report = self.schema_matcher.parameter_report()
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "relation_count_dependent_parameters": 0,
            "hop_count_dependent_parameters": 0,
            "fixed_factor_class_head_parameters": 0,
            "runtime_dynamic_relation_schema": True,
            "shared_query_schema_metric": False,
            "parameter_free_token_evidence_matcher": True,
            "candidate_conditioned_query_evidence": True,
            "ordered_query_evidence_coverage": True,
            "coverage_is_position_based_not_left_to_right": True,
            "coverage_nonzero_recovery_floor": 0.05,
            "symmetric_late_interaction": True,
            "authority_component_fusion": "equal_candidate_zscore_mean",
            "p1_schema_encoder_is_not_relation_match_authority": True,
            "p1_schema_relation_state_is_interface_only_for_operator": True,
            "relation_selection_decoupled_from_stop_unknown": True,
            "continuous_relation_hypotheses": True,
            "relation_sparsity_before_structural_binding": False,
            "exact_sparsity_owned_by_binder": True,
            "semantic_factor_schemas": True,
            "same_matcher_for_relations_and_factors": True,
            "schema_matcher_relation_identity_parameters": matcher_report[
                "relation_identity_parameters"
            ],
            "schema_matcher_trainable_parameters": matcher_report[
                "trainable_parameters"
            ],
            "frozen_semantic_authority_external": True,
            "frozen_authority_trainable_parameters": 0,
            "relation_identity_owned_by_trainable_operator": False,
            "factor_identity_owned_by_trainable_operator": False,
            "factor_schema_configured": self._factor_cache is not None,
            "continuous_operator_state": True,
            "role_count_ceiling": None,
            "traversal_count_ceiling": None,
            "direction_count_ceiling": None,
            "control_count_ceiling": None,
            "relation_count_ceiling": None,
            "runtime_step_count_ceiling": None,
        }
