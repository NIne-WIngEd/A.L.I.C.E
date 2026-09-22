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
    require_finite,
    require_unit_interval,
)


@dataclass(frozen=True)
class DynamicEvidenceViewConfig:
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


class DynamicEvidenceViewV2(nn.Module):
    """Evidence-specialist view with multi-layer token conditioning.

    This successor keeps the useful non-destructive evidence-view idea but
    removes the old single pooled frozen-query bottleneck. Query/field token
    interactions use every semantic layer. Runtime relation state and the
    continuous operator state condition field specialization without any
    relation-count parameter axis. No exact support sparsity is introduced here.
    """

    def __init__(self, config: DynamicEvidenceViewConfig | None = None) -> None:
        super().__init__()
        self.config = config or DynamicEvidenceViewConfig()
        self.config.validate()
        d = self.config.model_dim

        self.query_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.field_token_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.structured_projection = nn.Linear(d, d, bias=False)
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

        self.field_update = nn.Sequential(
            nn.Linear(5 * d + 3, 2 * d),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.selector = nn.Sequential(
            nn.Linear(4 * d + 4, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.summary_projection = nn.Linear(2 * d, d)

    def _query_summary(self, query: Tensor, mask: Tensor) -> Tensor:
        projected = self.query_projection(query.float())
        weight = mask[:, None, :, None].to(projected.dtype)
        per_layer = (
            (projected * weight).sum(dim=2)
            / weight.sum(dim=2).clamp_min(1.0)
        )
        layer_logit = self.query_layer_gate(
            torch.tanh(per_layer)
        ).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        return torch.einsum("bl,bld->bd", layer_weight, per_layer)

    def _late_interaction(
        self,
        *,
        query: Tensor,
        query_mask: Tensor,
        field: Tensor,
        field_mask: Tensor,
    ) -> Tensor:
        q = F.normalize(self.query_projection(query.float()), dim=-1)
        f = F.normalize(self.field_token_projection(field.float()), dim=-1)
        q_to_f, f_to_q = chunked_batched_bidirectional_late_max(
            query=q,
            query_mask=query_mask,
            items=f,
            item_mask=field_mask,
            chunk_tokens=self.config.interaction_chunk_tokens,
        )
        q_mask = query_mask[:, None, None, :].expand_as(q_to_f)
        f_mask = field_mask[:, :, None, :].expand_as(f_to_q)

        q_score = (
            q_to_f.masked_fill(~q_mask, 0.0).sum(dim=-1)
            / q_mask.sum(dim=-1).clamp_min(1).to(q_to_f.dtype)
        )
        f_score = (
            f_to_q.masked_fill(~f_mask, 0.0).sum(dim=-1)
            / f_mask.sum(dim=-1).clamp_min(1).to(f_to_q.dtype)
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

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        field_hidden_states: Tensor,
        field_token_mask: Tensor,
        structured_field_state: Tensor,
        field_valid_mask: Tensor,
        field_confidence: Tensor,
        field_missing: Tensor,
        field_reliability: Tensor,
        relation_schema_state: Tensor,
        relation_mass: Tensor,
        semantic_activity: Tensor,
        operator_state: Tensor,
    ) -> dict[str, Tensor]:
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        batch, layers, query_tokens, width = query_hidden_states.shape
        if layers != self.config.num_hidden_states or width != self.config.semantic_dim:
            raise ValueError("query semantic geometry drift")
        if query_token_mask.shape != (batch, query_tokens) or query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool [B,T]")

        if field_hidden_states.ndim != 5:
            raise ValueError("field_hidden_states must be [B,F,L,S,D]")
        if field_hidden_states.size(0) != batch or field_hidden_states.size(2) != layers:
            raise ValueError("field hidden-state geometry drift")
        fields = field_hidden_states.size(1)
        field_tokens = field_hidden_states.size(3)
        if field_hidden_states.size(-1) != self.config.semantic_dim:
            raise ValueError("field semantic width drift")
        if field_token_mask.shape != (batch, fields, field_tokens):
            raise ValueError("field_token_mask shape drift")
        if field_token_mask.dtype != torch.bool:
            raise ValueError("field_token_mask must be bool")

        if structured_field_state.shape != (batch, fields, self.config.model_dim):
            raise ValueError("structured_field_state shape drift")
        if field_valid_mask.shape != (batch, fields) or field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool [B,F]")
        if bool((field_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one valid field")
        for name, value in (
            ("field_confidence", field_confidence),
            ("field_missing", field_missing),
            ("field_reliability", field_reliability),
        ):
            if value.shape != (batch, fields):
                raise ValueError(f"{name} must be [B,F]")
            require_unit_interval(name, value)

        if relation_schema_state.ndim != 3 or relation_schema_state.size(0) != batch:
            raise ValueError("relation_schema_state must be [B,R,D]")
        if relation_schema_state.size(-1) != self.config.model_dim:
            raise ValueError("relation_schema_state width drift")
        relations = relation_schema_state.size(1)
        if relation_mass.shape != (batch, relations):
            raise ValueError("relation_mass shape drift")
        require_unit_interval("relation_mass", relation_mass)
        if semantic_activity.shape != (batch,):
            raise ValueError("semantic_activity must be [B]")
        if bool(
            (
                (semantic_activity < -1.0e-6)
                | (semantic_activity > 1.0 + 1.0e-6)
            ).any()
        ):
            raise ValueError("semantic_activity must stay inside [0,1]")
        if operator_state.shape != (batch, self.config.model_dim):
            raise ValueError("operator_state shape drift")

        late = self._late_interaction(
            query=query_hidden_states,
            query_mask=query_token_mask,
            field=field_hidden_states,
            field_mask=field_token_mask,
        )
        query_summary = self._query_summary(query_hidden_states, query_token_mask)
        structured = self.structured_projection(structured_field_state.float())
        expected_relation = torch.einsum(
            "br,brd->bd",
            relation_mass.float(),
            self.relation_projection(relation_schema_state.float()),
        )
        expected_relation = (
            expected_relation
            * semantic_activity[:, None].to(expected_relation.dtype)
        )
        operator = self.operator_projection(operator_state.float())

        q = query_summary[:, None, :].expand(batch, fields, -1)
        rel = expected_relation[:, None, :].expand(batch, fields, -1)
        op = operator[:, None, :].expand(batch, fields, -1)
        scalar = torch.stack(
            [
                field_confidence.float(),
                field_missing.float(),
                field_reliability.float(),
            ],
            dim=-1,
        )
        update = self.field_update(
            torch.cat([structured, q, rel, op, structured * q, scalar], dim=-1)
        )
        evidence_field_state = structured_field_state + update
        evidence_field_state = (
            evidence_field_state
            * field_valid_mask.unsqueeze(-1).to(evidence_field_state.dtype)
        )

        selector_scalar = torch.cat(
            [
                scalar,
                late.unsqueeze(-1),
            ],
            dim=-1,
        )
        selector_logit = self.selector(
            torch.cat([evidence_field_state, q, rel, op, selector_scalar], dim=-1)
        ).squeeze(-1)
        selector_logit = selector_logit + 1.5 * late
        selector_logit = selector_logit.masked_fill(~field_valid_mask, -1.0e4)

        # Deliberately soft. Exact zero support remains Binder's responsibility.
        field_weight = torch.softmax(selector_logit, dim=-1)
        field_weight = field_weight * field_valid_mask.to(field_weight.dtype)
        field_weight = field_weight / field_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)

        weighted = torch.einsum(
            "bf,bfd->bd",
            field_weight,
            evidence_field_state,
        )
        evidence_summary = self.summary_projection(
            torch.cat([weighted, operator], dim=-1)
        )

        return {
            "evidence_field_state": evidence_field_state,
            "field_weight": field_weight,
            "selector_logit": selector_logit,
            "late_interaction": late,
            "evidence_summary": evidence_summary,
            "semantic_activity": semantic_activity,
            "evidence_tokens": evidence_field_state,
            "evidence_token_mask": field_valid_mask,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_identity_parameters": 0,
            "relation_count_dependent_parameters": 0,
            "field_count_dependent_parameters": 0,
            "pooled_only_query_conditioning": False,
            "multi_layer_query_field_interaction": True,
            "content_conditioned_query_layer_read": True,
            "field_conditioned_interaction_layer_read": True,
            "global_static_layer_mixture": False,
            "exact_structural_sparsity": False,
            "token_interaction_chunk_is_operating_point": True,
            "query_token_count_ceiling": None,
            "field_token_count_ceiling": None,
            "field_count_ceiling": None,
            "relation_count_ceiling": None,
            "soft_relation_context_activity_gate": True,
        }
