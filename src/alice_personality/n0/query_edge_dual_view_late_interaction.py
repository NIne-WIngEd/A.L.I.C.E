from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .evidence_graph import EvidenceRelationType
from .relation_conditioned_multilayer_query import RELATION_ID_TO_NAME


class DualViewLateInteractionBridge(nn.Module):
    """Dual-view query-edge specialist with hierarchical routing.

    The frozen structured/graph path remains the reasoning view. Edge identity is
    selected through a separate token-level semantic binding view so fine-grained
    subject/attribute information is not forced through the pooled structured
    bottleneck before routing.

    Routing is factorized:

      P(parent/no-op) = 1 - P(specialist)
      P(edge=e)       = P(specialist) * P(e | specialist)

    P(e | specialist) is derived from symmetric query-to-field late interaction
    over the final semantic token layer. The source and target endpoint scores
    are averaged, so reversing an edge does not change which field pair is
    considered relevant. Directional source/target behavior stays in the
    existing relation-conditioned graph specialist.

    Exact-parent initialization is guaranteed by zero-initialized residual
    readouts. The route may be live at initialization without changing parent
    field weights or pooled state.
    """

    def __init__(
        self,
        *,
        semantic_size: int,
        graph_size: int,
        num_relation_types: int,
        num_hidden_states: int,
        layer_map: dict[str, Any],
        interface_size: int | None = None,
        dropout: float = 0.0,
        audit_prior_scale: float = 2.0,
        binding_logit_scale_init: float = 10.0,
    ) -> None:
        super().__init__()
        if semantic_size < 1 or graph_size < 1 or num_relation_types < 1:
            raise ValueError("semantic/graph/relation sizes must be positive")
        if num_hidden_states < 2:
            raise ValueError("dual-view bridge requires hidden-state depth")
        if not isinstance(layer_map, dict) or not layer_map:
            raise ValueError("relation-conditioned layer map is required")
        if binding_logit_scale_init <= 0.0:
            raise ValueError("binding_logit_scale_init must be positive")

        width = int(interface_size or graph_size)
        self.semantic_size = int(semantic_size)
        self.graph_size = int(graph_size)
        self.num_relation_types = int(num_relation_types)
        self.num_hidden_states = int(num_hidden_states)
        self.interface_size = width
        self.audit_prior_scale = float(audit_prior_scale)

        self.token_norm = nn.LayerNorm(semantic_size)
        self.token_key = nn.Linear(semantic_size, width, bias=False)
        self.token_value = nn.Linear(semantic_size, width, bias=False)

        self.relation_embedding = nn.Embedding(
            num_relation_types,
            width,
            padding_idx=int(EvidenceRelationType.PAD),
        )
        self.source_projection = nn.Linear(graph_size, width, bias=False)
        self.target_projection = nn.Linear(graph_size, width, bias=False)

        self.edge_query = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )

        self.layer_embedding = nn.Embedding(num_hidden_states, width)
        self.layer_score = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, 1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )

        # Only the confidence scale is learnable here. It cannot change edge
        # ordering implied by the frozen semantic late-interaction scores.
        self.binding_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(binding_logit_scale_init)))
        )

        # The learned routing problem is now only parent versus specialist.
        self.global_query_projection = nn.Linear(
            semantic_size,
            width,
            bias=False,
        )
        self.specialist_activation = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        activation_final = self.specialist_activation[-1]
        assert isinstance(activation_final, nn.Linear)
        nn.init.zeros_(activation_final.weight)
        nn.init.zeros_(activation_final.bias)

        self.source_residual_read = nn.Linear(width, 1)
        self.target_residual_read = nn.Linear(width, 1)
        nn.init.zeros_(self.source_residual_read.weight)
        nn.init.zeros_(self.source_residual_read.bias)
        nn.init.zeros_(self.target_residual_read.weight)
        nn.init.zeros_(self.target_residual_read.bias)

        candidate_mask = torch.zeros(
            num_relation_types,
            num_hidden_states,
            dtype=torch.bool,
        )
        audit_prior = torch.full(
            (num_relation_types, num_hidden_states),
            -1.0e4,
            dtype=torch.float32,
        )
        compiled: dict[int, dict[str, Any]] = {}

        for relation_id, relation_name in RELATION_ID_TO_NAME.items():
            if relation_id >= num_relation_types:
                continue
            info = layer_map.get(relation_name)
            if info is None:
                continue
            candidates = sorted({int(x) for x in info.get("candidate_layers", [])})
            ranked = {
                int(item["layer_index"]): float(item["token_accuracy"])
                for item in info.get("ranked_layers", [])
            }
            if not candidates:
                continue
            for layer in candidates:
                if not 0 <= layer < num_hidden_states:
                    raise ValueError(
                        f"candidate layer {layer} for {relation_name} exceeds "
                        f"hidden-state depth {num_hidden_states}"
                    )
                candidate_mask[relation_id, layer] = True
            values = torch.tensor(
                [ranked.get(layer, 0.0) for layer in candidates],
                dtype=torch.float32,
            )
            centered = values - values.mean()
            for layer, value in zip(candidates, centered.tolist()):
                audit_prior[relation_id, layer] = (
                    self.audit_prior_scale * float(value)
                )
            compiled[relation_id] = {
                "relation_name": relation_name,
                "candidate_layers": candidates,
                "audit_accuracy": {
                    str(layer): ranked.get(layer)
                    for layer in candidates
                },
            }

        candidate_mask[int(EvidenceRelationType.CONFLICTS_WITH)] = False
        audit_prior[int(EvidenceRelationType.CONFLICTS_WITH)] = -1.0e4
        candidate_mask[int(EvidenceRelationType.PAD)] = False
        audit_prior[int(EvidenceRelationType.PAD)] = -1.0e4

        self.register_buffer(
            "relation_layer_candidate_mask",
            candidate_mask,
            persistent=True,
        )
        self.register_buffer(
            "relation_layer_audit_prior",
            audit_prior,
            persistent=True,
        )
        self.compiled_relation_policy = compiled

    @classmethod
    def from_layer_map_file(
        cls,
        *,
        layer_map_path: str | Path,
        semantic_size: int,
        graph_size: int,
        num_relation_types: int,
        num_hidden_states: int,
        interface_size: int | None = None,
        dropout: float = 0.0,
        audit_prior_scale: float = 2.0,
        binding_logit_scale_init: float = 10.0,
    ) -> "DualViewLateInteractionBridge":
        payload = json.loads(Path(layer_map_path).read_text(encoding="utf-8"))
        if payload.get("schema") != "alice.eipm.n0.relation-conditioned-layer-map.v0.1":
            raise ValueError("relation-conditioned layer-map schema mismatch")
        if payload.get("training_authorized") is not False:
            raise ValueError("layer-map governance drift")
        return cls(
            semantic_size=semantic_size,
            graph_size=graph_size,
            num_relation_types=num_relation_types,
            num_hidden_states=num_hidden_states,
            layer_map=payload["relation_map"],
            interface_size=interface_size,
            dropout=dropout,
            audit_prior_scale=audit_prior_scale,
            binding_logit_scale_init=binding_logit_scale_init,
        )

    @staticmethod
    def _masked_mean(
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _late_interaction_field_scores(
        *,
        query_tokens: torch.Tensor,
        query_content_mask: torch.Tensor,
        field_tokens: torch.Tensor,
        field_content_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_tokens.ndim != 3:
            raise ValueError("query_tokens must be [batch,tokens,semantic]")
        if field_tokens.ndim != 4:
            raise ValueError(
                "field_tokens must be [batch,fields,tokens,semantic]"
            )
        batch, fields, _field_tokens, semantic = field_tokens.shape
        if query_tokens.size(0) != batch or query_tokens.size(-1) != semantic:
            raise ValueError("query/field token semantic shape mismatch")
        if query_content_mask.shape != query_tokens.shape[:2]:
            raise ValueError("query_content_mask shape mismatch")
        if field_content_mask.shape != field_tokens.shape[:3]:
            raise ValueError("field_content_mask shape mismatch")
        if not torch.all(query_content_mask.any(dim=1)):
            raise ValueError("every query needs at least one content token")
        if not torch.all(field_content_mask.any(dim=2)):
            raise ValueError("every field needs at least one content token")

        query = F.normalize(query_tokens.float(), p=2, dim=-1, eps=1e-8)
        field = F.normalize(field_tokens.float(), p=2, dim=-1, eps=1e-8)
        similarity = torch.einsum("bqd,bftd->bfqt", query, field)
        similarity = similarity.masked_fill(
            ~field_content_mask[:, :, None, :],
            -1.0e4,
        )
        per_query_token = similarity.max(dim=-1).values
        query_weights = query_content_mask.to(per_query_token.dtype)[:, None, :]
        return (
            (per_query_token * query_weights).sum(dim=-1)
            / query_weights.sum(dim=-1).clamp_min(1.0)
        )

    def forward(
        self,
        *,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        attention_mask: torch.Tensor,
        query_content_mask: torch.Tensor,
        field_token_states: torch.Tensor,
        field_content_mask: torch.Tensor,
        graph_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"expected {self.num_hidden_states} hidden states, "
                f"received {len(hidden_states)}"
            )
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be [batch,tokens]")
        if graph_states.ndim != 3 or graph_states.size(-1) != self.graph_size:
            raise ValueError("graph_states must be [batch,fields,graph_size]")
        if edge_index.ndim != 3 or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [batch,edges,2]")
        if edge_type_ids.shape != edge_valid_mask.shape:
            raise ValueError("edge type/mask shape mismatch")

        batch, fields, _ = graph_states.shape
        if attention_mask.size(0) != batch or edge_index.size(0) != batch:
            raise ValueError("query/graph/edge batch mismatch")
        if valid_mask.shape != (batch, fields):
            raise ValueError("valid_mask shape mismatch")
        if (
            field_token_states.size(0) != batch
            or field_token_states.size(1) != fields
        ):
            raise ValueError("field token state batch/field mismatch")
        if field_token_states.size(-1) != self.semantic_size:
            raise ValueError("field semantic token width mismatch")
        if query_content_mask.dtype != torch.bool:
            raise ValueError("query_content_mask must be bool")
        if field_content_mask.dtype != torch.bool:
            raise ValueError("field_content_mask must be bool")

        stack = torch.stack(list(hidden_states), dim=1)
        if stack.ndim != 4:
            raise ValueError("hidden states must stack to [batch,layers,tokens,width]")
        if stack.size(0) != batch or stack.size(2) != attention_mask.size(1):
            raise ValueError("hidden/query shape mismatch")
        if stack.size(3) != self.semantic_size:
            raise ValueError("semantic hidden width mismatch")

        source = edge_index[..., 0].clamp(0, fields - 1)
        target = edge_index[..., 1].clamp(0, fields - 1)
        batch_index = torch.arange(batch, device=graph_states.device).unsqueeze(1)
        batch_index = batch_index.expand_as(source)

        source_valid = valid_mask[batch_index, source]
        target_valid = valid_mask[batch_index, target]
        relation_ids = edge_type_ids.clamp(0, self.num_relation_types - 1)
        relation = self.relation_embedding(relation_ids)
        source_state = graph_states[batch_index, source]
        target_state = graph_states[batch_index, target]
        source_latent = self.source_projection(source_state)
        target_latent = self.target_projection(target_state)

        edge_query = self.edge_query(
            torch.cat(
                [
                    relation,
                    source_latent,
                    target_latent,
                    source_latent - target_latent,
                ],
                dim=-1,
            )
        )

        normalized = self.token_norm(stack)
        token_keys = self.token_key(normalized)
        token_values = self.token_value(normalized)

        token_scores = torch.einsum(
            "bew,bltw->belt",
            edge_query,
            token_keys,
        ) / (self.interface_size ** 0.5)
        token_valid = attention_mask.bool()[:, None, None, :]
        token_scores = token_scores.masked_fill(~token_valid, -1.0e4)
        token_weights = torch.softmax(token_scores, dim=-1)
        layer_summaries = torch.einsum(
            "belt,bltw->belw",
            token_weights,
            token_values,
        )

        candidate_mask = self.relation_layer_candidate_mask[relation_ids]
        layer_ids = torch.arange(self.num_hidden_states, device=stack.device)
        layer_emb = self.layer_embedding(layer_ids)[None, None, :, :]
        layer_emb = layer_emb.expand(
            batch,
            edge_index.size(1),
            -1,
            -1,
        )
        edge_query_expanded = edge_query[:, :, None, :].expand(
            -1, -1, self.num_hidden_states, -1
        )
        layer_features = torch.cat(
            [
                layer_summaries,
                edge_query_expanded,
                layer_summaries * edge_query_expanded,
                layer_emb,
            ],
            dim=-1,
        )
        learned_layer_score = self.layer_score(layer_features).squeeze(-1)
        layer_logits = (
            learned_layer_score
            + self.relation_layer_audit_prior[relation_ids]
        )

        conflict = edge_type_ids.eq(int(EvidenceRelationType.CONFLICTS_WITH))
        pad = edge_type_ids.eq(int(EvidenceRelationType.PAD))
        active_edge = (
            edge_valid_mask.bool()
            & source_valid
            & target_valid
            & ~conflict
            & ~pad
        )
        active_candidate = candidate_mask & active_edge[:, :, None]
        layer_logits = layer_logits.masked_fill(~active_candidate, -1.0e4)
        has_candidates = active_candidate.any(dim=-1)
        route_active = active_edge & has_candidates

        safe_logits = torch.where(
            has_candidates[:, :, None],
            layer_logits,
            torch.zeros_like(layer_logits),
        )
        layer_weights = torch.softmax(safe_logits, dim=-1)
        layer_weights = layer_weights * active_candidate.to(layer_weights.dtype)
        layer_weights = layer_weights / layer_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)

        query_edge_language = torch.einsum(
            "bel,belw->bew",
            layer_weights,
            layer_summaries,
        )
        fused = self.fusion(
            torch.cat(
                [
                    query_edge_language,
                    edge_query,
                    source_latent,
                    target_latent,
                ],
                dim=-1,
            )
        )

        # Canonical final token layer for content binding. We intentionally do
        # not hard-code the DEV-best layer from the audit.
        query_final = stack[:, -1]
        field_binding_scores = self._late_interaction_field_scores(
            query_tokens=query_final,
            query_content_mask=query_content_mask,
            field_tokens=field_token_states,
            field_content_mask=field_content_mask,
        )
        source_binding = field_binding_scores[batch_index, source]
        target_binding = field_binding_scores[batch_index, target]
        edge_binding_scores = 0.5 * (source_binding + target_binding)
        edge_binding_logits = (
            edge_binding_scores
            * self.binding_logit_scale.exp().clamp(max=100.0)
        )
        edge_binding_logits = edge_binding_logits.masked_fill(
            ~route_active,
            -1.0e4,
        )
        conditional_edge_probability = torch.softmax(
            edge_binding_logits,
            dim=-1,
        )

        final_query_summary = self._masked_mean(
            query_final,
            query_content_mask,
        )
        global_query = self.global_query_projection(final_query_summary)
        specialist_edge_summary = torch.einsum(
            "be,bew->bw",
            conditional_edge_probability,
            fused,
        )
        specialist_logit = self.specialist_activation(
            torch.cat([global_query, specialist_edge_summary], dim=-1)
        ).squeeze(-1)
        specialist_probability = torch.sigmoid(specialist_logit)
        noop_route_probability = 1.0 - specialist_probability
        edge_route_probability = (
            specialist_probability[:, None]
            * conditional_edge_probability
        )
        route_probabilities_with_noop = torch.cat(
            [noop_route_probability[:, None], edge_route_probability],
            dim=-1,
        )

        source_proposal = self.source_residual_read(fused).squeeze(-1)
        target_proposal = self.target_residual_read(fused).squeeze(-1)
        source_residual = edge_route_probability * source_proposal
        target_residual = edge_route_probability * target_proposal

        return {
            "edge_binding_query": edge_query,
            "edge_query_language_state": query_edge_language,
            "edge_fused_state": fused,
            "field_binding_scores": field_binding_scores,
            "edge_binding_scores": edge_binding_scores,
            "edge_binding_logits": edge_binding_logits,
            "conditional_edge_probability": conditional_edge_probability,
            "specialist_logit": specialist_logit,
            "specialist_probability": specialist_probability,
            "noop_route_probability": noop_route_probability,
            "edge_route_probability": edge_route_probability,
            "route_probabilities_with_noop": route_probabilities_with_noop,
            "source_residual_proposal": source_proposal,
            "target_residual_proposal": target_proposal,
            "source_residual": source_residual,
            "target_residual": target_residual,
            "relation_token_attention": token_weights,
            "relation_layer_weights": layer_weights,
            "relation_has_interface": has_candidates,
            "relation_layer_candidate_mask": candidate_mask,
            "route_active_mask": route_active,
            "binding_logit_scale": self.binding_logit_scale.exp(),
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "semantic_size": self.semantic_size,
            "graph_size": self.graph_size,
            "interface_size": self.interface_size,
            "num_hidden_states": self.num_hidden_states,
            "dual_view_binding": True,
            "field_binding_view": "final_semantic_token_late_interaction",
            "graph_reasoning_view": "frozen_structured_graph_parent",
            "hierarchical_parent_specialist_routing": True,
            "edge_identity_from_structured_graph_state": False,
            "edge_identity_from_token_late_interaction": True,
            "edge_binding_order_learnable": False,
            "edge_binding_scale_learnable": True,
            "source_target_pair_binding_symmetric": True,
            "relation_direction_reasoning_multilayer": True,
            "explicit_parent_noop_route": True,
            "supervised_route_equals_runtime_contribution_route": True,
            "independent_source_target_residuals": True,
            "forced_antisymmetry": False,
            "setwise_router_transformer": False,
            "flat_parent_plus_edge_softmax_router": False,
            "zero_initialized_residual_readouts": True,
            "hardcoded_dev_best_binding_layer": False,
            "hard_parameter_ceiling": None,
        }
