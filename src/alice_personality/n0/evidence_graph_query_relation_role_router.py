from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .evidence_graph import EvidenceGraphConfig
from .evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder


class QueryRelationRoleRouterEvidenceGraphEncoder(DualEndpointEvidenceGraphEncoder):
    """Frozen endpoint graph plus an independent raw-query relation-role router.

    The parent graph remains the authoritative expert for the capabilities it
    already has. A separate side network reads the *raw frozen semantic query*
    and the edge relation type, then predicts one of three routing intents per
    edge:

        SOURCE  -> use the relation source endpoint
        TARGET  -> use the relation target endpoint
        DEFER   -> leave the frozen parent expert in control

    The side route is composed at the probability level, not by adding an
    unbounded bias into the parent's endpoint logits. This gives the semantic
    expert a clean path to override a wrong parent preference when confident,
    while DEFER provides an explicit non-destructive route for existing skills.
    """

    SOURCE = 0
    TARGET = 1
    DEFER = 2

    def __init__(
        self,
        config: EvidenceGraphConfig | None = None,
        *,
        router_size: int | None = None,
    ) -> None:
        super().__init__(config)
        semantic = self.config.semantic_size
        router = int(router_size or max(256, min(semantic, self.config.graph_size)))

        self.router_size = router
        self.semantic_query_side = nn.Sequential(
            nn.LayerNorm(semantic),
            nn.Linear(semantic, router),
            nn.GELU(),
            nn.Linear(router, router),
            nn.LayerNorm(router),
        )
        self.semantic_relation_side = nn.Embedding(
            self.config.num_relation_types,
            router,
            padding_idx=0,
        )
        self.semantic_relation_norm = nn.LayerNorm(router)

        # q, r, q*r and |q-r| provide both additive and multiplicative
        # query/relation interactions without reusing the endpoint-specialized
        # parent query_projection.
        self.semantic_role_router = nn.Sequential(
            nn.Linear(4 * router, 2 * router),
            nn.GELU(),
            nn.Linear(2 * router, router),
            nn.GELU(),
            nn.Linear(router, 3),
        )

        # Start near-exactly in DEFER so the frozen parent remains behaviorally
        # dominant before the side expert has evidence. This is not a hard
        # gate: the router is fully trainable and the bias can move immediately.
        final = self.semantic_role_router[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[self.DEFER] = 12.0

    def _semantic_role_router_logits(
        self,
        *,
        query_semantic: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = self.semantic_query_side(query_semantic)
        relation = self.semantic_relation_norm(
            self.semantic_relation_side(edge_type_ids)
        )
        q_edge = q.unsqueeze(1).expand(-1, edge_type_ids.size(1), -1)
        features = torch.cat(
            [
                q_edge,
                relation,
                q_edge * relation,
                (q_edge - relation).abs(),
            ],
            dim=-1,
        )
        return self.semantic_role_router(features)

    def _semantic_field_distribution(
        self,
        *,
        query_semantic: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_confidence: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, fields = valid_mask.shape
        source = edge_index[..., 0].clamp(0, fields - 1)
        target = edge_index[..., 1].clamp(0, fields - 1)
        batch_index = torch.arange(
            batch, device=valid_mask.device
        ).unsqueeze(1).expand_as(source)
        active = (
            edge_valid_mask
            & valid_mask[batch_index, source]
            & valid_mask[batch_index, target]
        )

        logits = self._semantic_role_router_logits(
            query_semantic=query_semantic,
            edge_type_ids=edge_type_ids,
        )
        probabilities = torch.softmax(logits, dim=-1)
        source_probability = probabilities[..., self.SOURCE]
        target_probability = probabilities[..., self.TARGET]
        defer_probability = probabilities[..., self.DEFER]
        route_probability = 1.0 - defer_probability

        confidence = edge_confidence.to(query_semantic.dtype).squeeze(-1)
        edge_strength = confidence.clamp(0.0, 1.0) * active.to(query_semantic.dtype)
        source_mass = edge_strength * source_probability
        target_mass = edge_strength * target_probability

        flat_mass = torch.zeros(
            batch * fields,
            device=query_semantic.device,
            dtype=query_semantic.dtype,
        )
        offsets = (
            torch.arange(batch, device=query_semantic.device) * fields
        ).unsqueeze(1)
        source_flat = (source + offsets).reshape(-1)
        target_flat = (target + offsets).reshape(-1)
        flat_mass.index_add_(0, source_flat, source_mass.reshape(-1))
        flat_mass.index_add_(0, target_flat, target_mass.reshape(-1))
        field_mass = flat_mass.reshape(batch, fields)
        field_mass = field_mass * valid_mask.to(field_mass.dtype)

        mass_total = field_mass.sum(dim=-1, keepdim=True)
        semantic_weights = field_mass / mass_total.clamp_min(1e-8)

        active_strength = edge_strength.sum(dim=-1)
        routed_strength = (edge_strength * route_probability).sum(dim=-1)
        sample_route = torch.where(
            active_strength > 0.0,
            routed_strength / active_strength.clamp_min(1e-8),
            torch.zeros_like(active_strength),
        ).clamp(0.0, 1.0)

        no_semantic_mass = mass_total.squeeze(-1) <= 1e-8
        sample_route = torch.where(
            no_semantic_mass,
            torch.zeros_like(sample_route),
            sample_route,
        )
        return semantic_weights, sample_route, logits

    def forward(
        self,
        *,
        field_states: torch.Tensor,
        valid_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_confidence: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        query_semantic: torch.Tensor | None = None,
        base_field_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        parent = super().forward(
            field_states=field_states,
            valid_mask=valid_mask,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            query_semantic=query_semantic,
            base_field_weights=base_field_weights,
        )

        if query_semantic is None:
            out = dict(parent)
            out["semantic_router_field_weights"] = parent["field_weights"]
            out["semantic_router_route_probability"] = torch.zeros(
                field_states.size(0),
                device=field_states.device,
                dtype=field_states.dtype,
            )
            out["semantic_router_role_logits"] = torch.zeros(
                field_states.size(0),
                edge_type_ids.size(1),
                3,
                device=field_states.device,
                dtype=field_states.dtype,
            )
            return out

        semantic_weights, route_probability, role_logits = (
            self._semantic_field_distribution(
                query_semantic=query_semantic,
                edge_index=edge_index,
                edge_type_ids=edge_type_ids,
                edge_confidence=edge_confidence,
                edge_valid_mask=edge_valid_mask,
                valid_mask=valid_mask,
            )
        )

        alpha = route_probability.unsqueeze(-1)
        final_weights = (
            (1.0 - alpha) * parent["field_weights"]
            + alpha * semantic_weights
        )
        final_weights = final_weights * valid_mask.to(final_weights.dtype)
        final_weights = final_weights / final_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        side_pooled = torch.einsum(
            "bf,bfd->bd",
            semantic_weights,
            parent["field_states"],
        )
        pooled_alpha = route_probability.unsqueeze(-1)
        final_pooled = (
            (1.0 - pooled_alpha) * parent["pooled_state"]
            + pooled_alpha * side_pooled
        )

        out = dict(parent)
        out["field_weights"] = final_weights
        out["pooled_state"] = final_pooled
        out["semantic_router_field_weights"] = semantic_weights
        out["semantic_router_route_probability"] = route_probability
        out["semantic_router_role_logits"] = role_logits
        return out

    def router_parameter_names(self) -> list[str]:
        prefixes = (
            "semantic_query_side.",
            "semantic_relation_side.",
            "semantic_relation_norm.",
            "semantic_role_router.",
        )
        return [
            name
            for name, _parameter in self.named_parameters()
            if name.startswith(prefixes)
        ]

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        router_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name
            in set(self.router_parameter_names())
        )
        report["total_parameters"] = sum(
            parameter.numel() for parameter in self.parameters()
        )
        report["trainable_parameters"] = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        report["semantic_role_router_parameters"] = router_parameters
        report["semantic_role_router_input"] = (
            "raw_query_semantic_plus_relation_type_side_path"
        )
        report["semantic_role_router_outputs"] = [
            "source",
            "target",
            "defer_to_parent",
        ]
        report["composition"] = "probability_level_gated_expert_mixture"
        report["parent_endpoint_read_can_remain_frozen"] = True
        report["hard_parameter_ceiling"] = None
        return report
