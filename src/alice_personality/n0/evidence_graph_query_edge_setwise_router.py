from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .evidence_graph import EvidenceGraphConfig
from .evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from .query_edge_setwise_router_bridge import SetwiseQueryEdgeBridge


class SetwiseQueryEdgeEvidenceGraphEncoder(DualEndpointEvidenceGraphEncoder):
    """Frozen dual-endpoint parent plus setwise query-edge specialist.

    The specialist retains the qualified edge-specific representation path.
    Active directed edges are contextualized jointly before routing so route
    selection can compare competing edges rather than only normalize
    independently produced scalar scores.
    """

    def __init__(
        self,
        *,
        config: EvidenceGraphConfig,
        bridge: SetwiseQueryEdgeBridge,
    ) -> None:
        super().__init__(config)
        if bridge.semantic_size != config.semantic_size:
            raise ValueError("bridge semantic width mismatch")
        if bridge.graph_size != config.graph_size:
            raise ValueError("bridge graph width mismatch")
        if bridge.num_relation_types != config.num_relation_types:
            raise ValueError("bridge relation vocabulary mismatch")
        self.setwise_query_edge_bridge = bridge

    def _bridge_relation_bias(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_confidence: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        query_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
        query_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        batch, fields, _ = x.shape
        zero = torch.zeros(batch, fields, device=x.device, dtype=x.dtype)
        if query_hidden_states is None or query_attention_mask is None:
            return zero, None

        bridge = self.setwise_query_edge_bridge(
            hidden_states=query_hidden_states,
            attention_mask=query_attention_mask,
            graph_states=x,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
        )
        source_delta = bridge["source_residual"]
        target_delta = bridge["target_residual"]

        source = edge_index[..., 0].clamp(0, fields - 1)
        target = edge_index[..., 1].clamp(0, fields - 1)
        confidence = edge_confidence.to(x.dtype).clamp(0.0, 1.0).squeeze(-1)
        active = bridge["route_active_mask"].to(x.dtype)
        source_delta = source_delta * confidence * active
        target_delta = target_delta * confidence * active

        flat = torch.zeros(batch * fields, device=x.device, dtype=x.dtype)
        offsets = (torch.arange(batch, device=x.device) * fields).unsqueeze(1)
        source_flat = (source + offsets).reshape(-1)
        target_flat = (target + offsets).reshape(-1)
        flat.index_add_(0, source_flat, source_delta.reshape(-1))
        flat.index_add_(0, target_flat, target_delta.reshape(-1))
        bias = flat.reshape(batch, fields)

        out = dict(bridge)
        out["bridge_field_bias"] = bias
        return bias, out

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
        query_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, fields = self._validate_inputs(
            field_states=field_states,
            valid_mask=valid_mask,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            query_semantic=query_semantic,
            base_field_weights=base_field_weights,
        )
        if (query_hidden_states is None) != (query_attention_mask is None):
            raise ValueError(
                "query_hidden_states and query_attention_mask must be supplied together"
            )

        x = self.input_norm(self.input_projection(field_states))
        relation_update_norm = torch.zeros(
            batch,
            fields,
            device=x.device,
            dtype=x.dtype,
        )
        for layer in self.layers:
            x, layer_norm = layer(
                x,
                edge_index=edge_index,
                edge_type_ids=edge_type_ids,
                edge_confidence=edge_confidence,
                edge_valid_mask=edge_valid_mask,
                valid_mask=valid_mask,
            )
            relation_update_norm = relation_update_norm + layer_norm

        query = self.pool_query.unsqueeze(0).expand(batch, -1)
        if query_semantic is not None:
            query = query + self.query_projection(query_semantic)
        query = torch.nn.functional.normalize(query, dim=-1)

        parent_relation_bias = super()._query_conditioned_relation_bias(
            x=x,
            query=query,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
        )
        bridge_bias, bridge = self._bridge_relation_bias(
            x=x,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
            query_hidden_states=query_hidden_states,
            query_attention_mask=query_attention_mask,
        )
        relation_status_bias = parent_relation_bias + bridge_bias

        pool_scores = (
            torch.einsum("bfd,bd->bf", x, query)
            / math.sqrt(self.config.graph_size)
        )
        pool_scores = pool_scores + relation_status_bias
        if base_field_weights is not None and self.config.base_weight_scale > 0.0:
            base_prior = base_field_weights.to(x.dtype).clamp_min(1e-6).log()
            pool_scores = pool_scores + self.config.base_weight_scale * base_prior
        pool_scores = pool_scores.masked_fill(
            ~valid_mask,
            torch.finfo(pool_scores.dtype).min,
        )
        field_weights = torch.softmax(pool_scores, dim=-1)
        pooled = torch.einsum("bf,bfd->bd", field_weights, x)

        out: dict[str, torch.Tensor] = {
            "field_states": self.field_output_projection(x),
            "pooled_state": self.pooled_output_projection(pooled),
            "field_weights": field_weights,
            "relation_status_bias": relation_status_bias,
            "parent_relation_status_bias": parent_relation_bias,
            "bridge_relation_status_bias": bridge_bias,
            "relation_update_norm": relation_update_norm,
            "valid_mask": valid_mask,
        }
        if bridge is not None:
            out.update(bridge)
        return out

    def bridge_parameter_names(self) -> list[str]:
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith("setwise_query_edge_bridge.")
        ]

    def freeze_parent_for_bridge_training(self) -> list[nn.Parameter]:
        names = set(self.bridge_parameter_names())
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name in names
        trainable = [
            parameter
            for name, parameter in self.named_parameters()
            if name in names
        ]
        if not trainable:
            raise RuntimeError("setwise query-edge bridge has no trainable parameters")
        return trainable

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        names = set(self.bridge_parameter_names())
        report["total_parameters"] = sum(p.numel() for p in self.parameters())
        report["trainable_parameters"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        report["setwise_query_edge_bridge_parameters"] = sum(
            p.numel()
            for name, p in self.named_parameters()
            if name in names
        )
        report["edge_specific_query_binding"] = True
        report["setwise_contextual_routing"] = True
        report["joint_competitor_context"] = True
        report["permutation_equivariant_edge_set"] = True
        report["edge_order_positional_embedding"] = False
        report["explicit_parent_noop_route"] = True
        report["supervised_route_equals_runtime_contribution_route"] = True
        report["independent_source_target_residuals"] = True
        report["independent_per_edge_route_scoring"] = False
        report["independent_per_edge_tanh_gate"] = False
        report["proxy_dot_product_router"] = False
        report["parent_graph_can_remain_frozen"] = True
        report["hard_parameter_ceiling"] = None
        return report
