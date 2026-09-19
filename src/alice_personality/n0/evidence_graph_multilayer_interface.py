from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .evidence_graph import EvidenceGraphConfig, EvidenceRelationType
from .evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from .relation_conditioned_multilayer_query import (
    RelationConditionedMultiLayerQueryInterface,
)


class RelationConditionedMultiLayerEvidenceGraphEncoder(
    DualEndpointEvidenceGraphEncoder
):
    """Frozen-parent evidence graph with one edge-specific multi-layer query residual.

    The parent dual-endpoint graph remains the authoritative baseline expert.
    The new path receives relation-conditioned query states produced from
    intermediate frozen semantic token layers. A single signed scalar is read
    from each edge-specific state and applied complementarily to the relation
    source and target.

    The scalar readout is exactly zero-initialized. Therefore a parent graph
    checkpoint loaded into this class produces exact parent behavior before
    any interface training.

    This is not a generic endpoint router:
      - it has no SOURCE/TARGET class head;
      - it cannot see raw pooled query semantics;
      - it cannot bypass the compiled relation-layer policy;
      - it receives only the real edge_relation_query_state from the
        RelationConditionedMultiLayerQueryInterface.
    """

    def __init__(
        self,
        *,
        config: EvidenceGraphConfig,
        query_interface: RelationConditionedMultiLayerQueryInterface,
    ) -> None:
        super().__init__(config)
        if query_interface.semantic_size != config.semantic_size:
            raise ValueError("query-interface semantic width mismatch")
        if query_interface.graph_size != config.graph_size:
            raise ValueError("query-interface graph width mismatch")
        if query_interface.num_relation_types != config.num_relation_types:
            raise ValueError("query-interface relation vocabulary mismatch")

        self.query_interface = query_interface
        self.interface_endpoint_read = nn.Linear(config.graph_size, 1)
        nn.init.zeros_(self.interface_endpoint_read.weight)
        nn.init.zeros_(self.interface_endpoint_read.bias)

    def _interface_relation_bias(
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
        batch, fields, _width = x.shape
        zero = torch.zeros(batch, fields, device=x.device, dtype=x.dtype)
        if query_hidden_states is None or query_attention_mask is None:
            return zero, None

        interface = self.query_interface(
            hidden_states=query_hidden_states,
            attention_mask=query_attention_mask,
            edge_type_ids=edge_type_ids,
            edge_valid_mask=edge_valid_mask,
        )
        edge_state = interface["edge_relation_query_state"]
        has_interface = interface["relation_has_interface"]

        source = edge_index[..., 0].clamp(0, fields - 1)
        target = edge_index[..., 1].clamp(0, fields - 1)
        batch_index = torch.arange(batch, device=x.device).unsqueeze(1).expand_as(source)
        endpoint_valid = (
            valid_mask[batch_index, source]
            & valid_mask[batch_index, target]
        )
        conflict = edge_type_ids == int(EvidenceRelationType.CONFLICTS_WITH)
        active = (
            edge_valid_mask
            & endpoint_valid
            & has_interface
            & ~conflict
            & edge_type_ids.ne(int(EvidenceRelationType.PAD))
        )

        delta = self.interface_endpoint_read(edge_state).squeeze(-1)
        confidence = edge_confidence.to(x.dtype).clamp(0.0, 1.0).squeeze(-1)
        delta = delta * confidence * active.to(delta.dtype)

        flat = torch.zeros(batch * fields, device=x.device, dtype=x.dtype)
        offsets = (torch.arange(batch, device=x.device) * fields).unsqueeze(1)
        source_flat = (source + offsets).reshape(-1)
        target_flat = (target + offsets).reshape(-1)
        flat.index_add_(0, source_flat, delta.reshape(-1))
        flat.index_add_(0, target_flat, -delta.reshape(-1))
        bias = flat.reshape(batch, fields)

        interface = dict(interface)
        interface["interface_endpoint_delta"] = delta
        interface["interface_field_bias"] = bias
        return bias, interface

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
            batch, fields, device=x.device, dtype=x.dtype
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
        interface_bias, interface = self._interface_relation_bias(
            x=x,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
            query_hidden_states=query_hidden_states,
            query_attention_mask=query_attention_mask,
        )
        relation_status_bias = parent_relation_bias + interface_bias

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
            "interface_relation_status_bias": interface_bias,
            "relation_update_norm": relation_update_norm,
            "valid_mask": valid_mask,
        }
        if interface is not None:
            out.update(interface)
        return out

    def interface_parameter_names(self) -> list[str]:
        prefixes = ("query_interface.", "interface_endpoint_read.")
        return [
            name
            for name, _parameter in self.named_parameters()
            if name.startswith(prefixes)
        ]

    def freeze_parent_for_interface_training(self) -> list[nn.Parameter]:
        interface_names = set(self.interface_parameter_names())
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name in interface_names
        trainable = [
            parameter
            for name, parameter in self.named_parameters()
            if name in interface_names
        ]
        if not trainable:
            raise RuntimeError("multi-layer interface has no trainable parameters")
        return trainable

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        names = set(self.interface_parameter_names())
        report["total_parameters"] = sum(p.numel() for p in self.parameters())
        report["trainable_parameters"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        report["multilayer_interface_parameters"] = sum(
            p.numel()
            for name, p in self.named_parameters()
            if name in names
        )
        report["multilayer_query_input"] = (
            "edge_relation_query_state_from_relation_conditioned_intermediate_tokens"
        )
        report["composition"] = (
            "zero_initialized_signed_source_target_residual_on_frozen_parent_pool_logits"
        )
        report["generic_endpoint_router"] = False
        report["raw_mean_pool_router"] = False
        report["hardcoded_single_layer"] = False
        report["parent_graph_can_remain_frozen"] = True
        report["hard_parameter_ceiling"] = None
        return report
