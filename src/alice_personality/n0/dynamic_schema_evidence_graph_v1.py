from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from alice_personality.n0.numeric_contracts import (
    exact_masked_softmax,
    require_finite,
    require_unit_interval,
)
from torch import Tensor, nn


@dataclass(frozen=True)
class DynamicSchemaEvidenceGraphConfig:
    field_dim: int = 640
    relation_dim: int = 640
    operator_dim: int = 640
    model_dim: int = 640
    edge_metadata_dim: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("field_dim", self.field_dim),
            ("relation_dim", self.relation_dim),
            ("operator_dim", self.operator_dim),
            ("model_dim", self.model_dim),
            ("edge_metadata_dim", self.edge_metadata_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class DynamicSchemaEvidenceGraphV1(nn.Module):
    """Directed evidence graph whose relation meaning is supplied at runtime.

    edge_relation_index is only an index into the supplied runtime relation
    state. No learned relation-ID embedding or fixed relation ontology exists.
    Source and target messages remain distinct so directional endpoint meaning
    cannot collapse into a shared read path.
    """

    def __init__(self, config: DynamicSchemaEvidenceGraphConfig | None = None) -> None:
        super().__init__()
        self.config = config or DynamicSchemaEvidenceGraphConfig()
        self.config.validate()
        d = self.config.model_dim

        self.field_projection = nn.Linear(self.config.field_dim, d)
        self.relation_projection = nn.Linear(self.config.relation_dim, d, bias=False)
        self.operator_projection = nn.Linear(self.config.operator_dim, d, bias=False)
        self.edge_metadata_projection = nn.Linear(self.config.edge_metadata_dim, d)
        self.edge_update = nn.Sequential(
            nn.Linear(5 * d + 1, 2 * d),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.source_message = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.target_message = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.node_update = nn.GRUCell(2 * d, d)
        self.source_read = nn.Linear(2 * d, 1)
        self.target_read = nn.Linear(2 * d, 1)

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
        return exact_masked_softmax(logits, mask, dim=-1)

    def forward(
        self,
        *,
        field_state: Tensor,
        field_valid_mask: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        edge_metadata: Tensor,
        edge_valid_mask: Tensor,
        relation_schema_state: Tensor,
        relation_mass: Tensor,
        relation_symmetric: Tensor,
        semantic_activity: Tensor,
        operator_state: Tensor,
        message_steps: int,
    ) -> dict[str, Tensor]:
        if message_steps <= 0:
            raise ValueError("message_steps must be positive")
        if field_state.ndim != 3:
            raise ValueError("field_state must be [B,F,D]")
        batch, fields, width = field_state.shape
        if width != self.config.field_dim:
            raise ValueError("field state width drift")
        if field_valid_mask.shape != (batch, fields) or field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool [B,F]")
        if bool((field_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one valid field")
        if edge_index.ndim != 3 or edge_index.size(0) != batch or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [B,E,2]")
        edges = edge_index.size(1)
        if edge_relation_index.shape != (batch, edges):
            raise ValueError("edge_relation_index shape drift")
        if edge_metadata.shape != (batch, edges, self.config.edge_metadata_dim):
            raise ValueError("edge_metadata shape drift")
        if edge_valid_mask.shape != (batch, edges) or edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool [B,E]")
        if relation_schema_state.ndim != 3 or relation_schema_state.size(0) != batch:
            raise ValueError("relation_schema_state must be [B,R,D]")
        if relation_schema_state.size(-1) != self.config.relation_dim:
            raise ValueError("relation schema width drift")
        relations = relation_schema_state.size(1)
        if relations <= 0:
            raise ValueError("runtime relation schema is empty")
        if relation_mass.shape != (batch, relations):
            raise ValueError("relation_mass must be [B,R]")
        require_unit_interval("relation_mass", relation_mass)
        if relation_symmetric.shape != (batch, relations) or relation_symmetric.dtype != torch.bool:
            raise ValueError("relation_symmetric must be bool [B,R]")
        if semantic_activity.shape != (batch,):
            raise ValueError("semantic_activity must be [B]")
        if bool(
            (
                (semantic_activity < -1.0e-6)
                | (semantic_activity > 1.0 + 1.0e-6)
            ).any()
        ):
            raise ValueError("semantic_activity must stay inside [0,1]")
        if operator_state.shape != (batch, self.config.operator_dim):
            raise ValueError("operator_state shape drift")
        require_finite("field_state", field_state)
        require_finite("edge_metadata", edge_metadata)
        require_finite("relation_schema_state", relation_schema_state)
        require_finite("operator_state", operator_state)

        if bool(edge_valid_mask.any()):
            valid_edge = edge_index[edge_valid_mask]
            if int(valid_edge.min()) < 0 or int(valid_edge.max()) >= fields:
                raise ValueError("edge endpoint outside runtime field set")
            valid_relation = edge_relation_index[edge_valid_mask]
            if int(valid_relation.min()) < 0 or int(valid_relation.max()) >= relations:
                raise ValueError("edge relation index outside runtime schema")
            source_index_valid = edge_index[...,0].clamp(min=0,max=fields-1)
            target_index_valid = edge_index[...,1].clamp(min=0,max=fields-1)
            endpoint_field_valid = (
                field_valid_mask.gather(1,source_index_valid)
                & field_valid_mask.gather(1,target_index_valid)
            )
            if bool((edge_valid_mask & ~endpoint_field_valid).any()):
                raise ValueError("valid edge references padded invalid field")

        node = self.field_projection(field_state.float())
        node = node * field_valid_mask.unsqueeze(-1).to(node.dtype)
        relation = self.relation_projection(relation_schema_state.float())
        operator = self.operator_projection(operator_state.float())
        metadata = self.edge_metadata_projection(edge_metadata.float())

        source_index = edge_index[..., 0].clamp(min=0, max=max(fields - 1, 0))
        target_index = edge_index[..., 1].clamp(min=0, max=max(fields - 1, 0))
        relation_index = edge_relation_index.clamp(min=0, max=relations - 1)
        batch_index = torch.arange(batch, device=node.device)[:, None].expand(batch, edges)

        last_edge = torch.zeros(batch, edges, self.config.model_dim, device=node.device, dtype=node.dtype)
        for _ in range(message_steps):
            source = node[batch_index, source_index]
            target = node[batch_index, target_index]
            rel = relation[batch_index, relation_index]
            edge_relation_mass = relation_mass.gather(1, relation_index)
            edge_symmetric = relation_symmetric[batch_index, relation_index]
            pair_mean = 0.5 * (source + target)
            pair_delta = (source - target).abs()
            edge_source = torch.where(
                edge_symmetric.unsqueeze(-1),
                pair_mean,
                source,
            )
            edge_target = torch.where(
                edge_symmetric.unsqueeze(-1),
                pair_delta,
                target,
            )
            op = operator[:, None, :].expand(batch, edges, -1)
            edge = self.edge_update(
                torch.cat(
                    [
                        edge_source,
                        edge_target,
                        rel,
                        op,
                        metadata,
                        edge_relation_mass.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
            edge = edge * edge_valid_mask.unsqueeze(-1).to(edge.dtype)
            last_edge = edge

            source_msg = self.source_message(torch.cat([edge, op], dim=-1))
            target_msg = self.target_message(torch.cat([edge, op], dim=-1))
            symmetric_message = 0.5 * (source_msg + target_msg)
            source_msg = torch.where(
                edge_symmetric.unsqueeze(-1),
                symmetric_message,
                source_msg,
            )
            target_msg = torch.where(
                edge_symmetric.unsqueeze(-1),
                symmetric_message,
                target_msg,
            )
            semantic_gate_scalar = (
                edge_relation_mass.clamp(0.0, 1.0)
                * semantic_activity[:, None]
            )
            semantic_gate = semantic_gate_scalar.unsqueeze(-1)
            source_msg = (
                source_msg
                * semantic_gate
                * edge_valid_mask.unsqueeze(-1).to(source_msg.dtype)
            )
            target_msg = (
                target_msg
                * semantic_gate
                * edge_valid_mask.unsqueeze(-1).to(target_msg.dtype)
            )

            aggregate = torch.zeros_like(node)
            source_scatter = source_index.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
            target_scatter = target_index.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
            aggregate.scatter_add_(1, source_scatter, source_msg)
            aggregate.scatter_add_(1, target_scatter, target_msg)

            q = operator[:, None, :].expand(batch, fields, -1)
            updated = self.node_update(
                torch.cat([aggregate, q], dim=-1).reshape(batch * fields, -1),
                node.reshape(batch * fields, -1),
            ).reshape(batch, fields, -1)
            node_gate = torch.zeros(
                batch,
                fields,
                device=node.device,
                dtype=node.dtype,
            )
            edge_gate = (
                semantic_gate_scalar
                * edge_valid_mask.to(node.dtype)
            )
            node_gate.scatter_add_(1, source_index, edge_gate)
            node_gate.scatter_add_(1, target_index, edge_gate)
            node_gate = node_gate.clamp(0.0, 1.0)
            node_gate = node_gate * field_valid_mask.to(node_gate.dtype)
            node = (
                node_gate.unsqueeze(-1) * updated
                + (1.0 - node_gate.unsqueeze(-1)) * node
            )

        q = operator[:, None, :].expand(batch, fields, -1)
        read_input = torch.cat([node, q], dim=-1)
        source_logits = self.source_read(read_input).squeeze(-1)
        target_logits = self.target_read(read_input).squeeze(-1)
        source_weight = self._masked_softmax(source_logits, field_valid_mask)
        target_weight = self._masked_softmax(target_logits, field_valid_mask)
        relation_present = torch.zeros(
            batch,
            relations,
            device=node.device,
            dtype=relation_mass.dtype,
        )
        relation_present.scatter_add_(
            1,
            relation_index,
            edge_valid_mask.to(relation_mass.dtype),
        )
        relation_present = relation_present.gt(0)
        supported_relation_mass = (
            relation_mass
            * relation_present.to(relation_mass.dtype)
        ).sum(dim=-1).clamp(0.0, 1.0)
        graph_support_activity = (
            semantic_activity * supported_relation_mass
        ).clamp(0.0, 1.0)
        source_summary = (
            (node * source_weight.unsqueeze(-1)).sum(dim=1)
            * graph_support_activity[:, None]
        )
        target_summary = (
            (node * target_weight.unsqueeze(-1)).sum(dim=1)
            * graph_support_activity[:, None]
        )

        return {
            "field_states": node,
            "edge_states": last_edge,
            "source_weight": source_weight,
            "target_weight": target_weight,
            "source_summary": source_summary,
            "target_summary": target_summary,
            "semantic_activity": semantic_activity,
            "graph_support_activity": graph_support_activity,
            "supported_relation_mass": supported_relation_mass,
            "evidence_tokens": torch.stack([source_summary, target_summary], dim=1),
            "evidence_mask": torch.ones(batch, 2, dtype=torch.bool, device=node.device),
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "relation_identity_parameters": 0,
            "relation_count_dependent_parameters": 0,
            "message_step_dependent_parameters": 0,
            "field_count_dependent_parameters": 0,
            "edge_count_dependent_parameters": 0,
            "runtime_relation_schema": True,
            "continuous_relation_conditioning": True,
            "soft_relational_activity_gate": True,
            "zero_activity_preserves_pre_message_graph_state": True,
            "zero_edge_graph_summary_is_zero": True,
            "unsupported_relation_mass_cannot_activate_graph_views": True,
            "duplicate_edges_do_not_inflate_supported_relation_mass": True,
            "relation_mass_floor": 0.0,
            "continuous_node_update_gate": True,
            "runtime_relation_symmetry": True,
            "symmetric_edge_message_exchange": True,
            "symmetric_edge_endpoint_order_invariant": True,
            "exact_structural_sparsity": False,
            "dual_endpoint_read": True,
            "field_count_ceiling": None,
            "edge_count_ceiling": None,
            "relation_count_ceiling": None,
            "message_step_ceiling": None,
        }
