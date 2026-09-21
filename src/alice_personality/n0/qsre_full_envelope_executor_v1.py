from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from alice_personality.n0.full_envelope_structural_types import (
    CONTROL_RELATIONAL,
    DIRECTION_BIDIRECTIONAL,
    DIRECTION_FORWARD,
    DIRECTION_REVERSE,
    MOD_PROVENANCE_CONSTRAINT,
    MOD_RECENCY,
    MOD_RELIABILITY,
    MOD_TEMPORAL_CONSTRAINT,
    ROLE_NONE,
    ROLE_SOURCE,
    ROLE_SYMMETRIC,
    ROLE_TARGET,
    TRAVERSAL_AGGREGATE,
    TRAVERSAL_LOCAL,
    TRAVERSAL_PATH,
    FullEnvelopeOperatorState,
)


@dataclass(frozen=True)
class FullEnvelopeExecutorConfig:
    field_dim: int = 640
    model_dim: int = 640
    field_metadata_dim: int = 3
    edge_metadata_dim: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("field_dim", self.field_dim),
            ("model_dim", self.model_dim),
            ("field_metadata_dim", self.field_metadata_dim),
            ("edge_metadata_dim", self.edge_metadata_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class FullEnvelopeQSREExecutorV1(nn.Module):
    """Shared dynamic-schema executor with no learned factor-ID semantics.

    Relation meaning arrives as runtime schema state. Role/traversal/direction
    distributions are selected semantically upstream and consumed here as
    structural probabilities. No relation/role/traversal/direction embedding
    table owns their meaning.
    """

    def __init__(self, config: FullEnvelopeExecutorConfig | None = None) -> None:
        super().__init__()
        self.config = config or FullEnvelopeExecutorConfig()
        self.config.validate()
        d = self.config.model_dim

        self.node_projection = nn.Linear(self.config.field_dim, d)
        self.field_metadata_projection = nn.Linear(self.config.field_metadata_dim, d)
        self.relation_projection = nn.Linear(d, d, bias=False)
        self.operator_projection = nn.Linear(d, d, bias=False)
        self.source_position = nn.Parameter(torch.empty(d))
        self.target_position = nn.Parameter(torch.empty(d))

        structural_scalar_dim = 4 + 4 + 3 + 3 + 4 + 2
        self.edge_update = nn.Sequential(
            nn.Linear(6 * d + structural_scalar_dim, 2 * d),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.source_message = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.target_message = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.node_update = nn.GRUCell(2 * d, d)
        self.readout = nn.Sequential(
            nn.Linear(3 * d + 8, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.summary_projection = nn.Linear(2 * d, d)

        nn.init.normal_(self.source_position, std=0.02)
        nn.init.normal_(self.target_position, std=0.02)

    def forward(
        self,
        *,
        field_state: Tensor,
        field_metadata: Tensor,
        field_valid_mask: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        edge_valid_mask: Tensor,
        edge_support_weight: Tensor,
        support_available: Tensor,
        edge_reliability: Tensor,
        edge_recency: Tensor,
        edge_temporal_match: Tensor,
        edge_provenance_match: Tensor,
        relation_schema_state: Tensor,
        relation_symmetric: Tensor,
        operator: FullEnvelopeOperatorState,
        focus_field_weight: Tensor,
    ) -> dict[str, Tensor]:
        if field_state.ndim != 3:
            raise ValueError("field_state must be [B,F,D]")
        batch, fields, width = field_state.shape
        if width != self.config.field_dim:
            raise ValueError("field state width drift")
        if field_metadata.shape != (batch, fields, self.config.field_metadata_dim):
            raise ValueError("field metadata shape drift")
        if field_valid_mask.shape != (batch, fields) or field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool [B,F]")
        if bool((field_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one valid field")
        if relation_schema_state.ndim != 3 or relation_schema_state.size(0) != batch:
            raise ValueError("relation_schema_state must be [B,R,D]")
        relation_count = relation_schema_state.size(1)
        if relation_schema_state.size(-1) != self.config.model_dim:
            raise ValueError("relation schema width drift")
        if relation_symmetric.shape != (batch, relation_count) or relation_symmetric.dtype != torch.bool:
            raise ValueError("relation_symmetric must be bool [B,R]")
        operator.validate(relation_count=relation_count, model_dim=self.config.model_dim)

        if edge_index.ndim != 3 or edge_index.size(0) != batch or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [B,E,2]")
        edges = edge_index.size(1)
        expected = (batch, edges)
        for name, value in (
            ("edge_relation_index", edge_relation_index),
            ("edge_valid_mask", edge_valid_mask),
            ("edge_support_weight", edge_support_weight),
            ("edge_reliability", edge_reliability),
            ("edge_recency", edge_recency),
            ("edge_temporal_match", edge_temporal_match),
            ("edge_provenance_match", edge_provenance_match),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} shape drift")
        if edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool")
        if focus_field_weight.shape != (batch, fields):
            raise ValueError("focus_field_weight must be [B,F]")
        if support_available.shape != (batch,):
            raise ValueError("support_available must be [B]")

        if bool(edge_valid_mask.any()):
            valid_edge = edge_index[edge_valid_mask]
            if int(valid_edge.min()) < 0 or int(valid_edge.max()) >= fields:
                raise ValueError("edge endpoint outside field set")
            valid_relation = edge_relation_index[edge_valid_mask]
            if int(valid_relation.min()) < 0 or int(valid_relation.max()) >= relation_count:
                raise ValueError("edge relation outside runtime schema")

        node = (
            self.node_projection(field_state.float())
            + self.field_metadata_projection(field_metadata.float())
        )
        node = node * field_valid_mask.unsqueeze(-1).to(node.dtype)
        relation_state = self.relation_projection(relation_schema_state.float())
        operator_state = self.operator_projection(operator.continuous_state.float())

        source_index = edge_index[..., 0].clamp(min=0, max=fields - 1)
        target_index = edge_index[..., 1].clamp(min=0, max=fields - 1)
        relation_index = edge_relation_index.clamp(min=0, max=relation_count - 1)
        b = torch.arange(batch, device=edge_index.device)[:, None].expand(batch, edges)
        edge_relation_state = relation_state[b, relation_index]
        edge_symmetric_mask = relation_symmetric[b, relation_index]

        origin_focus = focus_field_weight.clamp(min=0.0, max=1.0)
        frontier = origin_focus
        local_probability = operator.traversal_distribution[:, TRAVERSAL_LOCAL]
        path_probability = operator.traversal_distribution[:, TRAVERSAL_PATH]
        aggregate_probability = operator.traversal_distribution[:, TRAVERSAL_AGGREGATE]
        last_edge_state = torch.zeros(
            batch, edges, self.config.model_dim, device=node.device, dtype=node.dtype
        )

        structural = torch.cat(
            [
                operator.role_distribution[:, :4],
                operator.traversal_distribution[:, :3],
                operator.direction_distribution[:, :3],
                operator.modifier_weight[:, :4],
                operator.applicability[:, None],
                operator.uncertainty[:, None],
            ],
            dim=-1,
        )

        for step in range(operator.relation_distribution.size(1)):
            relation_distribution = operator.relation_distribution[:, step]
            step_mass = operator.relation_step_mass[:, step]
            edge_relation_mass = relation_distribution.gather(1, relation_index)

            source_frontier = frontier.gather(1, source_index)
            target_frontier = frontier.gather(1, target_index)
            source_origin = origin_focus.gather(1, source_index)
            target_origin = origin_focus.gather(1, target_index)
            forward = operator.direction_distribution[:, DIRECTION_FORWARD][:, None]
            reverse = operator.direction_distribution[:, DIRECTION_REVERSE][:, None]
            bidir = operator.direction_distribution[:, DIRECTION_BIDIRECTIONAL][:, None]
            edge_symmetric = edge_symmetric_mask.to(forward.dtype)
            directed = 1.0 - edge_symmetric
            effective_forward = forward * directed
            effective_reverse = reverse * directed
            effective_bidir = (
                bidir + edge_symmetric * (forward + reverse)
            ).clamp(0.0, 1.0)

            reliability_multiplier = (
                1.0
                + operator.modifier_weight[:, MOD_RELIABILITY][:, None]
                * (edge_reliability - 0.5)
            ).clamp_min(0.0)
            recency_multiplier = (
                1.0
                + operator.modifier_weight[:, MOD_RECENCY][:, None]
                * (edge_recency - 0.5)
            ).clamp_min(0.0)
            temporal_multiplier = (
                1.0
                - operator.modifier_weight[:, MOD_TEMPORAL_CONSTRAINT][:, None]
                * (1.0 - edge_temporal_match)
            ).clamp_min(0.0)
            provenance_multiplier = (
                1.0
                - operator.modifier_weight[:, MOD_PROVENANCE_CONSTRAINT][:, None]
                * (1.0 - edge_provenance_match)
            ).clamp_min(0.0)

            common = (
                edge_support_weight
                * edge_valid_mask.to(edge_support_weight.dtype)
                * edge_relation_mass
                * step_mass[:, None]
                * reliability_multiplier
                * recency_multiplier
                * temporal_multiplier
                * provenance_multiplier
            )
            forward_gate = common * source_frontier * effective_forward
            reverse_gate = common * target_frontier * effective_reverse
            bidir_forward = common * source_frontier * effective_bidir
            bidir_reverse = common * target_frontier * effective_bidir
            path_gate = (
                forward_gate
                + reverse_gate
                + bidir_forward
                + bidir_reverse
            ).clamp(max=1.0)

            local_forward = common * source_origin * effective_forward
            local_reverse = common * target_origin * effective_reverse
            local_bidir = common * torch.maximum(
                source_origin,
                target_origin,
            ) * effective_bidir
            local_gate = (
                local_forward + local_reverse + local_bidir
            ).clamp(max=1.0)

            aggregate_gate = common
            gate = (
                local_probability[:, None] * local_gate
                + path_probability[:, None] * path_gate
                + aggregate_probability[:, None] * aggregate_gate
            ).clamp(max=1.0)

            source_node = node[b, source_index]
            target_node = node[b, target_index]
            pair_mean = 0.5 * (source_node + target_node)
            pair_delta = (source_node - target_node).abs()
            edge_source_node = torch.where(
                edge_symmetric_mask.unsqueeze(-1),
                pair_mean,
                source_node,
            )
            edge_target_node = torch.where(
                edge_symmetric_mask.unsqueeze(-1),
                pair_delta,
                target_node,
            )
            op = operator_state[:, None, :].expand(batch, edges, -1)
            source_pos = self.source_position.view(1, 1, -1).expand(batch, edges, -1)
            target_pos = self.target_position.view(1, 1, -1).expand(batch, edges, -1)
            position_mean = 0.5 * (source_pos + target_pos)
            position_delta = (source_pos - target_pos).abs()
            edge_source_pos = torch.where(
                edge_symmetric_mask.unsqueeze(-1),
                position_mean,
                source_pos,
            )
            edge_target_pos = torch.where(
                edge_symmetric_mask.unsqueeze(-1),
                position_delta,
                target_pos,
            )
            edge_scalar = torch.stack(
                [edge_reliability, edge_recency, edge_temporal_match, edge_provenance_match],
                dim=-1,
            )
            structural_edge = structural[:, None, :].expand(batch, edges, -1)
            edge_state = self.edge_update(
                torch.cat(
                    [
                        edge_source_node,
                        edge_target_node,
                        edge_relation_state,
                        op,
                        edge_source_pos,
                        edge_target_pos,
                        edge_scalar,
                        structural_edge,
                    ],
                    dim=-1,
                )
            )
            last_edge_state = edge_state

            source_message = self.source_message(torch.cat([edge_state, op], dim=-1))
            target_message = self.target_message(torch.cat([edge_state, op], dim=-1))
            symmetric_message = 0.5 * (source_message + target_message)
            source_message = torch.where(
                edge_symmetric_mask.unsqueeze(-1),
                symmetric_message,
                source_message,
            )
            target_message = torch.where(
                edge_symmetric_mask.unsqueeze(-1),
                symmetric_message,
                target_message,
            )
            source_message = source_message * gate.unsqueeze(-1)
            target_message = target_message * gate.unsqueeze(-1)

            aggregate = torch.zeros_like(node)
            source_scatter = source_index.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
            target_scatter = target_index.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
            aggregate.scatter_add_(1, source_scatter, source_message)
            aggregate.scatter_add_(1, target_scatter, target_message)

            active = torch.zeros(batch, fields, device=node.device, dtype=node.dtype)
            active.scatter_add_(1, source_index, gate)
            active.scatter_add_(1, target_index, gate)
            active = active.clamp(0.0, 1.0) * field_valid_mask.to(node.dtype)
            q_node = operator_state[:, None, :].expand(batch, fields, -1)
            updated = self.node_update(
                torch.cat([aggregate, q_node], dim=-1).reshape(batch * fields, -1),
                node.reshape(batch * fields, -1),
            ).reshape(batch, fields, -1)
            node = (
                active.unsqueeze(-1) * updated
                + (1.0 - active.unsqueeze(-1)) * node
            )

            next_frontier = torch.zeros_like(frontier)
            next_frontier.scatter_add_(1, target_index, forward_gate + bidir_forward)
            next_frontier.scatter_add_(1, source_index, reverse_gate + bidir_reverse)
            next_frontier = next_frontier.clamp(max=1.0)
            stepped = (
                next_frontier
                + (1.0 - step_mass[:, None]).clamp(0.0, 1.0) * frontier
            ).clamp(max=1.0)
            frontier = (
                path_probability[:, None] * stepped
                + (1.0 - path_probability[:, None]) * origin_focus
            )

        source_support = torch.zeros(batch, fields, device=node.device, dtype=node.dtype)
        target_support = torch.zeros_like(source_support)
        source_support.scatter_add_(1, source_index, edge_support_weight)
        target_support.scatter_add_(1, target_index, edge_support_weight)
        symmetric_support = (
            edge_support_weight
            * edge_symmetric_mask.to(edge_support_weight.dtype)
        )
        source_support.scatter_add_(1, target_index, symmetric_support)
        target_support.scatter_add_(1, source_index, symmetric_support)
        source_support = source_support.clamp(max=1.0)
        target_support = target_support.clamp(max=1.0)
        aggregate_union = (source_support + target_support).clamp(max=1.0)

        local_edge_focus = torch.maximum(
            origin_focus.gather(1, source_index),
            origin_focus.gather(1, target_index),
        )
        local_edge_support = edge_support_weight * local_edge_focus
        local_source_support = torch.zeros_like(source_support)
        local_target_support = torch.zeros_like(target_support)
        local_source_support.scatter_add_(1, source_index, local_edge_support)
        local_target_support.scatter_add_(1, target_index, local_edge_support)
        symmetric_local_support = (
            local_edge_support
            * edge_symmetric_mask.to(local_edge_support.dtype)
        )
        local_source_support.scatter_add_(
            1,
            target_index,
            symmetric_local_support,
        )
        local_target_support.scatter_add_(
            1,
            source_index,
            symmetric_local_support,
        )
        local_source_support = local_source_support.clamp(max=1.0)
        local_target_support = local_target_support.clamp(max=1.0)
        local_union = (
            local_source_support + local_target_support
        ).clamp(max=1.0)

        source_role = operator.role_distribution[:, ROLE_SOURCE][:, None]
        target_role = operator.role_distribution[:, ROLE_TARGET][:, None]
        symmetric_role = operator.role_distribution[:, ROLE_SYMMETRIC][:, None]
        none_role = operator.role_distribution[:, ROLE_NONE][:, None]
        local_role = (
            source_role * local_source_support
            + target_role * local_target_support
            + symmetric_role * local_union
            + none_role * local_union
        ).clamp(0.0, 1.0)
        aggregate_role = (
            source_role * source_support
            + target_role * target_support
            + symmetric_role * aggregate_union
            + none_role * aggregate_union
        ).clamp(0.0, 1.0)

        path_origin = origin_focus
        path_reached = frontier.clamp(0.0, 1.0)
        path_union = (path_origin + path_reached).clamp(max=1.0)
        forward = operator.direction_distribution[:, DIRECTION_FORWARD][:, None]
        reverse = operator.direction_distribution[:, DIRECTION_REVERSE][:, None]
        bidir = operator.direction_distribution[:, DIRECTION_BIDIRECTIONAL][:, None]
        semantic_source = (forward * path_origin + reverse * path_reached + bidir * path_union).clamp(max=1.0)
        semantic_target = (forward * path_reached + reverse * path_origin + bidir * path_union).clamp(max=1.0)
        path_role = (
            source_role * semantic_source
            + target_role * semantic_target
            + symmetric_role * path_union
            + none_role * path_union
        ).clamp(0.0, 1.0)
        role_weight = (
            local_probability[:, None] * local_role
            + path_probability[:, None] * path_role
            + aggregate_probability[:, None] * aggregate_role
        ).clamp(0.0, 1.0)
        readout_mask = (role_weight > 0) & field_valid_mask

        op_node = operator_state[:, None, :].expand(batch, fields, -1)
        structural_node = torch.cat(
            [
                operator.role_distribution[:, :4],
                operator.control_distribution[:, :3],
                operator.applicability[:, None],
            ],
            dim=-1,
        )[:, None, :].expand(batch, fields, -1)
        readout_logit = self.readout(
            torch.cat([node, op_node, node * op_node, structural_node], dim=-1)
        ).squeeze(-1)
        readout_logit = readout_logit + torch.log(role_weight.clamp_min(1.0e-8))
        probability = torch.softmax(readout_logit.masked_fill(~readout_mask, -1.0e4), dim=-1)
        probability = probability * readout_mask.to(probability.dtype)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

        known = (1.0 - operator.unknown_probability.sum(dim=1).clamp(max=1.0)).clamp(0.0, 1.0)
        program = operator.relation_step_mass.sum(dim=1).clamp(0.0, 1.0)
        execution_confidence = (
            operator.applicability
            * operator.control_distribution[:, CONTROL_RELATIONAL]
            * known
            * program
            * support_available.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        probability = probability * execution_confidence[:, None]

        normalized = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        node_summary = torch.einsum("bf,bfd->bd", normalized, node)
        relational_summary = self.summary_projection(
            torch.cat([node_summary, operator_state], dim=-1)
        )

        return {
            "node_state": node,
            "last_edge_state": last_edge_state,
            "path_frontier": frontier,
            "source_support_weight": source_support,
            "target_support_weight": target_support,
            "structural_role_weight": role_weight,
            "readout_mask": readout_mask,
            "relational_probability": probability,
            "relational_summary": relational_summary,
            "execution_confidence": execution_confidence,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_identity_parameters": 0,
            "role_identity_parameters": 0,
            "traversal_identity_parameters": 0,
            "direction_identity_parameters": 0,
            "factor_identity_parameters": 0,
            "relation_count_dependent_parameters": 0,
            "hop_count_dependent_parameters": 0,
            "shared_iterative_execution": True,
            "continuous_node_update_gate": True,
            "hard_active_edge_update_threshold": False,
            "runtime_relation_schema": True,
            "runtime_relation_symmetry": True,
            "symmetric_relation_direction_collapses_to_bidirectional": True,
            "symmetric_relation_endpoint_order_invariant": True,
            "structural_factor_probabilities": True,
            "continuous_traversal_mixture": True,
            "local_path_aggregate_distinct": True,
            "hard_traversal_threshold": False,
            "execution_confidence_requires_structural_support": True,
            "relation_count_ceiling": None,
            "hop_count_ceiling": None,
            "field_count_ceiling": None,
            "edge_count_ceiling": None,
        }
