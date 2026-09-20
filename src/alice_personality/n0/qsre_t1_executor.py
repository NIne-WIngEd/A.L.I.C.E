from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from alice_personality.n0.qsre_mechanics import (
    QSRE_RELATIONAL,
    qsre_control_state,
    support_local_field_mask,
    support_local_softmax,
)


QSRE_T1_OPERATION_ROLE_SELECT = 0
QSRE_T1_OPERATION_PATH_FOLLOW = 1
QSRE_T1_OPERATION_AGGREGATE = 2
QSRE_T1_OPERATION_PREFER_RELIABILITY = 3
QSRE_T1_OPERATION_PREFER_LATEST = 4


@dataclass(frozen=True)
class QSRET1Config:
    field_state_dim: int
    field_metadata_dim: int
    edge_metadata_dim: int
    operator_context_dim: int
    model_dim: int
    num_relations: int
    num_roles: int
    num_operations: int
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("field_state_dim", self.field_state_dim),
            ("field_metadata_dim", self.field_metadata_dim),
            ("edge_metadata_dim", self.edge_metadata_dim),
            ("operator_context_dim", self.operator_context_dim),
            ("model_dim", self.model_dim),
            ("num_relations", self.num_relations),
            ("num_roles", self.num_roles),
            ("num_operations", self.num_operations),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


@dataclass(frozen=True)
class QSRET1OracleOperator:
    relation_sequence_id: Tensor
    relation_sequence_mask: Tensor
    role_id: Tensor
    operation_id: Tensor
    focus_field_weight: Tensor
    context: Tensor
    applicability: Tensor

    def validate(
        self,
        *,
        batch: int,
        fields: int,
        num_relations: int,
        num_roles: int,
        num_operations: int,
        context_dim: int,
    ) -> None:
        if self.relation_sequence_id.ndim != 2:
            raise ValueError("relation_sequence_id must be [B,S]")
        if self.relation_sequence_id.size(0) != batch:
            raise ValueError("relation sequence batch drift")
        if self.relation_sequence_mask.shape != self.relation_sequence_id.shape:
            raise ValueError("relation_sequence_mask shape drift")
        if self.relation_sequence_mask.dtype != torch.bool:
            raise ValueError("relation_sequence_mask must be bool")
        if self.role_id.shape != (batch,):
            raise ValueError("role_id must be [B]")
        if self.operation_id.shape != (batch,):
            raise ValueError("operation_id must be [B]")
        if self.focus_field_weight.shape != (batch, fields):
            raise ValueError("focus_field_weight must be [B,F]")
        if self.context.shape != (batch, context_dim):
            raise ValueError("operator context shape drift")
        if self.applicability.shape != (batch,):
            raise ValueError("applicability must be [B]")

        if bool(self.relation_sequence_mask.any()):
            rel = self.relation_sequence_id[self.relation_sequence_mask]
            if int(rel.min()) < 0 or int(rel.max()) >= num_relations:
                raise ValueError("relation_sequence_id outside checkpoint vocabulary")
        if int(self.role_id.min()) < 0 or int(self.role_id.max()) >= num_roles:
            raise ValueError("role_id outside checkpoint vocabulary")
        if int(self.operation_id.min()) < 0 or int(self.operation_id.max()) >= num_operations:
            raise ValueError("operation_id outside checkpoint vocabulary")

        path_rows = self.operation_id.eq(QSRE_T1_OPERATION_PATH_FOLLOW)
        if bool(path_rows.any()):
            if bool((self.focus_field_weight[path_rows].sum(dim=-1) <= 0).any()):
                raise ValueError("PATH_FOLLOW requires a non-empty oracle focus frontier")


class QSRET1Executor(nn.Module):
    """T1 relational executor with oracle operator and oracle structural support.

    T1 intentionally has no natural-language operator extractor, learned support
    selector, or parent-global-logit input. Direct field support is forbidden:
    the T1 relational domain is induced only by oracle directed edges/paths.
    """

    def __init__(self, config: QSRET1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        self.node_projection = nn.Linear(config.field_state_dim, d)
        self.field_metadata_projection = nn.Linear(config.field_metadata_dim, d)
        self.edge_metadata_projection = nn.Linear(config.edge_metadata_dim, d)
        self.operator_context_projection = nn.Linear(config.operator_context_dim, d)

        self.relation_embedding = nn.Embedding(config.num_relations, d)
        self.role_embedding = nn.Embedding(config.num_roles, d)
        self.operation_embedding = nn.Embedding(config.num_operations, d)
        self.endpoint_position_embedding = nn.Embedding(2, d)
        self.focus_embedding = nn.Parameter(torch.zeros(d))

        self.edge_update = nn.Sequential(
            nn.Linear(7 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.source_message = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.target_message = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.node_update = nn.GRUCell(2 * d, d)
        self.readout = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def parameter_report(self) -> dict[str, int]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def forward(
        self,
        *,
        field_state: Tensor,
        field_metadata: Tensor,
        field_valid_mask: Tensor,
        edge_index: Tensor,
        edge_relation_id: Tensor,
        edge_metadata: Tensor,
        edge_valid_mask: Tensor,
        field_support_weight: Tensor,
        edge_support_weight: Tensor,
        operator: QSRET1OracleOperator,
    ) -> dict[str, Tensor]:
        if field_state.ndim != 3:
            raise ValueError("field_state must be [B,F,D]")
        batch, fields, field_dim = field_state.shape
        if field_dim != self.config.field_state_dim:
            raise ValueError("field_state width drift")
        if field_metadata.shape != (
            batch,
            fields,
            self.config.field_metadata_dim,
        ):
            raise ValueError("field_metadata shape drift")
        if (
            field_valid_mask.shape != (batch, fields)
            or field_valid_mask.dtype != torch.bool
        ):
            raise ValueError("field_valid_mask must be bool [B,F]")

        if edge_index.ndim != 3 or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [B,E,2]")
        if edge_index.size(0) != batch:
            raise ValueError("edge batch drift")
        edges = edge_index.size(1)
        if edge_relation_id.shape != (batch, edges):
            raise ValueError("edge_relation_id shape drift")
        if edge_metadata.shape != (
            batch,
            edges,
            self.config.edge_metadata_dim,
        ):
            raise ValueError("edge_metadata shape drift")
        if (
            edge_valid_mask.shape != (batch, edges)
            or edge_valid_mask.dtype != torch.bool
        ):
            raise ValueError("edge_valid_mask must be bool [B,E]")
        if field_support_weight.shape != (batch, fields):
            raise ValueError("field_support_weight shape drift")
        if edge_support_weight.shape != (batch, edges):
            raise ValueError("edge_support_weight shape drift")

        if bool((field_support_weight != 0).any()):
            raise ValueError(
                "T1 forbids direct field support; oracle relational support "
                "must be directed edge/path membership"
            )
        if bool(
            ((edge_support_weight != 0) & (edge_support_weight != 1)).any()
        ):
            raise ValueError(
                "T1 oracle edge_support_weight must be exact membership 0/1"
            )

        if bool(edge_valid_mask.any()):
            valid_index = edge_index[edge_valid_mask]
            if int(valid_index.min()) < 0 or int(valid_index.max()) >= fields:
                raise ValueError("valid edge endpoint outside field range")
            valid_rel = edge_relation_id[edge_valid_mask]
            if (
                int(valid_rel.min()) < 0
                or int(valid_rel.max()) >= self.config.num_relations
            ):
                raise ValueError("edge relation outside checkpoint vocabulary")

        operator.validate(
            batch=batch,
            fields=fields,
            num_relations=self.config.num_relations,
            num_roles=self.config.num_roles,
            num_operations=self.config.num_operations,
            context_dim=self.config.operator_context_dim,
        )

        support_mask = support_local_field_mask(
            field_valid_mask,
            field_support_weight,
            edge_index,
            edge_support_weight,
            edge_valid_mask,
        )

        node = (
            self.node_projection(field_state)
            + self.field_metadata_projection(field_metadata)
            + operator.focus_field_weight.unsqueeze(-1)
            * self.focus_embedding.view(1, 1, -1)
        )
        node = node * field_valid_mask.unsqueeze(-1).to(node.dtype)

        role_state = self.role_embedding(operator.role_id)
        operation_state = self.operation_embedding(operator.operation_id)
        context_state = self.operator_context_projection(operator.context)
        base_operator = role_state + operation_state + context_state

        edge_rel_state = self.relation_embedding(
            edge_relation_id.clamp(
                min=0,
                max=self.config.num_relations - 1,
            )
        )
        edge_meta_state = self.edge_metadata_projection(edge_metadata)
        source_position = self.endpoint_position_embedding.weight[0].view(
            1, 1, -1
        )
        target_position = self.endpoint_position_embedding.weight[1].view(
            1, 1, -1
        )

        batch_index = torch.arange(
            batch,
            device=field_state.device,
        ).unsqueeze(1)
        batch_index = batch_index.expand(batch, edges)

        source_index = edge_index[..., 0].clamp(
            min=0,
            max=max(fields - 1, 0),
        )
        target_index = edge_index[..., 1].clamp(
            min=0,
            max=max(fields - 1, 0),
        )

        last_edge_state = torch.zeros(
            batch,
            edges,
            self.config.model_dim,
            device=field_state.device,
            dtype=field_state.dtype,
        )

        is_path = operator.operation_id.eq(QSRE_T1_OPERATION_PATH_FOLLOW)
        frontier = (
            operator.focus_field_weight > 0
        ) & field_valid_mask

        steps = operator.relation_sequence_id.size(1)
        for step in range(steps):
            step_active = operator.relation_sequence_mask[:, step]
            requested_relation = operator.relation_sequence_id[:, step].clamp(
                min=0,
                max=self.config.num_relations - 1,
            )
            query_relation_state = self.relation_embedding(requested_relation)
            q_step = base_operator + query_relation_state

            source_node = node[batch_index, source_index]
            target_node = node[batch_index, target_index]
            q_edge = q_step[:, None, :].expand(batch, edges, -1)

            edge_input = torch.cat(
                [
                    source_node,
                    target_node,
                    edge_rel_state,
                    edge_meta_state,
                    q_edge,
                    source_position.expand(batch, edges, -1),
                    target_position.expand(batch, edges, -1),
                ],
                dim=-1,
            )
            edge_state = self.edge_update(edge_input)
            last_edge_state = edge_state

            relation_match = edge_relation_id.eq(
                requested_relation[:, None]
            )
            base_active = (
                (edge_support_weight > 0)
                & edge_valid_mask
                & relation_match
                & step_active[:, None]
            )
            source_on_frontier = frontier.gather(1, source_index)
            active_edge = base_active & (
                ~is_path[:, None] | source_on_frontier
            )
            gate = active_edge.to(edge_support_weight.dtype)

            msg_input = torch.cat([edge_state, q_edge], dim=-1)
            source_message = (
                self.source_message(msg_input) * gate.unsqueeze(-1)
            )
            target_message = (
                self.target_message(msg_input) * gate.unsqueeze(-1)
            )

            aggregate = torch.zeros_like(node)
            source_scatter = source_index.unsqueeze(-1).expand(
                -1, -1, self.config.model_dim
            )
            target_scatter = target_index.unsqueeze(-1).expand(
                -1, -1, self.config.model_dim
            )
            aggregate.scatter_add_(
                1,
                source_scatter,
                source_message,
            )
            aggregate.scatter_add_(
                1,
                target_scatter,
                target_message,
            )

            active_node_count = torch.zeros(
                batch,
                fields,
                device=field_state.device,
                dtype=torch.long,
            )
            active_node_count.scatter_add_(
                1,
                source_index,
                active_edge.long(),
            )
            active_node_count.scatter_add_(
                1,
                target_index,
                active_edge.long(),
            )
            active_node = active_node_count > 0

            q_node = q_step[:, None, :].expand(batch, fields, -1)
            update_input = torch.cat([aggregate, q_node], dim=-1)
            updated = self.node_update(
                update_input.reshape(batch * fields, -1),
                node.reshape(batch * fields, -1),
            ).reshape(batch, fields, -1)

            update_mask = (
                active_node
                & field_valid_mask
                & step_active[:, None]
            )
            node = torch.where(
                update_mask.unsqueeze(-1),
                updated,
                node,
            )

            if bool(is_path.any()):
                next_frontier_count = torch.zeros(
                    batch,
                    fields,
                    device=field_state.device,
                    dtype=torch.long,
                )
                path_active = active_edge & is_path[:, None]
                next_frontier_count.scatter_add_(
                    1,
                    target_index,
                    path_active.long(),
                )
                next_frontier = next_frontier_count > 0
                path_step = (
                    is_path[:, None] & step_active[:, None]
                )
                frontier = torch.where(
                    path_step,
                    next_frontier,
                    frontier,
                )

        relation_mask_f = operator.relation_sequence_mask.to(
            field_state.dtype
        )
        relation_states = self.relation_embedding(
            operator.relation_sequence_id.clamp(
                min=0,
                max=self.config.num_relations - 1,
            )
        )
        relation_sum = (
            relation_states * relation_mask_f.unsqueeze(-1)
        ).sum(dim=1)
        relation_count = relation_mask_f.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        operator_summary = (
            base_operator + relation_sum / relation_count
        )

        path_readout = (
            frontier & support_mask & field_valid_mask
        )
        readout_mask = torch.where(
            is_path[:, None],
            path_readout,
            support_mask,
        )

        control_state = qsre_control_state(
            operator.applicability,
            readout_mask.any(dim=-1),
        )

        q_readout = operator_summary[:, None, :].expand(
            batch,
            fields,
            -1,
        )
        readout_input = torch.cat(
            [node, q_readout],
            dim=-1,
        )
        relational_logit = self.readout(
            readout_input
        ).squeeze(-1)

        relational_probability = support_local_softmax(
            relational_logit,
            readout_mask,
        )
        relational_rows = control_state.eq(QSRE_RELATIONAL)
        relational_probability = (
            relational_probability
            * relational_rows[:, None].to(
                relational_probability.dtype
            )
        )

        return {
            "node_state": node,
            "last_edge_state": last_edge_state,
            "support_field_mask": support_mask,
            "readout_field_mask": readout_mask,
            "path_frontier": frontier,
            "control_state": control_state,
            "relational_logit": relational_logit,
            "relational_probability": relational_probability,
            "edge_support_used": edge_support_weight,
            "field_support_used": field_support_weight,
        }
