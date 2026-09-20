from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alice_personality.n0.qsre_t1_executor import QSRET1OracleOperator


QSRE_T2_CONTROL_FALLBACK = 0
QSRE_T2_CONTROL_RELATIONAL = 1
QSRE_T2_CONTROL_DEFER = 2


@dataclass(frozen=True)
class QSRET2OperatorConfig:
    semantic_dim: int = 640
    model_dim: int = 512
    num_hidden_states: int = 17
    num_attention_heads: int = 8
    refinement_layers: int = 2
    num_relations: int = 6
    num_roles: int = 4
    num_operations: int = 5
    num_controls: int = 3
    max_relation_steps: int = 2
    executor_context_dim: int = 4
    dropout: float = 0.05

    def validate(self) -> None:
        positive = {
            "semantic_dim": self.semantic_dim,
            "model_dim": self.model_dim,
            "num_hidden_states": self.num_hidden_states,
            "num_attention_heads": self.num_attention_heads,
            "refinement_layers": self.refinement_layers,
            "num_relations": self.num_relations,
            "num_roles": self.num_roles,
            "num_operations": self.num_operations,
            "num_controls": self.num_controls,
            "max_relation_steps": self.max_relation_steps,
            "executor_context_dim": self.executor_context_dim,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class QSRET2OperatorEncoder(nn.Module):
    """Learn a QSRE operator from frozen multi-layer semantic token states.

    T2 intentionally receives no graph support, edge identity, oracle relation,
    oracle role, or parent field logits. It learns only query/operator semantics.
    """

    def __init__(self, config: QSRET2OperatorConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        self.token_norm = nn.LayerNorm(config.semantic_dim)
        self.token_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.layer_embedding = nn.Embedding(config.num_hidden_states, d)

        self.relation_slot_query = nn.Parameter(
            torch.empty(config.max_relation_steps, d)
        )
        self.role_query = nn.Parameter(torch.empty(1, d))
        self.operation_query = nn.Parameter(torch.empty(1, d))
        self.control_query = nn.Parameter(torch.empty(1, d))
        self.continuous_query = nn.Parameter(torch.empty(1, d))

        self.cross_attention = nn.MultiheadAttention(
            d,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_attention_heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refiner = nn.TransformerEncoder(
            layer,
            num_layers=config.refinement_layers,
        )
        self.output_norm = nn.LayerNorm(d)

        self.relation_anchor = nn.Parameter(
            torch.empty(config.num_relations + 1, d)
        )
        self.role_anchor = nn.Parameter(torch.empty(config.num_roles, d))
        self.operation_anchor = nn.Parameter(
            torch.empty(config.num_operations, d)
        )
        self.control_anchor = nn.Parameter(torch.empty(config.num_controls, d))
        self.continuous_projection = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

        self.reset_parameters()

    @property
    def none_relation_id(self) -> int:
        return self.config.num_relations

    @property
    def latent_count(self) -> int:
        return self.config.max_relation_steps + 4

    def reset_parameters(self) -> None:
        for value in (
            self.relation_slot_query,
            self.role_query,
            self.operation_query,
            self.control_query,
            self.continuous_query,
            self.relation_anchor,
            self.role_anchor,
            self.operation_anchor,
            self.control_anchor,
        ):
            nn.init.normal_(value, mean=0.0, std=0.02)

    def _anchor_logits(self, state: Tensor, anchors: Tensor) -> Tensor:
        state = F.normalize(state, dim=-1)
        anchors = F.normalize(anchors, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * torch.einsum("...d,kd->...k", state, anchors)

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
    ) -> dict[str, Tensor]:
        if query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,T,D]")
        batch, layers, tokens, semantic_dim = query_hidden_states.shape
        if layers != self.config.num_hidden_states:
            raise ValueError(
                f"expected {self.config.num_hidden_states} hidden states, got {layers}"
            )
        if semantic_dim != self.config.semantic_dim:
            raise ValueError("semantic width drift")
        if query_token_mask.shape != (batch, tokens):
            raise ValueError("query_token_mask must be [B,T]")
        if query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool")
        if bool((query_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every query requires at least one valid token")

        x = self.token_projection(self.token_norm(query_hidden_states))
        layer_ids = torch.arange(layers, device=x.device)
        layer_state = self.layer_embedding(layer_ids).view(1, layers, 1, -1)
        x = x + layer_state
        memory = x.reshape(batch, layers * tokens, self.config.model_dim)

        flat_valid = (
            query_token_mask[:, None, :]
            .expand(batch, layers, tokens)
            .reshape(batch, layers * tokens)
        )

        latent_template = torch.cat(
            [
                self.relation_slot_query,
                self.role_query,
                self.operation_query,
                self.control_query,
                self.continuous_query,
            ],
            dim=0,
        )
        latent = latent_template.unsqueeze(0).expand(batch, -1, -1)

        attended, attention = self.cross_attention(
            latent,
            memory,
            memory,
            key_padding_mask=~flat_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        latent = self.cross_norm(latent + attended)
        latent = self.output_norm(self.refiner(latent))

        s = self.config.max_relation_steps
        relation_state = latent[:, :s]
        role_state = latent[:, s]
        operation_state = latent[:, s + 1]
        control_state = latent[:, s + 2]
        continuous_state = self.continuous_projection(latent[:, s + 3])

        relation_logits = self._anchor_logits(
            relation_state, self.relation_anchor
        )
        role_logits = self._anchor_logits(role_state, self.role_anchor)
        operation_logits = self._anchor_logits(
            operation_state, self.operation_anchor
        )
        control_logits = self._anchor_logits(
            control_state, self.control_anchor
        )

        attn = attention.mean(dim=1).reshape(
            batch, self.latent_count, layers, tokens
        )
        attn = attn * query_token_mask[:, None, None, :].to(attn.dtype)
        layer_attention = attn.sum(dim=-1)

        probabilities = [
            torch.softmax(relation_logits, dim=-1),
            torch.softmax(role_logits, dim=-1),
            torch.softmax(operation_logits, dim=-1),
            torch.softmax(control_logits, dim=-1),
        ]
        entropies = []
        for probability in probabilities:
            denom = torch.log(
                torch.tensor(
                    float(probability.size(-1)),
                    device=probability.device,
                    dtype=probability.dtype,
                )
            ).clamp_min(1.0e-6)
            entropy = -(
                probability.clamp_min(1.0e-12)
                * probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1) / denom
            if entropy.ndim > 1:
                entropy = entropy.mean(dim=-1)
            entropies.append(entropy)
        uncertainty = torch.stack(entropies, dim=-1).mean(dim=-1)

        return {
            "relation_logits": relation_logits,
            "role_logits": role_logits,
            "operation_logits": operation_logits,
            "control_logits": control_logits,
            "continuous_state": continuous_state,
            "uncertainty": uncertainty,
            "latent_state": latent,
            "layer_attention": layer_attention,
            "token_layer_attention": attn,
        }

    @torch.no_grad()
    def decode_for_frozen_t1(
        self,
        output: dict[str, Tensor],
        *,
        focus_field_weight: Tensor,
    ) -> QSRET1OracleOperator:
        relation = output["relation_logits"].argmax(dim=-1)
        role = output["role_logits"].argmax(dim=-1)
        operation = output["operation_logits"].argmax(dim=-1)
        control = output["control_logits"].argmax(dim=-1)

        relation_mask = relation.lt(self.config.num_relations)
        relation_mask = torch.cumprod(
            relation_mask.long(), dim=-1
        ).bool()
        relational_control = control.eq(QSRE_T2_CONTROL_RELATIONAL)
        relation_mask = relation_mask & relational_control[:, None]

        safe_relation = relation.clamp(
            min=0, max=self.config.num_relations - 1
        )
        centers = torch.tensor(
            [0.10, 0.90, 0.50],
            device=control.device,
            dtype=focus_field_weight.dtype,
        )
        applicability = centers[control]

        context = torch.zeros(
            focus_field_weight.size(0),
            self.config.executor_context_dim,
            device=focus_field_weight.device,
            dtype=focus_field_weight.dtype,
        )

        return QSRET1OracleOperator(
            relation_sequence_id=safe_relation,
            relation_sequence_mask=relation_mask,
            role_id=role,
            operation_id=operation,
            focus_field_weight=focus_field_weight,
            context=context,
            applicability=applicability,
        )

    def parameter_report(self) -> dict[str, int | bool]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "semantic_backbone_parameters": 0,
            "oracle_relation_input": False,
            "oracle_role_input": False,
            "graph_support_input": False,
            "multi_layer_token_input": True,
            "schema_anchor_readout": True,
            "continuous_residual_state": True,
            "hard_layer_mask": False,
            "max_relation_steps_is_product_ceiling": False,
        }
