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
class QSRET2SchemaOrderedConfig:
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


class QSRET2SchemaOrderedOperatorEncoder(nn.Module):
    """T2 operator encoder with schema-grounded ordered relation decoding.

    The frozen semantic backbone remains outside this module. Query tokens from
    all hidden states are consumed exactly as in T2 v0.2. Role, operation,
    control, and continuous-state heads retain the previous learned-anchor path.

    The causal architectural change is isolated to relation extraction:
      * relation classes are represented by frozen natural-language schema states;
      * each relation slot re-attends token/layer memory;
      * slot n is softly conditioned on the relation distribution from slot n-1;
      * relation scores come from token-level query/schema semantic similarity;
      * NONE/termination is a structural STOP decision, not a fake semantic class.

    Current relation count and relation-step count are fixture/runtime shapes,
    not product ceilings.
    """

    def __init__(
        self,
        config: QSRET2SchemaOrderedConfig,
        *,
        relation_schema_hidden_states: Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim

        expected = (
            config.num_relations,
            config.num_hidden_states,
            config.semantic_dim,
        )
        if tuple(relation_schema_hidden_states.shape) != expected:
            raise ValueError(
                "relation_schema_hidden_states must be "
                f"{expected}, got {tuple(relation_schema_hidden_states.shape)}"
            )
        self.register_buffer(
            "relation_schema_hidden_states",
            relation_schema_hidden_states.detach().float().clone(),
            persistent=True,
        )

        self.token_norm = nn.LayerNorm(config.semantic_dim)
        self.token_projection = nn.Linear(config.semantic_dim, d, bias=False)
        self.layer_embedding = nn.Embedding(config.num_hidden_states, d)

        # Preserve the v0.2 latent interface for all operator factors.
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

        # v0.2 heads intentionally retained for factors already shown learnable.
        self.role_anchor = nn.Parameter(torch.empty(config.num_roles, d))
        self.operation_anchor = nn.Parameter(
            torch.empty(config.num_operations, d)
        )
        self.control_anchor = nn.Parameter(torch.empty(config.num_controls, d))
        self.continuous_projection = nn.Linear(d, d)

        # Relation-only replacement: sequential schema-grounded late interaction.
        self.relation_cross_attention = nn.MultiheadAttention(
            d,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.relation_cross_norm = nn.LayerNorm(d)
        self.relation_transition = nn.Linear(d, d, bias=False)
        self.relation_stop_embedding = nn.Parameter(torch.empty(d))
        self.relation_stop_head = nn.Linear(d, 1)
        self.relation_logit_scale = nn.Parameter(torch.tensor(2.0))

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
            self.role_anchor,
            self.operation_anchor,
            self.control_anchor,
            self.relation_stop_embedding,
        ):
            nn.init.normal_(value, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.relation_transition.weight)
        nn.init.xavier_uniform_(self.relation_stop_head.weight)
        nn.init.zeros_(self.relation_stop_head.bias)

    def _anchor_logits(self, state: Tensor, anchors: Tensor) -> Tensor:
        state = F.normalize(state, dim=-1)
        anchors = F.normalize(anchors, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * torch.einsum("...d,kd->...k", state, anchors)

    def _project_query(
        self,
        query_hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        content = self.token_projection(self.token_norm(query_hidden_states))
        layer_ids = torch.arange(
            self.config.num_hidden_states,
            device=query_hidden_states.device,
        )
        layer_state = self.layer_embedding(layer_ids).view(
            1,
            self.config.num_hidden_states,
            1,
            self.config.model_dim,
        )
        memory_state = content + layer_state
        return content, memory_state

    def _project_schema(self) -> tuple[Tensor, Tensor]:
        raw = self.relation_schema_hidden_states.to(
            dtype=self.token_projection.weight.dtype,
        )
        content = self.token_projection(self.token_norm(raw))
        normalized = F.normalize(content, dim=-1)
        global_state = F.normalize(content.mean(dim=1), dim=-1)
        return normalized, global_state

    def _relation_logits(
        self,
        *,
        base_relation_state: Tensor,
        memory: Tensor,
        query_content: Tensor,
        flat_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode ordered relation slots with semantic schema grounding."""
        batch, layers, tokens, _ = query_content.shape
        schema_layer, schema_global = self._project_schema()
        query_normalized = F.normalize(query_content, dim=-1)

        logits_per_slot: list[Tensor] = []
        attention_per_slot: list[Tensor] = []
        states_per_slot: list[Tensor] = []

        previous_expected: Tensor | None = None
        stop_vector = F.normalize(self.relation_stop_embedding, dim=-1)
        schema_plus_stop = torch.cat(
            [schema_global, stop_vector.unsqueeze(0)],
            dim=0,
        )

        for slot in range(self.config.max_relation_steps):
            state = base_relation_state[:, slot]
            if previous_expected is not None:
                state = state + self.relation_transition(previous_expected)

            attended, attn = self.relation_cross_attention(
                state.unsqueeze(1),
                memory,
                memory,
                key_padding_mask=~flat_valid,
                need_weights=True,
                average_attn_weights=False,
            )
            state = self.relation_cross_norm(
                state + attended.squeeze(1)
            )

            # Average attention heads, then restore [L,T] structure.
            slot_attention = attn.mean(dim=1).squeeze(1).reshape(
                batch,
                layers,
                tokens,
            )
            slot_attention = (
                slot_attention
                * flat_valid.reshape(batch, layers, tokens).to(
                    slot_attention.dtype
                )
            )

            # Query-token x relation-schema similarity at matching layers.
            similarity = torch.einsum(
                "bltd,rld->brlt",
                query_normalized,
                schema_layer,
            )
            grounded = torch.einsum(
                "blt,brlt->br",
                slot_attention,
                similarity,
            )
            scale = self.relation_logit_scale.exp().clamp(max=100.0)
            grounded = grounded * scale

            stop_logit = self.relation_stop_head(state)
            slot_logits = torch.cat([grounded, stop_logit], dim=-1)

            probabilities = torch.softmax(slot_logits, dim=-1)
            previous_expected = torch.einsum(
                "br,rd->bd",
                probabilities,
                schema_plus_stop,
            )

            logits_per_slot.append(slot_logits)
            attention_per_slot.append(slot_attention)
            states_per_slot.append(state)

        return (
            torch.stack(logits_per_slot, dim=1),
            torch.stack(attention_per_slot, dim=1),
            torch.stack(states_per_slot, dim=1),
        )

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

        query_content, memory_state = self._project_query(query_hidden_states)
        memory = memory_state.reshape(
            batch,
            layers * tokens,
            self.config.model_dim,
        )

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
        base_relation_state = latent[:, :s]
        role_state = latent[:, s]
        operation_state = latent[:, s + 1]
        control_state = latent[:, s + 2]
        continuous_state = self.continuous_projection(latent[:, s + 3])

        relation_logits, relation_attention, relation_state = (
            self._relation_logits(
                base_relation_state=base_relation_state,
                memory=memory,
                query_content=query_content,
                flat_valid=flat_valid,
            )
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
            "relation_grounding_attention": relation_attention,
            "relation_ordered_state": relation_state,
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
            min=0,
            max=self.config.num_relations - 1,
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
        names = {name for name, _ in self.named_parameters()}
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
            "schema_grounded_relation_readout": True,
            "learned_relation_class_anchor": any(
                name.startswith("relation_anchor") for name in names
            ),
            "ordered_relation_transition": True,
            "structural_stop_decision": True,
            "continuous_residual_state": True,
            "hard_layer_mask": False,
            "max_relation_steps_is_product_ceiling": False,
        }
