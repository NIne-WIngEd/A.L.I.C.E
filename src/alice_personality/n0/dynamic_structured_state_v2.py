from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.semantic_operator_foundation import DynamicSemanticSchema
from alice_personality.n0.numeric_contracts import (
    require_finite,
    require_unit_interval,
)
from alice_personality.n0.chunked_set_attention_v1 import (
    ChunkedSetAttentionConfig,
    ChunkedSetTransformerEncoder,
)


@dataclass(frozen=True)
class DynamicStructuredStateConfig:
    semantic_dim: int = 640
    model_dim: int = 640
    num_hidden_states: int = 17
    num_attention_heads: int = 10
    num_layers: int = 2
    continuous_metadata_dim: int = 3
    query_chunk_fields: int = 64
    key_chunk_fields: int = 128
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("num_attention_heads", self.num_attention_heads),
            ("num_layers", self.num_layers),
            ("continuous_metadata_dim", self.continuous_metadata_dim),
            ("query_chunk_fields", self.query_chunk_fields),
            ("key_chunk_fields", self.key_chunk_fields),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class DynamicStructuredStateV2(nn.Module):
    """Permutation-safe structured state with runtime semantic descriptors.

    Field/type/provenance/temporal category semantics are supplied as runtime
    text-derived descriptor banks. Integer indices are only structural pointers
    into those banks; no category-ID embedding table owns their meaning.
    """

    def __init__(self, config: DynamicStructuredStateConfig | None = None) -> None:
        super().__init__()
        self.config = config or DynamicStructuredStateConfig()
        self.config.validate()
        d = self.config.model_dim

        self.content_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.descriptor_projection = nn.Linear(self.config.semantic_dim, d, bias=False)
        self.layer_gate = nn.Sequential(
            nn.Linear(self.config.semantic_dim + 1, self.config.semantic_dim),
            nn.SiLU(),
            nn.Linear(self.config.semantic_dim, 1),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(2, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.continuous_metadata_projection = nn.Sequential(
            nn.Linear(self.config.continuous_metadata_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.input_norm = nn.LayerNorm(d)
        self.encoder = ChunkedSetTransformerEncoder(
            ChunkedSetAttentionConfig(
                model_dim=d,
                num_attention_heads=self.config.num_attention_heads,
                query_chunk_fields=self.config.query_chunk_fields,
                key_chunk_fields=self.config.key_chunk_fields,
                feedforward_multiplier=4,
                dropout=self.config.dropout,
            ),
            num_layers=self.config.num_layers,
        )
        self.pool_query = nn.Parameter(torch.zeros(d))
        self.pool_score = nn.Linear(d, d, bias=False)
        self.output_norm = nn.LayerNorm(d)

    def _summarize_tokens(
        self,
        states: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # states [...,L,T,D], mask [...,T]
        if states.size(-3) != self.config.num_hidden_states:
            raise ValueError("hidden-state depth drift")
        if states.size(-1) != self.config.semantic_dim:
            raise ValueError("semantic width drift")
        if mask.shape != states.shape[:-3] + (states.size(-2),):
            raise ValueError("token mask geometry drift")
        if mask.dtype != torch.bool:
            raise ValueError("token mask must be bool")
        if bool((mask.sum(dim=-1) == 0).any()):
            raise ValueError("every structured semantic item requires content tokens")
        weight = mask.to(states.dtype).unsqueeze(-2).unsqueeze(-1)
        # [...,1,T,1] broadcasts across L.
        pooled_layer = (
            (states * weight).sum(dim=-2)
            / weight.sum(dim=-2).clamp_min(1.0)
        ).float()
        layers = pooled_layer.size(-2)
        layer_position = torch.linspace(
            -1.0,
            1.0,
            layers,
            device=pooled_layer.device,
            dtype=pooled_layer.dtype,
        )
        position_shape = (1,) * (pooled_layer.ndim - 2) + (layers, 1)
        position = layer_position.view(position_shape).expand(
            pooled_layer.shape[:-1] + (1,)
        )
        gate_input = torch.cat([pooled_layer, position], dim=-1)
        layer_logit = self.layer_gate(gate_input).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        return torch.einsum("...l,...ld->...d", layer_weight, pooled_layer)

    def _descriptor_summaries(
        self,
        bank: DynamicSemanticSchema,
    ) -> Tensor:
        bank.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        summary = self._summarize_tokens(bank.token_states, bank.token_mask)
        return self.descriptor_projection(summary)

    def forward(
        self,
        *,
        field_hidden_states: Tensor,
        field_token_mask: Tensor,
        field_valid_mask: Tensor,
        field_confidence: Tensor,
        field_missing: Tensor,
        field_metadata: Tensor,
        descriptor_banks: Mapping[str, DynamicSemanticSchema],
        descriptor_indices: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        if field_hidden_states.ndim != 5:
            raise ValueError("field_hidden_states must be [B,F,L,T,D]")
        batch, fields, layers, tokens, width = field_hidden_states.shape
        if layers != self.config.num_hidden_states or width != self.config.semantic_dim:
            raise ValueError("field semantic geometry drift")
        if field_token_mask.shape != (batch, fields, tokens) or field_token_mask.dtype != torch.bool:
            raise ValueError("field_token_mask must be bool [B,F,T]")
        if field_valid_mask.shape != (batch, fields) or field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool [B,F]")
        if bool((field_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one valid field")
        if field_confidence.shape != (batch, fields):
            raise ValueError("field_confidence must be [B,F]")
        if field_missing.shape != (batch, fields):
            raise ValueError("field_missing must be [B,F]")
        require_unit_interval("field_confidence", field_confidence)
        require_unit_interval("field_missing", field_missing)
        if field_metadata.shape != (
            batch,
            fields,
            self.config.continuous_metadata_dim,
        ):
            raise ValueError("field_metadata continuous geometry drift")
        require_finite("field_metadata", field_metadata)
        if set(descriptor_banks) != set(descriptor_indices):
            raise ValueError("descriptor bank/index names must match")

        # Invalid padded fields may have an empty token mask. Summarize only
        # valid fields by temporarily supplying one zero-safe mask position.
        safe_mask = field_token_mask.clone()
        invalid = ~field_valid_mask
        if bool(invalid.any()):
            first_token = torch.zeros(
                tokens,
                device=field_token_mask.device,
                dtype=torch.bool,
            )
            first_token[0] = True
            safe_mask = safe_mask | (
                invalid.unsqueeze(-1) & first_token.view(1, 1, tokens)
            )
        content = self._summarize_tokens(field_hidden_states, safe_mask)
        content = self.content_projection(content)

        descriptor_sum = torch.zeros_like(content)
        descriptor_count = torch.zeros(
            batch,
            fields,
            1,
            device=content.device,
            dtype=content.dtype,
        )
        for name, bank in descriptor_banks.items():
            index = descriptor_indices[name]
            if index.shape != (batch, fields):
                raise ValueError(f"descriptor index {name!r} must be [B,F]")
            summary = self._descriptor_summaries(bank)
            valid_index = index >= 0
            if bool(valid_index.any()):
                selected = index.clamp(min=0)
                if int(selected[valid_index].max()) >= summary.size(0):
                    raise ValueError(f"descriptor index {name!r} outside runtime bank")
                active = valid_index.unsqueeze(-1).to(summary.dtype)
                descriptor_sum = descriptor_sum + summary[selected] * active
                descriptor_count = descriptor_count + active
        descriptor = descriptor_sum / descriptor_count.clamp_min(1.0)

        scalar = torch.stack(
            [field_confidence.float(), field_missing.float()],
            dim=-1,
        )
        metadata = self.continuous_metadata_projection(field_metadata.float())
        value = self.input_norm(
            content
            + descriptor
            + self.scalar_projection(scalar)
            + metadata
        )
        value = value * field_valid_mask.unsqueeze(-1).to(value.dtype)

        encoded = self.encoder(
            value,
            field_valid_mask,
        )
        encoded = encoded * field_valid_mask.unsqueeze(-1).to(encoded.dtype)

        query = self.pool_score(self.pool_query).view(1, 1, -1)
        logits = (encoded * query).sum(dim=-1)
        logits = logits.masked_fill(~field_valid_mask, -1.0e4)
        weight = torch.softmax(logits, dim=-1) * field_valid_mask.to(logits.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = (encoded * weight.unsqueeze(-1)).sum(dim=1)
        pooled = self.output_norm(pooled)

        return {
            "field_states": encoded,
            "pooled_state": pooled,
            "pool_weight": weight,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "field_type_identity_parameters": 0,
            "provenance_identity_parameters": 0,
            "temporal_identity_parameters": 0,
            "descriptor_bank_count_dependent_parameters": 0,
            "descriptor_candidate_count_dependent_parameters": 0,
            "field_count_dependent_parameters": 0,
            "continuous_metadata_supported": True,
            "content_conditioned_layer_read": True,
            "global_static_layer_mixture": False,
            "descriptor_bank_count_normalized": True,
            "field_count_ceiling": None,
            "runtime_descriptor_semantics": True,
            "permutation_safe_field_encoder": True,
            "full_field_pair_matrix_materialized": False,
            "exact_dense_set_attention_semantics": True,
            "field_attention_chunk_is_operating_point": True,
        }
