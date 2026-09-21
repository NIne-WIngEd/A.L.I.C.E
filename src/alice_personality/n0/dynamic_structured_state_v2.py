from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.semantic_operator_foundation import DynamicSemanticSchema


@dataclass(frozen=True)
class DynamicStructuredStateConfig:
    semantic_dim: int = 640
    model_dim: int = 640
    num_hidden_states: int = 17
    num_attention_heads: int = 10
    num_layers: int = 2
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("num_attention_heads", self.num_attention_heads),
            ("num_layers", self.num_layers),
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
        self.layer_logits = nn.Parameter(torch.zeros(self.config.num_hidden_states))
        self.scalar_projection = nn.Sequential(
            nn.Linear(2, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.input_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.num_attention_heads,
            dim_feedforward=4 * d,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
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
        pooled_layer = (states * weight).sum(dim=-2) / weight.sum(dim=-2).clamp_min(1.0)
        layer_weight = torch.softmax(self.layer_logits, dim=0)
        return torch.einsum("l,...ld->...d", layer_weight, pooled_layer.float())

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
        if field_confidence.shape != (batch, fields):
            raise ValueError("field_confidence must be [B,F]")
        if field_missing.shape != (batch, fields):
            raise ValueError("field_missing must be [B,F]")
        if set(descriptor_banks) != set(descriptor_indices):
            raise ValueError("descriptor bank/index names must match")

        # Invalid padded fields may have an empty token mask. Summarize only
        # valid fields by temporarily supplying one zero-safe mask position.
        safe_mask = field_token_mask.clone()
        invalid = ~field_valid_mask
        if bool(invalid.any()):
            safe_mask[invalid, 0] = True
        content = self._summarize_tokens(field_hidden_states, safe_mask)
        content = self.content_projection(content)

        descriptor = torch.zeros_like(content)
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
                descriptor = descriptor + summary[selected] * valid_index.unsqueeze(-1).to(summary.dtype)

        scalar = torch.stack(
            [field_confidence.float(), field_missing.float()],
            dim=-1,
        )
        value = self.input_norm(content + descriptor + self.scalar_projection(scalar))
        value = value * field_valid_mask.unsqueeze(-1).to(value.dtype)

        encoded = self.encoder(
            value,
            src_key_padding_mask=~field_valid_mask,
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
            "field_count_ceiling": None,
            "runtime_descriptor_semantics": True,
            "permutation_safe_field_encoder": True,
        }
