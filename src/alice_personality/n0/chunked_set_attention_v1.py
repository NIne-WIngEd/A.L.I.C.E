from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ChunkedSetAttentionConfig:
    model_dim: int = 640
    num_attention_heads: int = 10
    query_chunk_fields: int = 64
    key_chunk_fields: int = 128
    feedforward_multiplier: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("model_dim", self.model_dim),
            ("num_attention_heads", self.num_attention_heads),
            ("query_chunk_fields", self.query_chunk_fields),
            ("key_chunk_fields", self.key_chunk_fields),
            ("feedforward_multiplier", self.feedforward_multiplier),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class ChunkedExactSetSelfAttention(nn.Module):
    """Exact permutation-equivariant self-attention with bounded score memory.

    The full F x F attention matrix is never materialized. Query and key fields
    are streamed in blocks with an online softmax, preserving the exact dense
    attention result (up to normal floating-point order differences).

    Chunk sizes are compute/memory operating points only. They do not create a
    field-count parameter axis or a product field-count ceiling.
    """

    def __init__(self, config: ChunkedSetAttentionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim
        self.head_dim = d // config.num_attention_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def _heads(self, value: Tensor) -> Tensor:
        batch, fields, width = value.shape
        return (
            value.view(
                batch,
                fields,
                self.config.num_attention_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        if value.ndim != 3:
            raise ValueError("set attention value must be [B,F,D]")
        batch, fields, width = value.shape
        if width != self.config.model_dim:
            raise ValueError("set attention width drift")
        if valid_mask.shape != (batch, fields) or valid_mask.dtype != torch.bool:
            raise ValueError("set attention valid_mask must be bool [B,F]")
        if bool((valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every set example requires one valid field")

        q = self._heads(self.q_proj(value.float()))
        k = self._heads(self.k_proj(value.float()))
        v = self._heads(self.v_proj(value.float()))
        scale = 1.0 / math.sqrt(float(self.head_dim))
        running_floor = torch.finfo(q.dtype).min

        query_outputs: list[Tensor] = []
        q_chunk_size = self.config.query_chunk_fields
        k_chunk_size = self.config.key_chunk_fields

        for q0 in range(0, fields, q_chunk_size):
            q1 = min(q0 + q_chunk_size, fields)
            q_chunk = q[:, :, q0:q1, :]
            q_valid = valid_mask[:, q0:q1]

            running_max = torch.full(
                (batch, self.config.num_attention_heads, q1 - q0),
                fill_value=running_floor,
                dtype=q.dtype,
                device=q.device,
            )
            running_denominator = torch.zeros_like(running_max)
            running_numerator = torch.zeros(
                batch,
                self.config.num_attention_heads,
                q1 - q0,
                self.head_dim,
                dtype=q.dtype,
                device=q.device,
            )

            for k0 in range(0, fields, k_chunk_size):
                k1 = min(k0 + k_chunk_size, fields)
                k_chunk = k[:, :, k0:k1, :]
                v_chunk = v[:, :, k0:k1, :]
                key_valid = valid_mask[:, k0:k1]

                score = torch.einsum(
                    "bhqd,bhkd->bhqk",
                    q_chunk,
                    k_chunk,
                ) * scale
                # Mask using the score storage dtype. Under autocast the
                # matmul/einsum result may be lower precision than q/k.
                score_floor = torch.finfo(score.dtype).min
                score = score.masked_fill(
                    ~key_valid[:, None, None, :],
                    score_floor,
                )
                chunk_max = score.max(dim=-1).values
                new_max = torch.maximum(running_max, chunk_max)

                old_scale = torch.exp(running_max - new_max)
                exponent = torch.exp(
                    score - new_max.unsqueeze(-1)
                )
                exponent = exponent * key_valid[:, None, None, :].to(
                    exponent.dtype
                )

                running_denominator = (
                    running_denominator * old_scale
                    + exponent.sum(dim=-1)
                )
                running_numerator = (
                    running_numerator * old_scale.unsqueeze(-1)
                    + torch.einsum(
                        "bhqk,bhkd->bhqd",
                        exponent,
                        v_chunk,
                    )
                )
                running_max = new_max

            out = running_numerator / running_denominator.unsqueeze(
                -1
            ).clamp_min(1.0e-12)
            out = out * q_valid[:, None, :, None].to(out.dtype)
            query_outputs.append(out)

        attended = torch.cat(query_outputs, dim=2)
        attended = attended.transpose(1, 2).reshape(batch, fields, width)
        attended = self.out_proj(attended)
        return attended * valid_mask.unsqueeze(-1).to(attended.dtype)

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "field_identity_parameters": 0,
            "field_count_dependent_parameters": 0,
            "full_pair_score_matrix_materialized": False,
            "exact_dense_attention_semantics": True,
            "query_chunk_is_operating_point": True,
            "key_chunk_is_operating_point": True,
            "field_count_ceiling": None,
        }


class ChunkedSetTransformerBlock(nn.Module):
    def __init__(self, config: ChunkedSetAttentionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim
        ff = config.feedforward_multiplier * d

        self.attention_norm = nn.LayerNorm(d)
        self.attention = ChunkedExactSetSelfAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)

        self.ff_norm = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, 2 * ff)
        self.ff_out = nn.Linear(ff, d)
        self.ff_dropout = nn.Dropout(config.dropout)

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        attended = self.attention(
            self.attention_norm(value),
            valid_mask,
        )
        value = value + self.attention_dropout(attended)
        value = value * valid_mask.unsqueeze(-1).to(value.dtype)

        gate, content = self.ff_in(self.ff_norm(value)).chunk(2, dim=-1)
        feedforward = torch.nn.functional.gelu(gate) * content
        value = value + self.ff_dropout(self.ff_out(feedforward))
        return value * valid_mask.unsqueeze(-1).to(value.dtype)


class ChunkedSetTransformerEncoder(nn.Module):
    def __init__(
        self,
        config: ChunkedSetAttentionConfig,
        *,
        num_layers: int,
    ) -> None:
        super().__init__()
        config.validate()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.config = config
        self.layers = nn.ModuleList(
            [ChunkedSetTransformerBlock(config) for _ in range(num_layers)]
        )

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value, valid_mask)
        return value

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "layers": len(self.layers),
            "field_identity_parameters": 0,
            "field_count_dependent_parameters": 0,
            "full_pair_score_matrix_materialized": False,
            "exact_dense_attention_semantics": True,
            "field_count_ceiling": None,
        }
