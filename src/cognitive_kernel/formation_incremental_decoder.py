"""Experimental incremental inference for the existing MFM specialist weights.

This module changes no model parameter or formation contract. It keeps the
prepared source projection and each decoder layer's cross-attention keys and
values, then appends only the newest self-attention key/value. Production
inference must keep using the established runner until parity and hardware
receipts qualify this implementation on the pinned runtime.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def _heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    batch, steps, width = value.shape
    return value.view(batch, steps, heads, width // heads).transpose(1, 2)


def _attention(attention: nn.MultiheadAttention, query: torch.Tensor,
               keys: torch.Tensor, values: torch.Tensor,
               allowed: torch.Tensor | None = None) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        query, keys, values, attn_mask=allowed, dropout_p=0.0, is_causal=False)
    return attention.out_proj(output.transpose(1, 2).contiguous().view(
        output.shape[0], 1, attention.embed_dim))


class CachedFormationDecoder:
    """One-case autoregressive decoder sharing a trained FormationSpecialist.

    Only batch size one is currently admitted. The original specialist remains
    responsible for training and weight loading; this object owns cache state
    for exactly one inference case and must never be reused across cases.
    """

    def __init__(self, specialist: nn.Module, *, base_states: torch.Tensor,
                 source_mask: torch.Tensor):
        config = specialist.config
        if specialist.training or any(layer.training for layer in specialist.decoder.layers):
            raise ValueError("cached specialist requires eval mode")
        if not torch.is_inference_mode_enabled():
            raise ValueError("cached specialist requires torch.inference_mode")
        if (base_states.ndim != 3 or base_states.shape[0] != 1 or
                base_states.shape[2] != config.base_hidden_size or
                source_mask.shape != base_states.shape[:2] or
                not base_states.shape[1] or not source_mask.bool().any().item()):
            raise ValueError("cached source states and mask disagree")
        if any(layer.dropout.p or layer.dropout1.p or layer.dropout2.p or
               layer.dropout3.p for layer in specialist.decoder.layers):
            raise ValueError("dropout cannot be cached with deterministic parity")
        if any(not layer.norm_first or not layer.self_attn.batch_first or
               not layer.multihead_attn.batch_first or
               layer.self_attn.in_proj_weight is None or
               layer.multihead_attn.in_proj_weight is None
               for layer in specialist.decoder.layers):
            raise ValueError("specialist decoder geometry has no verified cache mapping")
        self.specialist = specialist
        self.config = config
        self.position = 0
        self.allowed_source = source_mask.bool()[:, None, None, :]
        self.memory = specialist.source_projection(
            base_states.to(specialist.source_projection.weight.dtype))
        self.cross_keys_values = []
        for layer in specialist.decoder.layers:
            attention = layer.multihead_attn
            width = attention.embed_dim
            weight, bias = attention.in_proj_weight, attention.in_proj_bias
            k = F.linear(self.memory, weight[width:2 * width],
                         None if bias is None else bias[width:2 * width])
            v = F.linear(self.memory, weight[2 * width:],
                         None if bias is None else bias[2 * width:])
            self.cross_keys_values.append((_heads(k, attention.num_heads),
                                           _heads(v, attention.num_heads)))
        self.self_keys_values: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in specialist.decoder.layers]
        self.self_allowed = torch.zeros((1, 1, 1, config.max_target_tokens),
                                        dtype=torch.bool, device=self.memory.device)

    def step(self, token_id: int) -> torch.Tensor:
        """Consume one prefix token and return its next-token logits [1, vocab]."""
        if type(token_id) is not int or not 0 <= token_id < self.config.vocabulary_size:
            raise ValueError("cached decoder token ID is invalid")
        if self.position >= self.config.max_target_tokens:
            raise ValueError("cached decoder exceeds target budget")
        device = self.memory.device
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        position = torch.tensor([[self.position]], dtype=torch.long, device=device)
        x = self.specialist.token_embedding(token) + self.specialist.position_embedding(position)
        self.self_allowed[..., self.position] = token_id != self.config.pad_token_id
        for i, layer in enumerate(self.specialist.decoder.layers):
            attention = layer.self_attn
            width = attention.embed_dim
            weight, bias = attention.in_proj_weight, attention.in_proj_bias
            qkv = F.linear(layer.norm1(x), weight, bias)
            q, k, v = (_heads(part, attention.num_heads) for part in
                       qkv.split(width, dim=-1))
            cached = self.self_keys_values[i]
            if cached is None:
                cached = (torch.empty((1, attention.num_heads,
                                       self.config.max_target_tokens, q.shape[-1]),
                                      dtype=k.dtype, device=k.device),
                          torch.empty((1, attention.num_heads,
                                       self.config.max_target_tokens, q.shape[-1]),
                                      dtype=v.dtype, device=v.device))
                self.self_keys_values[i] = cached
            cached[0][..., self.position:self.position + 1, :] = k
            cached[1][..., self.position:self.position + 1, :] = v
            x = x + _attention(attention, q,
                               cached[0][..., :self.position + 1, :],
                               cached[1][..., :self.position + 1, :],
                               self.self_allowed[..., :self.position + 1])

            attention = layer.multihead_attn
            width = attention.embed_dim
            bias = attention.in_proj_bias
            q = F.linear(layer.norm2(x), attention.in_proj_weight[:width],
                         None if bias is None else bias[:width])
            cross_k, cross_v = self.cross_keys_values[i]
            x = x + _attention(attention, _heads(q, attention.num_heads),
                               cross_k, cross_v, self.allowed_source)

            feed_forward = layer.linear2(layer.dropout(
                layer.activation(layer.linear1(layer.norm3(x)))))
            x = x + layer.dropout3(feed_forward)
        if self.specialist.decoder.norm is not None:
            x = self.specialist.decoder.norm(x)
        self.position += 1
        return F.linear(x[:, -1], self.specialist.token_embedding.weight,
                        self.specialist.output_bias)


def greedy_tokens_cached(specialist: nn.Module, *, base_states: torch.Tensor,
                         source_mask: torch.Tensor, max_new_tokens: int
                         ) -> tuple[list[int], bool]:
    """Greedy BOS→EOS inference using only the specialist's output head."""
    if (type(max_new_tokens) is not int or
            not 1 <= max_new_tokens <= specialist.config.max_target_tokens):
        raise ValueError("invalid cached decode budget")
    generated: list[int] = []
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if base_states.device.type == "cuda" else nullcontext())
    with torch.inference_mode(), autocast:
        decoder = CachedFormationDecoder(specialist, base_states=base_states,
                                         source_mask=source_mask)
        current = specialist.config.start_token_id
        for _ in range(max_new_tokens):
            current = int(torch.argmax(decoder.step(current)[0]).item())
            if current == specialist.config.end_token_id:
                return generated, True
            generated.append(current)
    return generated, False
