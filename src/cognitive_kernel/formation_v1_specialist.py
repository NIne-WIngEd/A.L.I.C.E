"""Fresh MFM formation weights conditioned on a locally prepared representation base.

The base's language head is never used to produce formation decisions. This
module owns its decoder embeddings, cross attention, and output head. Source
representation ancestry remains licensed Gemma; the specialist parameters are
initialized independently. A separately enforced Claim gate still decides any
canonical memory write. Importing this module requires PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SpecialistConfig:
    base_hidden_size: int
    vocabulary_size: int
    width: int
    layers: int
    heads: int
    max_target_tokens: int
    pad_token_id: int
    start_token_id: int
    end_token_id: int
    logit_chunk_tokens: int = 64

    def validate(self) -> None:
        values = (self.base_hidden_size, self.vocabulary_size, self.width,
                  self.layers, self.heads, self.max_target_tokens,
                  self.logit_chunk_tokens)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("specialist sizes must be positive integers")
        if self.width % self.heads:
            raise ValueError("specialist width must divide the attention heads")
        if self.max_target_tokens < 2:
            raise ValueError("target budget must allow a decision and end token")
        for name in ("pad_token_id", "start_token_id", "end_token_id"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < self.vocabulary_size:
                raise ValueError(f"{name} exceeds the vocabulary")

    def record(self) -> dict[str, int]:
        self.validate()
        return asdict(self)


class FormationSpecialist(nn.Module):
    """Independent autoregressive decoder over exact-source base hidden states.

    ``base_states`` must come from the prepared source model's forward pass on
    every opened source, including native media tensors when present. The
    caller freezes that model and supplies its non-truncated attention mask.
    The target is the canonical, source-linked formation JSON contract.
    """

    def __init__(self, config: SpecialistConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.source_projection = nn.Linear(config.base_hidden_size, config.width)
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = nn.Embedding(config.max_target_tokens, config.width)
        layer = nn.TransformerDecoderLayer(
            d_model=config.width, nhead=config.heads,
            dim_feedforward=4 * config.width, batch_first=True,
            norm_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.layers,
                                             norm=nn.LayerNorm(config.width))
        self.output_bias = nn.Parameter(torch.zeros(config.vocabulary_size))

    def _decode(self, *, base_states: torch.Tensor, source_mask: torch.Tensor,
                input_ids: torch.Tensor) -> torch.Tensor:
        if base_states.ndim != 3 or input_ids.ndim != 2 or source_mask.ndim != 2:
            raise ValueError("specialist expects [batch, source, hidden] and 2D masks/tokens")
        batch, source_length, hidden_size = base_states.shape
        if (batch != input_ids.shape[0] or source_mask.shape != (batch, source_length)
                or hidden_size != self.config.base_hidden_size):
            raise ValueError("source hidden states and mask disagree with the specialist")
        target_length = input_ids.shape[1]
        if not source_length or not 0 < target_length <= self.config.max_target_tokens:
            raise ValueError("empty source or target exceeds the explicit specialist budget")
        if not torch.all(source_mask.any(dim=1)):
            raise ValueError("source attention mask hides all evidence")
        positions = torch.arange(target_length, device=input_ids.device)
        target = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        memory = self.source_projection(base_states.to(self.source_projection.weight.dtype))
        causal = torch.ones(target_length, target_length, device=input_ids.device,
                            dtype=torch.bool).triu_(diagonal=1)
        return self.decoder(
            target, memory, tgt_mask=causal,
            tgt_key_padding_mask=input_ids.eq(self.config.pad_token_id),
            memory_key_padding_mask=~source_mask.bool())

    def next_token_logits(self, *, base_states: torch.Tensor,
                          source_mask: torch.Tensor,
                          input_ids: torch.Tensor) -> torch.Tensor:
        """Project only the final decoder position during autoregressive inference."""
        decoded = self._decode(base_states=base_states, source_mask=source_mask,
                               input_ids=input_ids)
        return F.linear(decoded[:, -1], self.token_embedding.weight, self.output_bias)

    def forward(self, *, base_states: torch.Tensor, source_mask: torch.Tensor,
                input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        decoded = self._decode(base_states=base_states, source_mask=source_mask,
                               input_ids=input_ids)
        target_length = decoded.shape[1]
        # A tied head is a fresh trainable specialist head, not Gemma's LM head.
        if labels is None:
            return F.linear(decoded, self.token_embedding.weight, self.output_bias)
        if labels.shape != input_ids.shape:
            raise ValueError("specialist label shape differs from decoder inputs")
        supervised = labels.ne(-100).sum()
        if not supervised.item():
            raise ValueError("all specialist labels are ignored")
        # Recompute the large vocabulary projection one target chunk at a
        # time in backward; never retain a full 8K x 262K logit matrix.
        def chunk_loss(hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits = F.linear(hidden, self.token_embedding.weight, self.output_bias)
            return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   target.reshape(-1), ignore_index=-100,
                                   reduction="sum")

        total = decoded.new_zeros((), dtype=torch.float32)
        for start in range(0, target_length, self.config.logit_chunk_tokens):
            end = min(start + self.config.logit_chunk_tokens, target_length)
            hidden = decoded[:, start:end]
            target = labels[:, start:end]
            if target.ne(-100).any():
                total = total + checkpoint(chunk_loss, hidden, target,
                                           use_reentrant=False)
        return None, total / supervised
