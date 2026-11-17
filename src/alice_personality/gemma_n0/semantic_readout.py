"""First-party query/description readout over immutable all-layer token banks.

The learned parameters belong only to this readout. Candidate identity, order,
class IDs and labels are absent from its API. BF16 Gemma representations retain
inherited priors; successful mechanics do not establish personality neutrality.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ReadoutError(ValueError):
    """A bank or readout crossed the frozen semantic feature boundary."""


@dataclass(frozen=True)
class LayerTokenBank:
    layers: torch.Tensor  # [embedding+actual layers, complete source tokens, hidden width]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    source_mask: torch.Tensor | None = None
    query_mask: torch.Tensor | None = None
    head_mask: torch.Tensor | None = None
    tail_mask: torch.Tensor | None = None


def validate_bank(bank: LayerTokenBank, *, state_count: int, hidden_size: int,
                  source: bool = False) -> None:
    if not isinstance(bank, LayerTokenBank):
        raise ReadoutError("readout accepts only frozen token banks")
    layers, ids, mask = bank.layers, bank.input_ids, bank.attention_mask
    if (not isinstance(layers, torch.Tensor) or layers.ndim != 3
            or layers.shape[0] != state_count or layers.shape[2] != hidden_size
            or layers.shape[1] < 1 or layers.dtype != torch.bfloat16
            or layers.requires_grad or layers.grad_fn is not None
            or not bool(torch.isfinite(layers).all())):
        raise ReadoutError("every complete detached BF16 state is required")
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.int64
            or ids.shape[0] != layers.shape[1] or bool((ids < 0).any())
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != ids.shape or not bool(mask.any())
            or mask.requires_grad or ids.requires_grad
            or ids.device != layers.device or mask.device != layers.device):
        raise ReadoutError("complete source IDs and mask must align with every state")
    for name in ("source_mask", "query_mask", "head_mask", "tail_mask"):
        value = getattr(bank, name)
        if value is None and not source:
            continue
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                or value.shape != mask.shape or value.device != mask.device
                or value.requires_grad or not bool(value.any()) or bool((value & ~mask).any())):
            raise ReadoutError("source/query/ordered role masks must preserve actual tokens")
    if source and bool((bank.head_mask & bank.tail_mask).any()):
        raise ReadoutError("separately marked head and tail spans must not overlap")


def _attention(query: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.mv(tokens, query) / sqrt(tokens.shape[-1])
    return torch.sum(F.softmax(logits.masked_fill(~mask, -torch.inf), dim=0)[:, None] * tokens, dim=0)


class SemanticReadout(nn.Module):
    """Candidate-independent semantic interaction; no fixed relation axis."""

    def __init__(self, *, state_count: int, hidden_size: int, width: int = 64):
        super().__init__()
        for value in (state_count, hidden_size, width):
            if type(value) is not int or value < 1:
                raise ReadoutError("readout geometry must be positive integers")
        self.state_count, self.hidden_size, self.width = state_count, hidden_size, width
        self.layer_logits = nn.Parameter(torch.zeros(state_count, dtype=torch.float32))
        self.projection = nn.Linear(hidden_size, width, dtype=torch.float32)
        self.normalization = nn.LayerNorm(width, dtype=torch.float32)
        self.head_anchor = nn.Parameter(torch.randn(width, dtype=torch.float32) / sqrt(width))
        self.tail_anchor = nn.Parameter(torch.randn(width, dtype=torch.float32) / sqrt(width))
        self.query_anchor = nn.Parameter(torch.randn(width, dtype=torch.float32) / sqrt(width))
        self.query_projection = nn.Linear(width * 3, width, dtype=torch.float32)
        self.description_key = nn.Linear(width, width, bias=False, dtype=torch.float32)
        self.source_key = nn.Linear(width, width, bias=False, dtype=torch.float32)
        self.scorer = nn.Sequential(nn.Linear(width * 6, width, dtype=torch.float32), nn.GELU(),
                                    nn.Linear(width, 1, dtype=torch.float32))

    @property
    def geometry(self) -> dict:
        return {"state_count": self.state_count, "hidden_size": self.hidden_size, "width": self.width}

    def _tokens(self, bank: LayerTokenBank) -> torch.Tensor:
        # Explicit first-party FP32 mixture/projection; detached upstream BF16
        # tensors are never changed or given a gradient.
        weights = F.softmax(self.layer_logits, dim=0)
        mixed = torch.sum(weights[:, None, None] * bank.layers.detach().float(), dim=0)
        return self.normalization(self.projection(mixed))

    def forward(self, source: LayerTokenBank, candidates: Sequence[LayerTokenBank]) -> torch.Tensor:
        validate_bank(source, state_count=self.state_count, hidden_size=self.hidden_size, source=True)
        if not candidates:
            raise ReadoutError("runtime descriptions must be nonempty")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ReadoutError("first-party learned readout parameters must remain FP32")
        tokens = self._tokens(source)
        query = torch.tanh(self.query_projection(torch.cat((
            _attention(self.head_anchor, tokens, source.head_mask),
            _attention(self.tail_anchor, tokens, source.tail_mask),
            _attention(self.query_anchor, tokens, source.query_mask)))))
        scores = []
        for candidate in candidates:
            validate_bank(candidate, state_count=self.state_count, hidden_size=self.hidden_size)
            description_tokens = self._tokens(candidate)
            description = _attention(query, self.description_key(description_tokens), candidate.attention_mask)
            evidence = _attention(query + description, self.source_key(tokens), source.source_mask)
            interaction = torch.cat((query, description, evidence, query * description,
                                     evidence * description, query - evidence))
            scores.append(self.scorer(interaction).squeeze(-1))
        logits = torch.stack(scores)
        if not bool(torch.isfinite(logits).all()):
            raise ReadoutError("candidate scores must remain finite")
        return logits


def fixed_semantic_scores(source: LayerTokenBank, candidates: Sequence[LayerTokenBank]) -> torch.Tensor:
    """Untrained uniform-layer semantic control, not personality judgment."""
    if not candidates:
        raise ReadoutError("runtime descriptions must be nonempty")
    count, _, width = source.layers.shape
    validate_bank(source, state_count=count, hidden_size=width, source=True)
    with torch.no_grad():
        tokens = source.layers.float().mean(dim=0)
        # Entity-role span summaries are ordered; the whole sentence is never
        # collapsed into a mean as a substitute for governing evidence.
        head = tokens[source.head_mask].mean(dim=0)
        tail = tokens[source.tail_mask].mean(dim=0)
        query = F.normalize(head - tail + tokens[source.query_mask].mean(dim=0), dim=0)
        scores = []
        for candidate in candidates:
            validate_bank(candidate, state_count=count, hidden_size=width)
            description = _attention(query, candidate.layers.float().mean(dim=0), candidate.attention_mask)
            evidence = _attention(F.normalize(query + description, dim=0), tokens, source.source_mask)
            scores.append(F.cosine_similarity(description, evidence, dim=0)
                          + F.cosine_similarity(query, description, dim=0))
        return torch.stack(scores)
