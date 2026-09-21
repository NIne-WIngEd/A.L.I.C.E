from __future__ import annotations

import torch
from torch import Tensor


def chunked_batched_bidirectional_late_max(
    *,
    query: Tensor,
    query_mask: Tensor,
    items: Tensor,
    item_mask: Tensor,
    chunk_tokens: int = 128,
) -> tuple[Tensor, Tensor]:
    """Memory-bounded differentiable bidirectional token max interaction.

    This computes the same max statistics as the dense [B,N,L,T,S]
    interaction without materializing that full tensor. All accumulation is
    functional rather than in-place so gradients remain valid.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if query.ndim != 4:
        raise ValueError("query must be [B,L,T,D]")
    if items.ndim != 5:
        raise ValueError("items must be [B,N,L,S,D]")
    batch, layers, query_tokens, width = query.shape
    if items.size(0) != batch or items.size(2) != layers:
        raise ValueError("query/item batch or layer geometry drift")
    if items.size(-1) != width:
        raise ValueError("query/item width drift")
    item_count = items.size(1)
    item_tokens = items.size(3)
    if query_mask.shape != (batch, query_tokens):
        raise ValueError("query_mask must be [B,T]")
    if item_mask.shape != (batch, item_count, item_tokens):
        raise ValueError("item_mask must be [B,N,S]")
    if query_mask.dtype != torch.bool or item_mask.dtype != torch.bool:
        raise ValueError("late-interaction masks must be bool")
    if bool((query_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every query requires at least one valid token")
    item_available = item_mask.any(dim=-1)

    neg = torch.finfo(query.dtype).min
    query_parts: list[Tensor] = []
    item_running: list[Tensor | None] = []

    item_ranges = [
        (s0, min(s0 + chunk_tokens, item_tokens))
        for s0 in range(0, item_tokens, chunk_tokens)
    ]
    item_running = [None for _ in item_ranges]

    for q0 in range(0, query_tokens, chunk_tokens):
        q1 = min(q0 + chunk_tokens, query_tokens)
        q_chunk = query[:, :, q0:q1, :]
        q_valid = query_mask[:, q0:q1]
        q_running: Tensor | None = None

        for item_index, (s0, s1) in enumerate(item_ranges):
            item_chunk = items[:, :, :, s0:s1, :]
            item_valid = item_mask[:, :, s0:s1]
            similarity = torch.einsum(
                "blqd,bnlsd->bnlqs",
                q_chunk,
                item_chunk,
            )
            valid = (
                q_valid[:, None, None, :, None]
                & item_valid[:, :, None, None, :]
            )
            similarity = similarity.masked_fill(~valid, neg)

            q_max = similarity.max(dim=-1).values
            s_max = similarity.max(dim=-2).values
            q_running = (
                q_max
                if q_running is None
                else torch.maximum(q_running, q_max)
            )
            previous = item_running[item_index]
            item_running[item_index] = (
                s_max
                if previous is None
                else torch.maximum(previous, s_max)
            )

        assert q_running is not None
        query_parts.append(q_running)

    if any(value is None for value in item_running):
        raise RuntimeError("late-interaction item accumulation incomplete")

    query_to_item = torch.cat(query_parts, dim=-1)
    item_to_query = torch.cat(
        [value for value in item_running if value is not None],
        dim=-1,
    )
    query_to_item = query_to_item.masked_fill(
        ~query_mask[:, None, None, :],
        0.0,
    )
    query_to_item = query_to_item.masked_fill(
        ~item_available[:, :, None, None],
        0.0,
    )
    item_to_query = item_to_query.masked_fill(
        ~item_mask[:, :, None, :],
        0.0,
    )
    return query_to_item, item_to_query


def chunked_schema_bidirectional_late_max(
    *,
    query: Tensor,
    query_mask: Tensor,
    schema: Tensor,
    schema_mask: Tensor,
    chunk_tokens: int = 128,
) -> tuple[Tensor, Tensor]:
    """Schema convenience wrapper without copying the runtime bank.

    query: [B,L,T,D]
    schema: [C,L,S,D]
    """
    if schema.ndim != 4:
        raise ValueError("schema must be [C,L,S,D]")
    batch = query.size(0)
    items = schema.unsqueeze(0).expand(batch, -1, -1, -1, -1)
    item_mask = schema_mask.unsqueeze(0).expand(batch, -1, -1)
    return chunked_batched_bidirectional_late_max(
        query=query,
        query_mask=query_mask,
        items=items,
        item_mask=item_mask,
        chunk_tokens=chunk_tokens,
    )
