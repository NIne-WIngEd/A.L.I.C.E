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
    """Memory-bounded bidirectional token max interaction.

    Args:
        query: [B,L,T,D]
        query_mask: bool [B,T]
        items: [B,N,L,S,D]
        item_mask: bool [B,N,S]
        chunk_tokens: runtime compute chunk, never a semantic/context ceiling.

    Returns:
        query_to_item: [B,N,L,T], max over item tokens.
        item_to_query: [B,N,L,S], max over query tokens.

    This computes the same pairwise max statistics as materializing the full
    [B,N,L,T,S] tensor, but bounds peak interaction memory by chunk size.
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
    if bool((item_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every item requires at least one valid token")

    neg = torch.finfo(query.dtype).min
    query_to_item = torch.full(
        (batch, item_count, layers, query_tokens),
        fill_value=neg,
        dtype=query.dtype,
        device=query.device,
    )
    item_to_query = torch.full(
        (batch, item_count, layers, item_tokens),
        fill_value=neg,
        dtype=query.dtype,
        device=query.device,
    )

    for q0 in range(0, query_tokens, chunk_tokens):
        q1 = min(q0 + chunk_tokens, query_tokens)
        q_chunk = query[:, :, q0:q1, :]
        q_valid = query_mask[:, q0:q1]

        for s0 in range(0, item_tokens, chunk_tokens):
            s1 = min(s0 + chunk_tokens, item_tokens)
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
            query_to_item[:, :, :, q0:q1] = torch.maximum(
                query_to_item[:, :, :, q0:q1],
                q_max,
            )
            item_to_query[:, :, :, s0:s1] = torch.maximum(
                item_to_query[:, :, :, s0:s1],
                s_max,
            )

    query_to_item = query_to_item.masked_fill(
        ~query_mask[:, None, None, :],
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
