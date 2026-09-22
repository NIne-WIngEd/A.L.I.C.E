from __future__ import annotations

import math

import torch

from alice_personality.n0.chunked_set_attention_v1 import (
    ChunkedExactSetSelfAttention,
    ChunkedSetAttentionConfig,
)


def dense_reference(
    module: ChunkedExactSetSelfAttention,
    value: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    batch, fields, width = value.shape
    heads = module.config.num_attention_heads
    head_dim = module.head_dim

    def split(x: torch.Tensor) -> torch.Tensor:
        return x.view(batch, fields, heads, head_dim).transpose(1,2)

    q = split(module.q_proj(value.float()))
    k = split(module.k_proj(value.float()))
    v = split(module.v_proj(value.float()))
    score = torch.einsum("bhqd,bhkd->bhqk",q,k) / math.sqrt(float(head_dim))
    score = score.masked_fill(
        ~valid_mask[:,None,None,:],
        torch.finfo(score.dtype).min,
    )
    weight = torch.softmax(score,dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd",weight,v)
    out = out.transpose(1,2).reshape(batch,fields,width)
    out = module.out_proj(out)
    return out * valid_mask.unsqueeze(-1).to(out.dtype)


def test_chunked_set_attention_matches_dense_reference_and_gradient() -> None:
    torch.manual_seed(401)
    module=ChunkedExactSetSelfAttention(
        ChunkedSetAttentionConfig(
            model_dim=24,
            num_attention_heads=4,
            query_chunk_fields=2,
            key_chunk_fields=3,
            dropout=0.0,
        )
    )
    value=torch.randn(2,7,24,requires_grad=True)
    mask=torch.tensor(
        [
            [True,True,True,True,True,False,False],
            [True,True,True,True,True,True,True],
        ]
    )
    actual=module(value,mask)
    expected=dense_reference(module,value,mask)
    assert torch.allclose(actual,expected,atol=1e-5,rtol=1e-5)
    actual_grad=torch.autograd.grad(
        actual.square().mean(),
        value,
        retain_graph=True,
    )[0]
    dense_grad=torch.autograd.grad(
        expected.square().mean(),
        value,
    )[0]
    assert float(actual_grad.abs().sum()) > 0.0
    assert torch.allclose(
        actual_grad,
        dense_grad,
        atol=2e-5,
        rtol=2e-5,
    )


def test_chunked_set_attention_is_field_permutation_equivariant() -> None:
    torch.manual_seed(402)
    module=ChunkedExactSetSelfAttention(
        ChunkedSetAttentionConfig(
            model_dim=24,
            num_attention_heads=4,
            query_chunk_fields=3,
            key_chunk_fields=2,
            dropout=0.0,
        )
    ).eval()
    value=torch.randn(2,8,24)
    mask=torch.tensor(
        [
            [True,True,True,True,True,True,False,False],
            [True,True,True,True,True,True,True,True],
        ]
    )
    permutation=torch.tensor([5,0,7,1,6,2,4,3])
    with torch.no_grad():
        a=module(value,mask)
        b=module(value[:,permutation],mask[:,permutation])
    assert torch.allclose(
        b,
        a[:,permutation],
        atol=1e-5,
        rtol=1e-5,
    )


def test_chunked_set_attention_parameter_count_independent_of_field_count() -> None:
    module=ChunkedExactSetSelfAttention(
        ChunkedSetAttentionConfig(
            model_dim=24,
            num_attention_heads=4,
            query_chunk_fields=5,
            key_chunk_fields=7,
        )
    ).eval()
    before=sum(p.numel() for p in module.parameters())
    with torch.no_grad():
        small=module(
            torch.randn(1,5,24),
            torch.ones(1,5,dtype=torch.bool),
        )
        large=module(
            torch.randn(1,257,24),
            torch.ones(1,257,dtype=torch.bool),
        )
    assert small.shape==(1,5,24)
    assert large.shape==(1,257,24)
    assert sum(p.numel() for p in module.parameters())==before
    report=module.parameter_report()
    assert report["field_count_dependent_parameters"]==0
    assert report["full_pair_score_matrix_materialized"] is False
    assert report["exact_dense_attention_semantics"] is True
    assert report["field_count_ceiling"] is None
