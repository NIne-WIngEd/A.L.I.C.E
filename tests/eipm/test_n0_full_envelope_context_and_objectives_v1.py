from __future__ import annotations

import torch
import torch.nn.functional as F

from alice_personality.n0.chunked_late_interaction import (
    chunked_batched_bidirectional_late_max,
)
from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
    full_envelope_behavioral_objective,
    support_selection_loss,
)
from alice_personality.n0.full_envelope_loss_balancer_v1 import (
    MacroFamilyLossBalancer,
)
from alice_personality.n0.semantic_context_virtualizer_v1 import (
    SemanticContextVirtualizerConfig,
    SemanticContextVirtualizerV1,
)


def test_chunked_late_interaction_matches_dense_reference() -> None:
    torch.manual_seed(601)
    query = F.normalize(torch.randn(2, 3, 11, 8), dim=-1)
    items = F.normalize(torch.randn(2, 5, 3, 13, 8), dim=-1)
    query_mask = torch.ones(2, 11, dtype=torch.bool)
    query_mask[0, -2:] = False
    item_mask = torch.ones(2, 5, 13, dtype=torch.bool)
    item_mask[1, 2, -3:] = False

    dense = torch.einsum("bltd,bnlsd->bnlts", query, items)
    valid = (
        query_mask[:, None, None, :, None]
        & item_mask[:, :, None, None, :]
    )
    neg = torch.finfo(dense.dtype).min
    dense = dense.masked_fill(~valid, neg)
    expected_q = dense.max(dim=-1).values.masked_fill(
        ~query_mask[:, None, None, :], 0.0
    )
    expected_i = dense.max(dim=-2).values.masked_fill(
        ~item_mask[:, :, None, :], 0.0
    )

    for chunk in (1, 3, 7, 128):
        q_to_i, i_to_q = chunked_batched_bidirectional_late_max(
            query=query,
            query_mask=query_mask,
            items=items,
            item_mask=item_mask,
            chunk_tokens=chunk,
        )
        assert torch.allclose(q_to_i, expected_q, atol=1e-6, rtol=1e-6)
        assert torch.allclose(i_to_q, expected_i, atol=1e-6, rtol=1e-6)


def test_context_virtualizer_preserves_every_token_once_beyond_native_window() -> None:
    virtualizer = SemanticContextVirtualizerV1(
        SemanticContextVirtualizerConfig(
            native_window_tokens=16,
            overlap_tokens=4,
            pad_token_id=0,
        )
    )
    input_ids = torch.arange(1, 46).view(1, 45)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    segmented = virtualizer.segment(
        input_ids=input_ids,
        attention_mask=mask,
    )
    assert segmented["segment_input_ids"].shape[1] > 1

    reconstructed: list[int] = []
    for segment, content in zip(
        segmented["segment_input_ids"][0],
        segmented["segment_content_mask"][0],
    ):
        reconstructed.extend(segment[content].tolist())
    assert reconstructed == input_ids[0].tolist()
    assert int(segmented["segment_content_mask"].sum()) == 45

    report = virtualizer.parameter_report()
    assert report["total_parameters"] == 0
    assert report["segment_count_dependent_parameters"] == 0
    assert report["segment_count_ceiling"] is None
    assert report["product_context_token_ceiling"] is None


def test_context_virtualizer_handles_variable_lengths_and_encoded_geometry() -> None:
    virtualizer = SemanticContextVirtualizerV1(
        SemanticContextVirtualizerConfig(
            native_window_tokens=12,
            overlap_tokens=3,
        )
    )
    input_ids = torch.zeros(2, 31, dtype=torch.long)
    input_ids[0, :31] = torch.arange(1, 32)
    input_ids[1, :19] = torch.arange(101, 120)
    mask = torch.zeros(2, 31, dtype=torch.bool)
    mask[0, :31] = True
    mask[1, :19] = True

    segmented = virtualizer.segment(
        input_ids=input_ids,
        attention_mask=mask,
    )
    batch, segments, window = segmented["segment_input_ids"].shape
    encoded = torch.randn(batch, segments, 3, window, 8)
    fields = virtualizer.validate_encoded_segments(
        segment_hidden_states=encoded,
        segmented=segmented,
    )
    assert fields["field_hidden_states"].shape == (
        batch,
        segments,
        3,
        window,
        8,
    )
    assert fields["field_metadata"].shape == (batch, segments, 3)
    assert torch.equal(
        fields["field_valid_mask"],
        segmented["segment_valid_mask"],
    )


def test_context_virtualizer_stitches_long_query_content_without_overlap_duplication() -> None:
    virtualizer = SemanticContextVirtualizerV1(
        SemanticContextVirtualizerConfig(
            native_window_tokens=10,
            overlap_tokens=3,
        )
    )
    ids = torch.arange(1, 28).view(1, 27)
    mask = torch.ones_like(ids, dtype=torch.bool)
    segmented = virtualizer.segment(
        input_ids=ids,
        attention_mask=mask,
    )
    batch, segments, window = segmented["segment_input_ids"].shape
    encoded = torch.zeros(batch, segments, 2, window, 3)
    for g in range(segments):
        start = int(segmented["segment_absolute_start"][0, g])
        valid = segmented["segment_attention_mask"][0, g]
        absolute = torch.arange(
            start,
            start + int(valid.sum()),
            dtype=torch.float32,
        )
        encoded[0, g, 0, : absolute.numel(), 0] = absolute
        encoded[0, g, 1, : absolute.numel(), 0] = absolute + 100.0

    stitched, stitched_mask = virtualizer.stitch_owned_content(
        segment_hidden_states=encoded,
        segmented=segmented,
    )
    assert stitched.shape == (1, 2, 27, 3)
    assert bool(stitched_mask.all())
    assert torch.equal(
        stitched[0, 0, :, 0],
        torch.arange(27, dtype=torch.float32),
    )
    assert torch.equal(
        stitched[0, 1, :, 0],
        torch.arange(27, dtype=torch.float32) + 100.0,
    )


def test_context_virtualizer_fails_closed_on_noncontiguous_mask() -> None:
    virtualizer = SemanticContextVirtualizerV1(
        SemanticContextVirtualizerConfig(
            native_window_tokens=8,
            overlap_tokens=2,
        )
    )
    ids = torch.arange(10).view(1, 10)
    mask = torch.tensor(
        [[True, True, False, True, True, False, False, False, False, False]]
    )
    try:
        virtualizer.segment(input_ids=ids, attention_mask=mask)
    except ValueError as exc:
        assert "right-padded contiguous" in str(exc)
    else:
        raise AssertionError("noncontiguous token mask did not fail closed")


def test_behavioral_objective_backpropagates_without_parent_embedding_target() -> None:
    torch.manual_seed(602)
    batch, candidates, edges, fields, slots, views, dim = 3, 5, 7, 6, 4, 5, 12
    candidate_logits = torch.randn(batch, candidates, requires_grad=True)
    support_logits = torch.randn(batch, edges, requires_grad=True)
    source_weight = torch.softmax(
        torch.randn(batch, fields, requires_grad=True),
        dim=-1,
    )
    target_weight = torch.softmax(
        torch.randn(batch, fields, requires_grad=True),
        dim=-1,
    )
    latent_slots = torch.randn(batch, slots, dim, requires_grad=True)
    source_views = torch.randn(batch, views, dim, requires_grad=True)
    pooled = torch.randn(batch, dim, requires_grad=True)
    permuted = pooled + 0.01 * torch.randn_like(pooled)

    result = full_envelope_behavioral_objective(
        candidate_logits=candidate_logits,
        target_index=torch.tensor([0, 2, 4]),
        support_logits=support_logits,
        support_target=torch.tensor(
            [
                [1,0,1,0,0,0,0],
                [0,1,0,1,0,0,0],
                [1,0,0,0,1,0,0],
            ]
        ),
        support_valid_mask=torch.ones(batch, edges, dtype=torch.bool),
        source_weight=source_weight,
        target_weight=target_weight,
        source_target_index=torch.tensor([0, 1, 2]),
        target_target_index=torch.tensor([2, 3, 4]),
        endpoint_active_mask=torch.ones(batch, dtype=torch.bool),
        decisive_ablated_candidate_logits=candidate_logits - torch.tensor(
            [[0.5,0,0,0,0],[0,0,0.5,0,0],[0,0,0,0,0.5]],
            dtype=candidate_logits.dtype,
        ),
        irrelevant_removed_candidate_logits=candidate_logits + 0.001,
        latent_slots=latent_slots,
        source_views=source_views,
        view_available=torch.ones(batch, views, dtype=torch.bool),
        pooled_state_permuted=permuted,
        pooled_state_original=pooled,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert candidate_logits.grad is not None
    assert support_logits.grad is not None
    assert latent_slots.grad is not None


def test_macro_family_balancer_has_no_learned_task_weights() -> None:
    balancer = MacroFamilyLossBalancer(
        {
            "semantic": 1.0,
            "operator": 1.0,
            "fabric": 1.0,
        }
    )
    a = torch.tensor(100.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    c = torch.tensor(0.01, requires_grad=True)
    result = balancer(
        {
            "semantic": {"mlm": a},
            "operator": {"relation": b},
            "fabric": {"judgment": c},
        },
        update_ema=True,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert a.grad is not None and b.grad is not None and c.grad is not None
    report = balancer.parameter_report()
    assert report["total_parameters"] == 0
    assert report["learned_family_weight_parameters"] == 0
    assert report["test_adaptive_weights"] is False


def test_support_objective_explicitly_trains_null_support_head() -> None:
    support_logits=torch.tensor(
        [[-1.0,-2.0],[2.0,-1.0]],
        requires_grad=True,
    )
    target=torch.tensor(
        [[0.0,0.0],[1.0,0.0]],
    )
    valid=torch.ones(2,2,dtype=torch.bool)
    null_logit=torch.tensor([0.2,-0.3],requires_grad=True)
    loss=support_selection_loss(
        support_logits,
        target,
        valid,
        null_support_logit=null_logit,
    )
    loss.backward()
    assert null_logit.grad is not None
    assert float(null_logit.grad.abs().sum()) > 0.0
    assert support_logits.grad is not None


def test_chunked_late_interaction_allows_empty_padded_items_without_signal() -> None:
    torch.manual_seed(191)
    query = F.normalize(torch.randn(2,3,9,8),dim=-1)
    items = F.normalize(torch.randn(2,4,3,7,8),dim=-1)
    query_mask = torch.ones(2,9,dtype=torch.bool)
    item_mask = torch.ones(2,4,7,dtype=torch.bool)
    item_mask[0,3] = False
    item_mask[1,2:] = False
    q_to_i, i_to_q = chunked_batched_bidirectional_late_max(
        query=query,
        query_mask=query_mask,
        items=items,
        item_mask=item_mask,
        chunk_tokens=3,
    )
    assert torch.equal(
        q_to_i[0,3],
        torch.zeros_like(q_to_i[0,3]),
    )
    assert torch.equal(
        i_to_q[0,3],
        torch.zeros_like(i_to_q[0,3]),
    )
    assert torch.equal(
        q_to_i[1,2:],
        torch.zeros_like(q_to_i[1,2:]),
    )
    assert torch.equal(
        i_to_q[1,2:],
        torch.zeros_like(i_to_q[1,2:]),
    )
