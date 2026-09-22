from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from alice_personality.n0.chunked_late_interaction import (
    chunked_batched_bidirectional_late_max,
)
from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
    full_envelope_behavioral_objective,
    irrelevant_view_invariance_loss,
    latent_noncollapse_loss,
    support_selection_loss,
)
from alice_personality.n0.full_envelope_loss_balancer_v1 import (
    MacroFamilyLossBalancer,
)
from alice_personality.n0.semantic_context_virtualizer_v1 import (
    SemanticContextVirtualizerConfig,
    SemanticContextVirtualizerV1,
)
from alice_personality.n0.semantic_segment_context_bridge_v1 import (
    SemanticSegmentContextBridgeConfig,
    SemanticSegmentContextBridgeV1,
)
from alice_personality.n0.full_envelope_semantic_input_v1 import (
    FullEnvelopeSemanticInputConfig,
    FullEnvelopeSemanticInputV1,
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


def test_context_virtualizer_splits_overlap_at_midpoint_instead_of_later_window_bias() -> None:
    virtualizer = SemanticContextVirtualizerV1(
        SemanticContextVirtualizerConfig(
            native_window_tokens=10,
            overlap_tokens=4,
        )
    )
    ids = torch.arange(1, 23).view(1,22)
    mask = torch.ones_like(ids,dtype=torch.bool)
    segmented = virtualizer.segment(
        input_ids=ids,
        attention_mask=mask,
    )
    starts = segmented["segment_absolute_start"][0]
    ends = segmented["segment_absolute_end"][0]
    own_starts = segmented["content_absolute_start"][0]
    own_ends = segmented["content_absolute_end"][0]
    valid = segmented["segment_valid_mask"][0]
    indices = torch.nonzero(valid,as_tuple=False).flatten().tolist()
    assert len(indices) >= 2
    for left,right in zip(indices,indices[1:]):
        assert int(own_ends[left]) == int(own_starts[right])
        overlap_start = int(starts[right])
        overlap_end = int(ends[left])
        assert overlap_end > overlap_start
        midpoint = (overlap_start + overlap_end)//2
        assert int(own_ends[left]) == midpoint
        assert int(own_starts[right]) == midpoint
        assert midpoint > overlap_start
        assert midpoint < overlap_end
    report=virtualizer.parameter_report()
    assert report["overlap_midpoint_ownership"] is True
    assert report["boundary_context_bias_to_later_window"] is False


def _tiny_segment_bridge() -> SemanticSegmentContextBridgeV1:
    return SemanticSegmentContextBridgeV1(
        SemanticSegmentContextBridgeConfig(
            semantic_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            num_layers=1,
            metadata_dim=3,
            query_chunk_segments=2,
            key_chunk_segments=2,
            dropout=0.0,
            initial_context_scale=1.0e-2,
        )
    ).eval()


def test_segment_context_bridge_is_segment_permutation_equivariant() -> None:
    torch.manual_seed(221)
    model=_tiny_segment_bridge()
    states=torch.randn(2,5,3,7,24)
    attention=torch.ones(2,5,7,dtype=torch.bool)
    valid=torch.ones(2,5,dtype=torch.bool)
    metadata=torch.rand(2,5,3)
    perm=torch.tensor([3,0,4,1,2])
    inverse=torch.argsort(perm)
    with torch.no_grad():
        a=model(
            segment_hidden_states=states,
            segment_attention_mask=attention,
            segment_valid_mask=valid,
            segment_metadata=metadata,
        )
        b=model(
            segment_hidden_states=states[:,perm],
            segment_attention_mask=attention[:,perm],
            segment_valid_mask=valid[:,perm],
            segment_metadata=metadata[:,perm],
        )
    assert torch.allclose(
        a["contextualized_segment_hidden_states"],
        b["contextualized_segment_hidden_states"][:,inverse],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        a["segment_context_state"],
        b["segment_context_state"][:,inverse],
        atol=1e-5,
        rtol=1e-5,
    )


def test_segment_context_bridge_allows_distant_segment_to_change_local_tokens() -> None:
    torch.manual_seed(222)
    model=_tiny_segment_bridge()
    states=torch.randn(1,4,3,6,24)
    attention=torch.ones(1,4,6,dtype=torch.bool)
    valid=torch.ones(1,4,dtype=torch.bool)
    metadata=torch.tensor(
        [[[0.0,0.25,0.25],[0.25,0.50,0.25],[0.50,0.75,0.25],[0.75,1.0,0.25]]]
    )
    changed=states.clone()
    changed[:,3] = changed[:,3] + 4.0
    with torch.no_grad():
        a=model(
            segment_hidden_states=states,
            segment_attention_mask=attention,
            segment_valid_mask=valid,
            segment_metadata=metadata,
        )
        b=model(
            segment_hidden_states=changed,
            segment_attention_mask=attention,
            segment_valid_mask=valid,
            segment_metadata=metadata,
        )
    delta=(
        a["contextualized_segment_hidden_states"][:,0]
        - b["contextualized_segment_hidden_states"][:,0]
    ).abs().max()
    assert float(delta) > 1.0e-8


def test_segment_context_bridge_backpropagates_across_windows() -> None:
    torch.manual_seed(223)
    model=_tiny_segment_bridge().train()
    states=torch.randn(1,4,3,6,24,requires_grad=True)
    attention=torch.ones(1,4,6,dtype=torch.bool)
    valid=torch.ones(1,4,dtype=torch.bool)
    metadata=torch.tensor(
        [[[0.0,0.25,0.25],[0.25,0.50,0.25],[0.50,0.75,0.25],[0.75,1.0,0.25]]]
    )
    out=model(
        segment_hidden_states=states,
        segment_attention_mask=attention,
        segment_valid_mask=valid,
        segment_metadata=metadata,
    )
    loss=out["contextualized_segment_hidden_states"][:,0].square().mean()
    loss.backward()
    assert states.grad is not None
    assert float(states.grad[:,3].abs().sum()) > 0.0
    assert model.context_scale.grad is not None
    assert float(model.context_scale.grad.abs()) > 0.0


def test_segment_context_bridge_has_no_segment_identity_or_count_ceiling() -> None:
    model=_tiny_segment_bridge()
    report=model.parameter_report()
    assert report["segment_identity_parameters"] == 0
    assert report["segment_count_dependent_parameters"] == 0
    assert report["segment_count_ceiling"] is None
    assert report["product_context_token_ceiling"] is None
    assert report["cross_window_semantic_interaction"] is True
    assert report["global_segment_attention_memory_bounded"] is True
    assert report["full_segment_pair_matrix_materialized"] is False
    assert report["dense_unbounded_token_attention_equivalence_claimed"] is False


def test_virtualizer_report_requires_cross_window_bridge_for_long_query_semantics() -> None:
    report=SemanticContextVirtualizerV1().parameter_report()
    assert report["token_ownership_lossless"] is True
    assert report["standalone_cross_window_semantics_complete"] is False
    assert report["segment_context_bridge_required_for_long_query_semantics"] is True
    assert report["dense_unbounded_native_attention_equivalence_claimed"] is False


def test_source_recoverability_mask_does_not_force_irrelevant_views_into_latent() -> None:
    from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
        source_view_recoverability_loss,
    )
    torch.manual_seed(232)
    latent=torch.randn(1,3,12,requires_grad=True)
    views=torch.randn(1,4,12,requires_grad=True)
    available=torch.ones(1,4,dtype=torch.bool)
    recoverable=torch.tensor([[True,True,False,False]])
    loss=source_view_recoverability_loss(
        latent,
        views,
        available,
        recoverable,
    )
    loss.backward()
    assert views.grad is not None
    assert float(views.grad[:,:2].abs().sum()) > 0.0
    assert torch.equal(
        views.grad[:,2:],
        torch.zeros_like(views.grad[:,2:]),
    )


def test_segment_context_bridge_padded_segments_are_inert() -> None:
    torch.manual_seed(251)
    model=_tiny_segment_bridge()
    states=torch.randn(2,4,3,6,24)
    changed=states.clone()
    changed[1,2:] = torch.randn_like(changed[1,2:]) * 100.0
    attention=torch.ones(2,4,6,dtype=torch.bool)
    attention[1,2:]=False
    valid=torch.tensor(
        [[True,True,True,True],[True,True,False,False]],
        dtype=torch.bool,
    )
    metadata=torch.rand(2,4,3)
    metadata[1,2:]=0.0
    with torch.no_grad():
        a=model(
            segment_hidden_states=states,
            segment_attention_mask=attention,
            segment_valid_mask=valid,
            segment_metadata=metadata,
        )
        b=model(
            segment_hidden_states=changed,
            segment_attention_mask=attention,
            segment_valid_mask=valid,
            segment_metadata=metadata,
        )
    assert torch.allclose(
        a["segment_context_state"][1,:2],
        b["segment_context_state"][1,:2],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        a["segment_context_state"][1,2:],
        torch.zeros_like(a["segment_context_state"][1,2:]),
    )
    assert torch.allclose(
        a["contextualized_segment_hidden_states"][1,:2],
        b["contextualized_segment_hidden_states"][1,:2],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        a["contextualized_segment_hidden_states"][1,2:],
        torch.zeros_like(a["contextualized_segment_hidden_states"][1,2:]),
    )


class _FakeSemanticBackbone(torch.nn.Module):
    def __init__(self, *, vocab: int = 128, width: int = 24, hidden_states: int = 3) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, width)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(width, width) for _ in range(hidden_states - 1)]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        x = self.embedding(input_ids)
        states = [x]
        for layer in self.layers:
            x = torch.tanh(layer(x))
            states.append(x)
        mask = attention_mask.to(x.dtype).unsqueeze(-1)
        x = x * mask
        states = [state * mask for state in states]
        return SimpleNamespace(
            last_hidden_state=x,
            hidden_states=tuple(states) if output_hidden_states else None,
        )


def _tiny_semantic_input() -> FullEnvelopeSemanticInputV1:
    return FullEnvelopeSemanticInputV1(
        FullEnvelopeSemanticInputConfig(
            semantic_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            native_window_tokens=8,
            overlap_tokens=2,
            segment_bridge_layers=1,
            segment_query_chunk=2,
            segment_key_chunk=2,
            special_token_ids=(0,1,2,3,4),
            pad_token_id=0,
            dropout=0.0,
        )
    )


def test_full_semantic_input_short_path_preserves_all_layers() -> None:
    torch.manual_seed(271)
    backbone=_FakeSemanticBackbone()
    encoder=_tiny_semantic_input().eval()
    ids=torch.tensor(
        [[2,10,11,12,3,0],[2,20,21,22,23,3]],
        dtype=torch.long,
    )
    mask=ids.ne(0)
    with torch.no_grad():
        out=encoder.encode_items(
            backbone=backbone,
            input_ids=ids,
            attention_mask=mask,
        )
    assert out["used_virtualization"] is False
    assert out["hidden_states"].shape == (2,3,6,24)
    assert out["content_mask"].shape == (2,6)
    assert torch.equal(
        out["content_mask"][0],
        torch.tensor([False,True,True,True,False,False]),
    )


def test_full_semantic_input_virtualizes_long_relation_factor_and_descriptor_text() -> None:
    torch.manual_seed(272)
    backbone=_FakeSemanticBackbone()
    encoder=_tiny_semantic_input().eval()
    ids=torch.tensor(
        [
            [2,10,11,12,13,14,15,16,17,18,19,20,3],
            [2,30,31,32,33,34,35,36,37,38,39,40,3],
        ],
        dtype=torch.long,
    )
    mask=torch.ones_like(ids,dtype=torch.bool)
    domain=torch.tensor([[True,False],[False,True]])
    range_mask=torch.tensor([[False,True],[True,False]])
    symmetric=torch.tensor([False,True])
    with torch.no_grad():
        relation=encoder.encode_relation_bank(
            backbone=backbone,
            input_ids=ids,
            attention_mask=mask,
            domain_type_mask=domain,
            range_type_mask=range_mask,
            symmetric=symmetric,
        )
        factor=encoder.encode_semantic_bank(
            backbone=backbone,
            input_ids=ids,
            attention_mask=mask,
        )
    assert relation.token_states.shape[:3] == (2,3,13)
    assert factor.token_states.shape[:3] == (2,3,13)
    assert torch.equal(relation.token_mask,factor.token_mask)
    report=encoder.parameter_report()
    assert report["long_relation_description_supported"] is True
    assert report["long_factor_description_supported"] is True
    assert report["long_descriptor_supported"] is True
    assert report["product_context_token_ceiling"] is None


def test_full_semantic_input_padded_items_keep_invalid_fields_and_candidates_inert() -> None:
    torch.manual_seed(273)
    backbone=_FakeSemanticBackbone()
    encoder=_tiny_semantic_input().eval()
    ids=torch.tensor(
        [[
            [2,10,11,12,13,14,15,16,17,3],
            [2,20,21,22,23,24,25,26,27,3],
            [0,0,0,0,0,0,0,0,0,0],
        ]],
        dtype=torch.long,
    )
    changed=ids.clone()
    changed[0,2]=torch.tensor([2,90,91,92,93,94,95,96,97,3])
    valid=torch.tensor([[True,True,False]])
    mask=ids.ne(0)
    changed_mask=changed.ne(0)
    with torch.no_grad():
        a=encoder.encode_padded_items(
            backbone=backbone,
            input_ids=ids,
            attention_mask=mask,
            item_valid_mask=valid,
        )
        b=encoder.encode_padded_items(
            backbone=backbone,
            input_ids=changed,
            attention_mask=changed_mask,
            item_valid_mask=valid,
        )
    assert torch.allclose(
        a["hidden_states"][:,:2],
        b["hidden_states"][:,:2],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        a["hidden_states"][:,2],
        torch.zeros_like(a["hidden_states"][:,2]),
    )
    assert torch.equal(
        b["hidden_states"][:,2],
        torch.zeros_like(b["hidden_states"][:,2]),
    )
    assert not bool(a["token_mask"][:,2].any())
    assert not bool(b["token_mask"][:,2].any())


def test_full_semantic_input_long_path_gradient_reaches_backbone_and_segment_bridge() -> None:
    torch.manual_seed(274)
    backbone=_FakeSemanticBackbone()
    encoder=_tiny_semantic_input().train()
    ids=torch.tensor(
        [[2,10,11,12,13,14,15,16,17,18,19,20,21,22,3]],
        dtype=torch.long,
    )
    mask=torch.ones_like(ids,dtype=torch.bool)
    out=encoder.encode_items(
        backbone=backbone,
        input_ids=ids,
        attention_mask=mask,
    )
    assert out["used_virtualization"] is True
    loss=out["hidden_states"][:,:,-3:,:].square().mean()
    loss.backward()
    assert backbone.embedding.weight.grad is not None
    assert float(backbone.embedding.weight.grad.abs().sum()) > 0.0
    bridge_grads=[
        p.grad
        for p in encoder.segment_bridge.parameters()
        if p.requires_grad
    ]
    assert bridge_grads
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in bridge_grads)


def test_full_semantic_input_has_no_text_surface_native_window_product_ceiling() -> None:
    report=_tiny_semantic_input().parameter_report()
    assert report["single_shared_backbone_reference"] is True
    assert report["all_text_surfaces_share_virtualization_policy"] is True
    assert report["long_query_supported"] is True
    assert report["long_relation_description_supported"] is True
    assert report["long_factor_description_supported"] is True
    assert report["long_field_supported"] is True
    assert report["long_candidate_supported"] is True
    assert report["long_descriptor_supported"] is True
    assert report["item_count_dependent_parameters"] == 0
    assert report["token_count_dependent_parameters"] == 0
    assert report["item_count_ceiling"] is None
    assert report["product_context_token_ceiling"] is None


def test_full_semantic_input_allows_fully_unavailable_optional_bank() -> None:
    torch.manual_seed(275)
    backbone=_FakeSemanticBackbone()
    encoder=_tiny_semantic_input().eval()
    ids=torch.tensor(
        [[[2,10,11,3,0,0],[2,20,21,3,0,0]]],
        dtype=torch.long,
    )
    mask=ids.ne(0)
    valid=torch.tensor([[False,False]])
    with torch.no_grad():
        out=encoder.encode_padded_items(
            backbone=backbone,
            input_ids=ids,
            attention_mask=mask,
            item_valid_mask=valid,
        )
    assert out["hidden_states"].shape == (1,2,3,6,24)
    assert torch.equal(
        out["hidden_states"],
        torch.zeros_like(out["hidden_states"]),
    )
    assert not bool(out["token_mask"].any())


def test_latent_noncollapse_gradient_is_finite_on_repeated_and_degenerate_slots() -> None:
    torch.manual_seed(291)
    base=torch.randn(2,1,16)
    repeated=base.expand(2,6,16).clone().requires_grad_(True)
    loss=latent_noncollapse_loss(repeated)
    assert torch.isfinite(loss)
    loss.backward()
    assert repeated.grad is not None
    assert bool(torch.isfinite(repeated.grad).all())
    assert float(repeated.grad.abs().sum()) > 0.0

    zero=torch.zeros(2,6,16,requires_grad=True)
    zero_loss=latent_noncollapse_loss(zero)
    assert torch.isfinite(zero_loss)
    zero_loss.backward()
    assert zero.grad is not None
    assert bool(torch.isfinite(zero.grad).all())


def test_latent_noncollapse_penalizes_antipodal_rank_one_collapse() -> None:
    torch.manual_seed(292)
    direction=F.normalize(torch.randn(1,1,24),dim=-1)
    signs=torch.tensor([1.0,-1.0,1.0,-1.0,1.0,-1.0]).view(1,6,1)
    rank_one=(direction*signs).requires_grad_(True)
    diverse=F.normalize(torch.randn(1,6,24),dim=-1)
    rank_one_loss=latent_noncollapse_loss(rank_one)
    diverse_loss=latent_noncollapse_loss(diverse)
    assert torch.isfinite(rank_one_loss)
    assert torch.isfinite(diverse_loss)
    assert float(rank_one_loss) > float(diverse_loss)
    rank_one_loss.backward()
    assert rank_one.grad is not None
    assert bool(torch.isfinite(rank_one.grad).all())


def test_irrelevant_view_invariance_gradient_is_finite_with_masked_candidates() -> None:
    normal=torch.tensor(
        [[4.0,1.0,-3.0,2.0],[0.5,-1.0,3.0,-2.0]],
        requires_grad=True,
    )
    removed=torch.tensor(
        [[3.5,1.2,-2.5,2.2],[0.4,-0.8,2.7,-1.5]],
        requires_grad=True,
    )
    valid=torch.tensor(
        [[True,True,False,False],[True,False,True,False]],
        dtype=torch.bool,
    )
    active=torch.tensor([True,True],dtype=torch.bool)
    loss=irrelevant_view_invariance_loss(
        normal,
        removed,
        candidate_valid_mask=valid,
        active_mask=active,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert normal.grad is not None
    assert removed.grad is not None
    assert bool(torch.isfinite(normal.grad).all())
    assert bool(torch.isfinite(removed.grad).all())
    assert torch.equal(
        normal.grad.masked_select(~valid),
        torch.zeros_like(normal.grad.masked_select(~valid)),
    )
    assert torch.equal(
        removed.grad.masked_select(~valid),
        torch.zeros_like(removed.grad.masked_select(~valid)),
    )
