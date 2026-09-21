from __future__ import annotations

import torch

from alice_personality.n0.qsre_schema_matcher import QSRESchemaMatcher
from alice_personality.n0.qsre_semantic_authority import (
    candidate_zscore,
    combine_authority_components,
    slice_relation_authority,
)


def matcher() -> QSRESchemaMatcher:
    return QSRESchemaMatcher(
        semantic_dim=24,
        model_dim=24,
        num_hidden_states=3,
    ).eval()


def test_schema_matcher_is_parameter_free_and_identity_neutral() -> None:
    model = matcher()
    report = model.parameter_report()
    assert sum(p.numel() for p in model.parameters()) == 0
    assert report["total_parameters"] == 0
    assert report["trainable_parameters"] == 0
    assert report["relation_identity_parameters"] == 0
    assert report["factor_identity_parameters"] == 0
    assert report["candidate_count_dependent_parameters"] == 0
    assert report["reasoning_step_dependent_parameters"] == 0
    assert report["learned_semantic_metric"] is False
    assert report["frozen_semantic_authority_required"] is True
    assert report["runtime_schema_cardinality_ceiling"] is None


def test_parameter_free_token_evidence_prefers_matching_runtime_candidate() -> None:
    model = matcher()
    g = torch.Generator().manual_seed(2)
    target = torch.randn(3, 5, 24, generator=g)
    schema = torch.stack(
        [
            -target,
            torch.roll(target, shifts=7, dims=-1),
            target,
        ],
        dim=0,
    )
    out = model(
        query_hidden_states=target.unsqueeze(0),
        query_token_mask=torch.ones(1, 5, dtype=torch.bool),
        schema_hidden_states=schema,
        schema_token_mask=torch.ones(3, 5, dtype=torch.bool),
    )
    assert int(out["token_score"][0].argmax().item()) == 2
    assert out["query_evidence"].shape == (1, 3, 3, 5)
    assert out["remaining_support"].shape == (1, 3)


def test_parameter_free_matcher_is_schema_permutation_equivariant() -> None:
    model = matcher()
    g = torch.Generator().manual_seed(3)
    query = torch.randn(2, 3, 7, 24, generator=g)
    schema = torch.randn(5, 3, 6, 24, generator=g)
    qmask = torch.ones(2, 7, dtype=torch.bool)
    smask = torch.ones(5, 6, dtype=torch.bool)
    base = model(
        query_hidden_states=query,
        query_token_mask=qmask,
        schema_hidden_states=schema,
        schema_token_mask=smask,
    )["token_score"]

    perm = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.argsort(perm)
    moved = model(
        query_hidden_states=query,
        query_token_mask=qmask,
        schema_hidden_states=schema[perm],
        schema_token_mask=smask[perm],
    )["token_score"]
    assert torch.allclose(base, moved[:, inverse], atol=1e-6, rtol=1e-6)


def test_remaining_support_decreases_when_candidate_evidence_is_consumed() -> None:
    model = matcher()
    query = torch.zeros(1, 3, 3, 24)
    query[:, :, 0, 0] = 5.0
    query[:, :, 1, 1] = 1.0
    query[:, :, 2, 2] = 1.0
    schema = torch.zeros(1, 3, 2, 24)
    schema[:, :, :, 0] = 5.0
    qmask = torch.ones(1, 3, dtype=torch.bool)
    smask = torch.ones(1, 2, dtype=torch.bool)

    full = model(
        query_hidden_states=query,
        query_token_mask=qmask,
        schema_hidden_states=schema,
        schema_token_mask=smask,
    )
    consumed = model(
        query_hidden_states=query,
        query_token_mask=qmask,
        schema_hidden_states=schema,
        schema_token_mask=smask,
        query_remaining=torch.tensor([[0.05, 1.0, 1.0]]),
    )
    assert float(consumed["remaining_support"][0, 0]) < float(
        full["remaining_support"][0, 0]
    )


def test_frozen_authority_fusion_is_candidate_permutation_equivariant() -> None:
    joint = torch.tensor([[2.0, -1.0, 0.5]])
    semantic = torch.tensor([[0.1, 0.9, 0.4]])
    principle = torch.tensor([[0.4, 0.7, 0.1]])
    token = torch.tensor([[0.8, 0.2, 0.5]])
    base = combine_authority_components(
        joint_preference=joint,
        semantic_projection=semantic,
        principle_alignment=principle,
        token_evidence=token,
    )
    perm = torch.tensor([2, 0, 1])
    inverse = torch.argsort(perm)
    moved = combine_authority_components(
        joint_preference=joint[:, perm],
        semantic_projection=semantic[:, perm],
        principle_alignment=principle[:, perm],
        token_evidence=token[:, perm],
    )
    assert torch.allclose(base, moved[:, inverse], atol=1e-6, rtol=1e-6)


def test_runtime_subset_is_applied_before_candidate_normalization() -> None:
    cache = {
        "relation": {
            "joint_preference": torch.tensor([[[1.0, 2.0, 100.0]]]),
            "semantic_projection": torch.tensor([[[1.0, 3.0, -100.0]]]),
            "principle_alignment": torch.tensor([[[0.5, 0.8, 90.0]]]),
        }
    }
    sliced = slice_relation_authority(
        cache,
        indices=torch.tensor([0]),
        view=0,
        relation_count=2,
        device=torch.device("cpu"),
    )
    assert sliced["joint_preference"].shape == (1, 2)
    normalized = candidate_zscore(sliced["joint_preference"])
    assert torch.allclose(normalized, torch.tensor([[-1.0, 1.0]]), atol=1e-5)
