from __future__ import annotations

import torch

from alice_personality.n0.qsre_schema_matcher import QSRESchemaMatcher


def matcher() -> QSRESchemaMatcher:
    torch.manual_seed(1)
    return QSRESchemaMatcher(
        semantic_dim=24,
        model_dim=24,
        num_hidden_states=3,
    ).eval()


def test_schema_matcher_has_no_relation_or_hop_parameter_axis() -> None:
    report = matcher().parameter_report()
    assert report["relation_identity_parameters"] == 0
    assert report["candidate_count_dependent_parameters"] == 0
    assert report["reasoning_step_dependent_parameters"] == 0
    assert report["shared_query_schema_projection"] is True
    assert report["candidate_conditioned_query_evidence"] is True
    assert report["runtime_schema_cardinality_ceiling"] is None


def test_runtime_only_candidate_can_win_by_semantic_description() -> None:
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
    query = target.unsqueeze(0)
    qmask = torch.ones(1, 5, dtype=torch.bool)
    smask = torch.ones(3, 5, dtype=torch.bool)
    out = model(
        query_hidden_states=query,
        query_token_mask=qmask,
        schema_hidden_states=schema,
        schema_token_mask=smask,
    )
    assert int(out["logits"][0].argmax().item()) == 2
    assert out["query_evidence"].shape == (1, 3, 3, 5)
    assert torch.isfinite(out["logits"]).all()


def test_schema_candidate_permutation_is_equivariant() -> None:
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
    )["logits"]

    perm = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.argsort(perm)
    moved = model(
        query_hidden_states=query,
        query_token_mask=qmask,
        schema_hidden_states=schema[perm],
        schema_token_mask=smask[perm],
    )["logits"]
    assert torch.allclose(base, moved[:, inverse], atol=1e-5, rtol=1e-5)


def test_candidate_conditioned_query_evidence_changes_with_schema_meaning() -> None:
    model = matcher()
    query = torch.zeros(1, 3, 4, 24)
    query[:, :, 0, 0] = 4.0
    query[:, :, 1, 1] = 4.0
    query[:, :, 2, 2] = 1.0
    query[:, :, 3, 3] = 1.0

    schema = torch.zeros(2, 3, 3, 24)
    schema[0, :, :, 0] = 4.0
    schema[1, :, :, 1] = 4.0
    qmask = torch.ones(1, 4, dtype=torch.bool)
    smask = torch.ones(2, 3, dtype=torch.bool)

    evidence = model(
        query_hidden_states=query,
        query_token_mask=qmask,
        schema_hidden_states=schema,
        schema_token_mask=smask,
    )["query_evidence"].sum(dim=2)

    assert int(evidence[0, 0].argmax().item()) == 0
    assert int(evidence[0, 1].argmax().item()) == 1


def test_remaining_evidence_downweights_consumed_query_position() -> None:
    model = matcher()
    query = torch.zeros(1, 3, 3, 24)
    query[:, :, 0, 0] = 5.0
    query[:, :, 1, 0] = 4.0
    query[:, :, 2, 2] = 1.0
    schema = torch.zeros(1, 3, 2, 24)
    schema[:, :, :, 0] = 5.0
    qmask = torch.ones(1, 3, dtype=torch.bool)
    smask = torch.ones(1, 2, dtype=torch.bool)

    projected_query = model.project(query)
    projected_schema = model.project(schema)
    full = model.match_projected(
        query_projected=projected_query,
        query_token_mask=qmask,
        schema_projected=projected_schema,
        schema_token_mask=smask,
    )["query_evidence"].sum(dim=2)
    remaining = torch.tensor([[0.05, 1.0, 1.0]])
    consumed = model.match_projected(
        query_projected=projected_query,
        query_token_mask=qmask,
        schema_projected=projected_schema,
        schema_token_mask=smask,
        query_remaining=remaining,
    )["query_evidence"].sum(dim=2)

    assert float(consumed[0, 0, 0]) < float(full[0, 0, 0])
    assert float(consumed[0, 0, 1]) > float(full[0, 0, 1])
