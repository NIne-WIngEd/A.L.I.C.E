from __future__ import annotations

import torch

from alice_personality.n0.evidence_graph_objectives import (
    evidence_graph_objective,
    relation_counterfactual_margin_loss,
    soft_evidence_selection_loss,
)


def test_soft_evidence_selection_supports_multiple_valid_sources() -> None:
    weights = torch.tensor([[0.55, 0.35, 0.10]])
    target = torch.tensor([[0.60, 0.40, 0.00]])
    valid = torch.tensor([[True, True, True]])
    loss = soft_evidence_selection_loss(weights, target, valid)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_soft_evidence_selection_rejects_padding_target_mass() -> None:
    weights = torch.tensor([[0.8, 0.2]])
    target = torch.tensor([[0.5, 0.5]])
    valid = torch.tensor([[True, False]])
    try:
        soft_evidence_selection_loss(weights, target, valid)
    except ValueError as exc:
        assert "padding" in str(exc)
    else:
        raise AssertionError("padding target mass must be rejected")


def test_counterfactual_margin_rewards_relation_sensitive_read() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    correct = target.clone()
    corrupted = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    active = torch.tensor([True, True])
    good = relation_counterfactual_margin_loss(
        correct,
        corrupted,
        target,
        active_mask=active,
        margin=0.1,
    )
    bad = relation_counterfactual_margin_loss(
        corrupted,
        correct,
        target,
        active_mask=active,
        margin=0.1,
    )
    assert float(good) == 0.0
    assert float(bad) > 0.0


def test_counterfactual_can_be_disabled_per_example() -> None:
    target = torch.tensor([[1.0, 0.0]])
    value = relation_counterfactual_margin_loss(
        target,
        target,
        target,
        active_mask=torch.tensor([False]),
    )
    assert float(value) == 0.0


def test_joint_objective_is_finite_and_preserves_soft_targets() -> None:
    torch.manual_seed(5)
    batch, fields, hidden = 3, 4, 8
    field_weights = torch.softmax(torch.randn(batch, fields), dim=-1)
    target_distribution = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.6, 0.3, 0.1, 0.0],
        ]
    )
    valid_mask = torch.ones(batch, fields, dtype=torch.bool)
    pooled = torch.randn(batch, hidden)
    target = torch.randn(batch, hidden)
    corrupted = torch.randn(batch, hidden)
    field_semantic = torch.randn(batch, fields, hidden)
    field_states = field_semantic + 0.01 * torch.randn_like(field_semantic)

    losses = evidence_graph_objective(
        field_weights=field_weights,
        target_distribution=target_distribution,
        valid_mask=valid_mask,
        pooled_state=pooled,
        semantic_target=target,
        corrupted_pooled_state=corrupted,
        counterfactual_active_mask=torch.tensor([True, False, True]),
        field_states=field_states,
        field_semantic=field_semantic,
        permuted_pooled_state=pooled.clone(),
    )
    assert set(losses) == {
        "loss",
        "selection",
        "semantic",
        "counterfactual",
        "field_preservation",
        "permutation",
    }
    assert all(torch.isfinite(value) for value in losses.values())
