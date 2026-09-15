from __future__ import annotations

import pytest
import torch

from alice_personality.n0.structured_state_objectives import (
    binary_rationale_compatibility_loss,
    masked_field_preservation_loss,
    pooled_permutation_consistency_loss,
    structured_state_objective,
    symmetric_semantic_alignment_loss,
)


def test_semantic_alignment_prefers_matching_pairs() -> None:
    torch.manual_seed(3)
    target = torch.randn(4, 8)
    matching = target + 0.01 * torch.randn(4, 8)
    shuffled = target.roll(1, dims=0)

    matching_loss = symmetric_semantic_alignment_loss(matching, target, temperature=0.1)
    shuffled_loss = symmetric_semantic_alignment_loss(shuffled, target, temperature=0.1)

    assert torch.isfinite(matching_loss)
    assert matching_loss < shuffled_loss


def test_field_preservation_ignores_padding() -> None:
    semantic = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    field_states = semantic.clone()
    valid = torch.tensor([[True, True, False]])

    baseline = masked_field_preservation_loss(field_states, semantic, valid)
    field_states[:, 2, :] = torch.tensor([[-100.0, 100.0]])
    mutated_padding = masked_field_preservation_loss(field_states, semantic, valid)

    assert baseline.item() == pytest.approx(0.0, abs=1e-6)
    assert mutated_padding.item() == pytest.approx(0.0, abs=1e-6)


def test_permutation_consistency_is_zero_for_identical_pooled_state() -> None:
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = pooled_permutation_consistency_loss(state, state.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_compatibility_loss_prefers_correct_signs() -> None:
    structured = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    rationale = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([1.0, 0.0])
    loss = binary_rationale_compatibility_loss(structured, rationale, labels)
    assert loss.item() < 0.1


def test_combined_objective_is_finite_and_weighted() -> None:
    torch.manual_seed(11)
    pooled = torch.randn(4, 8)
    target = pooled + 0.05 * torch.randn(4, 8)
    fields = torch.randn(4, 3, 8)
    field_semantic = fields + 0.05 * torch.randn(4, 3, 8)
    valid = torch.ones(4, 3, dtype=torch.bool)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    result = structured_state_objective(
        pooled_state=pooled,
        target_semantic=target,
        field_states=fields,
        field_semantic=field_semantic,
        valid_mask=valid,
        permuted_pooled_state=pooled.clone(),
        compatibility_labels=labels,
    )
    assert set(result) == {
        "loss",
        "alignment",
        "compatibility",
        "field_preservation",
        "permutation",
    }
    assert all(torch.isfinite(value) for value in result.values())

    with pytest.raises(ValueError, match="sum to 1.0"):
        structured_state_objective(
            pooled_state=pooled,
            target_semantic=target,
            field_states=fields,
            field_semantic=field_semantic,
            valid_mask=valid,
            permuted_pooled_state=pooled.clone(),
            compatibility_labels=labels,
            alignment_weight=0.5,
            compatibility_weight=0.5,
            field_preservation_weight=0.5,
            permutation_weight=0.0,
        )
