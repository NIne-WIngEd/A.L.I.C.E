from __future__ import annotations

import torch

from alice_personality.n0.adaptive_multi_view_latent_pool_objectives_v0_2 import (
    best_slot_semantic_alignment_loss,
    centered_slot_effective_rank,
    normalized_view_specialization,
    slot_redundancy_loss,
)


def test_best_slot_semantic_loss_does_not_reward_duplicate_target_slots() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    one_good = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    all_copies = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    first = best_slot_semantic_alignment_loss(one_good, target)
    second = best_slot_semantic_alignment_loss(all_copies, target)
    assert torch.allclose(first, second, atol=1e-7, rtol=0.0)
    assert float(first) == 0.0


def test_redundancy_loss_rejects_near_duplicate_slots() -> None:
    diverse = torch.eye(4).unsqueeze(0)
    copied = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4])
    assert slot_redundancy_loss(diverse) < slot_redundancy_loss(copied)


def test_view_specialization_is_permutation_free_and_rewards_distinct_use() -> None:
    available = torch.tensor([[True, True, True]])
    uniform = torch.full((1, 6, 3), 1.0 / 3.0)
    specialized = torch.tensor(
        [[
            [0.98, 0.01, 0.01],
            [0.97, 0.02, 0.01],
            [0.01, 0.98, 0.01],
            [0.02, 0.97, 0.01],
            [0.01, 0.01, 0.98],
            [0.01, 0.02, 0.97],
        ]]
    )
    uniform_score = normalized_view_specialization(uniform, available)
    specialized_score = normalized_view_specialization(specialized, available)
    permuted_score = normalized_view_specialization(specialized[:, [4, 1, 5, 0, 3, 2]], available)
    assert specialized_score > uniform_score + 0.5
    assert torch.allclose(specialized_score, permuted_score, atol=1e-6, rtol=1e-6)


def test_centered_effective_rank_detects_collapse_without_penalizing_common_mode() -> None:
    common = torch.tensor([1.0, 2.0, 3.0, 4.0])
    collapsed = common.view(1, 1, 4).expand(1, 4, 4).clone()
    residuals = torch.eye(4)
    diverse = common.view(1, 1, 4) + residuals.unsqueeze(0)
    collapsed_rank = centered_slot_effective_rank(collapsed)
    diverse_rank = centered_slot_effective_rank(diverse)
    assert float(collapsed_rank) == 0.0
    assert float(diverse_rank) > 0.5
