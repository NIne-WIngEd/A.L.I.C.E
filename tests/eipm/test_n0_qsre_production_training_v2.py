from __future__ import annotations

import torch

from train_n0_v02_qsre_production_p2_v2 import (
    EVENT_CONTINUE,
    EVENT_STOP,
    EVENT_UNKNOWN,
    build_event_targets,
    macro_binary_factor_loss,
    macro_class_cross_entropy,
    macro_class_nll,
)
from train_n0_v02_qsre_production_p3_v2 import (
    balanced_binary_bce_with_logits,
)


def test_event_targets_separate_program_length_from_unknown_rejection() -> None:
    relation_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
            [True, True, True],
        ]
    )
    termination = torch.tensor([0, 1, 0])
    target, supervised = build_event_targets(
        relation_mask=relation_mask,
        termination_target=termination,
        pred_steps=4,
    )
    assert target[0].tolist() == [
        EVENT_CONTINUE,
        EVENT_CONTINUE,
        EVENT_STOP,
        EVENT_STOP,
    ]
    assert supervised[0].tolist() == [True, True, True, False]
    assert target[1, 0].item() == EVENT_UNKNOWN
    assert supervised[1].tolist() == [True, False, False, False]
    assert target[2].tolist() == [
        EVENT_CONTINUE,
        EVENT_CONTINUE,
        EVENT_CONTINUE,
        EVENT_STOP,
    ]
    assert supervised[2].tolist() == [True, True, True, True]


def test_dense_relation_supervision_can_recover_sparse_pruned_target() -> None:
    logits = torch.tensor(
        [[[12.0, -12.0]]],
        requires_grad=True,
    )
    target = torch.tensor([[1]])
    mask = torch.tensor([[True]])
    loss = macro_class_cross_entropy(
        logits,
        target,
        mask=mask,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert float(logits.grad[0, 0, 1]) < 0.0
    assert float(logits.grad[0, 0, 0]) > 0.0


def test_macro_class_nll_does_not_let_majority_class_dominate() -> None:
    # Nine easy class-0 rows and one deliberately weak class-1 row.
    # Macro reduction must give the minority class half of the objective.
    probability = torch.tensor(
        [[0.99, 0.01]] * 9 + [[0.60, 0.40]],
        dtype=torch.float32,
    )
    target = torch.tensor([0] * 9 + [1])
    macro = macro_class_nll(probability, target)
    expected = 0.5 * (
        -torch.log(torch.tensor(0.99))
        -torch.log(torch.tensor(0.40))
    )
    assert torch.allclose(macro, expected, atol=1e-6, rtol=1e-6)


def test_macro_binary_factor_loss_balances_positive_and_negative_states() -> None:
    probability = torch.tensor(
        [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.90, 0.10],
            [0.20, 0.40],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    value = macro_binary_factor_loss(probability, target)
    assert torch.isfinite(value)
    assert float(value) > 0.0


def test_p3_balanced_support_loss_preserves_rare_positive_signal() -> None:
    logits = torch.tensor([[5.0, -5.0, -5.0, -5.0]])
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = balanced_binary_bce_with_logits(logits, target, mask)
    positive = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :1],
        target[:, :1],
    )
    negative = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:],
        target[:, 1:],
    )
    expected = 0.5 * (positive + negative)
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
