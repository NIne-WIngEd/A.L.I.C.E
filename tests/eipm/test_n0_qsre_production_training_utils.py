from __future__ import annotations

import torch

from alice_personality.n0.qsre_production_core import (
    CONTROL_DEFER,
    CONTROL_RELATIONAL,
    QSREProductionOperatorState,
)
from qsre_production_training_utils import (
    downstream_success,
    focus_metrics,
    operator_metrics,
    relation_program_exact,
    support_metrics,
)


def make_operator(
    *,
    relations: int = 3,
    steps: int = 3,
    batch: int = 2,
) -> QSREProductionOperatorState:
    relation = torch.zeros(batch, steps, relations)
    relation[:, 0, 0] = 1.0
    mass = torch.zeros(batch, steps)
    mass[:, 0] = 1.0
    stop = torch.zeros(batch, steps)
    stop[:, 1] = 1.0
    unknown = torch.zeros(batch, steps)
    role = torch.zeros(batch, 4)
    role[:, 1] = 1.0
    traversal = torch.zeros(batch, 3)
    traversal[:, 0] = 1.0
    direction = torch.zeros(batch, 3)
    direction[:, 0] = 1.0
    modifiers = torch.zeros(batch, 4)
    control = torch.zeros(batch, 3)
    control[:, CONTROL_RELATIONAL] = 1.0
    return QSREProductionOperatorState(
        relation_distribution=relation,
        relation_step_mass=mass,
        stop_probability=stop,
        unknown_probability=unknown,
        role_distribution=role,
        traversal_distribution=traversal,
        direction_distribution=direction,
        modifier_weight=modifiers,
        applicability=torch.ones(batch),
        control_distribution=control,
        continuous_state=torch.zeros(batch, 8),
        uncertainty=torch.zeros(batch),
    )


def test_relation_program_exact_accepts_explicit_stop_budget() -> None:
    op = make_operator()
    target = torch.tensor([[0, 0], [0, 0]])
    target_mask = torch.tensor([[True, False], [True, False]])
    exact = relation_program_exact(op, target, target_mask)
    assert exact.tolist() == [True, True]


def test_operator_metrics_distinguish_stop_and_unknown() -> None:
    op = make_operator()
    op.unknown_probability[1, 0] = 1.0
    op.stop_probability[1, 0] = 0.0
    op.relation_step_mass[1, 0] = 0.0
    op.relation_distribution[1].zero_()
    op.control_distribution[1].zero_()
    op.control_distribution[1, CONTROL_DEFER] = 1.0

    split = {
        "relation_target": torch.tensor([[0, 0], [0, 0]]),
        "relation_target_mask": torch.tensor([[True, False], [False, False]]),
        "role_target": torch.tensor([1, 3]),
        "traversal_target": torch.tensor([0, 0]),
        "direction_target": torch.tensor([0, 2]),
        "modifier_target": torch.zeros(2, 4),
        "control_target": torch.tensor([CONTROL_RELATIONAL, CONTROL_DEFER]),
        "termination_target": torch.tensor([0, 1]),
        "open_schema": torch.tensor([False, False]),
    }
    metrics = operator_metrics(
        operator=op,
        split=split,
        indices=torch.arange(2),
    )
    assert metrics["termination_accuracy"] == 1.0
    assert metrics["unknown_termination_accuracy"] == 1.0


def test_downstream_nonrelational_mass_is_fail_closed() -> None:
    probability = torch.tensor(
        [[0.0, 0.0, 0.0], [0.004, 0.003, 0.0]]
    )
    target = torch.zeros(2, 3)
    control = torch.tensor([CONTROL_DEFER, CONTROL_DEFER])
    result = downstream_success(
        probability,
        target,
        control,
        nonrelational_mass_threshold=0.01,
    )
    assert result["success"].tolist() == [True, True]


def test_support_metrics_require_exact_type_compatible_support() -> None:
    predicted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    oracle = predicted.clone()
    type_ok = torch.tensor([[True, True], [True, False]])
    metrics = support_metrics(
        predicted=predicted,
        oracle=oracle,
        type_compatible=type_ok,
    )
    assert metrics["edge_f1"] == 1.0
    assert metrics["exact_set_accuracy"] == 1.0
    assert metrics["type_violation_rate"] > 0.0


def test_focus_metrics_score_only_path_rows() -> None:
    predicted = torch.tensor(
        [[0.0, 1.0, 0.0], [0.7, 0.3, 0.0], [0.0, 0.0, 1.0]]
    )
    oracle = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    traversal = torch.tensor([1, 0, 1])
    metrics = focus_metrics(
        predicted=predicted,
        oracle=oracle,
        traversal_target=traversal,
    )
    assert metrics["path_focus_top1_accuracy"] == 1.0
    assert metrics["path_focus_exact_set_accuracy"] == 1.0
