from __future__ import annotations

import pytest

from alice_personality.n0.training_schedule import (
    accelerated_scheduler_steps,
    validate_scheduler_horizon,
)


def test_ddp_scheduler_horizon_scales_with_processes_when_batches_are_sharded() -> None:
    assert accelerated_scheduler_steps(
        200,
        num_processes=2,
        split_batches=False,
    ) == 400
    assert accelerated_scheduler_steps(
        20,
        num_processes=2,
        split_batches=False,
    ) == 40


def test_split_batch_scheduler_horizon_does_not_scale() -> None:
    assert accelerated_scheduler_steps(
        200,
        num_processes=2,
        split_batches=True,
    ) == 200


def test_single_process_scheduler_horizon_is_unchanged() -> None:
    assert accelerated_scheduler_steps(
        1000,
        num_processes=1,
        split_batches=False,
    ) == 1000


def test_scheduler_horizon_must_cover_current_training_target() -> None:
    validate_scheduler_horizon(
        max_steps=1000,
        warmup_steps=100,
        scheduler_total_steps=10_000,
    )

    with pytest.raises(ValueError, match="scheduler_total_steps must be >= max_steps"):
        validate_scheduler_horizon(
            max_steps=1000,
            warmup_steps=100,
            scheduler_total_steps=999,
        )


def test_warmup_must_fit_inside_scheduler_horizon() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be smaller"):
        validate_scheduler_horizon(
            max_steps=100,
            warmup_steps=100,
            scheduler_total_steps=100,
        )
