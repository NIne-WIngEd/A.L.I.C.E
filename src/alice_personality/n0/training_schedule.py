from __future__ import annotations


def accelerated_scheduler_steps(
    global_optimizer_steps: int,
    *,
    num_processes: int,
    split_batches: bool,
) -> int:
    """Translate global optimizer steps into Accelerate scheduler steps.

    Accelerate's scheduler wrapper advances the wrapped scheduler once per process
    when ``split_batches`` is false. N0 expresses warmup and training horizons in
    global optimizer updates, so the wrapped scheduler horizon must be expanded by
    ``num_processes`` in that configuration.
    """
    if global_optimizer_steps < 0:
        raise ValueError("global_optimizer_steps must be non-negative")
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    return global_optimizer_steps if split_batches else global_optimizer_steps * num_processes


def validate_scheduler_horizon(
    *,
    max_steps: int,
    warmup_steps: int,
    scheduler_total_steps: int,
) -> None:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if scheduler_total_steps < max_steps:
        raise ValueError("scheduler_total_steps must be >= max_steps")
    if warmup_steps >= scheduler_total_steps:
        raise ValueError("warmup_steps must be smaller than scheduler_total_steps")
