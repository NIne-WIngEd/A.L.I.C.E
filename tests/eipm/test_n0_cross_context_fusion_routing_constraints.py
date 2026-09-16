from __future__ import annotations

import pytest

from alice_personality.n0.cross_context_fusion_routing_constraints import (
    evaluate_routing_constraints,
    validate_routing_constraints,
)


def test_allowed_top_set_does_not_force_exact_internal_split() -> None:
    constraints = {
        "allowed_top_views": [1, 2],
        "min_set_mass": [{"views": [1, 2], "min": 0.70}],
        "max_view_mass": {"0": 0.30},
    }
    assert evaluate_routing_constraints([0.12, 0.68, 0.20], constraints)["passed"] is True
    assert evaluate_routing_constraints([0.12, 0.22, 0.66], constraints)["passed"] is True


def test_pairwise_dominance_catches_stale_view_overweight() -> None:
    constraints = {
        "allowed_top_views": [2],
        "pairwise_margins": [{"higher": 2, "lower": 0, "margin": 0.15}],
    }
    result = evaluate_routing_constraints([0.44, 0.20, 0.36], constraints)
    assert result["passed"] is False
    assert any(item["kind"] == "pairwise_margin" for item in result["violations"])


def test_missing_view_mass_is_always_invalid() -> None:
    result = evaluate_routing_constraints(
        [0.51, 0.48, 0.01],
        {"allowed_top_views": [0, 1]},
        available=[True, True, False],
    )
    assert result["passed"] is False
    assert any(item["kind"] == "missing_view_mass" for item in result["violations"])


def test_consensus_can_have_no_top_view_constraint() -> None:
    constraints = {"min_set_mass": [{"views": [0, 1, 2], "min": 0.999}]}
    assert evaluate_routing_constraints([0.80, 0.10, 0.10], constraints)["passed"] is True
    assert evaluate_routing_constraints([0.05, 0.15, 0.80], constraints)["passed"] is True


def test_invalid_view_index_fails_closed() -> None:
    with pytest.raises(ValueError, match="view index"):
        validate_routing_constraints({"allowed_top_views": [3]}, view_count=3)


def test_set_mass_and_view_bounds_are_enforced() -> None:
    constraints = {
        "min_view_mass": {"2": 0.50},
        "max_view_mass": {"0": 0.20},
        "min_set_mass": [{"views": [1, 2], "min": 0.80}],
    }
    assert evaluate_routing_constraints([0.10, 0.30, 0.60], constraints)["passed"] is True
    result = evaluate_routing_constraints([0.25, 0.30, 0.45], constraints)
    assert result["passed"] is False
    kinds = {item["kind"] for item in result["violations"]}
    assert "min_view_mass" in kinds
    assert "max_view_mass" in kinds
