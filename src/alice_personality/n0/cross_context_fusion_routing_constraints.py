from __future__ import annotations

from typing import Any, Sequence


def _view_index(value: Any, *, view_count: int) -> int:
    index = int(value)
    if index < 0 or index >= view_count:
        raise ValueError(f"view index {index} outside [0, {view_count})")
    return index


def validate_routing_constraints(constraints: dict[str, Any], *, view_count: int) -> None:
    if view_count < 2:
        raise ValueError("routing constraints require at least two views")
    allowed = constraints.get("allowed_top_views")
    if allowed is not None:
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("allowed_top_views must be a non-empty list when present")
        for value in allowed:
            _view_index(value, view_count=view_count)

    for name in ("min_view_mass", "max_view_mass"):
        mapping = constraints.get(name, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"{name} must be an object")
        for raw_index, raw_value in mapping.items():
            _view_index(raw_index, view_count=view_count)
            value = float(raw_value)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} values must be in [0, 1]")

    for name in ("min_set_mass", "max_set_mass"):
        items = constraints.get(name, [])
        if not isinstance(items, list):
            raise ValueError(f"{name} must be a list")
        for item in items:
            views = item.get("views")
            if not isinstance(views, list) or not views:
                raise ValueError(f"{name}.views must be non-empty")
            for value in views:
                _view_index(value, view_count=view_count)
            threshold = float(item["min"] if name == "min_set_mass" else item["max"])
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError(f"{name} threshold must be in [0, 1]")

    margins = constraints.get("pairwise_margins", [])
    if not isinstance(margins, list):
        raise ValueError("pairwise_margins must be a list")
    for item in margins:
        higher = _view_index(item["higher"], view_count=view_count)
        lower = _view_index(item["lower"], view_count=view_count)
        if higher == lower:
            raise ValueError("pairwise margin views must differ")
        margin = float(item.get("margin", 0.0))
        if margin < 0.0 or margin > 1.0:
            raise ValueError("pairwise margin must be in [0, 1]")


def evaluate_routing_constraints(
    weights: Sequence[float],
    constraints: dict[str, Any],
    *,
    available: Sequence[bool] | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    values = [float(value) for value in weights]
    if not values:
        raise ValueError("routing weights may not be empty")
    if any(value < -tolerance or value > 1.0 + tolerance for value in values):
        raise ValueError("routing weights must be in [0, 1]")
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError("routing weights must sum to one")
    view_count = len(values)
    validate_routing_constraints(constraints, view_count=view_count)

    if available is None:
        availability = [True] * view_count
    else:
        availability = [bool(value) for value in available]
        if len(availability) != view_count:
            raise ValueError("availability length must match routing weights")

    violations: list[dict[str, Any]] = []

    # Missing views are a behavioral impossibility, not a preferred percentage.
    for index, is_available in enumerate(availability):
        if not is_available and values[index] > tolerance:
            violations.append(
                {
                    "kind": "missing_view_mass",
                    "view": index,
                    "observed": values[index],
                    "maximum": 0.0,
                    "magnitude": values[index],
                }
            )

    allowed = constraints.get("allowed_top_views")
    if allowed:
        maximum = max(values)
        top = {index for index, value in enumerate(values) if maximum - value <= tolerance}
        allowed_set = {int(value) for value in allowed}
        if not top.intersection(allowed_set):
            best_allowed = max(values[index] for index in allowed_set)
            violations.append(
                {
                    "kind": "allowed_top_views",
                    "allowed": sorted(allowed_set),
                    "observed_top": sorted(top),
                    "magnitude": max(0.0, maximum - best_allowed),
                }
            )

    for raw_index, raw_min in constraints.get("min_view_mass", {}).items():
        index = int(raw_index)
        threshold = float(raw_min)
        if values[index] + tolerance < threshold:
            violations.append(
                {
                    "kind": "min_view_mass",
                    "view": index,
                    "observed": values[index],
                    "minimum": threshold,
                    "magnitude": threshold - values[index],
                }
            )

    for raw_index, raw_max in constraints.get("max_view_mass", {}).items():
        index = int(raw_index)
        threshold = float(raw_max)
        if values[index] - tolerance > threshold:
            violations.append(
                {
                    "kind": "max_view_mass",
                    "view": index,
                    "observed": values[index],
                    "maximum": threshold,
                    "magnitude": values[index] - threshold,
                }
            )

    for item in constraints.get("pairwise_margins", []):
        higher = int(item["higher"])
        lower = int(item["lower"])
        margin = float(item.get("margin", 0.0))
        observed = values[higher] - values[lower]
        if observed + tolerance < margin:
            violations.append(
                {
                    "kind": "pairwise_margin",
                    "higher": higher,
                    "lower": lower,
                    "observed_margin": observed,
                    "minimum_margin": margin,
                    "magnitude": margin - observed,
                }
            )

    for item in constraints.get("min_set_mass", []):
        views = [int(value) for value in item["views"]]
        threshold = float(item["min"])
        observed = sum(values[index] for index in views)
        if observed + tolerance < threshold:
            violations.append(
                {
                    "kind": "min_set_mass",
                    "views": views,
                    "observed": observed,
                    "minimum": threshold,
                    "magnitude": threshold - observed,
                }
            )

    for item in constraints.get("max_set_mass", []):
        views = [int(value) for value in item["views"]]
        threshold = float(item["max"])
        observed = sum(values[index] for index in views)
        if observed - tolerance > threshold:
            violations.append(
                {
                    "kind": "max_set_mass",
                    "views": views,
                    "observed": observed,
                    "maximum": threshold,
                    "magnitude": observed - threshold,
                }
            )

    total_violation = sum(float(item["magnitude"]) for item in violations)
    return {
        "passed": not violations,
        "violations": violations,
        "violation_count": len(violations),
        "total_violation": total_violation,
    }
