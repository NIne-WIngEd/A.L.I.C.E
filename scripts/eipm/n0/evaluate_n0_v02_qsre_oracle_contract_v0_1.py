from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TOL = 1e-9


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL at line {line_no}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise SystemExit("oracle contract is empty")
    return rows


def normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in dist.values())
    if total <= 0:
        raise ValueError("cannot normalize zero-mass distribution")
    return {k: float(v) / total for k, v in dist.items() if float(v) > 0.0}


def parent_fallback(row: dict) -> dict:
    fields = row["fields"]
    if not fields:
        return {"mode": "defer"}
    best = max(float(f["parent_score"]) for f in fields)
    winners = [f["id"] for f in fields if abs(float(f["parent_score"]) - best) <= TOL]
    if len(winners) != 1:
        return {"mode": "defer"}
    return {"mode": "fallback", "field_distribution": {winners[0]: 1.0}}


def execute_path(row: dict, supported_edges: dict[str, dict]) -> dict:
    query = row["query"]
    current = query.get("start_field")
    relations = query.get("path_relations") or []
    if current is None or not relations:
        raise ValueError("path execution requires start_field and path_relations")

    for relation in relations:
        candidates = [
            edge
            for edge in supported_edges.values()
            if edge["source"] == current and edge["relation"] == relation
        ]
        if len(candidates) != 1:
            return {"mode": "defer"}
        current = candidates[0]["target"]

    return {"mode": "relational", "field_distribution": {current: 1.0}}


def execute(row: dict) -> dict:
    query = row["query"]
    applicability = float(query.get("applicability", 0.0))

    if 0.25 < applicability < 0.75:
        return {"mode": "defer"}

    support_items = row.get("oracle_support", [])
    if applicability <= 0.25 or not support_items:
        return parent_fallback(row)

    edges = {edge["id"]: edge for edge in row.get("edges", [])}
    supported_edges: dict[str, dict] = {}
    support_weight: dict[str, float] = {}
    for item in support_items:
        edge_id = item["edge_id"]
        if edge_id not in edges:
            raise ValueError(f"oracle support references unknown edge {edge_id!r}")
        supported_edges[edge_id] = edges[edge_id]
        support_weight[edge_id] = float(item["weight"])

    if query.get("path_relations"):
        return execute_path(row, supported_edges)

    relation_weight = defaultdict(float)
    for item in query.get("relation_hypotheses", []):
        relation_weight[item["relation"]] += float(item["weight"])

    role_weight = defaultdict(float)
    for item in query.get("role_hypotheses", []):
        role = str(item["role"]).upper()
        if role not in {"SOURCE", "TARGET"}:
            raise ValueError(f"unsupported role {role!r}")
        role_weight[role] += float(item["weight"])

    mass = defaultdict(float)
    for edge_id, edge in supported_edges.items():
        rw = relation_weight[edge["relation"]]
        if rw <= 0.0:
            continue
        sw = support_weight[edge_id]
        mass[edge["source"]] += sw * rw * role_weight["SOURCE"]
        mass[edge["target"]] += sw * rw * role_weight["TARGET"]

    if not mass or sum(mass.values()) <= 0.0:
        return {"mode": "defer"}

    return {"mode": "relational", "field_distribution": normalize(dict(mass))}


def dist_close(actual: dict[str, float], expected: dict[str, float]) -> bool:
    keys = set(actual) | set(expected)
    return all(abs(float(actual.get(k, 0.0)) - float(expected.get(k, 0.0))) <= TOL for k in keys)


def distribution_key(result: dict) -> tuple:
    if "field_distribution" not in result:
        return (result["mode"],)
    return (
        result["mode"],
        tuple(sorted((k, round(float(v), 12)) for k, v in result["field_distribution"].items())),
    )


def validate_row(row: dict) -> dict:
    for key in ("id", "diagnostic", "fields", "edges", "query", "oracle_support", "expected"):
        if key not in row:
            raise ValueError(f"{row.get('id', '<unknown>')}: missing {key}")

    field_ids = [f["id"] for f in row["fields"]]
    if len(field_ids) != len(set(field_ids)):
        raise ValueError(f"{row['id']}: duplicate field id")

    edge_ids = [e["id"] for e in row["edges"]]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError(f"{row['id']}: duplicate edge id")

    field_set = set(field_ids)
    for edge in row["edges"]:
        if edge["source"] not in field_set or edge["target"] not in field_set:
            raise ValueError(f"{row['id']}: edge endpoint not present in fields")

    result = execute(row)
    expected = row["expected"]

    if result["mode"] != expected["mode"]:
        raise AssertionError(
            f"{row['id']}: mode mismatch actual={result['mode']} expected={expected['mode']}"
        )

    if "field_distribution" in expected:
        actual_dist = result.get("field_distribution", {})
        expected_dist = expected["field_distribution"]
        if not dist_close(actual_dist, expected_dist):
            raise AssertionError(
                f"{row['id']}: distribution mismatch actual={actual_dist} expected={expected_dist}"
            )
        if abs(sum(actual_dist.values()) - 1.0) > TOL:
            raise AssertionError(f"{row['id']}: distribution does not sum to 1")

    if result["mode"] == "relational":
        edges = {e["id"]: e for e in row["edges"]}
        supported_fields = set()
        for item in row["oracle_support"]:
            edge = edges[item["edge_id"]]
            supported_fields.add(edge["source"])
            supported_fields.add(edge["target"])
        outside = set(result.get("field_distribution", {})) - supported_fields
        if outside:
            raise AssertionError(
                f"{row['id']}: global competition leak; output outside support {sorted(outside)}"
            )

    if row.get("parent_conflict_expected"):
        parent = parent_fallback(row)
        if parent["mode"] != "fallback":
            raise AssertionError(f"{row['id']}: expected a unique conflicting parent winner")
        if distribution_key(parent) == distribution_key(result):
            raise AssertionError(
                f"{row['id']}: relational result did not isolate from conflicting parent preference"
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixtures = Path(args.fixtures).resolve()
    output = Path(args.output).resolve()
    rows = read_jsonl(fixtures)

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate fixture ids")

    diagnostics = defaultdict(int)
    contrast_groups = defaultdict(list)
    permutation_groups = defaultdict(list)
    support_cardinalities = set()
    ambiguity_seen = False
    path_seen = False
    defer_seen = False
    fallback_seen = False
    parent_conflict_seen = False

    results = {}
    for row in rows:
        result = validate_row(row)
        results[row["id"]] = result
        diagnostics[row["diagnostic"]] += 1
        support_cardinalities.add(len(row["oracle_support"]))
        if len(result.get("field_distribution", {})) > 1:
            ambiguity_seen = True
        if row["query"].get("path_relations"):
            path_seen = True
        if result["mode"] == "defer":
            defer_seen = True
        if result["mode"] == "fallback":
            fallback_seen = True
        if row.get("parent_conflict_expected"):
            parent_conflict_seen = True
        if row.get("contrast_group"):
            contrast_groups[row["contrast_group"]].append(result)
        if row.get("permutation_group"):
            permutation_groups[row["permutation_group"]].append(result)

    expected_diags = {"D1", "D2", "D3", "D4", "D5"}
    if set(diagnostics) != expected_diags:
        raise AssertionError(
            f"diagnostic coverage mismatch actual={sorted(diagnostics)} expected={sorted(expected_diags)}"
        )

    for group, group_results in contrast_groups.items():
        keys = {distribution_key(r) for r in group_results}
        if len(keys) < 2:
            raise AssertionError(f"contrast group {group!r} failed to produce distinct outputs")

    for group, group_results in permutation_groups.items():
        keys = {distribution_key(r) for r in group_results}
        if len(keys) != 1:
            raise AssertionError(f"permutation group {group!r} changed semantic output")

    if 0 not in support_cardinalities or 1 not in support_cardinalities:
        raise AssertionError("must cover zero-support and one-support cases")
    if not any(value >= 2 for value in support_cardinalities):
        raise AssertionError("must cover multi-support cases")
    if not ambiguity_seen:
        raise AssertionError("must cover plural/ambiguous output")
    if not path_seen:
        raise AssertionError("must cover ordered path execution")
    if not defer_seen:
        raise AssertionError("must cover explicit defer state")
    if not fallback_seen:
        raise AssertionError("must cover general-parent fallback")
    if not parent_conflict_seen:
        raise AssertionError("must cover relational result against conflicting parent preference")

    receipt = {
        "schema": "alice.eipm.n0.qsre-oracle-contract-result.v0.1",
        "status": "PASS_QSRE_DETERMINISTIC_PREIMPLEMENTATION_CONTRACT",
        "fixture_count": len(rows),
        "diagnostic_counts": dict(sorted(diagnostics.items())),
        "support_cardinalities_observed": sorted(support_cardinalities),
        "ambiguity_observed": ambiguity_seen,
        "ordered_path_observed": path_seen,
        "explicit_defer_observed": defer_seen,
        "parent_fallback_observed": fallback_seen,
        "parent_conflict_isolation_observed": parent_conflict_seen,
        "contrast_groups": sorted(contrast_groups),
        "permutation_groups": sorted(permutation_groups),
        "new_learned_parameters": False,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
        "existing_model_weights_mutated": False,
        "causal_test_opened": False,
        "frozen_challenge_opened": False,
        "private_identity_gradient": False,
        "trainable_qsre_implementation_authorized": False,
        "next_action": "interpret deterministic contract and make a separate trainable-architecture decision",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
