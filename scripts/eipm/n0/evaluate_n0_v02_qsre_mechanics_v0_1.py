from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


TOL = 1e-9


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL line {line_no}: {exc}") from exc
    if not rows:
        raise SystemExit("mechanics fixture file is empty")
    return rows


def close_vec(actual: list[float], expected: list[float], tol: float = 1e-8) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(a) - float(b)) <= tol for a, b in zip(actual, expected)
    )


def masked_sparsemax(logits: list[float], mask: list[bool]) -> list[float]:
    if len(logits) != len(mask):
        raise ValueError("logits/mask length mismatch")
    valid = [(idx, float(value)) for idx, (value, keep) in enumerate(zip(logits, mask)) if keep]
    if not valid:
        return [0.0] * len(logits)

    sorted_vals = sorted((value for _idx, value in valid), reverse=True)
    cumsum = 0.0
    k_star = 0
    tau_sum = 0.0
    for k, value in enumerate(sorted_vals, 1):
        cumsum += value
        if 1.0 + k * value > cumsum:
            k_star = k
            tau_sum = cumsum
    if k_star == 0:
        raise RuntimeError("sparsemax support search failed")

    tau = (tau_sum - 1.0) / k_star
    out = [0.0] * len(logits)
    for idx, value in valid:
        out[idx] = max(value - tau, 0.0)

    total = sum(out)
    if total <= 0.0:
        raise RuntimeError("sparsemax produced zero valid mass")
    out = [value / total for value in out]
    return out


def role_incidence(field_count: int, edges: list[list[int]], edge_mask: list[bool]) -> tuple[list[list[int]], list[list[int]]]:
    if len(edges) != len(edge_mask):
        raise ValueError("edge/mask length mismatch")
    source = [[0 for _ in edges] for _ in range(field_count)]
    target = [[0 for _ in edges] for _ in range(field_count)]
    for e, ((u, v), keep) in enumerate(zip(edges, edge_mask)):
        if not keep:
            continue
        if not (0 <= u < field_count and 0 <= v < field_count):
            raise ValueError("edge endpoint outside field range")
        source[u][e] = 1
        target[v][e] = 1
    return source, target


def support_local_mask(
    field_valid: list[bool],
    field_support: list[float],
    edges: list[list[int]],
    edge_support: list[float],
) -> list[bool]:
    if len(field_valid) != len(field_support):
        raise ValueError("field valid/support mismatch")
    if len(edges) != len(edge_support):
        raise ValueError("edge support mismatch")

    active = [bool(v) and float(s) > 0.0 for v, s in zip(field_valid, field_support)]
    for (u, v), weight in zip(edges, edge_support):
        if float(weight) <= 0.0:
            continue
        if not (0 <= u < len(active) and 0 <= v < len(active)):
            raise ValueError("edge endpoint outside field range")
        if field_valid[u]:
            active[u] = True
        if field_valid[v]:
            active[v] = True
    return active


def masked_softmax(logits: list[float], mask: list[bool]) -> list[float]:
    valid = [float(x) for x, keep in zip(logits, mask) if keep]
    if not valid:
        return [0.0] * len(logits)
    maximum = max(valid)
    exps = [math.exp(float(x) - maximum) if keep else 0.0 for x, keep in zip(logits, mask)]
    total = sum(exps)
    return [value / total for value in exps]


def control_state(applicability: float, has_support: bool) -> str:
    applicability = float(applicability)
    if applicability <= 0.25:
        return "FALLBACK"
    if applicability >= 0.75:
        return "RELATIONAL" if has_support else "FALLBACK"
    return "DEFER"


def execute_path(start: str, relations: list[str], edges: list[list[str]]) -> str | None:
    current = start
    for relation in relations:
        matches = [
            (src, dst, rel)
            for src, dst, rel in edges
            if src == current and rel == relation
        ]
        if len(matches) != 1:
            return None
        current = matches[0][1]
    return current


def evaluate_fixture(row: dict) -> dict:
    kind = row["kind"]

    if kind == "sparsemax":
        actual = masked_sparsemax(row["logits"], row["mask"])
        if not close_vec(actual, row["expected"]):
            raise AssertionError(f"{row['id']}: sparsemax actual={actual} expected={row['expected']}")
        active = sum(value > TOL for value in actual)
        return {"active_support": active, "distribution": actual}

    if kind == "incidence":
        source, target = role_incidence(
            int(row["field_count"]),
            row["edges"],
            row["edge_mask"],
        )
        actual_source = []
        actual_target = []
        for e in range(len(row["edges"])):
            actual_source.append([f for f in range(len(source)) if source[f][e] == 1])
            actual_target.append([f for f in range(len(target)) if target[f][e] == 1])
        if actual_source != row["expected_source_fields"]:
            raise AssertionError(f"{row['id']}: source incidence mismatch")
        if actual_target != row["expected_target_fields"]:
            raise AssertionError(f"{row['id']}: target incidence mismatch")
        if source == target:
            raise AssertionError(f"{row['id']}: SOURCE/TARGET incidence collapsed")
        return {"source_fields": actual_source, "target_fields": actual_target}

    if kind == "readout":
        mask = support_local_mask(
            row["field_valid"],
            row["field_support"],
            row["edges"],
            row["edge_support"],
        )
        distribution = masked_softmax(row["field_logits"], mask)
        for index in row["expected_zero_fields"]:
            if abs(distribution[index]) > TOL:
                raise AssertionError(f"{row['id']}: outside-support field {index} received mass")
        winner = max(range(len(distribution)), key=lambda i: distribution[i])
        if winner != int(row["expected_winner"]):
            raise AssertionError(f"{row['id']}: winner={winner} expected={row['expected_winner']}")
        return {"mask": mask, "distribution": distribution, "winner": winner}

    if kind == "control":
        actual = control_state(row["applicability"], bool(row["has_support"]))
        if actual != row["expected"]:
            raise AssertionError(f"{row['id']}: control={actual} expected={row['expected']}")
        return {"control_state": actual}

    if kind == "path":
        actual = execute_path(row["start"], row["relations"], row["edges"])
        if actual != row["expected"]:
            raise AssertionError(f"{row['id']}: path={actual} expected={row['expected']}")
        return {"result": actual}

    if kind == "dynamic_shape":
        field_count = int(row["field_count"])
        edge_count = int(row["edge_count"])
        edges = [[e % field_count, (e * 7 + 1) % field_count] for e in range(edge_count)]
        source, target = role_incidence(field_count, edges, [True] * edge_count)
        if len(source) != row["expected_field_count"] or len(source[0]) != row["expected_edge_count"]:
            raise AssertionError(f"{row['id']}: source shape drift")
        if len(target) != row["expected_field_count"] or len(target[0]) != row["expected_edge_count"]:
            raise AssertionError(f"{row['id']}: target shape drift")
        return {"field_count": field_count, "edge_count": edge_count}

    if kind == "support_cardinality":
        active = sum(float(value) > 0.0 for value in row["weights"])
        if active != int(row["expected_active"]):
            raise AssertionError(f"{row['id']}: active={active} expected={row['expected_active']}")
        return {"active_support": active}

    raise ValueError(f"unknown fixture kind {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--workload-contract", required=True)
    parser.add_argument("--mechanics-contract", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixtures = Path(args.fixtures).resolve()
    workload = json.loads(Path(args.workload_contract).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.mechanics_contract).read_text(encoding="utf-8"))
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    output = Path(args.output).resolve()

    if workload.get("schema") != "alice.eipm.full-workload-envelope.v0.1":
        raise SystemExit("workload envelope schema drift")
    if contract.get("schema") != "alice.eipm.n0.qsre-static-mechanics-contract.v0.1":
        raise SystemExit("mechanics contract schema drift")
    if state.get("schema") != "alice.eipm.n0.latent-pool-stage-state.v0.42":
        raise SystemExit("v0.42 state drift")

    if contract["mechanics"]["fixed_top_k"] is not False:
        raise SystemExit("permanent fixed top-k entered mechanics contract")
    if contract["mechanics"]["source_target_incidence_explicit"] is not True:
        raise SystemExit("explicit role incidence missing")
    if contract["mechanics"]["relational_readout_support_local"] is not True:
        raise SystemExit("support-local readout missing")
    if contract["mechanics"]["applicability_separate_from_support_selection"] is not True:
        raise SystemExit("applicability/support separation missing")

    for key in (
        "learned_qsre_parameters_authorized",
        "optimizer_authorized",
        "gradient_authorized",
        "gpu_required",
        "causal_test_opening_authorized",
        "frozen_challenge_opening_authorized",
        "private_identity_gradient",
    ):
        if contract[key] is not False:
            raise SystemExit(f"mechanics boundary drift: {key}")

    rows = read_jsonl(fixtures)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate fixture ids")

    counts = defaultdict(int)
    support_cardinalities = set()
    path_groups = defaultdict(list)
    permutation_groups = defaultdict(list)
    dynamic_shape = None

    results = {}
    for row in rows:
        result = evaluate_fixture(row)
        results[row["id"]] = result
        counts[row["kind"]] += 1
        if "active_support" in result:
            support_cardinalities.add(int(result["active_support"]))
        if row.get("group"):
            path_groups[row["group"]].append(result)
        if row.get("permutation_group"):
            permutation_groups[row["permutation_group"]].append(result)
        if row["kind"] == "dynamic_shape":
            dynamic_shape = result

    for group, items in path_groups.items():
        outputs = {item["result"] for item in items}
        if len(outputs) < 2:
            raise AssertionError(f"path contrast group {group!r} did not preserve relation order")

    for group, items in permutation_groups.items():
        outputs = {item["result"] for item in items}
        if len(outputs) != 1:
            raise AssertionError(f"permutation group {group!r} changed result")

    if not {0, 1, 2, 3}.issubset(support_cardinalities):
        raise AssertionError(
            f"adaptive support coverage missing; observed={sorted(support_cardinalities)}"
        )

    if dynamic_shape is None:
        raise AssertionError("dynamic large-shape fixture missing")
    if dynamic_shape["field_count"] <= 64 or dynamic_shape["edge_count"] <= 256:
        raise AssertionError("dynamic-shape fixture did not exceed historical operating hints")

    required_kinds = {
        "sparsemax",
        "incidence",
        "readout",
        "control",
        "path",
        "dynamic_shape",
        "support_cardinality",
    }
    if set(counts) != required_kinds:
        raise AssertionError(
            f"mechanics kind coverage mismatch actual={sorted(counts)} expected={sorted(required_kinds)}"
        )

    receipt = {
        "schema": "alice.eipm.n0.qsre-mechanics-result.v0.1",
        "status": "PASS_QSRE_CPU_NO_GRADIENT_REFERENCE_MECHANICS",
        "fixture_count": len(rows),
        "kind_counts": dict(sorted(counts.items())),
        "support_cardinalities_observed": sorted(support_cardinalities),
        "dynamic_shape": dynamic_shape,
        "ordered_path_contrast_pass": True,
        "permutation_path_pass": True,
        "source_target_incidence_distinct": True,
        "support_local_readout_pass": True,
        "parent_global_distractor_exclusion_mechanic_pass": True,
        "fallback_defer_states_pass": True,
        "full_workload_envelope_bound": True,
        "learned_qsre_parameters": False,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
        "causal_test_opened": False,
        "frozen_challenge_opened": False,
        "private_identity_gradient": False,
        "trainable_qsre_implementation_authorized": False,
        "next_action": "ratify mechanics result, then decide whether CPU/no-gradient tensorized QSRE source skeleton is justified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
