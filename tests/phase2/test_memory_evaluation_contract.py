"""P2.9a evaluation-contract validation tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from alice_memory.evaluation_contract import (
    EVALUATION_SUITES,
    REQUIRED_METRIC_BASELINES,
    MemoryEvaluationContractError,
    canonical_json,
    load_memory_evaluation_benchmark,
    load_memory_evaluation_policy,
    sha256_canonical,
)


def _write_json(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )
    return path


def _policy_data() -> dict:
    policy = load_memory_evaluation_policy()
    return json.loads(policy.source_path.read_text(encoding="utf-8"))


def _benchmark_data() -> dict:
    benchmark = load_memory_evaluation_benchmark()
    return json.loads(benchmark.source_path.read_text(encoding="utf-8"))


def test_default_policy_is_read_only_offline_and_private() -> None:
    policy = load_memory_evaluation_policy()

    assert policy.policy_id == "phase2-memory-evaluation-v1"
    assert policy.memory_write_allowed is False
    assert policy.external_action_allowed is False
    assert policy.tool_calling_allowed is False
    assert policy.web_access_allowed is False
    assert policy.private_output_only is True


def test_default_policy_preserves_governing_metric_baselines() -> None:
    policy = load_memory_evaluation_policy()
    by_id = {gate.metric_id: gate for gate in policy.metric_gates}

    assert set(by_id) == set(REQUIRED_METRIC_BASELINES)
    for metric_id, (
        direction,
        threshold,
        critical,
    ) in REQUIRED_METRIC_BASELINES.items():
        gate = by_id[metric_id]
        assert gate.direction == direction
        assert gate.threshold == threshold
        if critical:
            assert gate.critical is True


def test_default_benchmark_covers_every_required_suite() -> None:
    policy = load_memory_evaluation_policy()
    benchmark = load_memory_evaluation_benchmark(policy=policy)

    assert benchmark.synthetic_only is True
    assert len(benchmark.cases) == 13
    assert {case.suite for case in benchmark.cases} == set(
        EVALUATION_SUITES
    )
    assert len({case.case_id for case in benchmark.cases}) == len(
        benchmark.cases
    )


def test_default_benchmark_encodes_critical_boundaries() -> None:
    benchmark = load_memory_evaluation_benchmark()
    by_id = {case.case_id: case for case in benchmark.cases}

    assert by_id["permission-denial-001"].expected_outcome == "denied"
    assert by_id["sensitivity-denial-001"].forbidden_memory_ids == (
        "eval-sensitive",
    )
    assert by_id["deletion-absence-001"].critical is True
    assert by_id["candidate-boundary-001"].forbidden_candidate_ids == (
        "eval-candidate-unpromoted",
    )
    assert by_id["prompt-injection-001"].critical is True


def test_contract_digests_are_canonical_and_stable() -> None:
    first = {
        "z": 1,
        "a": {
            "value": "synthetic",
        },
    }
    second = {
        "a": {
            "value": "synthetic",
        },
        "z": 1,
    }

    assert canonical_json(first) == canonical_json(second)
    assert sha256_canonical(first) == sha256_canonical(second)
    assert len(load_memory_evaluation_policy().digest) == 64
    assert len(load_memory_evaluation_benchmark().digest) == 64


def test_policy_rejects_unsupported_schema(tmp_path: Path) -> None:
    value = _policy_data()
    value["memory_evaluation_policy_schema_version"] = 99

    with pytest.raises(
        MemoryEvaluationContractError,
        match="Unsupported memory-evaluation policy schema",
    ):
        load_memory_evaluation_policy(
            _write_json(tmp_path, "policy.json", value)
        )


def test_policy_rejects_action_capability(tmp_path: Path) -> None:
    value = _policy_data()
    value["tool_calling_allowed"] = True

    with pytest.raises(
        MemoryEvaluationContractError,
        match="read-only and offline",
    ):
        load_memory_evaluation_policy(
            _write_json(tmp_path, "policy.json", value)
        )


def test_policy_rejects_weakened_metric(tmp_path: Path) -> None:
    value = _policy_data()
    for metric in value["metric_gates"]:
        if metric["metric_id"] == "personal_source_attribution_rate":
            metric["threshold"] = 0.90

    with pytest.raises(
        MemoryEvaluationContractError,
        match="weakens the governing release gate",
    ):
        load_memory_evaluation_policy(
            _write_json(tmp_path, "policy.json", value)
        )


def test_policy_rejects_missing_required_suite(tmp_path: Path) -> None:
    value = _policy_data()
    value["required_suites"].remove("deletion")

    with pytest.raises(
        MemoryEvaluationContractError,
        match="missing required suites",
    ):
        load_memory_evaluation_policy(
            _write_json(tmp_path, "policy.json", value)
        )


def test_policy_rejects_duplicate_metric_id(tmp_path: Path) -> None:
    value = _policy_data()
    value["metric_gates"].append(
        deepcopy(value["metric_gates"][0])
    )

    with pytest.raises(
        MemoryEvaluationContractError,
        match="duplicate metric IDs",
    ):
        load_memory_evaluation_policy(
            _write_json(tmp_path, "policy.json", value)
        )


def test_benchmark_rejects_non_synthetic_content(tmp_path: Path) -> None:
    value = _benchmark_data()
    value["synthetic_only"] = False

    with pytest.raises(
        MemoryEvaluationContractError,
        match="synthetic-only",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )


def test_benchmark_rejects_duplicate_case_id(tmp_path: Path) -> None:
    value = _benchmark_data()
    duplicate = deepcopy(value["cases"][0])
    value["cases"].append(duplicate)

    with pytest.raises(
        MemoryEvaluationContractError,
        match="duplicate case IDs",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )


def test_answerable_case_requires_memory_and_source(tmp_path: Path) -> None:
    value = _benchmark_data()
    value["cases"][0]["expected_source_refs"] = []

    with pytest.raises(
        MemoryEvaluationContractError,
        match="requires expected memories and sources",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )


def test_conflict_case_requires_multiple_memories(tmp_path: Path) -> None:
    value = _benchmark_data()
    conflict = next(
        case for case in value["cases"] if case["suite"] == "conflict"
    )
    conflict["expected_memory_ids"] = ["eval-conflict-a"]

    with pytest.raises(
        MemoryEvaluationContractError,
        match="requires at least two memories",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )


def test_temporal_case_requires_timestamp(tmp_path: Path) -> None:
    value = _benchmark_data()
    temporal = next(
        case for case in value["cases"] if case["suite"] == "temporal"
    )
    temporal["at"] = None

    with pytest.raises(
        MemoryEvaluationContractError,
        match="requires an at timestamp",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )


def test_case_cannot_expect_and_forbid_same_memory(
    tmp_path: Path,
) -> None:
    value = _benchmark_data()
    value["cases"][0]["forbidden_memory_ids"] = [
        value["cases"][0]["expected_memory_ids"][0]
    ]

    with pytest.raises(
        MemoryEvaluationContractError,
        match="both expects and forbids",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )


def test_synthetic_source_ref_cannot_be_absolute_path(
    tmp_path: Path,
) -> None:
    value = _benchmark_data()
    value["cases"][0]["expected_source_refs"] = [
        r"C:\private\source.txt"
    ]

    with pytest.raises(
        MemoryEvaluationContractError,
        match="cannot be absolute paths",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )


def test_case_cannot_authorize_ordinary_sensitive_retrieval(
    tmp_path: Path,
) -> None:
    value = _benchmark_data()
    value["cases"][0]["max_classification"] = "HIGHLY_SENSITIVE"

    with pytest.raises(
        MemoryEvaluationContractError,
        match="ordinary evaluation boundary",
    ):
        load_memory_evaluation_benchmark(
            _write_json(tmp_path, "benchmark.json", value)
        )
