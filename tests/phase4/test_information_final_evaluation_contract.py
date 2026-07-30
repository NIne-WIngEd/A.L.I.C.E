from __future__ import annotations

import json

import pytest

from alice_information.final_evaluation_contract import (
    EVALUATION_SUITES,
    REQUIRED_METRIC_BASELINES,
    InformationFinalEvaluationContractError,
    information_evaluation_observation_digest,
    load_information_evaluation_submissions,
    load_information_final_evaluation_benchmark,
    load_information_final_evaluation_policy,
    parse_information_evaluation_submission,
)

from _information_final_evaluation_helpers import ROOT, benchmark, policy


def test_policy_and_benchmark_load() -> None:
    selected_policy = policy()
    selected_benchmark = benchmark()
    assert set(selected_policy.required_suites) == set(EVALUATION_SUITES)
    assert len(selected_benchmark.cases) == 24
    assert selected_policy.runtime_backed_release_required is True
    assert selected_policy.external_submission_bundle_allowed is False
    assert selected_policy.minimum_collected_test_count == 640


def test_policy_has_exact_metric_coverage() -> None:
    assert {item.metric_id for item in policy().metric_gates} == set(
        REQUIRED_METRIC_BASELINES
    )


def test_benchmark_is_synthetic_only_and_two_cases_per_suite() -> None:
    selected = benchmark()
    assert selected.synthetic_only is True
    counts = {suite: 0 for suite in EVALUATION_SUITES}
    for item in selected.cases:
        counts[item.suite] += 1
    assert set(counts.values()) == {2}


def test_weakened_policy_is_rejected(tmp_path) -> None:
    raw = json.loads(
        (ROOT / "policies/information_final_evaluation_policy.json").read_text()
    )
    raw["metric_gates"][0]["threshold"] = 0.99
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(InformationFinalEvaluationContractError):
        load_information_final_evaluation_policy(path)


@pytest.mark.parametrize(
    "field",
    [
        "live_network_allowed",
        "real_private_query_allowed",
        "raw_query_text_allowed",
        "raw_source_content_allowed",
        "source_body_persistence_allowed",
        "memory_write_allowed",
        "external_action_allowed",
        "repository_write_allowed",
        "background_execution_allowed",
    ],
)
def test_enabled_prohibited_boundary_is_rejected(tmp_path, field: str) -> None:
    raw = json.loads(
        (ROOT / "policies/information_final_evaluation_policy.json").read_text()
    )
    raw["boundaries"][field] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(InformationFinalEvaluationContractError):
        load_information_final_evaluation_policy(path)




@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_backed_release_required", False),
        ("external_submission_bundle_allowed", True),
        ("pinned_test_manifest_required", False),
        ("network_guard_required", False),
        ("repository_snapshot_stability_required", False),
        ("minimum_collected_test_count", 639),
    ],
)
def test_weakened_runtime_evidence_policy_is_rejected(
    tmp_path, field: str, value: object
) -> None:
    raw = json.loads(
        (ROOT / "policies/information_final_evaluation_policy.json").read_text()
    )
    raw["runtime_evidence"][field] = value
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(InformationFinalEvaluationContractError):
        load_information_final_evaluation_policy(path)


def test_duplicate_json_key_is_rejected(tmp_path) -> None:
    source = ROOT / "policies/information_final_evaluation_policy.json"
    text = source.read_text()
    path = tmp_path / "policy.json"
    path.write_text(text.replace('"phase": "4",', '"phase": "4",\n  "phase": "4",'))
    with pytest.raises(InformationFinalEvaluationContractError):
        load_information_final_evaluation_policy(path)


def test_missing_suite_case_is_rejected(tmp_path) -> None:
    raw = json.loads(
        (ROOT / "benchmarks/phase4/information_final_evaluation_v1.json").read_text()
    )
    raw["cases"] = [item for item in raw["cases"] if item["suite"] != "injection"]
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(InformationFinalEvaluationContractError):
        load_information_final_evaluation_benchmark(path, policy=policy())


def test_unknown_submission_content_field_is_rejected() -> None:
    value = {
        "case_id": "p48-case-001",
        "actual_outcome": "completed",
        "signals": [],
        "violation_codes": [],
        "observation_digest": "0" * 64,
        "raw_source": "secret",
    }
    with pytest.raises(InformationFinalEvaluationContractError):
        parse_information_evaluation_submission(value)


def test_invalid_observation_digest_is_rejected() -> None:
    value = {
        "case_id": "p48-case-001",
        "actual_outcome": "completed",
        "signals": [],
        "violation_codes": [],
        "observation_digest": "bad",
    }
    with pytest.raises(InformationFinalEvaluationContractError):
        parse_information_evaluation_submission(value)


def test_submission_bundle_must_match_benchmark(tmp_path) -> None:
    value = {
        "information_final_evaluation_submission_schema_version": 1,
        "benchmark_id": "other-benchmark",
        "test_set_version": "p4.8-v1",
        "submissions": [],
    }
    path = tmp_path / "submissions.json"
    path.write_text(json.dumps(value))
    with pytest.raises(InformationFinalEvaluationContractError):
        load_information_evaluation_submissions(path, benchmark=benchmark())


def test_observation_digest_must_bind_submission() -> None:
    value = {
        "case_id": "p48-case-001",
        "actual_outcome": "completed",
        "signals": ["all_good"],
        "violation_codes": [],
        "observation_digest": "0" * 64,
    }
    with pytest.raises(InformationFinalEvaluationContractError):
        parse_information_evaluation_submission(value)


def test_canonical_observation_digest_is_accepted() -> None:
    digest = information_evaluation_observation_digest(
        case_id="p48-case-001",
        actual_outcome="completed",
        signals=("all_good",),
        violation_codes=(),
    )
    value = {
        "case_id": "p48-case-001",
        "actual_outcome": "completed",
        "signals": ["all_good"],
        "violation_codes": [],
        "observation_digest": digest,
    }
    assert parse_information_evaluation_submission(value).observation_digest == digest


def test_substituted_benchmark_is_rejected(tmp_path) -> None:
    raw = json.loads(
        (ROOT / "benchmarks/phase4/information_final_evaluation_v1.json").read_text()
    )
    for item in raw["cases"]:
        item["expected_outcome"] = "completed"
        item["required_signals"] = ["all_good"]
        item["forbidden_signals"] = []
        item["critical"] = False
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(InformationFinalEvaluationContractError):
        load_information_final_evaluation_benchmark(path, policy=policy())
