from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from alice_information.final_evaluation import build_expected_observation_fixture
from alice_information.final_evaluation_contract import (
    information_evaluation_observation_digest,
    load_information_final_evaluation_benchmark,
    load_information_final_evaluation_policy,
)

ROOT = Path(__file__).resolve().parents[2]


def policy():
    return load_information_final_evaluation_policy(
        ROOT / "policies/information_final_evaluation_policy.json"
    )


def benchmark():
    return load_information_final_evaluation_benchmark(
        ROOT / "benchmarks/phase4/information_final_evaluation_v1.json",
        policy=policy(),
    )


def passing_submissions():
    return build_expected_observation_fixture(benchmark())


def replace_submission(items, index: int, **changes):
    values = list(items)
    updated = replace(values[index], **changes)
    if "observation_digest" not in changes:
        updated = replace(
            updated,
            observation_digest=information_evaluation_observation_digest(
                case_id=updated.case_id,
                actual_outcome=updated.actual_outcome,
                signals=updated.signals,
                violation_codes=updated.violation_codes,
            ),
        )
    values[index] = updated
    return tuple(values)
