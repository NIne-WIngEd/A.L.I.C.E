from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from alice_conversation.final_evaluation import (
    build_expected_observation_fixture,
    run_conversation_final_evaluation,
)
from alice_conversation.final_evaluation_contract import (
    load_conversation_final_evaluation_benchmark,
    load_conversation_final_evaluation_policy,
)
from alice_conversation.release_audit import (
    Phase3ReleaseMetadata,
    audit_phase3_release,
    load_phase3_release_audit_policy,
)

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
ROLLBACK = "b" * 40


def evaluation_policy():
    return load_conversation_final_evaluation_policy(
        ROOT / "policies" / "conversation_final_evaluation_policy.json"
    )


def benchmark():
    policy = evaluation_policy()
    return load_conversation_final_evaluation_benchmark(
        ROOT / "benchmarks" / "phase3" / "conversation_final_evaluation_v1.json",
        policy=policy,
    )


def passing_report():
    policy = evaluation_policy()
    value = benchmark()
    return run_conversation_final_evaluation(
        submissions=build_expected_observation_fixture(value),
        benchmark=value,
        policy=policy,
    )


def failed_report():
    policy = evaluation_policy()
    value = benchmark()
    submissions = list(build_expected_observation_fixture(value))
    submissions[0] = replace(submissions[0], actual_outcome="rejected")
    return run_conversation_final_evaluation(
        submissions=tuple(submissions),
        benchmark=value,
        policy=policy,
    )


def release_policy():
    return load_phase3_release_audit_policy(
        ROOT / "policies" / "conversation_release_audit_policy.json"
    )


def metadata(**changes):
    policy = release_policy()
    values = {
        "repository_commit": COMMIT,
        "repository_head_commit": COMMIT,
        "repository_clean": True,
        "evaluated_at": "2026-07-26T22:00:00Z",
        "policy_versions": (
            evaluation_policy().policy_id,
            policy.policy_id,
        ),
        "package_version": "0.12.0",
        "evidence_manifest_id": policy.required_evidence_manifest_id,
        "evidence_manifest_digest": "c" * 64,
        "evidence_run_digest": "d" * 64,
        "evidence_target_count": 20,
        "evidence_passed_target_count": 20,
        "known_limitations": ("Synthetic benchmark only.",),
        "rollback_commit": ROLLBACK,
    }
    values.update(changes)
    return Phase3ReleaseMetadata(**values)


def passing_decision():
    return audit_phase3_release(
        passing_report(),
        metadata=metadata(),
        release_policy=release_policy(),
    )
