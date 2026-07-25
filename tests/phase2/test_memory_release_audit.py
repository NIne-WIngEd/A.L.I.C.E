"""P2.9d final Phase 2 release audit and private record gates."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from alice_memory.cited_answer import MemoryAnswerAuthorization
from alice_memory.evaluation_contract import (
    load_memory_evaluation_benchmark,
    load_memory_evaluation_policy,
)
from alice_memory.evaluation_fixtures import build_memory_evaluation_fixture
from alice_memory.final_evaluation import (
    memory_core_final_report_digest,
    run_memory_core_final_evaluation,
)
from alice_memory.release_audit import (
    Phase2ReleaseAuditError,
    Phase2ReleaseMetadata,
    audit_phase2_release,
    load_phase2_release_record,
    phase2_release_record_json,
    verify_phase2_release_decision,
    write_phase2_release_record,
)
from alice_memory.sensitive_crypto import InMemoryTestKeyProtector
from alice_memory.store import open_memory_store


@contextmanager
def _report(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    benchmark = load_memory_evaluation_benchmark()
    policy = load_memory_evaluation_policy()
    with open_memory_store(vault, repository_root=repository) as connection:
        fixture = build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
            benchmark=benchmark,
            key_protector=InMemoryTestKeyProtector(),
        )
        report = run_memory_core_final_evaluation(
            connection,
            fixture=fixture,
            benchmark=benchmark,
            policy=policy,
            authorization=MemoryAnswerAuthorization(
                actor="p2.9d-auditor",
                allowed=True,
                purpose="offline Phase 2 release audit",
                max_classification="PRIVATE",
            ),
        )
        yield repository, vault, policy, report


def _metadata(policy_id: str, **changes):
    values = {
        "repository_commit": "abcdef1234567890",
        "evaluated_at": "2026-07-25T21:00:00Z",
        "policy_versions": (policy_id,),
        "known_limitations": ("Synthetic benchmark only.",),
        "rollback_commit": "1234567abcdef890",
    }
    values.update(changes)
    return Phase2ReleaseMetadata(**values)


def test_passing_final_report_is_approved(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
    assert decision.approved is True
    assert decision.decision_reasons == ()
    assert decision.evaluation_report_digest == report.report_digest
    verify_phase2_release_decision(decision)


def test_failed_final_report_is_rejected(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        bad = replace(report, passed=False, report_digest="")
        bad = replace(bad, report_digest=memory_core_final_report_digest(bad))
        with pytest.raises(Exception, match="inconsistent"):
            audit_phase2_release(bad, metadata=_metadata(policy.policy_id))


def test_release_metadata_rejects_invalid_commit(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        with pytest.raises(Phase2ReleaseAuditError, match="commit"):
            audit_phase2_release(
                report,
                metadata=_metadata(policy.policy_id, repository_commit="not-a-commit"),
            )


def test_release_metadata_requires_utc_time(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        with pytest.raises(Phase2ReleaseAuditError, match="UTC"):
            audit_phase2_release(
                report,
                metadata=_metadata(policy.policy_id, evaluated_at="2026-07-25T17:00:00-04:00"),
            )


def test_release_audit_rejects_model_dependency(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        with pytest.raises(Phase2ReleaseAuditError, match="model"):
            audit_phase2_release(
                report,
                metadata=_metadata(policy.policy_id, model="qwen"),
            )


def test_release_audit_rejects_tool_dependency(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        with pytest.raises(Phase2ReleaseAuditError, match="tool"):
            audit_phase2_release(
                report,
                metadata=_metadata(policy.policy_id, tool_versions=("web:1",)),
            )


def test_release_audit_requires_policy_versions(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        with pytest.raises(Phase2ReleaseAuditError, match="policy"):
            audit_phase2_release(
                report,
                metadata=_metadata(policy.policy_id, policy_versions=()),
            )


def test_rollback_commit_must_differ(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        with pytest.raises(Phase2ReleaseAuditError, match="differ"):
            audit_phase2_release(
                report,
                metadata=_metadata(
                    policy.policy_id,
                    repository_commit="abcdef1234567890",
                    rollback_commit="abcdef1234567890",
                ),
            )


def test_release_record_writes_under_private_root(tmp_path: Path) -> None:
    with _report(tmp_path) as (repository, vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
        output = vault / "reports" / "phase2-release.json"
        written = write_phase2_release_record(
            decision,
            output,
            private_root=vault,
            repository_root=repository,
        )
        loaded = load_phase2_release_record(written)
    assert loaded == decision


def test_release_record_refuses_repository_output(tmp_path: Path) -> None:
    with _report(tmp_path) as (repository, vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
        with pytest.raises(Phase2ReleaseAuditError, match="repository"):
            write_phase2_release_record(
                decision,
                repository / "phase2-release.json",
                private_root=tmp_path,
                repository_root=repository,
            )


def test_release_record_refuses_output_outside_private_root(tmp_path: Path) -> None:
    with _report(tmp_path) as (repository, vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
        with pytest.raises(Phase2ReleaseAuditError, match="private"):
            write_phase2_release_record(
                decision,
                tmp_path / "outside.json",
                private_root=vault,
                repository_root=repository,
            )


def test_release_record_requires_json_suffix(tmp_path: Path) -> None:
    with _report(tmp_path) as (repository, vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
        with pytest.raises(Phase2ReleaseAuditError, match=".json"):
            write_phase2_release_record(
                decision,
                vault / "release.txt",
                private_root=vault,
                repository_root=repository,
            )


def test_release_record_write_is_idempotent(tmp_path: Path) -> None:
    with _report(tmp_path) as (repository, vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
        output = vault / "release.json"
        first = write_phase2_release_record(
            decision,
            output,
            private_root=vault,
            repository_root=repository,
        )
        second = write_phase2_release_record(
            decision,
            output,
            private_root=vault,
            repository_root=repository,
        )
    assert first == second


def test_release_record_refuses_different_overwrite(tmp_path: Path) -> None:
    with _report(tmp_path) as (repository, vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
        output = vault / "release.json"
        output.write_text("{}\n", encoding="utf-8")
        with pytest.raises(Phase2ReleaseAuditError, match="overwrite"):
            write_phase2_release_record(
                decision,
                output,
                private_root=vault,
                repository_root=repository,
            )


def test_tampered_release_record_fails_digest_check(tmp_path: Path) -> None:
    with _report(tmp_path) as (repository, vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
        output = vault / "release.json"
        write_phase2_release_record(
            decision,
            output,
            private_root=vault,
            repository_root=repository,
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        value["approved"] = False
        output.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(Phase2ReleaseAuditError, match="digest"):
            load_phase2_release_record(output)


def test_release_record_json_is_stable(tmp_path: Path) -> None:
    with _report(tmp_path) as (_repository, _vault, policy, report):
        decision = audit_phase2_release(report, metadata=_metadata(policy.policy_id))
    assert phase2_release_record_json(decision) == phase2_release_record_json(decision)
    assert decision.release_id in phase2_release_record_json(decision)
