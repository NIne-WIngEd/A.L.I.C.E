from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alice_conversation.release_audit import (
    Phase3ReleaseAuditError,
    audit_phase3_release,
    load_phase3_release_audit_policy,
    load_phase3_release_record,
    phase3_release_record_json,
    verify_phase3_release_decision,
    write_phase3_release_record,
)
from _release_audit_helpers import (
    COMMIT,
    ROOT,
    failed_report,
    metadata,
    passing_decision,
    passing_report,
    release_policy,
)


def test_release_policy_loads_with_required_boundaries():
    policy = release_policy()
    assert policy.policy_id == "phase3-conversation-release-audit-v1"
    assert policy.private_output_only is True
    assert policy.repository_output_allowed is False
    assert policy.required_evidence_manifest_id == "phase3-conversation-release-evidence-v1"


def test_weakened_release_policy_is_rejected(tmp_path):
    value = json.loads((ROOT / "policies/conversation_release_audit_policy.json").read_text())
    value["boundaries"]["web_access_allowed"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase3ReleaseAuditError, match="weakened"):
        load_phase3_release_audit_policy(path)


def test_passing_evaluation_is_approved():
    decision = passing_decision()
    assert decision.approved is True
    assert decision.decision_reasons == ()
    verify_phase3_release_decision(decision)


def test_failed_evaluation_creates_rejected_decision():
    decision = audit_phase3_release(
        failed_report(), metadata=metadata(), release_policy=release_policy()
    )
    assert decision.approved is False
    assert "final_conversation_evaluation_failed" in decision.decision_reasons
    assert "critical_case_failure" in decision.decision_reasons


def test_invalid_repository_commit_is_rejected():
    with pytest.raises(Phase3ReleaseAuditError, match="40-character"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(repository_commit="abc"),
            release_policy=release_policy(),
        )


def test_release_time_requires_utc():
    with pytest.raises(Phase3ReleaseAuditError, match="UTC"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(evaluated_at="2026-07-26T18:00:00-04:00"),
            release_policy=release_policy(),
        )


def test_repository_head_must_match_release_commit():
    with pytest.raises(Phase3ReleaseAuditError, match="exactly match"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(repository_head_commit="c" * 40),
            release_policy=release_policy(),
        )


def test_repository_must_be_clean():
    with pytest.raises(Phase3ReleaseAuditError, match="clean"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(repository_clean=False),
            release_policy=release_policy(),
        )


def test_rollback_commit_is_required():
    with pytest.raises(Phase3ReleaseAuditError, match="rollback"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(rollback_commit=None),
            release_policy=release_policy(),
        )


def test_rollback_commit_must_differ():
    with pytest.raises(Phase3ReleaseAuditError, match="differ"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(rollback_commit=COMMIT),
            release_policy=release_policy(),
        )


def test_package_version_must_match_policy():
    with pytest.raises(Phase3ReleaseAuditError, match="package version"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(package_version="0.11.0"),
            release_policy=release_policy(),
        )


def test_policy_versions_are_required():
    with pytest.raises(Phase3ReleaseAuditError, match="policy"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(policy_versions=()),
            release_policy=release_policy(),
        )


def test_release_record_writes_and_loads_under_private_root(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    decision = passing_decision()
    output = vault / "reports" / "phase3-release.json"
    written = write_phase3_release_record(
        decision, output, private_root=vault, repository_root=repository
    )
    assert load_phase3_release_record(written) == decision


def test_release_record_refuses_repository_output(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    decision = passing_decision()
    with pytest.raises(Phase3ReleaseAuditError, match="repository"):
        write_phase3_release_record(
            decision,
            repository / "release.json",
            private_root=tmp_path,
            repository_root=repository,
        )


def test_release_record_refuses_output_outside_private_root(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    with pytest.raises(Phase3ReleaseAuditError, match="private"):
        write_phase3_release_record(
            passing_decision(),
            tmp_path / "outside.json",
            private_root=vault,
            repository_root=repository,
        )


def test_release_record_requires_json_suffix(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    with pytest.raises(Phase3ReleaseAuditError, match=".json"):
        write_phase3_release_record(
            passing_decision(),
            vault / "release.txt",
            private_root=vault,
            repository_root=repository,
        )


def test_release_record_write_is_idempotent(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    output = vault / "release.json"
    decision = passing_decision()
    first = write_phase3_release_record(
        decision, output, private_root=vault, repository_root=repository
    )
    second = write_phase3_release_record(
        decision, output, private_root=vault, repository_root=repository
    )
    assert first == second


def test_release_record_refuses_different_overwrite(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    output = vault / "release.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Phase3ReleaseAuditError, match="overwrite"):
        write_phase3_release_record(
            passing_decision(), output, private_root=vault, repository_root=repository
        )


def test_tampered_release_record_fails_digest_check(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    output = vault / "release.json"
    write_phase3_release_record(
        passing_decision(), output, private_root=vault, repository_root=repository
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["approved"] = False
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase3ReleaseAuditError, match="digest"):
        load_phase3_release_record(output)



def test_evidence_manifest_must_match_policy():
    with pytest.raises(Phase3ReleaseAuditError, match="evidence manifest"):
        audit_phase3_release(
            passing_report(),
            metadata=metadata(evidence_manifest_id="wrong-manifest"),
            release_policy=release_policy(),
        )


def test_incomplete_evidence_rejects_release():
    decision = audit_phase3_release(
        passing_report(),
        metadata=metadata(evidence_passed_target_count=19),
        release_policy=release_policy(),
    )
    assert decision.approved is False
    assert "release_evidence_test_failed" in decision.decision_reasons


def test_release_record_json_is_stable():
    decision = passing_decision()
    assert phase3_release_record_json(decision) == phase3_release_record_json(decision)
    assert decision.release_id in phase3_release_record_json(decision)
