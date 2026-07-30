from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alice_information.final_evaluation_runtime import (
    information_runtime_backed_report_digest,
    information_runtime_evidence_digest,
)
from alice_information.release_audit import (
    Phase4ReleaseAuditError,
    audit_phase4_release,
    load_phase4_release_audit_policy,
    load_phase4_release_record,
    phase4_release_record_json,
    verify_phase4_release_decision,
    write_phase4_release_record,
)
from _information_release_audit_helpers import (
    COMMIT,
    ROOT,
    metadata,
    passing_decision,
    release_policy,
    runtime_report,
)


def test_release_policy_loads_with_required_bindings():
    policy = release_policy()
    assert policy.policy_id == "phase4-information-release-audit-v1"
    assert policy.required_package_version == "0.15.0"
    assert policy.required_runtime_manifest_id == "phase4-information-runtime-probes-v1"
    assert policy.minimum_runtime_collected_test_count == 640
    assert policy.private_output_only is True
    assert policy.repository_output_allowed is False


def test_weakened_release_boundary_is_rejected(tmp_path):
    value = json.loads(
        (ROOT / "policies/information_release_audit_policy.json").read_text()
    )
    value["boundaries"]["live_network_allowed"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase4ReleaseAuditError, match="weakened"):
        load_phase4_release_audit_policy(path)


def test_weakened_runtime_floor_is_rejected(tmp_path):
    value = json.loads(
        (ROOT / "policies/information_release_audit_policy.json").read_text()
    )
    value["minimum_runtime_collected_test_count"] = 1
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase4ReleaseAuditError, match="weakened"):
        load_phase4_release_audit_policy(path)


def test_policy_substitution_is_rejected_by_digest(tmp_path):
    value = json.loads(
        (ROOT / "policies/information_release_audit_policy.json").read_text()
    )
    value["minimum_runtime_collected_test_count"] = 641
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase4ReleaseAuditError, match="canonical policy digest"):
        load_phase4_release_audit_policy(path)


def test_duplicate_policy_key_is_rejected(tmp_path):
    text = (ROOT / "policies/information_release_audit_policy.json").read_text()
    path = tmp_path / "policy.json"
    path.write_text(text.replace('"phase": "4",', '"phase": "4",\n  "phase": "4",'), encoding="utf-8")
    with pytest.raises(Phase4ReleaseAuditError, match="duplicate"):
        load_phase4_release_audit_policy(path)


def test_passing_runtime_evaluation_is_approved(tmp_path):
    decision = passing_decision(tmp_path)
    assert decision.approved is True
    assert decision.decision_reasons == ()
    assert decision.runtime_passed_test_count == 640
    assert decision.runtime_collected_test_count == 640
    assert decision.runtime_skipped_test_count == 0
    assert decision.runtime_network_guard_active is True
    verify_phase4_release_decision(decision)


def test_failed_runtime_evaluation_creates_rejected_decision(tmp_path):
    decision = audit_phase4_release(
        runtime_report(tmp_path, passed=False),
        metadata=metadata(),
        release_policy=release_policy(),
    )
    assert decision.approved is False
    assert "runtime_test_failure" in decision.decision_reasons
    assert "runtime_evidence_failed" in decision.decision_reasons
    assert "runtime_backed_evaluation_failed" in decision.decision_reasons
    verify_phase4_release_decision(decision)



def test_missing_runtime_case_evidence_rejects_release(tmp_path):
    report = runtime_report(tmp_path)
    evidence = replace(report.runtime_evidence, case_evidence=(), evidence_digest="")
    evidence = replace(
        evidence,
        evidence_digest=information_runtime_evidence_digest(evidence),
    )
    tampered = replace(report, runtime_evidence=evidence, report_digest="")
    tampered = replace(
        tampered,
        report_digest=information_runtime_backed_report_digest(tampered),
    )
    decision = audit_phase4_release(
        tampered,
        metadata=metadata(),
        release_policy=release_policy(),
    )
    assert decision.approved is False
    assert "runtime_case_evidence_incomplete" in decision.decision_reasons


def test_invalid_repository_commit_is_rejected(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="40-character"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(repository_commit="abc"),
            release_policy=release_policy(),
        )


def test_release_time_requires_utc(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="UTC"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(evaluated_at="2026-07-30T00:30:00-05:00"),
            release_policy=release_policy(),
        )


def test_repository_head_must_match_release_commit(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="exactly match"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(repository_head_commit="c" * 40),
            release_policy=release_policy(),
        )


def test_repository_must_be_clean(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="clean"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(repository_clean=False),
            release_policy=release_policy(),
        )


def test_rollback_commit_is_required(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="rollback"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(rollback_commit=None),
            release_policy=release_policy(),
        )


def test_rollback_commit_must_differ(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="differ"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(rollback_commit=COMMIT),
            release_policy=release_policy(),
        )


def test_package_version_must_match_policy(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="package version"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(package_version="0.14.0"),
            release_policy=release_policy(),
        )


def test_policy_versions_are_required(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="policy"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(policy_versions=()),
            release_policy=release_policy(),
        )


def test_limitations_must_be_compact_metadata(tmp_path):
    with pytest.raises(Phase4ReleaseAuditError, match="single-line"):
        audit_phase4_release(
            runtime_report(tmp_path),
            metadata=metadata(known_limitations=("line one\nline two",)),
            release_policy=release_policy(),
        )


def test_release_record_writes_and_loads_under_private_root(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    decision = passing_decision(tmp_path / "decision")
    output = vault / "reports" / "phase4-information-release.json"
    written = write_phase4_release_record(
        decision,
        output,
        private_root=vault,
        repository_root=repository,
    )
    assert load_phase4_release_record(written) == decision


def test_release_record_refuses_repository_output(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(Phase4ReleaseAuditError, match="repository"):
        write_phase4_release_record(
            passing_decision(tmp_path / "decision"),
            repository / "release.json",
            private_root=tmp_path,
            repository_root=repository,
        )


def test_release_record_refuses_output_outside_private_root(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    with pytest.raises(Phase4ReleaseAuditError, match="private"):
        write_phase4_release_record(
            passing_decision(tmp_path / "decision"),
            tmp_path / "outside.json",
            private_root=vault,
            repository_root=repository,
        )


def test_release_record_requires_json_suffix(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    with pytest.raises(Phase4ReleaseAuditError, match=".json"):
        write_phase4_release_record(
            passing_decision(tmp_path / "decision"),
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
    decision = passing_decision(tmp_path / "decision")
    first = write_phase4_release_record(
        decision, output, private_root=vault, repository_root=repository
    )
    second = write_phase4_release_record(
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
    with pytest.raises(Phase4ReleaseAuditError, match="overwrite"):
        write_phase4_release_record(
            passing_decision(tmp_path / "decision"),
            output,
            private_root=vault,
            repository_root=repository,
        )


def test_tampered_release_record_fails_digest_check(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    output = vault / "release.json"
    write_phase4_release_record(
        passing_decision(tmp_path / "decision"),
        output,
        private_root=vault,
        repository_root=repository,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["approved"] = False
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase4ReleaseAuditError, match="digest"):
        load_phase4_release_record(output)


def test_release_record_json_is_stable(tmp_path):
    decision = passing_decision(tmp_path)
    assert phase4_release_record_json(decision) == phase4_release_record_json(decision)
    assert decision.release_id in phase4_release_record_json(decision)
