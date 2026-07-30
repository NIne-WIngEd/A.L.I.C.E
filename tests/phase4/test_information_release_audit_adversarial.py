from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alice_information.release_audit import (
    Phase4ReleaseAuditError,
    _phase4_release_id,
    load_phase4_release_record,
    phase4_release_record_digest,
    phase4_release_record_json,
    verify_phase4_release_decision,
)
from _information_release_audit_helpers import passing_decision


def _resign(decision, **changes):
    value = replace(decision, **changes, record_digest="")
    return replace(value, record_digest=phase4_release_record_digest(value))


def test_approved_flag_cannot_disagree_with_reasons(tmp_path):
    decision = _resign(passing_decision(tmp_path), approved=False)
    with pytest.raises(Phase4ReleaseAuditError, match="inconsistent"):
        verify_phase4_release_decision(decision)



def test_release_id_is_bound_to_release_material(tmp_path):
    decision = _resign(passing_decision(tmp_path), release_id="0" * 32)
    with pytest.raises(Phase4ReleaseAuditError, match="ID binding"):
        verify_phase4_release_decision(decision)


def test_duplicate_metric_ids_are_rejected(tmp_path):
    original = passing_decision(tmp_path)
    decision = _resign(
        original,
        metric_results=original.metric_results + (original.metric_results[0],),
    )
    with pytest.raises(Phase4ReleaseAuditError, match="metric"):
        verify_phase4_release_decision(decision)


def test_wrong_audit_version_is_rejected(tmp_path):
    decision = _resign(passing_decision(tmp_path), audit_version="p4.9-v0")
    with pytest.raises(Phase4ReleaseAuditError, match="version"):
        verify_phase4_release_decision(decision)


def test_repository_clean_binding_cannot_be_disabled(tmp_path):
    decision = _resign(passing_decision(tmp_path), repository_clean=False)
    with pytest.raises(Phase4ReleaseAuditError, match="binding"):
        verify_phase4_release_decision(decision)


def test_repository_output_boundary_cannot_be_enabled(tmp_path):
    decision = _resign(passing_decision(tmp_path), repository_output_allowed=True)
    with pytest.raises(Phase4ReleaseAuditError, match="boundaries"):
        verify_phase4_release_decision(decision)


def test_raw_query_boundary_cannot_be_enabled(tmp_path):
    decision = _resign(passing_decision(tmp_path), raw_query_text_allowed=True)
    with pytest.raises(Phase4ReleaseAuditError, match="boundaries"):
        verify_phase4_release_decision(decision)


def test_runtime_network_guard_cannot_be_disabled_for_approval(tmp_path):
    decision = _resign(
        passing_decision(tmp_path),
        runtime_network_guard_active=False,
    )
    with pytest.raises(Phase4ReleaseAuditError, match="incomplete"):
        verify_phase4_release_decision(decision)


def test_runtime_counts_cannot_be_inflated(tmp_path):
    decision = _resign(
        passing_decision(tmp_path),
        runtime_passed_test_count=641,
    )
    with pytest.raises(Phase4ReleaseAuditError, match="counts"):
        verify_phase4_release_decision(decision)


def test_runtime_report_digest_must_be_sha256(tmp_path):
    original = passing_decision(tmp_path)
    bad_digest = "0" * 63
    release_id = _phase4_release_id(
        repository_commit=original.repository_commit,
        rollback_commit=original.rollback_commit,
        evaluated_at=original.evaluated_at,
        runtime_backed_report_digest=bad_digest,
        release_policy_digest=original.release_policy_digest,
    )
    decision = _resign(
        original,
        runtime_backed_report_digest=bad_digest,
        release_id=release_id,
    )
    with pytest.raises(Phase4ReleaseAuditError, match="digest"):
        verify_phase4_release_decision(decision)


def test_duplicate_release_record_key_is_rejected(tmp_path):
    path = tmp_path / "release.json"
    text = phase4_release_record_json(passing_decision(tmp_path / "decision"))
    path.write_text(
        text.replace('"approved": true,', '"approved": true,\n  "approved": true,'),
        encoding="utf-8",
    )
    with pytest.raises(Phase4ReleaseAuditError, match="duplicate"):
        load_phase4_release_record(path)


def test_string_boolean_is_rejected_in_record(tmp_path):
    path = tmp_path / "release.json"
    value = json.loads(
        phase4_release_record_json(passing_decision(tmp_path / "decision"))
    )
    value["approved"] = "true"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase4ReleaseAuditError, match="boolean"):
        load_phase4_release_record(path)
