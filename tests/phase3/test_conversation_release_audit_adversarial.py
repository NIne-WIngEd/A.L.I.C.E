from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alice_conversation.final_evaluation import conversation_final_report_digest
from alice_conversation.release_audit import (
    Phase3ReleaseAuditError,
    phase3_release_record_digest,
    verify_phase3_release_decision,
)
from _release_audit_helpers import passing_decision, passing_report


def _resign(decision, **changes):
    value = replace(decision, **changes, record_digest="")
    return replace(value, record_digest=phase3_release_record_digest(value))


def test_approved_flag_cannot_disagree_with_reasons():
    decision = _resign(passing_decision(), approved=False)
    with pytest.raises(Phase3ReleaseAuditError, match="inconsistent"):
        verify_phase3_release_decision(decision)


def test_duplicate_metric_ids_are_rejected():
    original = passing_decision()
    metrics = original.metric_results + (original.metric_results[0],)
    decision = _resign(original, metric_results=metrics)
    with pytest.raises(Phase3ReleaseAuditError, match="metric"):
        verify_phase3_release_decision(decision)


def test_wrong_audit_version_is_rejected():
    decision = _resign(passing_decision(), audit_version="p3.11-v0")
    with pytest.raises(Phase3ReleaseAuditError, match="version"):
        verify_phase3_release_decision(decision)


def test_repository_clean_binding_cannot_be_disabled():
    decision = _resign(passing_decision(), repository_clean=False)
    with pytest.raises(Phase3ReleaseAuditError, match="binding"):
        verify_phase3_release_decision(decision)


def test_repository_output_boundary_cannot_be_enabled():
    decision = _resign(passing_decision(), repository_output_allowed=True)
    with pytest.raises(Phase3ReleaseAuditError, match="boundaries"):
        verify_phase3_release_decision(decision)


def test_raw_conversation_boundary_cannot_be_enabled():
    decision = _resign(passing_decision(), raw_conversation_content_allowed=True)
    with pytest.raises(Phase3ReleaseAuditError, match="boundaries"):
        verify_phase3_release_decision(decision)


def test_tampered_final_report_boundary_fails_before_release_audit():
    report = passing_report()
    tampered = replace(report, web_access_allowed=True, report_digest="")
    tampered = replace(tampered, report_digest=conversation_final_report_digest(tampered))
    from alice_conversation.final_evaluation import verify_conversation_final_report
    with pytest.raises(Exception, match="inconsistent"):
        verify_conversation_final_report(tampered)
