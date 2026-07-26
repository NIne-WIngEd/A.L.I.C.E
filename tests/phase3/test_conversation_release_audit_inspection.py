from __future__ import annotations

from alice_conversation.release_audit_inspection import (
    inspect_phase3_release,
    render_phase3_release_inspection,
)
from _release_audit_helpers import passing_decision


def test_release_inspection_is_metadata_only():
    value = inspect_phase3_release(passing_decision())
    assert value.approved is True
    assert value.case_count == value.passed_case_count
    assert not hasattr(value, "case_results")
    assert value.evidence_target_count == value.evidence_passed_target_count


def test_release_inspection_lists_failed_metrics_as_empty_for_approval():
    value = inspect_phase3_release(passing_decision())
    assert value.failed_metric_ids == ()


def test_release_inspection_render_is_stable():
    value = inspect_phase3_release(passing_decision())
    rendered = render_phase3_release_inspection(value)
    assert "approved=true" in rendered
    assert f"release_id={value.release_id}" in rendered
    assert "record_digest=" in rendered
