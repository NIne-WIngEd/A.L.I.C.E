from __future__ import annotations

from alice_information.release_audit_inspection import (
    inspect_phase4_release,
    render_phase4_release_inspection,
)
from _information_release_audit_helpers import passing_decision


def test_release_inspection_is_metadata_only(tmp_path):
    value = inspect_phase4_release(passing_decision(tmp_path))
    assert value.approved is True
    assert value.case_count == value.passed_case_count
    assert value.runtime_collected_test_count == value.runtime_passed_test_count
    assert not hasattr(value, "case_results")
    assert not hasattr(value, "target_files")


def test_release_inspection_lists_failed_metrics_as_empty_for_approval(tmp_path):
    value = inspect_phase4_release(passing_decision(tmp_path))
    assert value.failed_metric_ids == ()


def test_release_inspection_render_is_stable(tmp_path):
    value = inspect_phase4_release(passing_decision(tmp_path))
    rendered = render_phase4_release_inspection(value)
    assert "approved=true" in rendered
    assert f"release_id={value.release_id}" in rendered
    assert "runtime_network_guard_active=true" in rendered
    assert "record_digest=" in rendered
