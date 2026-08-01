from alice_information.live_acceptance_inspection import inspect_live_acceptance, render_live_acceptance_inspection
from _information_live_acceptance_helpers import record


def test_inspection_is_metadata_only_and_surfaces_exact_live_gates():
    inspection = inspect_live_acceptance(record())
    rendered = render_live_acceptance_inspection(inspection)
    assert "approved=true" in rendered
    assert "repository_regression=2124/2126" in rendered
    assert "repository_subtests_passed=14" in rendered
    assert "live_search_results=1" in rendered
    assert "live_fetch_attempts=1" in rendered
    assert "live_fetches=1" in rendered
    assert "live_fetch_failures=0" in rendered
    assert "grounded_sources=1" in rendered
    assert "p36_pre_commit_validations=1" in rendered
    assert "p45b_validation_outcome=accepted" in rendered
    assert "source_body" not in rendered
