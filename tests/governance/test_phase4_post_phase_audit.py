from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_phase4_status_is_truthful_and_phase5_is_blocked() -> None:
    readme = _text("README.md")
    roadmap = _text("docs/ROADMAP.md")
    assert "P4.10 operational live-public-information closure" in readme
    assert "Phase 5.0" in readme and "blocked" in readme.lower()
    assert "P4.10 operational live-public-information closure" in roadmap
    assert "Phase 5" in roadmap and "blocked" in roadmap.lower()


def test_post_phase_audit_and_standard_exist() -> None:
    audit = _text("docs/PHASE_4_POST_PHASE_AUDIT.md")
    standard = _text("docs/PHASE_BOUNDARY_AUDIT_STANDARD.md")
    adr = _text("docs/decisions/ADR-009-phase4-live-public-information-closure.md")
    assert "operational live-public-information closure remains required" in audit
    assert "Every top-level phase ends with an adversarial audit" in standard
    assert "Phase 5 is blocked until P4.10" in adr


def test_live_acceptance_policy_is_narrow_and_blocks_phase5() -> None:
    payload = json.loads(
        _text("policies/information_live_provider_acceptance_policy.json")
    )
    assert payload["milestone"] == "P4.10"
    assert payload["initial_query_classifications"] == ["PUBLIC"]
    assert payload["required_live_search_provider_count"] >= 1
    assert payload["required_live_fetch_provider_count"] >= 1
    assert payload["phase5_start_gate"] == "blocked_until_p4_10_approved"
    controls = set(payload["required_controls"])
    assert "no_silent_fallback" in controls
    assert "no_source_body_persistence" in controls
    assert "no_phase5_storage_activation" in controls
    assert "no_external_action" in controls
    assert payload["capability_ceiling"] is False


def test_capability_catalog_does_not_claim_live_web_is_available() -> None:
    catalog = _text("docs/CAPABILITY_CATALOG.md")
    assert "Fresh web research with exact citations — `IN DEVELOPMENT / P4.10 LIVE ACCEPTANCE`" in catalog
    assert "Read-only public information tools — `IN DEVELOPMENT / P4.10 LIVE ACCEPTANCE`" in catalog


def test_p49_report_is_preserved_as_fixture_release() -> None:
    report = _text("docs/PHASE_4_FINAL_RELEASE_REPORT.md")
    assert "Approved fixture-governed compatibility release" in report
    assert "P4.10 operational live acceptance remains required" in report
    assert "The release audit blocks live network access" in report
