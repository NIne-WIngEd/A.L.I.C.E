from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_phase4_status_is_truthful_and_phase5_is_active() -> None:
    readme = _text("README.md")
    roadmap = _text("docs/ROADMAP.md")
    catalog = _text("docs/CAPABILITY_CATALOG.md")
    assert "Phase 4 operational closure" in readme
    assert "P4.10 operational live-public-information closure is approved and merged" in catalog
    assert "Phase 5.0" in readme and "active" in readme.lower()
    assert "Operationally complete" in roadmap
    assert "Active after approved P4.10 closure and merge" in roadmap


def test_post_phase_audit_is_preserved_as_discovery_record() -> None:
    audit = _text("docs/PHASE_4_POST_PHASE_AUDIT.md")
    standard = _text("docs/PHASE_BOUNDARY_AUDIT_STANDARD.md")
    adr = _text("docs/decisions/ADR-009-phase4-live-public-information-closure.md")
    assert "operational live-public-information closure remains required" in audit
    assert "Every top-level phase ends with an adversarial audit" in standard
    assert "Phase 5 is blocked until P4.10" in adr


def test_live_acceptance_policy_remains_narrow_historical_gate() -> None:
    payload = json.loads(_text("policies/information_live_provider_acceptance_policy.json"))
    assert payload["milestone"] == "P4.10"
    assert payload["initial_query_classifications"] == ["PUBLIC"]
    assert payload["phase5_start_gate"] == "blocked_until_p4_10_approved"
    assert payload["capability_ceiling"] is False


def test_operational_release_evidence_exists() -> None:
    report = _text("docs/PHASE_4_LIVE_OPERATIONAL_RELEASE_REPORT.md")
    policy = json.loads(_text("policies/information_live_acceptance_release_policy.json"))
    assert "P4.10c" in report
    assert policy["milestone"] == "P4.10c"
