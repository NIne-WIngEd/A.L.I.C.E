from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from alice_information.live_research import InformationLiveResearchError, InformationLiveSourceOutcome
from _information_live_research_helpers import DIGEST, receipt


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome", "unbounded"),
        ("query_sha256", "0" * 63),
        ("pre_commit_validation_count", 2),
        ("fetch_attempt_count", 0),
        ("fetch_failure_sha256s", [DIGEST]),
        ("policy_bindings", ("missing-version",)),
        ("fetch_receipt_sha256s", [DIGEST]),
    ],
)
def test_receipt_rejects_tampered_critical_fields(field, value):
    original = receipt()
    with pytest.raises(InformationLiveResearchError):
        replace(original, **{field: value}).validate()


def test_blocked_source_can_never_support_a_claim():
    value = InformationLiveSourceOutcome(
        source_id="source-1",
        canonical_url="https://example.com/source",
        source_content_sha256=DIGEST,
        temporal_verdict="accepted",
        inspection_verdict="blocked",
        freshness_verdict=None,
        supports_claim=True,
        disposition="blocked_injection",
        reason_code="prompt_injection_blocked",
    )
    with pytest.raises(InformationLiveResearchError):
        value.validate()

def test_operational_boundary_rejects_substituted_registry():
    fake = SimpleNamespace(registry=object())
    with pytest.raises(InformationLiveResearchError, match="registry was substituted"):
        from alice_information.live_research import LiveInformationResearchExecutor

        LiveInformationResearchExecutor.validate_operational_boundary(fake)
