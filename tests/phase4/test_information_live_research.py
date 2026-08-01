from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from alice_information.live_research import (
    InformationLiveFetchFailure,
    InformationLiveResearchError,
    InformationLiveResearchReceipt,
    InformationLiveSourceOutcome,
    LiveInformationResearchExecutor,
)
from _information_live_research_helpers import DIGEST, command, policy, receipt, request


def test_metadata_only_source_outcome_validates():
    outcome = InformationLiveSourceOutcome(
        source_id="source-1",
        canonical_url="https://example.com/source",
        source_content_sha256=DIGEST,
        temporal_verdict="accepted",
        inspection_verdict="clear",
        freshness_verdict="fresh",
        supports_claim=True,
        disposition="grounded",
        reason_code=None,
    )
    outcome.validate()
    record = outcome.metadata_record()
    assert "normalized_text" not in record
    assert record["disposition"] == "grounded"


def test_receipt_is_tamper_evident_and_metadata_only():
    value = receipt()
    value.validate()
    record = value.to_metadata_record()
    assert "query" not in record
    assert "source_body" not in record
    with pytest.raises(InformationLiveResearchError):
        replace(value, pre_commit_validation_count=0).validate()


def test_receipt_requires_one_temporal_binding_per_fetch():
    with pytest.raises(InformationLiveResearchError):
        receipt(temporal_resolution_sha256s=())


def test_request_gate_requires_explicit_available_research_mode():
    fake = SimpleNamespace(research_policy=policy())
    with pytest.raises(InformationLiveResearchError, match="explicit available"):
        LiveInformationResearchExecutor._validate_request(
            fake, command(), mode="local_only", availability="available", request=request()
        )
    with pytest.raises(InformationLiveResearchError, match="explicit available"):
        LiveInformationResearchExecutor._validate_request(
            fake, command(), mode="research", availability="offline", request=request()
        )


def test_request_gate_rejects_preinjected_grounding_and_budget_substitution():
    fake = SimpleNamespace(research_policy=policy())
    with pytest.raises(InformationLiveResearchError, match="only grounding"):
        LiveInformationResearchExecutor._validate_request(
            fake, command(grounding=object()), mode="research", availability="available", request=request()
        )
    altered = request(max_sources=2)
    altered = replace(altered, max_fetch_calls=1)
    with pytest.raises(InformationLiveResearchError, match="budget"):
        LiveInformationResearchExecutor._validate_request(
            fake, command(), mode="research", availability="available", request=altered
        )


def test_request_gate_rejects_operation_substitution():
    fake = SimpleNamespace(research_policy=policy())
    with pytest.raises(InformationLiveResearchError, match="search/fetch"):
        LiveInformationResearchExecutor._validate_request(
            fake, command(), mode="research", availability="available", request=request(operations=("fetch", "search"))
        )


def test_fetch_failure_is_tamper_evident_and_metadata_only():
    failure = InformationLiveFetchFailure.create(
        result_id="result-1",
        result_rank=1,
        canonical_url="https://example.com/rejected",
        result_sha256=DIGEST,
        failure_code="http_status_rejected",
    )
    record = failure.metadata_record()
    assert record["failure_code"] == "http_status_rejected"
    assert "source_body" not in record
    assert "query" not in record
    with pytest.raises(InformationLiveResearchError):
        replace(failure, result_rank=2).validate()


def test_receipt_accounts_for_successes_and_rejected_candidates():
    value = receipt(
        search_result_count=2,
        fetch_attempt_count=2,
        fetch_receipt_sha256s=(DIGEST,),
        fetch_failure_sha256s=(DIGEST,),
        temporal_resolution_sha256s=(DIGEST,),
    )
    value.validate()
    with pytest.raises(InformationLiveResearchError):
        replace(value, fetch_attempt_count=1).validate()


def test_source_local_rejections_are_skippable_but_unknown_failures_are_not():
    class StatusRejected(RuntimeError):
        code = "http_status_rejected"

    class HeaderRejected(RuntimeError):
        code = "response_header_invalid"

    class Unknown(RuntimeError):
        code = "transport_failed"

    fake = SimpleNamespace(research_policy=policy())
    assert (
        LiveInformationResearchExecutor._skippable_fetch_failure_code(
            fake, StatusRejected("rejected")
        )
        == "http_status_rejected"
    )
    assert (
        LiveInformationResearchExecutor._skippable_fetch_failure_code(
            fake, HeaderRejected("rejected")
        )
        == "response_header_invalid"
    )
    assert (
        LiveInformationResearchExecutor._skippable_fetch_failure_code(
            fake, Unknown("failed")
        )
        is None
    )
