"""P4.6a bounded deterministic research-orchestration tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import pytest

from alice_information.contracts import (
    InformationActivityRecord,
    InformationQuery,
    InformationResearchRequest,
    InformationSearchResult,
    InformationSourceDocument,
)
from alice_information.policy import load_information_policy
from alice_information.provider_policy import load_information_provider_policy
from alice_information.providers import (
    DeterministicInformationFetchProvider,
    DeterministicInformationSearchProvider,
    InformationCancellationToken,
    InformationProviderExecutionError,
    InformationProviderFailure,
    InformationSearchFixture,
    InformationSourceFixture,
)
from alice_information.registry import InformationProviderRegistry
from alice_information.research_orchestration import (
    InformationResearchOrchestrationError,
    InformationResearchOrchestrator,
    InformationResearchRunReceipt,
)
from alice_information.research_orchestration_policy import (
    InformationResearchOrchestrationPolicyError,
    load_information_research_orchestration_policy,
)

NOW = "2026-07-28T15:00:00Z"
PROVIDER = "deterministic-fixture-v1"
QUERY_TEXT = "bounded orchestration fixture"
URLS = (
    "https://example.com/a",
    "https://example.com/b",
    "https://example.com/c",
)


def _query() -> InformationQuery:
    return InformationQuery.create(
        query_id="query-p46a-001",
        text=QUERY_TEXT,
        created_at=NOW,
    )


def _request(
    *,
    max_fetch_calls: int = 3,
    max_sources: int = 3,
    request_timeout_seconds: float = 10,
    total_timeout_seconds: float = 45,
    operations: tuple[str, ...] = ("search", "fetch"),
) -> InformationResearchRequest:
    request = InformationResearchRequest(
        request_id="research-request-p46a-001",
        query=_query(),
        operations=operations,
        max_search_calls=1,
        max_fetch_calls=max_fetch_calls,
        max_sources=max_sources,
        request_timeout_seconds=request_timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
    )
    request.validate()
    return request


def _search_fixtures(
    *,
    include_duplicate_url: bool = False,
) -> tuple[InformationSearchFixture, ...]:
    fixtures = [
        InformationSearchFixture(
            result_id="result-c",
            rank=3,
            title="C",
            canonical_url=URLS[2],
            snippet="Snippet C",
            retrieved_at=NOW,
        ),
        InformationSearchFixture(
            result_id="result-a",
            rank=1,
            title="A",
            canonical_url=URLS[0],
            snippet="Snippet A",
            retrieved_at=NOW,
        ),
        InformationSearchFixture(
            result_id="result-b",
            rank=2,
            title="B",
            canonical_url=URLS[1],
            snippet="Snippet B",
            retrieved_at=NOW,
        ),
    ]
    if include_duplicate_url:
        fixtures.append(
            InformationSearchFixture(
                result_id="result-a-copy",
                rank=4,
                title="A copy",
                canonical_url=URLS[0],
                snippet="Duplicate canonical URL",
                retrieved_at=NOW,
            )
        )
    return tuple(fixtures)


def _search_provider(
    *,
    include_duplicate_url: bool = False,
) -> DeterministicInformationSearchProvider:
    return DeterministicInformationSearchProvider(
        provider=PROVIDER,
        fixtures={
            _query().content_sha256: _search_fixtures(
                include_duplicate_url=include_duplicate_url
            )
        },
    )


def _fetch_provider(
    *,
    missing: tuple[str, ...] = (),
    duplicate_source_id: bool = False,
) -> DeterministicInformationFetchProvider:
    fixtures: dict[str, InformationSourceFixture] = {}
    for index, url in enumerate(URLS):
        if url in missing:
            continue
        source_id = "source-duplicate" if duplicate_source_id and index < 2 else f"source-{index + 1}"
        fixtures[url] = InformationSourceFixture(
            source_id=source_id,
            canonical_url=url,
            title=f"Source {index + 1}",
            normalized_text=f"Normalized source content {index + 1}.",
            retrieved_at=NOW,
        )
    return DeterministicInformationFetchProvider(provider=PROVIDER, fixtures=fixtures)


def _registry(
    search: object | None = None,
    fetch: object | None = None,
) -> InformationProviderRegistry:
    registry = InformationProviderRegistry(
        information_policy=load_information_policy(),
        provider_policy=load_information_provider_policy(),
    )
    registry.register_search(search or _search_provider())
    registry.register_fetch(fetch or _fetch_provider())
    return registry


def _orchestrator(
    *,
    search: object | None = None,
    fetch: object | None = None,
    clock: Callable[[], float] = lambda: 0.0,
    timestamp_factory: Callable[[], str] = lambda: NOW,
) -> InformationResearchOrchestrator:
    return InformationResearchOrchestrator(
        policy=load_information_research_orchestration_policy(),
        registry=_registry(search, fetch),
        clock=clock,
        timestamp_factory=timestamp_factory,
    )


def test_complete_fixture_run_is_ranked_bounded_and_digest_bound() -> None:
    run = _orchestrator().execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "completed"
    assert run.receipt.stopping_reason == "all_selected_sources_fetched"
    assert [result.result_id for result in run.search_results] == [
        "result-a",
        "result-b",
        "result-c",
    ]
    assert [source.source_id for source in run.sources] == [
        "source-1",
        "source-2",
        "source-3",
    ]
    assert run.receipt.search_calls == 1
    assert run.receipt.fetch_calls == 3
    assert run.receipt.failed_fetch_calls == 0
    run.validate(policy=load_information_research_orchestration_policy())


def test_source_budget_limits_selected_results_and_fetches() -> None:
    fetch = _fetch_provider()
    run = _orchestrator(fetch=fetch).execute(
        _request(max_sources=2),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert len(run.search_results) == 2
    assert len(run.sources) == 2
    assert fetch.result_ids == ["result-a", "result-b"]


def test_fetch_budget_limits_selected_results_and_fetches() -> None:
    fetch = _fetch_provider()
    run = _orchestrator(fetch=fetch).execute(
        _request(max_fetch_calls=1),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert tuple(result.result_id for result in run.search_results) == ("result-a",)
    assert fetch.result_ids == ["result-a"]


def test_duplicate_canonical_search_urls_are_deduplicated() -> None:
    run = _orchestrator(search=_search_provider(include_duplicate_url=True)).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert len({result.canonical_url for result in run.search_results}) == 3
    assert "result-a-copy" not in run.receipt.selected_result_ids


@dataclass
class EmptySearch:
    provider: str = PROVIDER
    provider_type: str = "deterministic_fixture"
    calls: int = 0

    def search(self, query, *, max_results, timeout_seconds, cancellation=None):
        self.calls += 1
        return ()


def test_empty_search_returns_insufficient_sources() -> None:
    search = EmptySearch()
    run = _orchestrator(search=search).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "insufficient_sources"
    assert run.receipt.stopping_reason == "search_returned_no_results"
    assert run.sources == ()
    assert search.calls == 1


def test_missing_search_fixture_returns_sanitized_failed_run() -> None:
    search = DeterministicInformationSearchProvider(
        provider=PROVIDER,
        fixtures={
            InformationQuery.create(
                query_id="other",
                text="other",
                created_at=NOW,
            ).content_sha256: _search_fixtures()
        },
    )
    run = _orchestrator(search=search).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "failed"
    assert run.receipt.stopping_reason == "search_failed"
    assert run.receipt.activity_records[0].error_code == "information_integrity_failed"
    assert QUERY_TEXT not in repr(run.receipt)


def test_partial_fetch_failure_preserves_successful_sources() -> None:
    fetch = _fetch_provider(missing=(URLS[1],))
    run = _orchestrator(fetch=fetch).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "partial"
    assert run.receipt.stopping_reason == "partial_fetch_failure"
    assert tuple(source.source_id for source in run.sources) == ("source-1", "source-3")
    assert run.receipt.failed_result_ids == ("result-b",)
    assert run.receipt.failed_fetch_calls == 1


def test_all_fetch_failures_return_insufficient_sources() -> None:
    run = _orchestrator(fetch=_fetch_provider(missing=URLS)).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "insufficient_sources"
    assert run.receipt.stopping_reason == "all_fetches_failed"
    assert run.receipt.failed_fetch_calls == 3
    assert run.sources == ()


def test_initial_cancellation_prevents_provider_calls() -> None:
    search = _search_provider()
    fetch = _fetch_provider()
    token = InformationCancellationToken()
    token.cancel()
    run = _orchestrator(search=search, fetch=fetch).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
        cancellation=token,
    )
    assert run.receipt.outcome == "cancelled"
    assert run.receipt.search_calls == 0
    assert search.query_digests == []
    assert fetch.result_ids == []


@dataclass
class CancellingSearch:
    token: InformationCancellationToken
    provider: str = PROVIDER
    provider_type: str = "deterministic_fixture"

    def search(self, query, *, max_results, timeout_seconds, cancellation=None):
        self.token.cancel()
        return ()


def test_cancellation_during_search_is_sanitized() -> None:
    token = InformationCancellationToken()
    run = _orchestrator(search=CancellingSearch(token)).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
        cancellation=token,
    )
    assert run.receipt.outcome == "cancelled"
    assert run.receipt.activity_records[0].status == "cancelled"
    assert run.receipt.activity_records[0].error_code == "research_cancelled"


class StepClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.last = 0.0

    def __call__(self) -> float:
        try:
            self.last = next(self.values)
        except StopIteration:
            pass
        return self.last


def test_total_timeout_before_search_returns_failed_run() -> None:
    clock = StepClock([0.0, 46.0])
    run = _orchestrator(clock=clock).execute(
        _request(total_timeout_seconds=45),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "failed"
    assert run.receipt.stopping_reason == "total_timeout"
    assert run.receipt.activity_records[0].error_code == "research_budget_exhausted"


def test_total_timeout_after_search_stops_before_fetch() -> None:
    clock = StepClock([0.0, 0.0, 46.0])
    run = _orchestrator(clock=clock).execute(
        _request(total_timeout_seconds=45),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "failed"
    assert run.receipt.stopping_reason == "total_timeout"
    assert run.receipt.fetch_calls == 0


def test_total_timeout_during_fetch_rejects_out_of_budget_source() -> None:
    clock = StepClock([0.0, 0.0, 0.0, 0.0, 46.0])
    run = _orchestrator(clock=clock).execute(
        _request(max_fetch_calls=1, max_sources=1, total_timeout_seconds=45),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "failed"
    assert run.receipt.stopping_reason == "total_timeout"
    assert run.sources == ()
    assert run.receipt.failed_result_ids == ("result-a",)
    assert run.receipt.activity_records[-1].status == "failed"
    assert run.receipt.activity_records[-1].error_code == "research_budget_exhausted"


def test_wrong_operation_order_is_rejected() -> None:
    with pytest.raises(InformationResearchOrchestrationError, match="operation sequence"):
        _orchestrator().execute(
            _request(operations=("fetch", "search")),
            search_provider=PROVIDER,
            fetch_provider=PROVIDER,
        )


def test_request_exceeding_policy_budget_is_rejected() -> None:
    request = replace(_request(), max_search_calls=2)
    request.validate()
    with pytest.raises(InformationResearchOrchestrationPolicyError, match="exceeds"):
        _orchestrator().execute(
            request,
            search_provider=PROVIDER,
            fetch_provider=PROVIDER,
        )


def test_exact_provider_names_are_required_without_fallback() -> None:
    with pytest.raises(Exception, match="not registered"):
        _orchestrator().execute(
            _request(),
            search_provider="missing-provider",
            fetch_provider=PROVIDER,
        )


@dataclass
class CapturingSearch:
    results: tuple[InformationSearchResult, ...]
    provider: str = PROVIDER
    provider_type: str = "deterministic_fixture"
    max_results_seen: int | None = None
    timeout_seen: float | None = None
    calls: int = 0

    def search(self, query, *, max_results, timeout_seconds, cancellation=None):
        self.calls += 1
        self.max_results_seen = max_results
        self.timeout_seen = timeout_seconds
        return self.results[:max_results]


def _result(*, result_id: str, rank: int, url: str, provider: str = PROVIDER, query_id: str | None = None):
    return InformationSearchResult.create(
        result_id=result_id,
        query_id=query_id or _query().query_id,
        provider=provider,
        rank=rank,
        title=result_id,
        url=url,
        snippet=result_id,
        retrieved_at=NOW,
    )


def test_search_provider_is_called_once_with_exact_source_cap() -> None:
    search = CapturingSearch(
        (
            _result(result_id="r1", rank=1, url=URLS[0]),
            _result(result_id="r2", rank=2, url=URLS[1]),
            _result(result_id="r3", rank=3, url=URLS[2]),
        )
    )
    _orchestrator(search=search).execute(
        _request(max_sources=2),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert search.calls == 1
    assert search.max_results_seen == 2
    assert search.timeout_seen == 10


def _assert_sanitized_search_integrity_failure(run) -> None:
    assert run.receipt.outcome == "failed"
    assert run.receipt.stopping_reason == "search_failed"
    assert run.search_results == ()
    assert run.sources == ()
    assert len(run.receipt.activity_records) == 1
    assert run.receipt.activity_records[0].operation == "search"
    assert run.receipt.activity_records[0].status == "failed"
    assert run.receipt.activity_records[0].error_code == "information_integrity_failed"


@dataclass
class OverproducingSearch:
    provider: str = PROVIDER
    provider_type: str = "deterministic_fixture"

    def search(self, query, *, max_results, timeout_seconds, cancellation=None):
        return (
            _result(result_id="r1", rank=1, url=URLS[0]),
            _result(result_id="r2", rank=2, url=URLS[1]),
        )


def test_search_provider_cannot_exceed_returned_result_budget() -> None:
    run = _orchestrator(search=OverproducingSearch()).execute(
        _request(max_fetch_calls=1, max_sources=1),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    _assert_sanitized_search_integrity_failure(run)


def test_search_result_provider_substitution_returns_sanitized_failed_run() -> None:
    search = CapturingSearch(
        (_result(result_id="r1", rank=1, url=URLS[0], provider="other"),)
    )
    run = _orchestrator(search=search).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    _assert_sanitized_search_integrity_failure(run)


def test_search_result_query_substitution_returns_sanitized_failed_run() -> None:
    search = CapturingSearch(
        (_result(result_id="r1", rank=1, url=URLS[0], query_id="other-query"),)
    )
    run = _orchestrator(search=search).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    _assert_sanitized_search_integrity_failure(run)


def test_duplicate_search_rank_returns_sanitized_failed_run() -> None:
    search = CapturingSearch(
        (
            _result(result_id="r1", rank=1, url=URLS[0]),
            _result(result_id="r2", rank=1, url=URLS[1]),
        )
    )
    run = _orchestrator(search=search).execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    _assert_sanitized_search_integrity_failure(run)


@dataclass
class MismatchedFetch:
    provider: str = PROVIDER
    provider_type: str = "deterministic_fixture"
    calls: int = 0

    def fetch(self, result, *, timeout_seconds, max_response_bytes, cancellation=None):
        self.calls += 1
        return InformationSourceDocument.create(
            source_id=f"bad-{self.calls}",
            provider=self.provider,
            url="https://example.com/not-selected",
            title="Bad",
            normalized_text="Bad source",
            retrieved_at=NOW,
        )


def test_fetch_cannot_follow_an_unselected_url() -> None:
    fetch = MismatchedFetch()
    run = _orchestrator(fetch=fetch).execute(
        _request(max_sources=1),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "insufficient_sources"
    assert run.receipt.failed_fetch_calls == 1
    assert fetch.calls == 1


def test_duplicate_fetched_source_identity_is_partial_not_silently_collapsed() -> None:
    @dataclass
    class DuplicateIdentityFetch:
        provider: str = PROVIDER
        provider_type: str = "deterministic_fixture"
        calls: int = 0

        def fetch(self, result, *, timeout_seconds, max_response_bytes, cancellation=None):
            self.calls += 1
            source_id = "source-duplicate" if self.calls <= 2 else "source-3"
            return InformationSourceDocument.create(
                source_id=source_id,
                provider=self.provider,
                url=result.canonical_url,
                title=result.title,
                normalized_text=f"Duplicate test source {self.calls}.",
                retrieved_at=NOW,
            )

    run = _orchestrator(fetch=DuplicateIdentityFetch()).execute(
        _request(),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "partial"
    assert len(run.sources) == 2
    assert run.receipt.failed_fetch_calls == 1


def test_receipt_run_id_tampering_is_rejected() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(run.receipt, run_id="research-forged")
    with pytest.raises(InformationResearchOrchestrationError, match="run ID"):
        forged.validate()


def test_failed_result_ids_must_match_exact_attempted_sequence() -> None:
    run = _orchestrator(fetch=_fetch_provider(missing=(URLS[1],))).execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged_receipt = InformationResearchRunReceipt.create(
        request_id=run.receipt.request_id,
        query_id=run.receipt.query_id,
        query_sha256=run.receipt.query_sha256,
        search_provider=run.receipt.search_provider,
        fetch_provider=run.receipt.fetch_provider,
        outcome=run.receipt.outcome,
        stopping_reason=run.receipt.stopping_reason,
        search_calls=run.receipt.search_calls,
        fetch_calls=run.receipt.fetch_calls,
        failed_fetch_calls=run.receipt.failed_fetch_calls,
        selected_result_ids=run.receipt.selected_result_ids,
        failed_result_ids=("result-a",),
        source_ids=run.receipt.source_ids,
        source_content_sha256s=run.receipt.source_content_sha256s,
        started_at=run.receipt.started_at,
        finished_at=run.receipt.finished_at,
        policy_version=run.receipt.policy_version,
        activity_records=run.receipt.activity_records,
    )
    forged = replace(run, receipt=forged_receipt)
    with pytest.raises(InformationResearchOrchestrationError, match="attempted result"):
        forged.validate(policy=load_information_research_orchestration_policy())


def test_inconsistent_outcome_and_stopping_reason_are_rejected() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    with pytest.raises(InformationResearchOrchestrationError, match="inconsistent"):
        InformationResearchRunReceipt.create(
            request_id=run.receipt.request_id,
            query_id=run.receipt.query_id,
            query_sha256=run.receipt.query_sha256,
            search_provider=run.receipt.search_provider,
            fetch_provider=run.receipt.fetch_provider,
            outcome="completed",
            stopping_reason="partial_fetch_failure",
            search_calls=run.receipt.search_calls,
            fetch_calls=run.receipt.fetch_calls,
            failed_fetch_calls=run.receipt.failed_fetch_calls,
            selected_result_ids=run.receipt.selected_result_ids,
            failed_result_ids=run.receipt.failed_result_ids,
            source_ids=run.receipt.source_ids,
            source_content_sha256s=run.receipt.source_content_sha256s,
            started_at=run.receipt.started_at,
            finished_at=run.receipt.finished_at,
            policy_version=run.receipt.policy_version,
            activity_records=run.receipt.activity_records,
        )


def test_selected_results_require_successful_search_activity() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    altered_search = replace(
        run.receipt.activity_records[0],
        status="failed",
        error_code="information_integrity_failed",
    )
    forged = replace(
        run.receipt,
        activity_records=(altered_search, *run.receipt.activity_records[1:]),
    )
    with pytest.raises(InformationResearchOrchestrationError, match="successful search"):
        forged.validate()


def test_receipt_digest_tampering_is_rejected() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(run.receipt, receipt_sha256="0" * 64)
    with pytest.raises(InformationResearchOrchestrationError, match="digest"):
        forged.validate()


def test_receipt_request_binding_tampering_is_rejected() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(run, receipt=replace(run.receipt, request_id="other"))
    with pytest.raises(InformationResearchOrchestrationError):
        forged.validate(policy=load_information_research_orchestration_policy())


def test_receipt_rejects_reordered_activity_sequence() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    activities = run.receipt.activity_records
    forged = replace(
        run.receipt,
        activity_records=(activities[0], activities[2], activities[1], activities[3]),
    )
    with pytest.raises(InformationResearchOrchestrationError, match="activity order"):
        forged.validate()


def test_receipt_rejects_activity_outside_run_interval() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged_search = replace(
        run.receipt.activity_records[0],
        started_at="2026-07-28T14:59:59Z",
    )
    forged = replace(
        run.receipt,
        activity_records=(forged_search, *run.receipt.activity_records[1:]),
    )
    with pytest.raises(InformationResearchOrchestrationError, match="run interval"):
        forged.validate()


def test_receipt_rejects_successful_fetch_without_exactly_one_source() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged_fetch = replace(run.receipt.activity_records[1], source_ids=())
    forged = replace(
        run.receipt,
        activity_records=(run.receipt.activity_records[0], forged_fetch, *run.receipt.activity_records[2:]),
    )
    with pytest.raises(InformationResearchOrchestrationError, match="exactly one source"):
        forged.validate()


def test_receipt_rejects_all_fetches_failed_without_attempts() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    with pytest.raises(InformationResearchOrchestrationError, match="every selected result"):
        InformationResearchRunReceipt.create(
            request_id=run.receipt.request_id,
            query_id=run.receipt.query_id,
            query_sha256=run.receipt.query_sha256,
            search_provider=run.receipt.search_provider,
            fetch_provider=run.receipt.fetch_provider,
            outcome="insufficient_sources",
            stopping_reason="all_fetches_failed",
            search_calls=1,
            fetch_calls=0,
            failed_fetch_calls=0,
            selected_result_ids=run.receipt.selected_result_ids,
            failed_result_ids=(),
            source_ids=(),
            source_content_sha256s=(),
            started_at=run.receipt.started_at,
            finished_at=run.receipt.finished_at,
            policy_version=run.receipt.policy_version,
            activity_records=(run.receipt.activity_records[0],),
        )


def test_run_rejects_duplicate_selected_ranks_after_receipt_creation() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    duplicate_rank = replace(run.search_results[1], rank=run.search_results[0].rank)
    forged = replace(
        run,
        search_results=(run.search_results[0], duplicate_rank, run.search_results[2]),
    )
    with pytest.raises(InformationResearchOrchestrationError, match="unique ranks"):
        forged.validate(policy=load_information_research_orchestration_policy())


def test_receipt_duplicate_activity_ids_are_rejected() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    activities = list(run.receipt.activity_records)
    activities[1] = replace(activities[1], activity_id=activities[0].activity_id)
    forged = replace(run.receipt, activity_records=tuple(activities))
    with pytest.raises(InformationResearchOrchestrationError, match="activity IDs"):
        forged.validate()


def test_receipt_duplicate_selected_result_ids_are_rejected() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(run.receipt, selected_result_ids=("result-a", "result-a"))
    with pytest.raises(InformationResearchOrchestrationError, match="duplicates"):
        forged.validate()


def test_receipt_source_digest_cardinality_is_enforced() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(run.receipt, source_content_sha256s=())
    with pytest.raises(InformationResearchOrchestrationError, match="equal length"):
        forged.validate()


def test_completed_receipt_cannot_omit_sources() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(
        run.receipt,
        source_ids=(),
        source_content_sha256s=(),
        fetch_calls=run.receipt.failed_fetch_calls,
    )
    with pytest.raises(InformationResearchOrchestrationError):
        forged.validate()


def test_run_rejects_source_outside_selected_results() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    outside = InformationSourceDocument.create(
        source_id="outside",
        provider=PROVIDER,
        url="https://example.com/outside",
        title="Outside",
        normalized_text="Outside",
        retrieved_at=NOW,
    )
    forged = replace(run, sources=(outside,))
    with pytest.raises(InformationResearchOrchestrationError, match="not selected"):
        forged.validate(policy=load_information_research_orchestration_policy())


def test_run_rejects_reordered_selected_results() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(run, search_results=tuple(reversed(run.search_results)))
    with pytest.raises(InformationResearchOrchestrationError, match="ordered"):
        forged.validate(policy=load_information_research_orchestration_policy())


def test_run_rejects_reordered_sources() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    forged = replace(run, sources=tuple(reversed(run.sources)))
    with pytest.raises(InformationResearchOrchestrationError):
        forged.validate(policy=load_information_research_orchestration_policy())


def test_metadata_record_contains_no_raw_query_or_source_text() -> None:
    run = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    record = run.receipt.to_metadata_record()
    serialized = repr(record)
    assert QUERY_TEXT not in serialized
    assert "Normalized source content" not in serialized
    assert run.request.query.content_sha256 in serialized
    assert run.sources[0].content_sha256 in serialized


def test_activity_records_are_terminal_sanitized_and_counted() -> None:
    run = _orchestrator(fetch=_fetch_provider(missing=(URLS[1],))).execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    assert all(record.status != "started" for record in run.receipt.activity_records)
    assert all(record.finished_at == NOW for record in run.receipt.activity_records)
    assert run.receipt.search_calls == 1
    assert run.receipt.fetch_calls == 3
    assert run.receipt.activity_records[2].error_code == "information_integrity_failed"


def test_provider_response_too_large_uses_approved_error_code() -> None:
    failure = InformationProviderExecutionError(
        InformationProviderFailure(
            provider=PROVIDER,
            operation="fetch",
            code="response_too_large",
            message="Information source exceeded the approved byte limit.",
            retryable=False,
        )
    )

    @dataclass
    class FailingFetch:
        provider: str = PROVIDER
        provider_type: str = "deterministic_fixture"

        def fetch(self, *args, **kwargs):
            raise failure

    run = _orchestrator(fetch=FailingFetch()).execute(
        _request(max_sources=1), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    assert run.receipt.activity_records[-1].error_code == "response_too_large"


def test_fetch_provider_cannot_bypass_response_byte_budget() -> None:
    @dataclass
    class OversizedFetch:
        provider: str = PROVIDER
        provider_type: str = "deterministic_fixture"

        def fetch(self, result, *, timeout_seconds, max_response_bytes, cancellation=None):
            return InformationSourceDocument.create(
                source_id="oversized-source",
                provider=self.provider,
                url=result.canonical_url,
                title="Oversized",
                normalized_text="x" * (max_response_bytes + 1),
                retrieved_at=NOW,
            )

    run = _orchestrator(fetch=OversizedFetch()).execute(
        _request(max_fetch_calls=1, max_sources=1),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert run.receipt.outcome == "insufficient_sources"
    assert run.receipt.stopping_reason == "all_fetches_failed"
    assert run.sources == ()
    assert run.receipt.failed_result_ids == ("result-a",)
    assert run.receipt.activity_records[-1].error_code == "response_too_large"


def test_unexpected_provider_exception_is_sanitized_without_retry() -> None:
    @dataclass
    class ExplodingFetch:
        provider: str = PROVIDER
        provider_type: str = "deterministic_fixture"
        calls: int = 0

        def fetch(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("secret provider detail")

    fetch = ExplodingFetch()
    run = _orchestrator(fetch=fetch).execute(
        _request(max_sources=1), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    assert fetch.calls == 1
    assert run.receipt.activity_records[-1].error_code == "information_integrity_failed"
    assert "secret provider detail" not in repr(run.receipt)


def test_same_inputs_and_times_produce_same_receipt_digest() -> None:
    first = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    second = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    assert first.receipt.run_id == second.receipt.run_id
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256


def test_activity_ids_are_deterministic() -> None:
    run = _orchestrator().execute(
        _request(max_sources=2), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    assert tuple(record.activity_id for record in run.receipt.activity_records) == (
        "research-request-p46a-001:search:1",
        "research-request-p46a-001:fetch:1",
        "research-request-p46a-001:fetch:2",
    )


def test_request_timeout_is_capped_by_remaining_total_time() -> None:
    search = CapturingSearch((_result(result_id="r1", rank=1, url=URLS[0]),))
    clock = StepClock([0.0, 40.0, 40.0, 40.0])
    _orchestrator(search=search, clock=clock).execute(
        _request(request_timeout_seconds=10, total_timeout_seconds=45, max_sources=1),
        search_provider=PROVIDER,
        fetch_provider=PROVIDER,
    )
    assert search.timeout_seen == 5


def test_registry_selection_does_not_mutate_registered_identities() -> None:
    orchestrator = _orchestrator()
    before = orchestrator.registry.identities()
    orchestrator.execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    assert orchestrator.registry.identities() == before


def test_orchestrator_rejects_policy_budget_above_selected_foundation() -> None:
    foundation = replace(load_information_policy(), max_sources=2)
    foundation.validate()
    registry = InformationProviderRegistry(
        information_policy=foundation,
        provider_policy=load_information_provider_policy(),
    )
    registry.register_search(_search_provider())
    registry.register_fetch(_fetch_provider())
    with pytest.raises(InformationResearchOrchestrationError, match="foundation policy"):
        InformationResearchOrchestrator(
            policy=load_information_research_orchestration_policy(),
            registry=registry,
            clock=lambda: 0.0,
            timestamp_factory=lambda: NOW,
        )


def test_run_id_changes_when_the_research_outcome_changes() -> None:
    complete = _orchestrator().execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    partial = _orchestrator(fetch=_fetch_provider(missing=(URLS[1],))).execute(
        _request(), search_provider=PROVIDER, fetch_provider=PROVIDER
    )
    assert complete.receipt.run_id != partial.receipt.run_id
    assert complete.receipt.receipt_sha256 != partial.receipt.receipt_sha256
