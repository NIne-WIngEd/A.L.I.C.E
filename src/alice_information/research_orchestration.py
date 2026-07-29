"""Deterministic, fixture-only research orchestration for Phase 4 P4.6a."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Callable

from .contracts import (
    InformationActivityRecord,
    InformationResearchRequest,
    InformationSearchResult,
    InformationSourceDocument,
    utc_now_text,
)
from .providers import (
    InformationCancellationToken,
    InformationProviderCancelledError,
    InformationProviderConfigurationError,
    InformationProviderExecutionError,
    InformationProviderProtocolError,
    InformationProviderTimeoutError,
)
from .registry import InformationProviderRegistry
from .research_orchestration_policy import InformationResearchOrchestrationPolicy

RESEARCH_RUN_OUTCOMES = (
    "completed",
    "partial",
    "insufficient_sources",
    "cancelled",
    "failed",
)
RESEARCH_STOPPING_REASONS = (
    "all_selected_sources_fetched",
    "search_returned_no_results",
    "search_failed",
    "all_fetches_failed",
    "partial_fetch_failure",
    "cancelled",
    "total_timeout",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class InformationResearchOrchestrationError(RuntimeError):
    """Raised when a P4.6a research run cannot be executed or verified."""

    def __init__(self, message: str, *, code: str = "information_integrity_failed") -> None:
        self.code = code
        super().__init__(message)


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchOrchestrationError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _require_digest(value: object, *, field: str) -> str:
    digest = _require_text(value, field=field).lower()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise InformationResearchOrchestrationError(
            f"{field} must be a lowercase SHA-256 digest."
        )
    return digest


def _parse_timestamp(value: object, *, field: str) -> datetime:
    text = _require_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InformationResearchOrchestrationError(
            f"{field} must be valid ISO-8601 text."
        ) from exc
    if parsed.tzinfo is None:
        raise InformationResearchOrchestrationError(
            f"{field} must include a timezone offset."
        )
    return parsed.astimezone(timezone.utc)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _activity_payload(record: InformationActivityRecord) -> dict[str, object]:
    return {
        "activity_id": record.activity_id,
        "request_id": record.request_id,
        "operation": record.operation,
        "provider": record.provider,
        "status": record.status,
        "started_at": record.started_at,
        "query_sha256": record.query_sha256,
        "finished_at": record.finished_at,
        "source_ids": list(record.source_ids),
        "error_code": record.error_code,
    }


def _receipt_digest_payload(receipt: "InformationResearchRunReceipt") -> dict[str, object]:
    return {
        "run_id": receipt.run_id,
        "request_id": receipt.request_id,
        "query_id": receipt.query_id,
        "query_sha256": receipt.query_sha256,
        "search_provider": receipt.search_provider,
        "fetch_provider": receipt.fetch_provider,
        "outcome": receipt.outcome,
        "stopping_reason": receipt.stopping_reason,
        "search_calls": receipt.search_calls,
        "fetch_calls": receipt.fetch_calls,
        "failed_fetch_calls": receipt.failed_fetch_calls,
        "selected_result_ids": list(receipt.selected_result_ids),
        "failed_result_ids": list(receipt.failed_result_ids),
        "source_ids": list(receipt.source_ids),
        "source_content_sha256s": list(receipt.source_content_sha256s),
        "started_at": receipt.started_at,
        "finished_at": receipt.finished_at,
        "policy_version": receipt.policy_version,
        "activity_records": [_activity_payload(item) for item in receipt.activity_records],
    }


def _run_id_payload(receipt: "InformationResearchRunReceipt") -> dict[str, object]:
    payload = _receipt_digest_payload(receipt)
    payload.pop("run_id")
    return payload


def _compute_run_id(receipt: "InformationResearchRunReceipt") -> str:
    digest = hashlib.sha256(
        _canonical_json(_run_id_payload(receipt)).encode("utf-8")
    ).hexdigest()
    return f"research-{digest[:20]}"


def _compute_receipt_digest(receipt: "InformationResearchRunReceipt") -> str:
    return hashlib.sha256(
        _canonical_json(_receipt_digest_payload(receipt)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class InformationResearchRunReceipt:
    """Metadata-safe, digest-bound record of one bounded P4.6a run."""

    run_id: str
    request_id: str
    query_id: str
    query_sha256: str
    search_provider: str
    fetch_provider: str
    outcome: str
    stopping_reason: str
    search_calls: int
    fetch_calls: int
    failed_fetch_calls: int
    selected_result_ids: tuple[str, ...]
    failed_result_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    source_content_sha256s: tuple[str, ...]
    started_at: str
    finished_at: str
    policy_version: str
    activity_records: tuple[InformationActivityRecord, ...]
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        query_id: str,
        query_sha256: str,
        search_provider: str,
        fetch_provider: str,
        outcome: str,
        stopping_reason: str,
        search_calls: int,
        fetch_calls: int,
        failed_fetch_calls: int,
        selected_result_ids: tuple[str, ...],
        failed_result_ids: tuple[str, ...],
        source_ids: tuple[str, ...],
        source_content_sha256s: tuple[str, ...],
        started_at: str,
        finished_at: str,
        policy_version: str,
        activity_records: tuple[InformationActivityRecord, ...],
    ) -> "InformationResearchRunReceipt":
        draft = cls(
            run_id="research-pending",
            request_id=request_id,
            query_id=query_id,
            query_sha256=query_sha256,
            search_provider=search_provider,
            fetch_provider=fetch_provider,
            outcome=outcome,
            stopping_reason=stopping_reason,
            search_calls=search_calls,
            fetch_calls=fetch_calls,
            failed_fetch_calls=failed_fetch_calls,
            selected_result_ids=selected_result_ids,
            failed_result_ids=failed_result_ids,
            source_ids=source_ids,
            source_content_sha256s=source_content_sha256s,
            started_at=started_at,
            finished_at=finished_at,
            policy_version=policy_version,
            activity_records=activity_records,
            receipt_sha256="0" * 64,
        )
        identified = cls(**{**draft.__dict__, "run_id": _compute_run_id(draft)})
        receipt = cls(
            **{
                **identified.__dict__,
                "receipt_sha256": _compute_receipt_digest(identified),
            }
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        _require_text(self.run_id, field="run_id")
        _require_text(self.request_id, field="request_id")
        _require_text(self.query_id, field="query_id")
        _require_digest(self.query_sha256, field="query_sha256")
        _require_text(self.search_provider, field="search_provider")
        _require_text(self.fetch_provider, field="fetch_provider")
        if self.outcome not in RESEARCH_RUN_OUTCOMES:
            raise InformationResearchOrchestrationError(
                "Research receipt outcome is not recognized."
            )
        if self.stopping_reason not in RESEARCH_STOPPING_REASONS:
            raise InformationResearchOrchestrationError(
                "Research receipt stopping reason is not recognized."
            )
        allowed_stops = {
            "completed": {"all_selected_sources_fetched"},
            "partial": {"partial_fetch_failure", "cancelled", "total_timeout"},
            "insufficient_sources": {
                "search_returned_no_results",
                "all_fetches_failed",
            },
            "cancelled": {"cancelled"},
            "failed": {"search_failed", "total_timeout"},
        }
        if self.stopping_reason not in allowed_stops[self.outcome]:
            raise InformationResearchOrchestrationError(
                "Research outcome and stopping reason are inconsistent."
            )
        for field_name, value in (
            ("search_calls", self.search_calls),
            ("fetch_calls", self.fetch_calls),
            ("failed_fetch_calls", self.failed_fetch_calls),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InformationResearchOrchestrationError(
                    f"{field_name} must be a non-negative integer."
                )
        if self.search_calls not in {0, 1}:
            raise InformationResearchOrchestrationError(
                "P4.6a permits at most one search call."
            )
        if self.failed_fetch_calls > self.fetch_calls:
            raise InformationResearchOrchestrationError(
                "Failed fetch calls cannot exceed total fetch calls."
            )
        collections = (
            ("selected_result_ids", self.selected_result_ids),
            ("failed_result_ids", self.failed_result_ids),
            ("source_ids", self.source_ids),
        )
        for field_name, values in collections:
            if len(set(values)) != len(values):
                raise InformationResearchOrchestrationError(
                    f"{field_name} cannot contain duplicates."
                )
            for value in values:
                _require_text(value, field=field_name)
        if len(self.source_ids) != len(self.source_content_sha256s):
            raise InformationResearchOrchestrationError(
                "Source IDs and source digests must have equal length."
            )
        for digest in self.source_content_sha256s:
            _require_digest(digest, field="source_content_sha256")
        if self.fetch_calls != len(self.source_ids) + self.failed_fetch_calls:
            raise InformationResearchOrchestrationError(
                "Fetch counters must equal successful plus failed fetches."
            )
        if len(self.failed_result_ids) != self.failed_fetch_calls:
            raise InformationResearchOrchestrationError(
                "Failed result IDs must match failed fetch count."
            )
        if self.fetch_calls > len(self.selected_result_ids):
            raise InformationResearchOrchestrationError(
                "Fetch calls cannot exceed selected search results."
            )
        started = _parse_timestamp(self.started_at, field="started_at")
        finished = _parse_timestamp(self.finished_at, field="finished_at")
        if finished < started:
            raise InformationResearchOrchestrationError(
                "Research receipt cannot finish before it starts."
            )
        if self.policy_version != "1.0.0":
            raise InformationResearchOrchestrationError(
                "Research receipt must bind P4.6a policy version 1.0.0."
            )
        activity_ids: set[str] = set()
        search_activities = 0
        fetch_activities = 0
        activity_source_ids: list[str] = []
        for record in self.activity_records:
            record.validate()
            if record.status not in {"succeeded", "failed", "cancelled"}:
                raise InformationResearchOrchestrationError(
                    "Research receipt activities must be terminal orchestration records."
                )
            if record.activity_id in activity_ids:
                raise InformationResearchOrchestrationError(
                    "Research receipt activity IDs must be unique."
                )
            activity_ids.add(record.activity_id)
            if record.request_id != self.request_id:
                raise InformationResearchOrchestrationError(
                    "Activity request binding does not match the receipt."
                )
            if record.query_sha256 != self.query_sha256:
                raise InformationResearchOrchestrationError(
                    "Activity query digest does not match the receipt."
                )
            activity_started = _parse_timestamp(
                record.started_at, field="activity.started_at"
            )
            activity_finished = _parse_timestamp(
                record.finished_at, field="activity.finished_at"
            )
            if activity_started < started or activity_finished > finished:
                raise InformationResearchOrchestrationError(
                    "Research activity timestamps must remain inside the run interval."
                )
            if record.operation == "search":
                search_activities += 1
                if record.provider != self.search_provider:
                    raise InformationResearchOrchestrationError(
                        "Search activity provider does not match the receipt."
                    )
                if record.source_ids:
                    raise InformationResearchOrchestrationError(
                        "Search activity records cannot contain source IDs."
                    )
            else:
                fetch_activities += 1
                if record.provider != self.fetch_provider:
                    raise InformationResearchOrchestrationError(
                        "Fetch activity provider does not match the receipt."
                    )
                if record.status == "succeeded" and len(record.source_ids) != 1:
                    raise InformationResearchOrchestrationError(
                        "Successful fetch activity requires exactly one source ID."
                    )
                if record.status != "succeeded" and record.source_ids:
                    raise InformationResearchOrchestrationError(
                        "Failed or cancelled fetch activity cannot contain source IDs."
                    )
                activity_source_ids.extend(record.source_ids)
        failed_fetch_activities = sum(
            record.operation == "fetch" and record.status != "succeeded"
            for record in self.activity_records
        )
        if failed_fetch_activities != self.failed_fetch_calls:
            raise InformationResearchOrchestrationError(
                "Failed fetch activity count does not match the receipt."
            )
        if search_activities != self.search_calls or fetch_activities != self.fetch_calls:
            raise InformationResearchOrchestrationError(
                "Activity counts do not match the research receipt counters."
            )
        expected_activity_ids: list[str] = []
        if self.search_calls:
            expected_activity_ids.append(f"{self.request_id}:search:1")
        expected_activity_ids.extend(
            f"{self.request_id}:fetch:{index}"
            for index in range(1, self.fetch_calls + 1)
        )
        if [record.activity_id for record in self.activity_records] != expected_activity_ids:
            raise InformationResearchOrchestrationError(
                "Research activity order or deterministic identity does not match the receipt."
            )
        expected_operations = (["search"] if self.search_calls else []) + [
            "fetch"
        ] * self.fetch_calls
        if [record.operation for record in self.activity_records] != expected_operations:
            raise InformationResearchOrchestrationError(
                "Research activity operation order does not match the receipt."
            )
        search_records = tuple(
            record for record in self.activity_records if record.operation == "search"
        )
        if self.selected_result_ids:
            if len(search_records) != 1 or search_records[0].status != "succeeded":
                raise InformationResearchOrchestrationError(
                    "Selected results require one successful search activity."
                )
        elif self.search_calls == 0:
            if self.outcome != "cancelled":
                raise InformationResearchOrchestrationError(
                    "A zero-search run must be cancelled before execution."
                )
        elif self.stopping_reason == "search_returned_no_results":
            if search_records[0].status != "succeeded":
                raise InformationResearchOrchestrationError(
                    "An empty search result set still requires a successful search activity."
                )
        elif self.stopping_reason == "cancelled":
            if search_records[0].status != "cancelled":
                raise InformationResearchOrchestrationError(
                    "Search cancellation requires a cancelled search activity."
                )
        elif search_records[0].status != "failed":
            raise InformationResearchOrchestrationError(
                "Failed search termination requires a failed search activity."
            )
        if tuple(activity_source_ids) != self.source_ids:
            raise InformationResearchOrchestrationError(
                "Successful fetch activity sources do not match the receipt."
            )
        if self.stopping_reason == "all_selected_sources_fetched":
            if (
                not self.selected_result_ids
                or self.fetch_calls != len(self.selected_result_ids)
                or self.failed_fetch_calls != 0
                or len(self.source_ids) != self.fetch_calls
            ):
                raise InformationResearchOrchestrationError(
                    "Completed stopping metadata does not match the selected fetch set."
                )
        if self.stopping_reason == "search_returned_no_results" and (
            self.selected_result_ids or self.fetch_calls
        ):
            raise InformationResearchOrchestrationError(
                "Empty-search stopping metadata cannot contain selected or fetched results."
            )
        if self.stopping_reason == "all_fetches_failed" and (
            not self.selected_result_ids
            or self.fetch_calls != len(self.selected_result_ids)
            or self.failed_fetch_calls != self.fetch_calls
        ):
            raise InformationResearchOrchestrationError(
                "All-fetch-failed metadata must bind every selected result."
            )
        if self.stopping_reason == "partial_fetch_failure" and (
            not self.source_ids
            or self.failed_fetch_calls == 0
            or self.fetch_calls != len(self.selected_result_ids)
        ):
            raise InformationResearchOrchestrationError(
                "Partial-fetch metadata must bind successful and failed selected results."
            )
        if self.outcome == "completed" and not self.source_ids:
            raise InformationResearchOrchestrationError(
                "Completed research requires at least one source."
            )
        if self.outcome == "partial" and not self.source_ids:
            raise InformationResearchOrchestrationError(
                "Partial research must preserve at least one source."
            )
        if self.outcome in {"cancelled", "failed", "insufficient_sources"} and self.source_ids:
            raise InformationResearchOrchestrationError(
                f"{self.outcome} research cannot contain sources."
            )
        if self.run_id != _compute_run_id(self):
            raise InformationResearchOrchestrationError(
                "Research run ID does not match the complete receipt metadata."
            )
        digest = _require_digest(self.receipt_sha256, field="receipt_sha256")
        if _compute_receipt_digest(self) != digest:
            raise InformationResearchOrchestrationError(
                "Research receipt digest does not match its metadata."
            )

    def to_metadata_record(self) -> dict[str, object]:
        """Return a raw-content-free receipt suitable for later ledgers."""

        self.validate()
        return {**_receipt_digest_payload(self), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class InformationResearchRun:
    """Selected search results, fetched sources, and a verified run receipt."""

    request: InformationResearchRequest
    search_results: tuple[InformationSearchResult, ...]
    sources: tuple[InformationSourceDocument, ...]
    receipt: InformationResearchRunReceipt

    def validate(self, *, policy: InformationResearchOrchestrationPolicy) -> None:
        self.request.validate()
        policy.validate_request_budget(self.request)
        self.receipt.validate()
        if self.request.operations != ("search", "fetch"):
            raise InformationResearchOrchestrationError(
                "P4.6a research requests must use search followed by fetch."
            )
        if self.receipt.request_id != self.request.request_id:
            raise InformationResearchOrchestrationError(
                "Research receipt request ID does not match."
            )
        if self.receipt.query_id != self.request.query.query_id:
            raise InformationResearchOrchestrationError(
                "Research receipt query ID does not match."
            )
        if self.receipt.query_sha256 != self.request.query.content_sha256:
            raise InformationResearchOrchestrationError(
                "Research receipt query digest does not match."
            )
        if self.receipt.search_calls > self.request.max_search_calls:
            raise InformationResearchOrchestrationError(
                "Research receipt exceeds the search-call budget."
            )
        if self.receipt.fetch_calls > self.request.max_fetch_calls:
            raise InformationResearchOrchestrationError(
                "Research receipt exceeds the fetch-call budget."
            )
        if len(self.sources) > self.request.max_sources:
            raise InformationResearchOrchestrationError(
                "Research run exceeds the source budget."
            )
        selection_cap = min(
            self.request.max_sources, self.request.max_fetch_calls
        )
        if len(self.search_results) > selection_cap:
            raise InformationResearchOrchestrationError(
                "Research run exceeds the selected-result budget."
            )
        result_ids: set[str] = set()
        result_urls: set[str] = set()
        result_ranks: set[int] = set()
        result_by_url: dict[str, InformationSearchResult] = {}
        previous_key: tuple[int, str, str] | None = None
        for result in self.search_results:
            result.validate()
            if result.provider != self.receipt.search_provider:
                raise InformationResearchOrchestrationError(
                    "Search result provider does not match the selected provider."
                )
            if result.query_id != self.request.query.query_id:
                raise InformationResearchOrchestrationError(
                    "Search result query binding does not match."
                )
            if (
                result.result_id in result_ids
                or result.canonical_url in result_urls
                or result.rank in result_ranks
            ):
                raise InformationResearchOrchestrationError(
                    "Selected search results must have unique ranks, IDs, and canonical URLs."
                )
            result_ids.add(result.result_id)
            result_urls.add(result.canonical_url)
            result_ranks.add(result.rank)
            result_by_url[result.canonical_url] = result
            key = (result.rank, result.canonical_url, result.result_id)
            if previous_key is not None and key < previous_key:
                raise InformationResearchOrchestrationError(
                    "Selected search results are not deterministically ordered."
                )
            previous_key = key
        if tuple(item.result_id for item in self.search_results) != self.receipt.selected_result_ids:
            raise InformationResearchOrchestrationError(
                "Selected result IDs do not match the research receipt."
            )
        source_ids: set[str] = set()
        source_urls: set[str] = set()
        source_positions: list[int] = []
        selected_position = {
            result.canonical_url: index for index, result in enumerate(self.search_results)
        }
        for source in self.sources:
            source.validate()
            if source.provider != self.receipt.fetch_provider:
                raise InformationResearchOrchestrationError(
                    "Fetched source provider does not match the selected provider."
                )
            if source.canonical_url not in result_by_url:
                raise InformationResearchOrchestrationError(
                    "Fetched source was not selected from the search results."
                )
            if source.source_id in source_ids or source.canonical_url in source_urls:
                raise InformationResearchOrchestrationError(
                    "Fetched sources must have unique IDs and canonical URLs."
                )
            source_ids.add(source.source_id)
            source_urls.add(source.canonical_url)
            source_positions.append(selected_position[source.canonical_url])
        if source_positions != sorted(source_positions):
            raise InformationResearchOrchestrationError(
                "Fetched sources must preserve selected-result order."
            )
        if tuple(source.source_id for source in self.sources) != self.receipt.source_ids:
            raise InformationResearchOrchestrationError(
                "Source IDs do not match the research receipt."
            )
        if (
            tuple(source.content_sha256 for source in self.sources)
            != self.receipt.source_content_sha256s
        ):
            raise InformationResearchOrchestrationError(
                "Source digests do not match the research receipt."
            )
        attempted = self.search_results[: self.receipt.fetch_calls]
        expected_failed_result_ids = tuple(
            result.result_id
            for result in attempted
            if result.canonical_url not in source_urls
        )
        if expected_failed_result_ids != self.receipt.failed_result_ids:
            raise InformationResearchOrchestrationError(
                "Failed result IDs do not match the exact attempted result sequence."
            )


class InformationResearchOrchestrator:
    """Execute one bounded foreground search-and-fetch fixture run."""

    def __init__(
        self,
        *,
        policy: InformationResearchOrchestrationPolicy,
        registry: InformationProviderRegistry,
        clock: Callable[[], float] = monotonic,
        timestamp_factory: Callable[[], str] = utc_now_text,
    ) -> None:
        policy.validate()
        registry.information_policy.validate()
        registry.provider_policy.validate()
        foundation = registry.information_policy
        for field_name, value, maximum in (
            ("max_fetch_calls", policy.max_fetch_calls, foundation.max_fetch_calls),
            ("max_sources", policy.max_sources, foundation.max_sources),
            ("max_response_bytes", policy.max_response_bytes, foundation.max_response_bytes),
            (
                "max_request_timeout_seconds",
                policy.max_request_timeout_seconds,
                foundation.request_timeout_seconds,
            ),
            (
                "max_total_timeout_seconds",
                policy.max_total_timeout_seconds,
                foundation.total_timeout_seconds,
            ),
        ):
            if value > maximum:
                raise InformationResearchOrchestrationError(
                    f"P4.6a {field_name} exceeds the selected foundation policy."
                )
        self.policy = policy
        self.registry = registry
        self.clock = clock
        self.timestamp_factory = timestamp_factory

    def _timestamp(self) -> str:
        value = self.timestamp_factory()
        _parse_timestamp(value, field="timestamp_factory result")
        return value

    @staticmethod
    def _error_code(exc: Exception) -> str:
        if isinstance(exc, InformationProviderCancelledError):
            return "research_cancelled"
        if isinstance(exc, InformationResearchOrchestrationError):
            if exc.code in {"research_budget_exhausted", "response_too_large"}:
                return exc.code
            return "information_integrity_failed"
        if isinstance(exc, InformationProviderTimeoutError):
            return "provider_timeout"
        if isinstance(exc, InformationProviderExecutionError):
            return {
                "provider_timeout": "provider_timeout",
                "response_too_large": "response_too_large",
            }.get(exc.failure.code, "information_integrity_failed")
        return "information_integrity_failed"

    def _activity(
        self,
        *,
        request: InformationResearchRequest,
        activity_id: str,
        operation: str,
        provider: str,
        status: str,
        started_at: str,
        source_ids: tuple[str, ...] = (),
        error_code: str | None = None,
    ) -> InformationActivityRecord:
        record = InformationActivityRecord(
            activity_id=activity_id,
            request_id=request.request_id,
            operation=operation,
            provider=provider,
            status=status,
            started_at=started_at,
            query_sha256=request.query.content_sha256,
            finished_at=self._timestamp(),
            source_ids=source_ids,
            error_code=error_code,
        )
        record.validate()
        return record

    def _build_run(
        self,
        *,
        request: InformationResearchRequest,
        search_provider: str,
        fetch_provider: str,
        outcome: str,
        stopping_reason: str,
        selected: tuple[InformationSearchResult, ...],
        sources: tuple[InformationSourceDocument, ...],
        failed_result_ids: tuple[str, ...],
        activities: tuple[InformationActivityRecord, ...],
        started_at: str,
    ) -> InformationResearchRun:
        receipt = InformationResearchRunReceipt.create(
            request_id=request.request_id,
            query_id=request.query.query_id,
            query_sha256=request.query.content_sha256,
            search_provider=search_provider,
            fetch_provider=fetch_provider,
            outcome=outcome,
            stopping_reason=stopping_reason,
            search_calls=sum(item.operation == "search" for item in activities),
            fetch_calls=sum(item.operation == "fetch" for item in activities),
            failed_fetch_calls=len(failed_result_ids),
            selected_result_ids=tuple(item.result_id for item in selected),
            failed_result_ids=failed_result_ids,
            source_ids=tuple(item.source_id for item in sources),
            source_content_sha256s=tuple(item.content_sha256 for item in sources),
            started_at=started_at,
            finished_at=self._timestamp(),
            policy_version=self.policy.version,
            activity_records=activities,
        )
        run = InformationResearchRun(
            request=request,
            search_results=selected,
            sources=sources,
            receipt=receipt,
        )
        run.validate(policy=self.policy)
        return run

    def execute(
        self,
        request: InformationResearchRequest,
        *,
        search_provider: str,
        fetch_provider: str,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationResearchRun:
        request.validate()
        self.policy.validate_request_budget(request)
        if request.operations != ("search", "fetch"):
            raise InformationResearchOrchestrationError(
                "P4.6a requires the exact operation sequence search, fetch."
            )
        search = self.registry.resolve_search(provider=search_provider)
        fetch = self.registry.resolve_fetch(provider=fetch_provider)
        if self.policy.deterministic_fixture_only and (
            getattr(search, "provider_type", None) != "deterministic_fixture"
            or getattr(fetch, "provider_type", None) != "deterministic_fixture"
        ):
            raise InformationResearchOrchestrationError(
                "P4.6a permits deterministic fixture providers only."
            )
        started_tick = self.clock()
        started_at = self._timestamp()
        activities: list[InformationActivityRecord] = []

        def elapsed() -> float:
            return max(0.0, float(self.clock() - started_tick))

        def remaining_timeout() -> float:
            remaining = request.total_timeout_seconds - elapsed()
            if remaining <= 0:
                raise InformationResearchOrchestrationError(
                    "Research total timeout was exhausted.",
                    code="research_budget_exhausted",
                )
            return min(request.request_timeout_seconds, remaining)

        if cancellation is not None and cancellation.cancelled:
            return self._build_run(
                request=request,
                search_provider=search_provider,
                fetch_provider=fetch_provider,
                outcome="cancelled",
                stopping_reason="cancelled",
                selected=(),
                sources=(),
                failed_result_ids=(),
                activities=(),
                started_at=started_at,
            )

        search_started = self._timestamp()
        try:
            timeout = remaining_timeout()
            results = search.search(
                request.query,
                max_results=min(request.max_sources, request.max_fetch_calls),
                timeout_seconds=timeout,
                cancellation=cancellation,
            )
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if elapsed() > request.total_timeout_seconds:
                raise InformationResearchOrchestrationError(
                    "Research total timeout was exhausted.",
                    code="research_budget_exhausted",
                )
        except InformationResearchOrchestrationError as exc:
            activities.append(
                self._activity(
                    request=request,
                    activity_id=f"{request.request_id}:search:1",
                    operation="search",
                    provider=search_provider,
                    status="failed",
                    started_at=search_started,
                    error_code=exc.code,
                )
            )
            return self._build_run(
                request=request,
                search_provider=search_provider,
                fetch_provider=fetch_provider,
                outcome="failed",
                stopping_reason="total_timeout",
                selected=(),
                sources=(),
                failed_result_ids=(),
                activities=tuple(activities),
                started_at=started_at,
            )
        except InformationProviderCancelledError as exc:
            activities.append(
                self._activity(
                    request=request,
                    activity_id=f"{request.request_id}:search:1",
                    operation="search",
                    provider=search_provider,
                    status="cancelled",
                    started_at=search_started,
                    error_code=self._error_code(exc),
                )
            )
            return self._build_run(
                request=request,
                search_provider=search_provider,
                fetch_provider=fetch_provider,
                outcome="cancelled",
                stopping_reason="cancelled",
                selected=(),
                sources=(),
                failed_result_ids=(),
                activities=tuple(activities),
                started_at=started_at,
            )
        except Exception as exc:
            activities.append(
                self._activity(
                    request=request,
                    activity_id=f"{request.request_id}:search:1",
                    operation="search",
                    provider=search_provider,
                    status="failed",
                    started_at=search_started,
                    error_code=self._error_code(exc),
                )
            )
            return self._build_run(
                request=request,
                search_provider=search_provider,
                fetch_provider=fetch_provider,
                outcome="failed",
                stopping_reason="search_failed",
                selected=(),
                sources=(),
                failed_result_ids=(),
                activities=tuple(activities),
                started_at=started_at,
            )

        try:
            result_limit = min(request.max_sources, request.max_fetch_calls)
            returned_results = tuple(results)
            if len(returned_results) > result_limit:
                raise InformationResearchOrchestrationError(
                    "Search provider exceeded the approved result budget."
                )
            sorted_results = sorted(
                returned_results,
                key=lambda item: (item.rank, item.canonical_url, item.result_id),
            )
            ranks: set[int] = set()
            result_ids: set[str] = set()
            urls: set[str] = set()
            selected_list: list[InformationSearchResult] = []
            for result in sorted_results:
                result.validate()
                if (
                    result.provider != search_provider
                    or result.query_id != request.query.query_id
                ):
                    raise InformationResearchOrchestrationError(
                        "Search provider changed result identity."
                    )
                if result.rank in ranks or result.result_id in result_ids:
                    raise InformationResearchOrchestrationError(
                        "Search results contain duplicate rank or result identity."
                    )
                ranks.add(result.rank)
                result_ids.add(result.result_id)
                if result.canonical_url in urls:
                    continue
                urls.add(result.canonical_url)
                selected_list.append(result)
                if len(selected_list) >= min(
                    request.max_sources, request.max_fetch_calls
                ):
                    break
            selected = tuple(selected_list)
        except Exception:
            activities.append(
                self._activity(
                    request=request,
                    activity_id=f"{request.request_id}:search:1",
                    operation="search",
                    provider=search_provider,
                    status="failed",
                    started_at=search_started,
                    error_code="information_integrity_failed",
                )
            )
            return self._build_run(
                request=request,
                search_provider=search_provider,
                fetch_provider=fetch_provider,
                outcome="failed",
                stopping_reason="search_failed",
                selected=(),
                sources=(),
                failed_result_ids=(),
                activities=tuple(activities),
                started_at=started_at,
            )
        activities.append(
            self._activity(
                request=request,
                activity_id=f"{request.request_id}:search:1",
                operation="search",
                provider=search_provider,
                status="succeeded",
                started_at=search_started,
            )
        )
        if not selected:
            return self._build_run(
                request=request,
                search_provider=search_provider,
                fetch_provider=fetch_provider,
                outcome="insufficient_sources",
                stopping_reason="search_returned_no_results",
                selected=(),
                sources=(),
                failed_result_ids=(),
                activities=tuple(activities),
                started_at=started_at,
            )

        sources: list[InformationSourceDocument] = []
        failed_result_ids: list[str] = []
        stop_reason: str | None = None
        for index, result in enumerate(selected, start=1):
            if cancellation is not None and cancellation.cancelled:
                stop_reason = "cancelled"
                break
            try:
                timeout = remaining_timeout()
            except InformationResearchOrchestrationError:
                stop_reason = "total_timeout"
                break
            fetch_started = self._timestamp()
            try:
                source = fetch.fetch(
                    result,
                    timeout_seconds=timeout,
                    max_response_bytes=self.policy.max_response_bytes,
                    cancellation=cancellation,
                )
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                if elapsed() > request.total_timeout_seconds:
                    raise InformationResearchOrchestrationError(
                        "Research total timeout was exhausted.",
                        code="research_budget_exhausted",
                    )
                source.validate()
                if len(source.normalized_text.encode("utf-8")) > self.policy.max_response_bytes:
                    raise InformationResearchOrchestrationError(
                        "Fetched source exceeded the approved byte limit.",
                        code="response_too_large",
                    )
                if source.provider != fetch_provider:
                    raise InformationProviderProtocolError(
                        "Fetch provider changed source identity."
                    )
                if source.canonical_url != result.canonical_url:
                    raise InformationProviderProtocolError(
                        "Fetch provider returned a source for an unselected URL."
                    )
                if any(
                    existing.source_id == source.source_id
                    or existing.canonical_url == source.canonical_url
                    for existing in sources
                ):
                    raise InformationProviderProtocolError(
                        "Fetch provider returned a duplicate source identity."
                    )
                sources.append(source)
                activities.append(
                    self._activity(
                        request=request,
                        activity_id=f"{request.request_id}:fetch:{index}",
                        operation="fetch",
                        provider=fetch_provider,
                        status="succeeded",
                        started_at=fetch_started,
                        source_ids=(source.source_id,),
                    )
                )
            except InformationProviderCancelledError as exc:
                failed_result_ids.append(result.result_id)
                activities.append(
                    self._activity(
                        request=request,
                        activity_id=f"{request.request_id}:fetch:{index}",
                        operation="fetch",
                        provider=fetch_provider,
                        status="cancelled",
                        started_at=fetch_started,
                        error_code=self._error_code(exc),
                    )
                )
                stop_reason = "cancelled"
                break
            except InformationResearchOrchestrationError as exc:
                failed_result_ids.append(result.result_id)
                activities.append(
                    self._activity(
                        request=request,
                        activity_id=f"{request.request_id}:fetch:{index}",
                        operation="fetch",
                        provider=fetch_provider,
                        status="failed",
                        started_at=fetch_started,
                        error_code=self._error_code(exc),
                    )
                )
                if exc.code == "research_budget_exhausted":
                    stop_reason = "total_timeout"
                    break
                continue
            except Exception as exc:
                failed_result_ids.append(result.result_id)
                activities.append(
                    self._activity(
                        request=request,
                        activity_id=f"{request.request_id}:fetch:{index}",
                        operation="fetch",
                        provider=fetch_provider,
                        status="failed",
                        started_at=fetch_started,
                        error_code=self._error_code(exc),
                    )
                )
                continue

        if sources and (failed_result_ids or stop_reason is not None):
            outcome = "partial"
            stopping_reason = stop_reason or "partial_fetch_failure"
        elif sources:
            outcome = "completed"
            stopping_reason = "all_selected_sources_fetched"
        elif stop_reason == "cancelled":
            outcome = "cancelled"
            stopping_reason = "cancelled"
        elif stop_reason == "total_timeout":
            outcome = "failed"
            stopping_reason = "total_timeout"
        else:
            outcome = "insufficient_sources"
            stopping_reason = "all_fetches_failed"
        return self._build_run(
            request=request,
            search_provider=search_provider,
            fetch_provider=fetch_provider,
            outcome=outcome,
            stopping_reason=stopping_reason,
            selected=selected,
            sources=tuple(sources),
            failed_result_ids=tuple(failed_result_ids),
            activities=tuple(activities),
            started_at=started_at,
        )
