"""Exact P4.10a fetch adapter over LiveControlledInformationHttpRetriever."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Callable
from urllib.parse import urlsplit

from .contracts import InformationSearchResult, InformationSourceDocument
from .live_provider_contracts import (
    InformationLiveEgressReceipt,
    InformationLiveFetchResponse,
    InformationLiveRateLimitState,
    sequence_sha256,
)
from .live_provider_policy import InformationLiveProviderRuntimePolicy
from .providers import InformationCancellationToken, validate_provider_identity
from .retrieval import LiveControlledInformationHttpRetriever


class InformationLiveFetchProviderError(ValueError):
    """Raised when the exact P4.10a live fetch boundary is substituted."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass
class LiveControlledInformationFetchProvider:
    """One credential-free page fetch through the exact P4.2b live retriever."""

    policy: InformationLiveProviderRuntimePolicy
    retriever: LiveControlledInformationHttpRetriever
    clock: Callable[[], str] = _now
    monotonic_clock: Callable[[], float] = monotonic
    provider: str = field(default="controlled-live-http-v1", init=False)
    provider_type: str = field(default="live", init=False)
    last_response: InformationLiveFetchResponse | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.policy.validate()
        if validate_provider_identity(self.provider) != self.policy.fetch_provider_id:
            raise InformationLiveFetchProviderError(
                "Live fetch provider identity changed."
            )
        if type(self.retriever) is not LiveControlledInformationHttpRetriever:
            raise InformationLiveFetchProviderError(
                "P4.10a requires the exact LiveControlledInformationHttpRetriever."
            )
        if not callable(self.clock) or not callable(self.monotonic_clock):
            raise InformationLiveFetchProviderError(
                "Live fetch provider clocks must be callable."
            )

    def fetch(
        self,
        result: InformationSearchResult,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationSourceDocument:
        """Provider-protocol compatibility method.

        The protocol does not expose the original query digest. P4.10b always
        calls ``fetch_with_receipt`` with the exact query digest instead.
        """

        return self.fetch_with_receipt(
            result,
            query_sha256=result.content_sha256,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            cancellation=cancellation,
        ).source_document  # type: ignore[return-value]

    def fetch_with_receipt(
        self,
        result: InformationSearchResult,
        *,
        query_sha256: str,
        timeout_seconds: float,
        max_response_bytes: int,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationLiveFetchResponse:
        result.validate()
        if result.provider != self.policy.search_provider_id:
            raise InformationLiveFetchProviderError(
                "Live fetch accepts only results from the exact live search provider."
            )
        if not isinstance(query_sha256, str) or len(query_sha256) != 64:
            raise InformationLiveFetchProviderError(
                "Live fetch requires the exact original query SHA-256 digest."
            )
        try:
            int(query_sha256, 16)
        except ValueError as exc:
            raise InformationLiveFetchProviderError(
                "Live fetch query digest is invalid."
            ) from exc
        parsed_result = urlsplit(result.canonical_url)
        if parsed_result.scheme != "https":
            raise InformationLiveFetchProviderError(
                "Initial P4.10 live fetch requires HTTPS sources."
            )
        retrieval_policy = self.retriever.retrieval_policy
        if float(timeout_seconds) != float(retrieval_policy.request_timeout_seconds):
            raise InformationLiveFetchProviderError(
                "Fetch timeout must match the exact controlled retrieval policy."
            )
        if max_response_bytes != retrieval_policy.max_decoded_bytes:
            raise InformationLiveFetchProviderError(
                "Fetch byte budget must match the exact controlled retrieval policy."
            )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        started_at = self.clock()
        started_monotonic = self.monotonic_clock()
        resource = self.retriever.retrieve(
            result.canonical_url,
            cancellation=cancellation,
        )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        completed_at = self.clock()
        source_id = "live-source-" + hashlib.sha256(
            (
                f"{result.result_id}\n{resource.final_url}\n"
                f"{resource.content_sha256}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        source = resource.to_source_document(
            source_id=source_id,
            provider=self.provider,
            retrieved_at=completed_at,
        )
        source.validate()
        final = urlsplit(resource.final_url)
        receipt = InformationLiveEgressReceipt.create(
            provider=self.provider,
            operation="fetch",
            endpoint_host=final.hostname or "unknown",
            endpoint_path=final.path or "/",
            query_id=result.query_id,
            query_sha256=query_sha256.lower(),
            configuration_sha256=self.policy.policy_sha256,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_milliseconds=max(
                0,
                int((self.monotonic_clock() - started_monotonic) * 1000),
            ),
            status_code=resource.status_code,
            peer_address=resource.peer_address,
            item_count=1,
            response_sha256=resource.content_sha256,
            item_sequence_sha256=sequence_sha256([source.content_sha256]),
            rate_limit=InformationLiveRateLimitState(None, None, None, None),
            policy_binding=self.policy.binding,
        )
        response = InformationLiveFetchResponse(
            search_result=result,
            source_document=source,
            resource=resource,
            receipt=receipt,
        )
        response.validate()
        self.last_response = response
        return response
