"""P4.10b exact live PUBLIC research path through P3.6 and P4.5b.

This module is additive. It does not alter the frozen P4.6a, P4.7a, or P4.7b
fixture profiles. One explicit foreground research turn performs exactly one
Brave search, bounded credential-free public-page fetches, deterministic
inspection/freshness/grounding, Phase 3 projection, and pre-commit response
validation. No retry, fallback, persistence, memory write, action, recursive
browse, or background execution is available here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from time import monotonic
from typing import Callable

from alice_conversation.contracts import (
    ConversationContractError,
    ConversationGroundingPacket,
    ModelResponse,
)
from alice_conversation.orchestration import ConversationTurnCommand, ConversationTurnResult
from alice_conversation.response_validation_policy import ConversationResponseValidationPolicy

from .conversation_bridge import (
    InformationConversationGroundingProjection,
    InformationConversationResponseValidation,
    project_information_grounding_to_conversation,
    validate_information_conversation_response,
)
from .conversation_bridge_policy import InformationConversationBridgePolicy
from .contracts import InformationResearchRequest
from .freshness import (
    DeterministicInformationFreshnessEvaluator,
    DeterministicInformationTemporalClassifier,
    InformationTemporallyQualifiedSource,
)
from .freshness_policy import InformationFreshnessPolicy
from .grounding import (
    DeterministicInformationGroundingBuilder,
    InformationVerifiedGroundingPacket,
)
from .grounding_policy import InformationGroundingPolicy
from .injection_firewall import (
    DeterministicInformationInjectionFirewall,
    InformationInspectedSource,
)
from .injection_policy import InformationInjectionFirewallPolicy
from .live_claims import DeterministicLiveExtractiveClaimPlanner
from .live_provider_contracts import (
    InformationLiveFetchResponse,
    InformationLiveSearchResponse,
    canonical_sha256,
    sequence_sha256,
)
from .live_provider_registry import ExactInformationLiveProviderRegistry
from .live_research_policy import InformationLiveResearchPolicy
from .policy import InformationPolicy
from .temporal_metadata import (
    DeterministicInformationTemporalMetadataResolver,
    InformationResolvedTemporalResource,
    InformationTemporalMetadataError,
)
from .temporal_metadata_policy import InformationTemporalMetadataPolicy

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_DISPOSITIONS = {
    "grounded",
    "eligible_not_selected",
    "blocked_injection",
    "temporal_rejected",
    "freshness_rejected",
}


class InformationLiveResearchError(RuntimeError):
    """Sanitized P4.10b execution or binding failure."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


def _text(value: object, field: str, *, maximum: int = 2048) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise InformationLiveResearchError(
            f"{field} is invalid.", code="live_research_binding_invalid"
        )
    return value.strip()


def _digest(value: object, field: str) -> str:
    text = _text(value, field, maximum=64).lower()
    if _SHA256.fullmatch(text) is None:
        raise InformationLiveResearchError(
            f"{field} is invalid.", code="live_research_binding_invalid"
        )
    return text


def _timestamp(value: object, field: str) -> str:
    text = _text(value, field, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InformationLiveResearchError(
            f"{field} is invalid.", code="live_research_binding_invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise InformationLiveResearchError(
            f"{field} is invalid.", code="live_research_binding_invalid"
        )
    return text


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _model_response_sha256(response: ModelResponse) -> str:
    response.validate()
    return _sha(
        {
            "request_id": response.request_id,
            "provider": response.provider,
            "model": response.model,
            "content": response.content,
            "finish_reason": response.finish_reason,
            "created_at": response.created_at,
        }
    )


def _raise_if_cancelled(cancellation: object | None) -> None:
    if cancellation is None:
        return
    checker = getattr(cancellation, "raise_if_cancelled", None)
    if callable(checker):
        checker()


@dataclass(frozen=True)
class InformationLiveSourceOutcome:
    """Metadata-only disposition of one fetched source."""

    source_id: str
    canonical_url: str
    source_content_sha256: str
    temporal_verdict: str | None
    inspection_verdict: str | None
    freshness_verdict: str | None
    supports_claim: bool
    disposition: str
    reason_code: str | None

    def validate(self) -> None:
        _text(self.source_id, "source_id", maximum=512)
        url = _text(self.canonical_url, "canonical_url")
        if not url.startswith("https://"):
            raise InformationLiveResearchError(
                "Live source URL must be HTTPS.",
                code="live_research_binding_invalid",
            )
        _digest(self.source_content_sha256, "source_content_sha256")
        if self.disposition not in _ALLOWED_DISPOSITIONS:
            raise InformationLiveResearchError(
                "Live source disposition changed.",
                code="live_research_binding_invalid",
            )
        if not isinstance(self.supports_claim, bool):
            raise InformationLiveResearchError(
                "Live source support flag is invalid.",
                code="live_research_binding_invalid",
            )
        for value, field in (
            (self.temporal_verdict, "temporal_verdict"),
            (self.inspection_verdict, "inspection_verdict"),
            (self.freshness_verdict, "freshness_verdict"),
            (self.reason_code, "reason_code"),
        ):
            if value is not None:
                _text(value, field, maximum=128)
        if self.disposition == "grounded" and not self.supports_claim:
            raise InformationLiveResearchError(
                "Grounded source cannot be unsupported.",
                code="live_research_binding_invalid",
            )
        if self.disposition in {
            "blocked_injection",
            "temporal_rejected",
            "freshness_rejected",
        } and self.supports_claim:
            raise InformationLiveResearchError(
                "Rejected source cannot support a claim.",
                code="live_research_binding_invalid",
            )

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "source_content_sha256": self.source_content_sha256,
            "temporal_verdict": self.temporal_verdict,
            "inspection_verdict": self.inspection_verdict,
            "freshness_verdict": self.freshness_verdict,
            "supports_claim": self.supports_claim,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class InformationLiveFetchFailure:
    """Metadata-only record for one rejected live source candidate."""

    failure_id: str
    result_id: str
    result_rank: int
    canonical_url: str
    result_sha256: str
    failure_code: str
    failure_sha256: str

    @classmethod
    def create(
        cls,
        *,
        result_id: str,
        result_rank: int,
        canonical_url: str,
        result_sha256: str,
        failure_code: str,
    ) -> "InformationLiveFetchFailure":
        draft = cls(
            failure_id="live-fetch-failure-pending",
            result_id=result_id,
            result_rank=result_rank,
            canonical_url=canonical_url,
            result_sha256=result_sha256,
            failure_code=failure_code,
            failure_sha256="0" * 64,
        )
        identified = replace(
            draft,
            failure_id=(
                "live-fetch-failure-"
                + _sha(draft._payload(include_id=False))[:20]
            ),
        )
        failure = replace(
            identified,
            failure_sha256=_sha(identified._payload(include_id=True)),
        )
        failure.validate()
        return failure

    def _payload(self, *, include_id: bool) -> dict[str, object]:
        value: dict[str, object] = {
            "failure_id": self.failure_id,
            "result_id": self.result_id,
            "result_rank": self.result_rank,
            "canonical_url": self.canonical_url,
            "result_sha256": self.result_sha256,
            "failure_code": self.failure_code,
        }
        if not include_id:
            value.pop("failure_id")
        return value

    def validate(self) -> None:
        _text(self.result_id, "result_id", maximum=512)
        if (
            not isinstance(self.result_rank, int)
            or isinstance(self.result_rank, bool)
            or self.result_rank < 1
        ):
            raise InformationLiveResearchError(
                "Live fetch failure rank is invalid.",
                code="live_research_binding_invalid",
            )
        url = _text(self.canonical_url, "canonical_url")
        if not url.startswith("https://"):
            raise InformationLiveResearchError(
                "Rejected live source URL must be HTTPS.",
                code="live_research_binding_invalid",
            )
        _digest(self.result_sha256, "result_sha256")
        code = _text(self.failure_code, "failure_code", maximum=128)
        if re.fullmatch(r"[a-z][a-z0-9_]*", code) is None:
            raise InformationLiveResearchError(
                "Live fetch failure code is invalid.",
                code="live_research_binding_invalid",
            )
        expected_id = (
            "live-fetch-failure-"
            + _sha(self._payload(include_id=False))[:20]
        )
        if self.failure_id != expected_id:
            raise InformationLiveResearchError(
                "Live fetch failure identity changed.",
                code="live_research_binding_invalid",
            )
        if self.failure_sha256 != _sha(self._payload(include_id=True)):
            raise InformationLiveResearchError(
                "Live fetch failure digest changed.",
                code="live_research_binding_invalid",
            )

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            **self._payload(include_id=True),
            "failure_sha256": self.failure_sha256,
        }


@dataclass(frozen=True)
class InformationLiveEvidenceResult:
    """In-memory live evidence with a metadata-only deterministic digest."""

    request: InformationResearchRequest
    search_response: InformationLiveSearchResponse
    fetch_responses: tuple[InformationLiveFetchResponse, ...]
    fetch_failures: tuple[InformationLiveFetchFailure, ...]
    resolved_resources: tuple[InformationResolvedTemporalResource, ...]
    inspected_sources: tuple[InformationInspectedSource, ...]
    eligible_sources: tuple[InformationTemporallyQualifiedSource, ...]
    grounded_sources: tuple[InformationTemporallyQualifiedSource, ...]
    source_outcomes: tuple[InformationLiveSourceOutcome, ...]
    grounding: InformationVerifiedGroundingPacket
    evidence_sha256: str

    def validate(self, *, executor: "LiveInformationResearchExecutor") -> None:
        executor._validate_evidence(self)


@dataclass(frozen=True)
class InformationLiveResearchReceipt:
    """Raw-query- and source-body-free binding for one exact P4.10b turn."""

    receipt_id: str
    policy_version: str
    request_id: str
    query_id: str
    query_sha256: str
    outcome: str
    search_result_count: int
    search_receipt_sha256: str
    fetch_attempt_count: int
    fetch_attempt_sequence_sha256: str
    fetch_receipt_sha256s: tuple[str, ...]
    fetch_failure_sha256s: tuple[str, ...]
    temporal_resolution_sha256s: tuple[str, ...]
    source_outcome_sha256: str
    grounded_source_sha256s: tuple[str, ...]
    grounding_sha256: str
    projection_sha256: str
    conversation_packet_sha256: str
    response_sha256: str
    validation_sha256: str
    citation_validation_outcome: str
    pre_commit_validation_count: int
    policy_bindings: tuple[str, ...]
    created_at: str
    receipt_sha256: str

    @classmethod
    def create(cls, **values: object) -> "InformationLiveResearchReceipt":
        draft = cls(
            receipt_id="live-research-pending",
            receipt_sha256="0" * 64,
            **values,
        )  # type: ignore[arg-type]
        identified = replace(
            draft,
            receipt_id=f"live-research-{_sha(draft._payload(include_id=False))[:20]}",
        )
        receipt = replace(
            identified, receipt_sha256=_sha(identified._payload(include_id=True))
        )
        receipt.validate()
        return receipt

    def _payload(self, *, include_id: bool) -> dict[str, object]:
        value: dict[str, object] = {
            "receipt_id": self.receipt_id,
            "policy_version": self.policy_version,
            "request_id": self.request_id,
            "query_id": self.query_id,
            "query_sha256": self.query_sha256,
            "outcome": self.outcome,
            "search_result_count": self.search_result_count,
            "search_receipt_sha256": self.search_receipt_sha256,
            "fetch_attempt_count": self.fetch_attempt_count,
            "fetch_attempt_sequence_sha256": self.fetch_attempt_sequence_sha256,
            "fetch_receipt_sha256s": list(self.fetch_receipt_sha256s),
            "fetch_failure_sha256s": list(self.fetch_failure_sha256s),
            "temporal_resolution_sha256s": list(
                self.temporal_resolution_sha256s
            ),
            "source_outcome_sha256": self.source_outcome_sha256,
            "grounded_source_sha256s": list(self.grounded_source_sha256s),
            "grounding_sha256": self.grounding_sha256,
            "projection_sha256": self.projection_sha256,
            "conversation_packet_sha256": self.conversation_packet_sha256,
            "response_sha256": self.response_sha256,
            "validation_sha256": self.validation_sha256,
            "citation_validation_outcome": self.citation_validation_outcome,
            "pre_commit_validation_count": self.pre_commit_validation_count,
            "policy_bindings": list(self.policy_bindings),
            "created_at": self.created_at,
        }
        if not include_id:
            value.pop("receipt_id")
        return value

    def validate(self) -> None:
        if self.policy_version != "1.0.0":
            raise InformationLiveResearchError(
                "Live research policy version changed.",
                code="live_research_binding_invalid",
            )
        _text(self.request_id, "request_id", maximum=512)
        _text(self.query_id, "query_id", maximum=512)
        if self.outcome not in {"answerable", "insufficient_sources"}:
            raise InformationLiveResearchError(
                "Live research outcome is invalid.",
                code="live_research_binding_invalid",
            )
        if (
            not isinstance(self.search_result_count, int)
            or isinstance(self.search_result_count, bool)
            or self.search_result_count < 0
        ):
            raise InformationLiveResearchError(
                "Live search result count is invalid.",
                code="live_research_binding_invalid",
            )
        if (
            not isinstance(self.fetch_attempt_count, int)
            or isinstance(self.fetch_attempt_count, bool)
            or self.fetch_attempt_count < 0
            or self.fetch_attempt_count > self.search_result_count
        ):
            raise InformationLiveResearchError(
                "Live fetch attempt count is invalid.",
                code="live_research_binding_invalid",
            )
        if self.citation_validation_outcome != "accepted":
            raise InformationLiveResearchError(
                "P4.5b citation validation was not accepted.",
                code="live_research_response_invalid",
            )
        for value, field in (
            (self.query_sha256, "query_sha256"),
            (self.search_receipt_sha256, "search_receipt_sha256"),
            (
                self.fetch_attempt_sequence_sha256,
                "fetch_attempt_sequence_sha256",
            ),
            (self.source_outcome_sha256, "source_outcome_sha256"),
            (self.grounding_sha256, "grounding_sha256"),
            (self.projection_sha256, "projection_sha256"),
            (self.conversation_packet_sha256, "conversation_packet_sha256"),
            (self.response_sha256, "response_sha256"),
            (self.validation_sha256, "validation_sha256"),
            (self.receipt_sha256, "receipt_sha256"),
        ):
            _digest(value, field)
        for sequence, field in (
            (self.fetch_receipt_sha256s, "fetch_receipt_sha256s"),
            (self.fetch_failure_sha256s, "fetch_failure_sha256s"),
            (self.temporal_resolution_sha256s, "temporal_resolution_sha256s"),
            (self.grounded_source_sha256s, "grounded_source_sha256s"),
        ):
            if not isinstance(sequence, tuple):
                raise InformationLiveResearchError(
                    f"{field} must be immutable.",
                    code="live_research_binding_invalid",
                )
            for digest in sequence:
                _digest(digest, field)
        if len(self.fetch_receipt_sha256s) != len(
            self.temporal_resolution_sha256s
        ):
            raise InformationLiveResearchError(
                "Every successful fetch requires temporal resolution evidence.",
                code="live_research_binding_invalid",
            )
        if self.fetch_attempt_count != (
            len(self.fetch_receipt_sha256s)
            + len(self.fetch_failure_sha256s)
        ):
            raise InformationLiveResearchError(
                "Live fetch attempts are not fully accounted for.",
                code="live_research_binding_invalid",
            )
        if self.pre_commit_validation_count != 1:
            raise InformationLiveResearchError(
                "P3.6 must invoke P4.5b exactly once before commit.",
                code="live_research_precommit_missing",
            )
        if not self.policy_bindings or len(set(self.policy_bindings)) != len(
            self.policy_bindings
        ):
            raise InformationLiveResearchError(
                "Live research policy bindings are invalid.",
                code="live_research_binding_invalid",
            )
        for binding in self.policy_bindings:
            if "@" not in _text(binding, "policy_binding", maximum=512):
                raise InformationLiveResearchError(
                    "Live research policy binding is invalid.",
                    code="live_research_binding_invalid",
                )
        _timestamp(self.created_at, "created_at")
        expected_id = f"live-research-{_sha(self._payload(include_id=False))[:20]}"
        if self.receipt_id != expected_id:
            raise InformationLiveResearchError(
                "Live research receipt identity changed.",
                code="live_research_binding_invalid",
            )
        if self.receipt_sha256 != _sha(self._payload(include_id=True)):
            raise InformationLiveResearchError(
                "Live research receipt digest changed.",
                code="live_research_binding_invalid",
            )

    def to_metadata_record(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(include_id=True), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class InformationLiveResearchTurnResult:
    evidence: InformationLiveEvidenceResult
    projection: InformationConversationGroundingProjection
    conversation_result: ConversationTurnResult
    response_validation: InformationConversationResponseValidation
    receipt: InformationLiveResearchReceipt

    def validate(self, *, executor: "LiveInformationResearchExecutor") -> None:
        self.evidence.validate(executor=executor)
        self.receipt.validate()
        if self.receipt.grounding_sha256 != self.evidence.grounding.grounding_sha256:
            raise InformationLiveResearchError(
                "Live turn grounding binding changed.",
                code="live_research_binding_invalid",
            )
        if self.receipt.projection_sha256 != self.projection.receipt.projection_sha256:
            raise InformationLiveResearchError(
                "Live turn projection binding changed.",
                code="live_research_binding_invalid",
            )
        if (
            self.receipt.validation_sha256
            != self.response_validation.receipt.validation_sha256
        ):
            raise InformationLiveResearchError(
                "Live turn validation binding changed.",
                code="live_research_binding_invalid",
            )
        if self.receipt.fetch_attempt_count != (
            len(self.evidence.fetch_responses)
            + len(self.evidence.fetch_failures)
        ):
            raise InformationLiveResearchError(
                "Live turn fetch-attempt binding changed.",
                code="live_research_binding_invalid",
            )
        if self.receipt.fetch_failure_sha256s != tuple(
            item.failure_sha256 for item in self.evidence.fetch_failures
        ):
            raise InformationLiveResearchError(
                "Live turn fetch-failure binding changed.",
                code="live_research_binding_invalid",
            )
        expected_attempt_sequence = executor._fetch_attempt_sequence_sha256(
            self.evidence.fetch_responses,
            self.evidence.fetch_failures,
        )
        if (
            self.receipt.fetch_attempt_sequence_sha256
            != expected_attempt_sequence
        ):
            raise InformationLiveResearchError(
                "Live turn fetch-attempt sequence changed.",
                code="live_research_binding_invalid",
            )


@dataclass(frozen=True)
class LiveInformationResearchExecutor:
    """One exact additive live PUBLIC research execution profile."""

    registry: ExactInformationLiveProviderRegistry
    information_policy: InformationPolicy
    firewall_policy: InformationInjectionFirewallPolicy
    freshness_policy: InformationFreshnessPolicy
    temporal_metadata_policy: InformationTemporalMetadataPolicy
    grounding_policy: InformationGroundingPolicy
    bridge_policy: InformationConversationBridgePolicy
    response_validation_policy: ConversationResponseValidationPolicy
    research_policy: InformationLiveResearchPolicy
    conversation_runner: object
    claim_planner: DeterministicLiveExtractiveClaimPlanner
    clock: Callable[[], str]
    monotonic_clock: Callable[[], float] = monotonic

    def __post_init__(self) -> None:
        self.registry.validate()
        self.research_policy.validate(provider_policy=self.registry.policy)
        self.temporal_metadata_policy.validate(
            information_policy=self.information_policy,
            freshness_policy=self.freshness_policy,
        )
        self.grounding_policy.validate(
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
        )
        self.bridge_policy.validate(
            response_validation_policy=self.response_validation_policy,
        )
        if self.claim_planner.grounding_policy != self.grounding_policy:
            raise InformationLiveResearchError(
                "Live claim planner policy was substituted.",
                code="live_research_configuration_invalid",
            )
        if not callable(getattr(self.conversation_runner, "run_turn", None)):
            raise InformationLiveResearchError(
                "Live conversation runner is unavailable.",
                code="live_research_configuration_invalid",
            )
        if not callable(self.clock) or not callable(self.monotonic_clock):
            raise InformationLiveResearchError(
                "Live research clocks are invalid.",
                code="live_research_configuration_invalid",
            )

    def validate_operational_boundary(self) -> None:
        """Require the exact live provider and controlled-fetch implementations."""

        from .brave_search import BraveInformationSearchProvider
        from .brave_search_live import StrictBraveSearchHttpsTransport
        from .live_fetch_provider import LiveControlledInformationFetchProvider
        from .retrieval import LiveControlledInformationHttpRetriever

        if type(self.registry) is not ExactInformationLiveProviderRegistry:
            raise InformationLiveResearchError(
                "Live provider registry was substituted.",
                code="live_research_configuration_invalid",
            )
        search_provider = self.registry.search_provider
        fetch_provider = self.registry.fetch_provider
        if type(search_provider) is not BraveInformationSearchProvider:
            raise InformationLiveResearchError(
                "Live search provider was substituted.",
                code="live_research_configuration_invalid",
            )
        if type(search_provider.transport) is not StrictBraveSearchHttpsTransport:
            raise InformationLiveResearchError(
                "Live search transport was substituted.",
                code="live_research_configuration_invalid",
            )
        if type(fetch_provider) is not LiveControlledInformationFetchProvider:
            raise InformationLiveResearchError(
                "Live fetch provider was substituted.",
                code="live_research_configuration_invalid",
            )
        if type(fetch_provider.retriever) is not LiveControlledInformationHttpRetriever:
            raise InformationLiveResearchError(
                "Controlled live retriever was substituted.",
                code="live_research_configuration_invalid",
            )
        self.registry.validate()
        search_provider.transport.validate_live_boundary()
        fetch_provider.retriever._validate_runtime_components()

    def run_turn(
        self,
        command: ConversationTurnCommand,
        *,
        mode: str,
        availability: str,
        request: InformationResearchRequest,
        reference_time: str,
        created_at: str,
        window_start: str | None = None,
        window_end: str | None = None,
        cancellation: object | None = None,
    ) -> InformationLiveResearchTurnResult:
        command.validate()
        request.validate()
        _timestamp(reference_time, "reference_time")
        _timestamp(created_at, "created_at")
        self.validate_operational_boundary()
        self.research_policy.validate(provider_policy=self.registry.policy)
        self._validate_request(command, mode=mode, availability=availability, request=request)

        started = self.monotonic_clock()
        search_provider = self.registry.resolve_search(
            self.research_policy.search_provider
        )
        fetch_provider = self.registry.resolve_fetch(
            self.research_policy.fetch_provider
        )
        _raise_if_cancelled(cancellation)
        search_response = search_provider.search_with_receipt(
            request.query,
            max_results=request.max_sources,
            timeout_seconds=self._remaining_request_timeout(request, started),
            cancellation=cancellation,
        )
        search_response.validate()

        fetch_responses: list[InformationLiveFetchResponse] = []
        fetch_failures: list[InformationLiveFetchFailure] = []
        fetch_attempt_count = 0
        for search_result in search_response.results:
            if fetch_attempt_count >= request.max_fetch_calls:
                break
            fetch_attempt_count += 1
            _raise_if_cancelled(cancellation)
            try:
                fetched = fetch_provider.fetch_with_receipt(
                    search_result,
                    query_sha256=request.query.content_sha256,
                    timeout_seconds=self._remaining_request_timeout(
                        request, started
                    ),
                    max_response_bytes=(
                        fetch_provider.retriever.retrieval_policy.max_decoded_bytes
                    ),
                    cancellation=cancellation,
                )
            except Exception as exc:
                failure_code = self._skippable_fetch_failure_code(exc)
                if failure_code is None:
                    raise
                fetch_failures.append(
                    InformationLiveFetchFailure.create(
                        result_id=search_result.result_id,
                        result_rank=search_result.rank,
                        canonical_url=search_result.canonical_url,
                        result_sha256=search_result.content_sha256,
                        failure_code=failure_code,
                    )
                )
                continue
            fetched.validate()
            fetch_responses.append(fetched)

        temporal = DeterministicInformationTemporalMetadataResolver(
            self.information_policy,
            self.freshness_policy,
            self.temporal_metadata_policy,
        )
        firewall = DeterministicInformationInjectionFirewall(
            self.information_policy, self.firewall_policy
        )
        classifier = DeterministicInformationTemporalClassifier(self.freshness_policy)
        intent = classifier.classify(
            request.query,
            reference_time=reference_time,
            window_start=window_start,
            window_end=window_end,
        )
        freshness = DeterministicInformationFreshnessEvaluator(
            self.information_policy, self.firewall_policy, self.freshness_policy
        )

        resolved: list[InformationResolvedTemporalResource] = []
        inspected: list[InformationInspectedSource] = []
        eligible: list[InformationTemporallyQualifiedSource] = []
        provisional_outcomes: list[InformationLiveSourceOutcome] = []
        for fetched in fetch_responses:
            _raise_if_cancelled(cancellation)
            resolved_item = temporal.resolve(fetched.resource)
            resolved.append(resolved_item)
            result = fetched.search_result
            source_id = getattr(fetched.source_document, "source_id")
            provider = getattr(fetched.source_document, "provider")
            retrieved_at = getattr(fetched.source_document, "retrieved_at")
            try:
                source = resolved_item.to_source_document(
                    source_id=source_id,
                    provider=provider,
                    retrieved_at=retrieved_at,
                    policy=self.temporal_metadata_policy,
                    freshness_policy=self.freshness_policy,
                )
            except InformationTemporalMetadataError as exc:
                provisional_outcomes.append(
                    InformationLiveSourceOutcome(
                        source_id=source_id,
                        canonical_url=fetched.resource.final_url,
                        source_content_sha256=fetched.resource.content_sha256,
                        temporal_verdict=resolved_item.resolution.verdict,
                        inspection_verdict=None,
                        freshness_verdict=None,
                        supports_claim=False,
                        disposition="temporal_rejected",
                        reason_code=exc.code,
                    )
                )
                continue
            checked = firewall.inspect(source)
            inspected.append(checked)
            if checked.inspection.verdict == "blocked":
                provisional_outcomes.append(
                    InformationLiveSourceOutcome(
                        source_id=source.source_id,
                        canonical_url=source.canonical_url,
                        source_content_sha256=source.content_sha256,
                        temporal_verdict=resolved_item.resolution.verdict,
                        inspection_verdict="blocked",
                        freshness_verdict=None,
                        supports_claim=False,
                        disposition="blocked_injection",
                        reason_code="prompt_injection_blocked",
                    )
                )
                continue
            assessed = freshness.assess(checked, intent=intent, query=request.query)
            if assessed.assessment.supports_claim:
                eligible.append(assessed)
                disposition = "eligible_not_selected"
                reason = None
            else:
                disposition = "freshness_rejected"
                reason = "freshness_insufficient"
            provisional_outcomes.append(
                InformationLiveSourceOutcome(
                    source_id=source.source_id,
                    canonical_url=source.canonical_url,
                    source_content_sha256=source.content_sha256,
                    temporal_verdict=resolved_item.resolution.verdict,
                    inspection_verdict="clear",
                    freshness_verdict=assessed.assessment.verdict,
                    supports_claim=assessed.assessment.supports_claim,
                    disposition=disposition,
                    reason_code=reason,
                )
            )

        grounded_sources, claim_drafts = self.claim_planner.plan(
            query=request.query,
            qualified_sources=tuple(eligible),
            maximum_sources=self.research_policy.maximum_grounded_sources,
        )
        grounded_ids = {
            item.inspected_source.source.source_id for item in grounded_sources
        }
        outcomes = tuple(
            replace(item, disposition="grounded")
            if item.source_id in grounded_ids
            else item
            for item in provisional_outcomes
        )
        outcome = "answerable" if claim_drafts else "insufficient_sources"
        builder = DeterministicInformationGroundingBuilder(
            self.information_policy,
            self.firewall_policy,
            self.freshness_policy,
            self.grounding_policy,
        )
        grounding = builder.build(
            packet_id=f"live-grounding-{request.request_id}",
            request_id=request.request_id,
            outcome=outcome,
            query=request.query,
            qualified_sources=grounded_sources,
            claim_drafts=claim_drafts,
            created_at=created_at,
        )
        evidence = InformationLiveEvidenceResult(
            request=request,
            search_response=search_response,
            fetch_responses=tuple(fetch_responses),
            fetch_failures=tuple(fetch_failures),
            resolved_resources=tuple(resolved),
            inspected_sources=tuple(inspected),
            eligible_sources=tuple(eligible),
            grounded_sources=grounded_sources,
            source_outcomes=outcomes,
            grounding=grounding,
            evidence_sha256=self._evidence_sha256(
                request=request,
                search_response=search_response,
                fetch_responses=tuple(fetch_responses),
                fetch_failures=tuple(fetch_failures),
                resolved_resources=tuple(resolved),
                source_outcomes=outcomes,
                grounded_sources=grounded_sources,
                grounding=grounding,
            ),
        )
        evidence.validate(executor=self)

        projection = project_information_grounding_to_conversation(
            verified_grounding=grounding,
            query=request.query,
            qualified_sources=grounded_sources,
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
            grounding_policy=self.grounding_policy,
            bridge_policy=self.bridge_policy,
        )
        selected = replace(command, grounding=projection.conversation_packet)
        validations: list[InformationConversationResponseValidation] = []

        def validate_before_commit(
            response: ModelResponse,
            grounding_packet: ConversationGroundingPacket | None,
        ) -> None:
            if grounding_packet != projection.conversation_packet:
                raise ConversationContractError(
                    "Live response validation received substituted grounding."
                )
            try:
                validation = validate_information_conversation_response(
                    response=response,
                    projection=projection,
                    verified_grounding=evidence.grounding,
                    query=request.query,
                    qualified_sources=evidence.grounded_sources,
                    information_policy=self.information_policy,
                    firewall_policy=self.firewall_policy,
                    freshness_policy=self.freshness_policy,
                    grounding_policy=self.grounding_policy,
                    bridge_policy=self.bridge_policy,
                    response_validation_policy=self.response_validation_policy,
                )
            except Exception as exc:
                raise ConversationContractError(
                    "Live response failed the P4.5b validation boundary."
                ) from exc
            if validation.report.outcome == "rejected":
                raise ConversationContractError(
                    "Live response was rejected before Phase 3 commit."
                )
            validations.append(validation)

        conversation = self.conversation_runner.run_turn(
            selected,
            cancellation=cancellation,
            response_validation_hook=validate_before_commit,
        )
        if conversation.replayed:
            validation = validate_information_conversation_response(
                response=conversation.response,
                projection=projection,
                verified_grounding=evidence.grounding,
                query=request.query,
                qualified_sources=evidence.grounded_sources,
                information_policy=self.information_policy,
                firewall_policy=self.firewall_policy,
                freshness_policy=self.freshness_policy,
                grounding_policy=self.grounding_policy,
                bridge_policy=self.bridge_policy,
                response_validation_policy=self.response_validation_policy,
            )
            if validation.report.outcome == "rejected":
                raise InformationLiveResearchError(
                    "Replayed live response failed P4.5b.",
                    code="live_research_response_invalid",
                )
        else:
            if len(validations) != 1:
                raise InformationLiveResearchError(
                    "P3.6 did not invoke P4.5b exactly once.",
                    code="live_research_precommit_missing",
                )
            validation = validations[0]

        receipt = InformationLiveResearchReceipt.create(
            policy_version=self.research_policy.version,
            request_id=request.request_id,
            query_id=request.query.query_id,
            query_sha256=request.query.content_sha256,
            outcome=outcome,
            search_result_count=len(search_response.results),
            search_receipt_sha256=search_response.receipt.receipt_sha256,
            fetch_attempt_count=fetch_attempt_count,
            fetch_attempt_sequence_sha256=(
                self._fetch_attempt_sequence_sha256(
                    tuple(fetch_responses),
                    tuple(fetch_failures),
                )
            ),
            fetch_receipt_sha256s=tuple(
                item.receipt.receipt_sha256 for item in fetch_responses
            ),
            fetch_failure_sha256s=tuple(
                item.failure_sha256 for item in fetch_failures
            ),
            temporal_resolution_sha256s=tuple(
                item.resolution.resolution_sha256 for item in resolved
            ),
            source_outcome_sha256=canonical_sha256(
                [item.metadata_record() for item in outcomes]
            ),
            grounded_source_sha256s=tuple(
                item.inspected_source.source.content_sha256
                for item in grounded_sources
            ),
            grounding_sha256=grounding.grounding_sha256,
            projection_sha256=projection.receipt.projection_sha256,
            conversation_packet_sha256=projection.receipt.conversation_packet_sha256,
            response_sha256=_model_response_sha256(conversation.response),
            validation_sha256=validation.receipt.validation_sha256,
            citation_validation_outcome=validation.report.outcome,
            pre_commit_validation_count=1,
            policy_bindings=self._policy_bindings(),
            created_at=self.clock(),
        )
        result = InformationLiveResearchTurnResult(
            evidence=evidence,
            projection=projection,
            conversation_result=conversation,
            response_validation=validation,
            receipt=receipt,
        )
        result.validate(executor=self)
        return result

    def _validate_request(
        self,
        command: ConversationTurnCommand,
        *,
        mode: str,
        availability: str,
        request: InformationResearchRequest,
    ) -> None:
        if mode != self.research_policy.required_mode or availability != self.research_policy.required_availability:
            raise InformationLiveResearchError(
                "P4.10b requires explicit available research mode.",
                code="live_research_mode_required",
            )
        if command.grounding is not None:
            raise InformationLiveResearchError(
                "P4.10b must inject the only grounding packet.",
                code="live_research_grounding_preinjected",
            )
        if request.query.data_classification != "PUBLIC":
            raise InformationLiveResearchError(
                "P4.10b accepts PUBLIC queries only.",
                code="live_research_request_invalid",
            )
        if request.operations != self.research_policy.required_operations:
            raise InformationLiveResearchError(
                "P4.10b requires exact search/fetch operations.",
                code="live_research_request_invalid",
            )
        if (
            request.max_search_calls != 1
            or request.max_search_calls > self.research_policy.maximum_search_calls
            or request.max_fetch_calls > self.research_policy.maximum_fetch_calls
            or request.max_sources > self.research_policy.maximum_sources
            or request.max_fetch_calls != request.max_sources
        ):
            raise InformationLiveResearchError(
                "P4.10b request budget changed.",
                code="live_research_budget_invalid",
            )

    def _skippable_fetch_failure_code(
        self, exc: Exception
    ) -> str | None:
        code = getattr(exc, "code", None)
        if not isinstance(code, str) or not code:
            prefix = str(exc).split(":", 1)[0].strip()
            code = prefix if re.fullmatch(r"[a-z][a-z0-9_]*", prefix) else None
        if code in self.research_policy.skippable_fetch_failure_codes:
            return code
        return None

    def _fetch_attempt_sequence_sha256(
        self,
        fetch_responses: tuple[InformationLiveFetchResponse, ...],
        fetch_failures: tuple[InformationLiveFetchFailure, ...],
    ) -> str:
        attempts: dict[int, str] = {}
        for item in fetch_responses:
            rank = int(getattr(item.search_result, "rank", 0))
            if rank in attempts:
                raise InformationLiveResearchError(
                    "Live fetch attempt rank was duplicated.",
                    code="live_research_binding_invalid",
                )
            attempts[rank] = (
                f"success:{rank}:{item.receipt.receipt_sha256}"
            )
        for item in fetch_failures:
            item.validate()
            if item.result_rank in attempts:
                raise InformationLiveResearchError(
                    "Live fetch attempt rank was duplicated.",
                    code="live_research_binding_invalid",
                )
            attempts[item.result_rank] = (
                f"rejected:{item.result_rank}:{item.failure_sha256}"
            )
        ordered = [attempts[index] for index in sorted(attempts)]
        if sorted(attempts) != list(range(1, len(attempts) + 1)):
            raise InformationLiveResearchError(
                "Live fetch attempts must consume a contiguous search-result prefix.",
                code="live_research_binding_invalid",
            )
        return sequence_sha256(ordered)

    def _remaining_request_timeout(
        self, request: InformationResearchRequest, started: float
    ) -> float:
        elapsed = self.monotonic_clock() - started
        remaining = request.total_timeout_seconds - elapsed
        if remaining <= 0:
            raise InformationLiveResearchError(
                "Live research total timeout was exhausted.",
                code="live_research_timeout",
            )
        return min(float(request.request_timeout_seconds), float(remaining))

    def _policy_bindings(self) -> tuple[str, ...]:
        values = (
            self.research_policy,
            self.registry.policy,
            self.information_policy,
            self.firewall_policy,
            self.freshness_policy,
            self.temporal_metadata_policy,
            self.grounding_policy,
            self.bridge_policy,
            self.response_validation_policy,
        )
        return tuple(f"{item.policy_name}@{item.version}" for item in values)

    def _evidence_sha256(
        self,
        *,
        request: InformationResearchRequest,
        search_response: InformationLiveSearchResponse,
        fetch_responses: tuple[InformationLiveFetchResponse, ...],
        fetch_failures: tuple[InformationLiveFetchFailure, ...],
        resolved_resources: tuple[InformationResolvedTemporalResource, ...],
        source_outcomes: tuple[InformationLiveSourceOutcome, ...],
        grounded_sources: tuple[InformationTemporallyQualifiedSource, ...],
        grounding: InformationVerifiedGroundingPacket,
    ) -> str:
        return canonical_sha256(
            {
                "request_id": request.request_id,
                "query_id": request.query.query_id,
                "query_sha256": request.query.content_sha256,
                "search_receipt_sha256": search_response.receipt.receipt_sha256,
                "fetch_attempt_sequence_sha256": (
                    self._fetch_attempt_sequence_sha256(
                        fetch_responses,
                        fetch_failures,
                    )
                ),
                "fetch_receipt_sha256s": [
                    item.receipt.receipt_sha256 for item in fetch_responses
                ],
                "fetch_failure_sha256s": [
                    item.failure_sha256 for item in fetch_failures
                ],
                "temporal_resolution_sha256s": [
                    item.resolution.resolution_sha256
                    for item in resolved_resources
                ],
                "source_outcomes": [
                    item.metadata_record() for item in source_outcomes
                ],
                "grounded_source_sha256s": [
                    item.inspected_source.source.content_sha256
                    for item in grounded_sources
                ],
                "grounding_sha256": grounding.grounding_sha256,
            }
        )

    def _validate_evidence(self, evidence: InformationLiveEvidenceResult) -> None:
        evidence.request.validate()
        evidence.search_response.validate()
        if (
            len(evidence.fetch_responses) + len(evidence.fetch_failures)
            > evidence.request.max_fetch_calls
        ):
            raise InformationLiveResearchError(
                "Live fetch budget changed.", code="live_research_binding_invalid"
            )
        if len(evidence.fetch_responses) != len(evidence.resolved_resources):
            raise InformationLiveResearchError(
                "Every live fetch requires one temporal resolution.",
                code="live_research_binding_invalid",
            )
        for item in evidence.fetch_responses:
            item.validate()
            if item.receipt.query_sha256 != evidence.request.query.content_sha256:
                raise InformationLiveResearchError(
                    "Live fetch query binding changed.",
                    code="live_research_binding_invalid",
                )
        for item in evidence.fetch_failures:
            item.validate()
            if (
                item.failure_code
                not in self.research_policy.skippable_fetch_failure_codes
            ):
                raise InformationLiveResearchError(
                    "Live fetch failure code was not approved.",
                    code="live_research_binding_invalid",
                )
        attempted_ranks = {
            int(item.search_result.rank) for item in evidence.fetch_responses
        } | {item.result_rank for item in evidence.fetch_failures}
        if sorted(attempted_ranks) != list(
            range(1, len(attempted_ranks) + 1)
        ):
            raise InformationLiveResearchError(
                "Live fetch attempts did not consume a contiguous result prefix.",
                code="live_research_binding_invalid",
            )
        if len(attempted_ranks) != (
            len(evidence.fetch_responses) + len(evidence.fetch_failures)
        ):
            raise InformationLiveResearchError(
                "Live fetch attempts contain duplicate result ranks.",
                code="live_research_binding_invalid",
            )
        search_results_by_rank = {
            int(item.rank): item for item in evidence.search_response.results
        }
        for failure in evidence.fetch_failures:
            result = search_results_by_rank.get(failure.result_rank)
            if (
                result is None
                or result.result_id != failure.result_id
                or result.canonical_url != failure.canonical_url
                or result.content_sha256 != failure.result_sha256
            ):
                raise InformationLiveResearchError(
                    "Live fetch failure lost its search-result binding.",
                    code="live_research_binding_invalid",
                )
        for item in evidence.source_outcomes:
            item.validate()
        if len({item.source_id for item in evidence.source_outcomes}) != len(
            evidence.source_outcomes
        ):
            raise InformationLiveResearchError(
                "Live source outcomes contain duplicate identities.",
                code="live_research_binding_invalid",
            )
        eligible_ids = {
            item.inspected_source.source.source_id for item in evidence.eligible_sources
        }
        grounded_ids = {
            item.inspected_source.source.source_id for item in evidence.grounded_sources
        }
        if not grounded_ids.issubset(eligible_ids):
            raise InformationLiveResearchError(
                "Grounded sources must be freshness-eligible.",
                code="live_research_binding_invalid",
            )
        if len(grounded_ids) > self.research_policy.maximum_grounded_sources:
            raise InformationLiveResearchError(
                "Grounded-source budget changed.",
                code="live_research_binding_invalid",
            )
        outcome_by_id = {item.source_id: item for item in evidence.source_outcomes}
        if grounded_ids != {
            source_id
            for source_id, item in outcome_by_id.items()
            if item.disposition == "grounded"
        }:
            raise InformationLiveResearchError(
                "Grounded source outcomes do not match the grounding set.",
                code="live_research_binding_invalid",
            )
        evidence.grounding.validate(
            query=evidence.request.query,
            qualified_sources=evidence.grounded_sources,
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
            grounding_policy=self.grounding_policy,
        )
        expected = self._evidence_sha256(
            request=evidence.request,
            search_response=evidence.search_response,
            fetch_responses=evidence.fetch_responses,
            fetch_failures=evidence.fetch_failures,
            resolved_resources=evidence.resolved_resources,
            source_outcomes=evidence.source_outcomes,
            grounded_sources=evidence.grounded_sources,
            grounding=evidence.grounding,
        )
        if evidence.evidence_sha256 != expected:
            raise InformationLiveResearchError(
                "Live evidence digest changed.",
                code="live_research_binding_invalid",
            )
