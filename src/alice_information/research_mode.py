"""Explicit local-only and research conversation turns for Phase 4 P4.7a."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Protocol
from urllib.parse import urlsplit

from alice_conversation.contracts import (
    ConversationContractError,
    ConversationGroundingPacket,
    ModelResponse,
)
from alice_conversation.orchestration import (
    ConversationTurnCommand,
    ConversationTurnResult,
)
from alice_conversation.response_validation_policy import (
    ConversationResponseValidationPolicy,
)

from .conversation_bridge import (
    InformationConversationGroundingProjection,
    InformationConversationResponseValidation,
    project_information_grounding_to_conversation,
    validate_information_conversation_response,
)
from .conversation_bridge_policy import InformationConversationBridgePolicy
from .research_evidence import (
    DeterministicInformationResearchEvidencePipeline,
    InformationResearchEvidenceResult,
)
from .research_mode_policy import (
    APPROVED_MAX_SOURCE_SUMMARIES,
    InformationResearchModePolicy,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WEB_SOURCE_KIND = "web_source"


class InformationResearchModeError(RuntimeError):
    """Raised when a P4.7a turn cannot be safely prepared or verified."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


class InformationConversationTurnRunner(Protocol):
    """Narrow Phase 3 runner contract used by the P4.7a adapter."""

    def run_turn(
        self,
        command: ConversationTurnCommand,
        *,
        cancellation: object | None = None,
        response_validation_hook: (
            Callable[[ModelResponse, ConversationGroundingPacket | None], None]
            | None
        ) = None,
    ) -> ConversationTurnResult: ...


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchModeError(
            f"{field} must be non-empty text.",
            code="research_mode_binding_invalid",
        )
    return value.strip()


def _digest(value: object, field: str) -> str:
    text = _text(value, field).lower()
    if _SHA256.fullmatch(text) is None:
        raise InformationResearchModeError(
            f"{field} must be a lowercase SHA-256 digest.",
            code="research_mode_binding_invalid",
        )
    return text


def _timestamp(value: object, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InformationResearchModeError(
            f"{field} must be valid ISO-8601 text.",
            code="research_mode_binding_invalid",
        ) from exc
    if parsed.tzinfo is None:
        raise InformationResearchModeError(
            f"{field} must include a timezone offset.",
            code="research_mode_binding_invalid",
        )
    return text


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _model_response_payload(response: ModelResponse) -> dict[str, str]:
    response.validate()
    return {
        "request_id": response.request_id,
        "provider": response.provider,
        "model": response.model,
        "content": response.content,
        "finish_reason": response.finish_reason,
        "created_at": response.created_at,
    }


def _model_response_sha256(response: ModelResponse) -> str:
    return hashlib.sha256(
        _canonical_json(_model_response_payload(response)).encode("utf-8")
    ).hexdigest()


def _contains_web_grounding(grounding: ConversationGroundingPacket | None) -> bool:
    if grounding is None:
        return False
    grounding.validate()
    return any(
        citation.source_kind == _WEB_SOURCE_KIND
        for claim in grounding.claims
        for citation in claim.citations
    )


@dataclass(frozen=True)
class InformationResearchModeSourceSummary:
    """Metadata-only user-facing binding for one exact web citation token."""

    citation_token: str
    source_id: str
    canonical_url: str
    source_content_sha256: str
    freshness_verdict: str

    def validate(self) -> None:
        _text(self.citation_token, "citation_token")
        _text(self.source_id, "source_id")
        canonical_url = _text(self.canonical_url, "canonical_url")
        parsed = urlsplit(canonical_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise InformationResearchModeError(
                "Research-mode source summaries require canonical credential-free HTTPS URLs.",
                code="research_mode_binding_invalid",
            )
        _digest(self.source_content_sha256, "source_content_sha256")
        _text(self.freshness_verdict, "freshness_verdict")
        if not self.citation_token.startswith("[WEB:") or not self.citation_token.endswith("]"):
            raise InformationResearchModeError(
                "Research-mode source summaries require exact WEB citation tokens.",
                code="research_mode_binding_invalid",
            )

    def metadata_record(self) -> dict[str, str]:
        self.validate()
        return {
            "citation_token": self.citation_token,
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "source_content_sha256": self.source_content_sha256,
            "freshness_verdict": self.freshness_verdict,
        }


def _source_summaries(
    projection: InformationConversationGroundingProjection,
) -> tuple[InformationResearchModeSourceSummary, ...]:
    source_by_id = {
        binding.source_id: binding
        for binding in projection.receipt.source_bindings
    }
    if len(source_by_id) != len(projection.receipt.source_bindings):
        raise InformationResearchModeError(
            "Conversation projection contains duplicate source bindings.",
            code="research_mode_binding_invalid",
        )
    summaries: list[InformationResearchModeSourceSummary] = []
    seen_tokens: set[str] = set()
    for citation in projection.receipt.citation_bindings:
        source = source_by_id.get(citation.source_id)
        if source is None:
            raise InformationResearchModeError(
                "Conversation citation is missing its source binding.",
                code="research_mode_binding_invalid",
            )
        if citation.token in seen_tokens:
            continue
        seen_tokens.add(citation.token)
        summary = InformationResearchModeSourceSummary(
            citation_token=citation.token,
            source_id=citation.source_id,
            canonical_url=citation.canonical_url,
            source_content_sha256=citation.source_content_sha256,
            freshness_verdict=source.freshness_verdict,
        )
        summary.validate()
        summaries.append(summary)
    return tuple(summaries)


def _receipt_payload(
    receipt: "InformationResearchModeReceipt",
    *,
    include_ids: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "adapter_id": receipt.adapter_id,
        "policy_version": receipt.policy_version,
        "mode": receipt.mode,
        "availability": receipt.availability,
        "status": receipt.status,
        "unavailable_reason": receipt.unavailable_reason,
        "session_id": receipt.session_id,
        "turn_id": receipt.turn_id,
        "request_id": receipt.request_id,
        "generation_id": receipt.generation_id,
        "research_run_id": receipt.research_run_id,
        "research_receipt_sha256": receipt.research_receipt_sha256,
        "evidence_pipeline_id": receipt.evidence_pipeline_id,
        "evidence_receipt_sha256": receipt.evidence_receipt_sha256,
        "information_grounding_sha256": receipt.information_grounding_sha256,
        "projection_sha256": receipt.projection_sha256,
        "conversation_packet_id": receipt.conversation_packet_id,
        "conversation_packet_sha256": receipt.conversation_packet_sha256,
        "response_sha256": receipt.response_sha256,
        "validation_sha256": receipt.validation_sha256,
        "source_summaries": [item.metadata_record() for item in receipt.source_summaries],
        "policy_versions": list(receipt.policy_versions),
        "created_at": receipt.created_at,
    }
    if not include_ids:
        payload.pop("adapter_id")
    return payload


def _adapter_id(receipt: "InformationResearchModeReceipt") -> str:
    digest = hashlib.sha256(
        _canonical_json(_receipt_payload(receipt, include_ids=False)).encode("utf-8")
    ).hexdigest()
    return f"research-turn-{digest[:20]}"


def _receipt_sha256(receipt: "InformationResearchModeReceipt") -> str:
    return hashlib.sha256(
        _canonical_json(_receipt_payload(receipt)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class InformationResearchModeReceipt:
    """Raw-content-free binding for one explicit P4.7a conversation turn."""

    adapter_id: str
    policy_version: str
    mode: str
    availability: str
    status: str
    unavailable_reason: str | None
    session_id: str
    turn_id: str
    request_id: str
    generation_id: str
    research_run_id: str | None
    research_receipt_sha256: str | None
    evidence_pipeline_id: str | None
    evidence_receipt_sha256: str | None
    information_grounding_sha256: str | None
    projection_sha256: str | None
    conversation_packet_id: str | None
    conversation_packet_sha256: str | None
    response_sha256: str | None
    validation_sha256: str | None
    source_summaries: tuple[InformationResearchModeSourceSummary, ...]
    policy_versions: tuple[str, ...]
    created_at: str
    receipt_sha256: str

    @classmethod
    def create(cls, **values: object) -> "InformationResearchModeReceipt":
        draft = cls(
            adapter_id="research-turn-pending",
            receipt_sha256="0" * 64,
            **values,
        )  # type: ignore[arg-type]
        identified = cls(**{**draft.__dict__, "adapter_id": _adapter_id(draft)})
        receipt = cls(
            **{
                **identified.__dict__,
                "receipt_sha256": _receipt_sha256(identified),
            }
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        _text(self.adapter_id, "adapter_id")
        if self.policy_version != "1.0.0":
            raise InformationResearchModeError(
                "Research-mode receipt must bind policy version 1.0.0.",
                code="research_mode_binding_invalid",
            )
        if self.mode not in {"local_only", "research"}:
            raise InformationResearchModeError(
                "Research-mode receipt mode is not recognized.",
                code="research_mode_binding_invalid",
            )
        if self.availability not in {"not_requested", "available", "offline", "unavailable"}:
            raise InformationResearchModeError(
                "Research-mode availability is not recognized.",
                code="research_mode_binding_invalid",
            )
        if self.status not in {"completed", "unavailable"}:
            raise InformationResearchModeError(
                "Research-mode status is not recognized.",
                code="research_mode_binding_invalid",
            )
        for field_name in ("session_id", "turn_id", "request_id", "generation_id"):
            _text(getattr(self, field_name), field_name)
        for field_name in (
            "research_receipt_sha256",
            "evidence_receipt_sha256",
            "information_grounding_sha256",
            "projection_sha256",
            "conversation_packet_sha256",
            "response_sha256",
            "validation_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _digest(value, field_name)
        for field_name in (
            "research_run_id",
            "evidence_pipeline_id",
            "conversation_packet_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)
        if len(self.policy_versions) != 4 or any(
            not isinstance(value, str) or "@" not in value
            for value in self.policy_versions
        ):
            raise InformationResearchModeError(
                "Research-mode policy-version bindings are incomplete.",
                code="research_mode_binding_invalid",
            )
        if len(self.source_summaries) > APPROVED_MAX_SOURCE_SUMMARIES:
            raise InformationResearchModeError(
                "Research-mode source-summary budget exceeded.",
                code="research_mode_binding_invalid",
            )
        seen_tokens: set[str] = set()
        for summary in self.source_summaries:
            summary.validate()
            if summary.citation_token in seen_tokens:
                raise InformationResearchModeError(
                    "Research-mode source summaries cannot repeat citation tokens.",
                    code="research_mode_binding_invalid",
                )
            seen_tokens.add(summary.citation_token)
        _timestamp(self.created_at, "created_at")
        web_fields = (
            self.research_run_id,
            self.research_receipt_sha256,
            self.evidence_pipeline_id,
            self.evidence_receipt_sha256,
            self.information_grounding_sha256,
            self.projection_sha256,
            self.validation_sha256,
        )
        if self.status == "unavailable":
            if (
                self.mode != "research"
                or self.availability not in {"offline", "unavailable"}
                or self.unavailable_reason != self.availability
                or any(value is not None for value in web_fields)
                or self.conversation_packet_id is not None
                or self.conversation_packet_sha256 is not None
                or self.response_sha256 is not None
                or self.source_summaries
            ):
                raise InformationResearchModeError(
                    "Unavailable research-mode metadata is inconsistent.",
                    code="research_mode_binding_invalid",
                )
        elif self.mode == "local_only":
            if (
                self.availability != "not_requested"
                or self.unavailable_reason is not None
                or any(value is not None for value in web_fields)
                or self.source_summaries
                or self.response_sha256 is None
            ):
                raise InformationResearchModeError(
                    "Local-only turn metadata is inconsistent.",
                    code="research_mode_binding_invalid",
                )
        else:
            required = (
                self.research_run_id,
                self.research_receipt_sha256,
                self.evidence_pipeline_id,
                self.evidence_receipt_sha256,
                self.information_grounding_sha256,
                self.projection_sha256,
                self.conversation_packet_id,
                self.conversation_packet_sha256,
                self.response_sha256,
                self.validation_sha256,
            )
            if (
                self.availability != "available"
                or self.unavailable_reason is not None
                or any(value is None for value in required)
            ):
                raise InformationResearchModeError(
                    "Completed research-turn metadata is incomplete.",
                    code="research_mode_binding_invalid",
                )
        if self.adapter_id != _adapter_id(self):
            raise InformationResearchModeError(
                "Research-mode adapter ID does not match its metadata.",
                code="research_mode_binding_invalid",
            )
        if _digest(self.receipt_sha256, "receipt_sha256") != _receipt_sha256(self):
            raise InformationResearchModeError(
                "Research-mode receipt digest does not match its metadata.",
                code="research_mode_binding_invalid",
            )

    def to_metadata_record(self) -> dict[str, object]:
        self.validate()
        return {**_receipt_payload(self), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class InformationResearchModeTurnResult:
    """One explicit local-only, web-grounded, or unavailable turn outcome."""

    mode: str
    availability: str
    status: str
    conversation_result: ConversationTurnResult | None
    projection: InformationConversationGroundingProjection | None
    response_validation: InformationConversationResponseValidation | None
    source_summaries: tuple[InformationResearchModeSourceSummary, ...]
    receipt: InformationResearchModeReceipt

    def validate(
        self,
        *,
        adapter: "DeterministicInformationResearchModeAdapter",
        command: ConversationTurnCommand,
        evidence_result: InformationResearchEvidenceResult | None,
    ) -> None:
        adapter._validate_result(
            result=self,
            command=command,
            evidence_result=evidence_result,
        )


@dataclass(frozen=True)
class DeterministicInformationResearchModeAdapter:
    """Register verified P4 evidence in one explicit Phase 3 turn."""

    orchestrator: InformationConversationTurnRunner
    evidence_pipeline: DeterministicInformationResearchEvidencePipeline
    bridge_policy: InformationConversationBridgePolicy
    response_validation_policy: ConversationResponseValidationPolicy
    mode_policy: InformationResearchModePolicy
    clock: Callable[[], str]

    def __post_init__(self) -> None:
        if not callable(getattr(self.orchestrator, "run_turn", None)):
            raise InformationResearchModeError(
                "Research-mode adapter requires a conversation turn runner.",
                code="research_mode_configuration_invalid",
            )
        if not callable(self.clock):
            raise InformationResearchModeError(
                "Research-mode adapter clock must be callable.",
                code="research_mode_configuration_invalid",
            )
        self.mode_policy.validate(
            evidence_policy=self.evidence_pipeline.evidence_policy,
            bridge_policy=self.bridge_policy,
            response_validation_policy=self.response_validation_policy,
        )

    def run_turn(
        self,
        command: ConversationTurnCommand,
        *,
        mode: str,
        availability: str,
        evidence_result: InformationResearchEvidenceResult | None = None,
        cancellation: object | None = None,
    ) -> InformationResearchModeTurnResult:
        command.validate()
        self.mode_policy.validate(
            evidence_policy=self.evidence_pipeline.evidence_policy,
            bridge_policy=self.bridge_policy,
            response_validation_policy=self.response_validation_policy,
        )
        if mode not in self.mode_policy.allowed_modes:
            raise InformationResearchModeError(
                "An explicit recognized research mode is required.",
                code="research_mode_required",
            )
        if availability not in self.mode_policy.allowed_availability_states:
            raise InformationResearchModeError(
                "Research availability state is not recognized.",
                code="research_availability_invalid",
            )
        if mode == "local_only":
            if availability != "not_requested" or evidence_result is not None:
                raise InformationResearchModeError(
                    "Local-only turns cannot receive research evidence or availability.",
                    code="local_only_research_forbidden",
                )
            if _contains_web_grounding(command.grounding):
                raise InformationResearchModeError(
                    "Local-only turns cannot contain web grounding.",
                    code="local_only_web_grounding_forbidden",
                )
            conversation_result = self.orchestrator.run_turn(
                command,
                cancellation=cancellation,
            )
            receipt = self._completed_local_receipt(
                command=command,
                result=conversation_result,
            )
            output = InformationResearchModeTurnResult(
                mode=mode,
                availability=availability,
                status="completed",
                conversation_result=conversation_result,
                projection=None,
                response_validation=None,
                source_summaries=(),
                receipt=receipt,
            )
            self._validate_result(
                result=output,
                command=command,
                evidence_result=None,
            )
            return output
        if command.grounding is not None:
            raise InformationResearchModeError(
                "P4.7a research turns require the adapter to inject the only grounding packet.",
                code="research_grounding_preinjected",
            )
        if availability in {"offline", "unavailable"}:
            if evidence_result is not None:
                raise InformationResearchModeError(
                    "Unavailable research cannot consume a supplied evidence result.",
                    code="research_unavailable_evidence_forbidden",
                )
            receipt = InformationResearchModeReceipt.create(
                policy_version=self.mode_policy.version,
                mode="research",
                availability=availability,
                status="unavailable",
                unavailable_reason=availability,
                session_id=command.session_id,
                turn_id=command.turn_id,
                request_id=command.request_id,
                generation_id=command.generation_id,
                research_run_id=None,
                research_receipt_sha256=None,
                evidence_pipeline_id=None,
                evidence_receipt_sha256=None,
                information_grounding_sha256=None,
                projection_sha256=None,
                conversation_packet_id=None,
                conversation_packet_sha256=None,
                response_sha256=None,
                validation_sha256=None,
                source_summaries=(),
                policy_versions=self._policy_versions(),
                created_at=self.clock(),
            )
            output = InformationResearchModeTurnResult(
                mode="research",
                availability=availability,
                status="unavailable",
                conversation_result=None,
                projection=None,
                response_validation=None,
                source_summaries=(),
                receipt=receipt,
            )
            self._validate_result(
                result=output,
                command=command,
                evidence_result=None,
            )
            return output
        if availability != "available" or evidence_result is None:
            raise InformationResearchModeError(
                "Available research mode requires one verified evidence result.",
                code="research_evidence_required",
            )
        evidence_result.validate(pipeline=self.evidence_pipeline)
        query = evidence_result.research_run.request.query
        projection = project_information_grounding_to_conversation(
            verified_grounding=evidence_result.grounding,
            query=query,
            qualified_sources=evidence_result.qualified_sources,
            information_policy=self.evidence_pipeline.information_policy,
            firewall_policy=self.evidence_pipeline.firewall_policy,
            freshness_policy=self.evidence_pipeline.freshness_policy,
            grounding_policy=self.evidence_pipeline.grounding_policy,
            bridge_policy=self.bridge_policy,
        )
        summaries = _source_summaries(projection)
        if len(summaries) > self.mode_policy.max_source_summaries:
            raise InformationResearchModeError(
                "Research-mode source-summary budget exceeded.",
                code="research_mode_budget_exceeded",
            )
        selected = replace(command, grounding=projection.conversation_packet)
        validations: list[InformationConversationResponseValidation] = []

        def validate_before_commit(
            response: ModelResponse,
            grounding: ConversationGroundingPacket | None,
        ) -> None:
            if grounding != projection.conversation_packet:
                raise ConversationContractError(
                    "Research response validation received substituted grounding."
                )
            try:
                validation = validate_information_conversation_response(
                    response=response,
                    projection=projection,
                    verified_grounding=evidence_result.grounding,
                    query=query,
                    qualified_sources=evidence_result.qualified_sources,
                    information_policy=self.evidence_pipeline.information_policy,
                    firewall_policy=self.evidence_pipeline.firewall_policy,
                    freshness_policy=self.evidence_pipeline.freshness_policy,
                    grounding_policy=self.evidence_pipeline.grounding_policy,
                    bridge_policy=self.bridge_policy,
                    response_validation_policy=self.response_validation_policy,
                )
            except Exception as exc:
                raise ConversationContractError(
                    "Research response failed the registered P4 validation boundary."
                ) from exc
            if validation.report.outcome == "rejected":
                raise ConversationContractError(
                    "Research response was rejected by the registered P4 validation boundary."
                )
            validations.append(validation)

        conversation_result = self.orchestrator.run_turn(
            selected,
            cancellation=cancellation,
            response_validation_hook=validate_before_commit,
        )
        if conversation_result.replayed:
            validation = validate_information_conversation_response(
                response=conversation_result.response,
                projection=projection,
                verified_grounding=evidence_result.grounding,
                query=query,
                qualified_sources=evidence_result.qualified_sources,
                information_policy=self.evidence_pipeline.information_policy,
                firewall_policy=self.evidence_pipeline.firewall_policy,
                freshness_policy=self.evidence_pipeline.freshness_policy,
                grounding_policy=self.evidence_pipeline.grounding_policy,
                bridge_policy=self.bridge_policy,
                response_validation_policy=self.response_validation_policy,
            )
            if validation.report.outcome == "rejected":
                raise InformationResearchModeError(
                    "Replayed research response failed P4 validation.",
                    code="research_response_validation_failed",
                )
        else:
            if len(validations) != 1:
                raise InformationResearchModeError(
                    "Research response was not validated exactly once before commit.",
                    code="research_response_validation_missing",
                )
            validation = validations[0]
        receipt = self._completed_research_receipt(
            command=command,
            result=conversation_result,
            evidence_result=evidence_result,
            projection=projection,
            validation=validation,
            summaries=summaries,
        )
        output = InformationResearchModeTurnResult(
            mode="research",
            availability="available",
            status="completed",
            conversation_result=conversation_result,
            projection=projection,
            response_validation=validation,
            source_summaries=summaries,
            receipt=receipt,
        )
        self._validate_result(
            result=output,
            command=command,
            evidence_result=evidence_result,
        )
        return output

    def _policy_versions(self) -> tuple[str, ...]:
        return (
            f"{self.mode_policy.policy_name}@{self.mode_policy.version}",
            f"{self.evidence_pipeline.evidence_policy.policy_name}@{self.evidence_pipeline.evidence_policy.version}",
            f"{self.bridge_policy.policy_name}@{self.bridge_policy.version}",
            f"{self.response_validation_policy.policy_name}@{self.response_validation_policy.version}",
        )

    def _completed_local_receipt(
        self,
        *,
        command: ConversationTurnCommand,
        result: ConversationTurnResult,
    ) -> InformationResearchModeReceipt:
        return InformationResearchModeReceipt.create(
            policy_version=self.mode_policy.version,
            mode="local_only",
            availability="not_requested",
            status="completed",
            unavailable_reason=None,
            session_id=command.session_id,
            turn_id=command.turn_id,
            request_id=result.request_id,
            generation_id=result.generation_id,
            research_run_id=None,
            research_receipt_sha256=None,
            evidence_pipeline_id=None,
            evidence_receipt_sha256=None,
            information_grounding_sha256=None,
            projection_sha256=None,
            conversation_packet_id=result.grounding_packet_id,
            conversation_packet_sha256=result.grounding_packet_sha256,
            response_sha256=_model_response_sha256(result.response),
            validation_sha256=None,
            source_summaries=(),
            policy_versions=self._policy_versions(),
            created_at=self.clock(),
        )

    def _completed_research_receipt(
        self,
        *,
        command: ConversationTurnCommand,
        result: ConversationTurnResult,
        evidence_result: InformationResearchEvidenceResult,
        projection: InformationConversationGroundingProjection,
        validation: InformationConversationResponseValidation,
        summaries: tuple[InformationResearchModeSourceSummary, ...],
    ) -> InformationResearchModeReceipt:
        return InformationResearchModeReceipt.create(
            policy_version=self.mode_policy.version,
            mode="research",
            availability="available",
            status="completed",
            unavailable_reason=None,
            session_id=command.session_id,
            turn_id=command.turn_id,
            request_id=result.request_id,
            generation_id=result.generation_id,
            research_run_id=evidence_result.research_run.receipt.run_id,
            research_receipt_sha256=evidence_result.research_run.receipt.receipt_sha256,
            evidence_pipeline_id=evidence_result.receipt.pipeline_id,
            evidence_receipt_sha256=evidence_result.receipt.receipt_sha256,
            information_grounding_sha256=evidence_result.grounding.grounding_sha256,
            projection_sha256=projection.receipt.projection_sha256,
            conversation_packet_id=projection.conversation_packet.packet_id,
            conversation_packet_sha256=projection.receipt.conversation_packet_sha256,
            response_sha256=_model_response_sha256(result.response),
            validation_sha256=validation.receipt.validation_sha256,
            source_summaries=summaries,
            policy_versions=self._policy_versions(),
            created_at=self.clock(),
        )

    def _validate_result(
        self,
        *,
        result: InformationResearchModeTurnResult,
        command: ConversationTurnCommand,
        evidence_result: InformationResearchEvidenceResult | None,
    ) -> None:
        command.validate()
        self.mode_policy.validate(
            evidence_policy=self.evidence_pipeline.evidence_policy,
            bridge_policy=self.bridge_policy,
            response_validation_policy=self.response_validation_policy,
        )
        result.receipt.validate()
        if (
            result.mode != result.receipt.mode
            or result.availability != result.receipt.availability
            or result.status != result.receipt.status
            or result.source_summaries != result.receipt.source_summaries
        ):
            raise InformationResearchModeError(
                "Research-mode result does not match its receipt.",
                code="research_mode_binding_invalid",
            )
        if (
            result.receipt.session_id != command.session_id
            or result.receipt.turn_id != command.turn_id
            or result.receipt.request_id != command.request_id
            or result.receipt.generation_id != command.generation_id
            or result.receipt.policy_versions != self._policy_versions()
        ):
            raise InformationResearchModeError(
                "Research-mode command binding does not match.",
                code="research_mode_binding_invalid",
            )
        if result.status == "unavailable":
            if any(
                value is not None
                for value in (
                    result.conversation_result,
                    result.projection,
                    result.response_validation,
                    evidence_result,
                )
            ):
                raise InformationResearchModeError(
                    "Unavailable research result contains execution artifacts.",
                    code="research_mode_binding_invalid",
                )
            return
        if result.conversation_result is None:
            raise InformationResearchModeError(
                "Completed research-mode result requires a conversation result.",
                code="research_mode_binding_invalid",
            )
        result.conversation_result.validate()
        if (
            result.conversation_result.session_id != command.session_id
            or result.conversation_result.turn_id != command.turn_id
            or result.receipt.request_id != result.conversation_result.request_id
            or result.receipt.generation_id != result.conversation_result.generation_id
            or result.receipt.response_sha256
            != _model_response_sha256(result.conversation_result.response)
        ):
            raise InformationResearchModeError(
                "Conversation result binding does not match the research-mode receipt.",
                code="research_mode_binding_invalid",
            )
        if (
            result.receipt.conversation_packet_id
            != result.conversation_result.grounding_packet_id
            or result.receipt.conversation_packet_sha256
            != result.conversation_result.grounding_packet_sha256
        ):
            raise InformationResearchModeError(
                "Conversation grounding does not match the research-mode receipt.",
                code="research_mode_binding_invalid",
            )
        if result.mode == "local_only":
            if any(
                value is not None
                for value in (
                    evidence_result,
                    result.projection,
                    result.response_validation,
                )
            ) or _contains_web_grounding(command.grounding):
                raise InformationResearchModeError(
                    "Local-only result contains web research artifacts.",
                    code="research_mode_binding_invalid",
                )
            return
        if evidence_result is None or result.projection is None or result.response_validation is None:
            raise InformationResearchModeError(
                "Completed research result is missing verified artifacts.",
                code="research_mode_binding_invalid",
            )
        evidence_result.validate(pipeline=self.evidence_pipeline)
        query = evidence_result.research_run.request.query
        result.projection.validate(
            verified_grounding=evidence_result.grounding,
            query=query,
            qualified_sources=evidence_result.qualified_sources,
            information_policy=self.evidence_pipeline.information_policy,
            firewall_policy=self.evidence_pipeline.firewall_policy,
            freshness_policy=self.evidence_pipeline.freshness_policy,
            grounding_policy=self.evidence_pipeline.grounding_policy,
            bridge_policy=self.bridge_policy,
        )
        result.response_validation.validate(
            response=result.conversation_result.response,
            projection=result.projection,
            verified_grounding=evidence_result.grounding,
            query=query,
            qualified_sources=evidence_result.qualified_sources,
            information_policy=self.evidence_pipeline.information_policy,
            firewall_policy=self.evidence_pipeline.firewall_policy,
            freshness_policy=self.evidence_pipeline.freshness_policy,
            grounding_policy=self.evidence_pipeline.grounding_policy,
            bridge_policy=self.bridge_policy,
            response_validation_policy=self.response_validation_policy,
        )
        expected_summaries = _source_summaries(result.projection)
        if result.source_summaries != expected_summaries:
            raise InformationResearchModeError(
                "Research source summaries do not match the exact projection.",
                code="research_mode_binding_invalid",
            )
        receipt = result.receipt
        expected = (
            evidence_result.research_run.receipt.run_id,
            evidence_result.research_run.receipt.receipt_sha256,
            evidence_result.receipt.pipeline_id,
            evidence_result.receipt.receipt_sha256,
            evidence_result.grounding.grounding_sha256,
            result.projection.receipt.projection_sha256,
            result.projection.conversation_packet.packet_id,
            result.projection.receipt.conversation_packet_sha256,
            result.response_validation.receipt.validation_sha256,
        )
        supplied = (
            receipt.research_run_id,
            receipt.research_receipt_sha256,
            receipt.evidence_pipeline_id,
            receipt.evidence_receipt_sha256,
            receipt.information_grounding_sha256,
            receipt.projection_sha256,
            receipt.conversation_packet_id,
            receipt.conversation_packet_sha256,
            receipt.validation_sha256,
        )
        if supplied != expected:
            raise InformationResearchModeError(
                "Research-mode evidence and projection bindings do not match.",
                code="research_mode_binding_invalid",
            )
        if (
            result.conversation_result.grounding_packet_id
            != result.projection.conversation_packet.packet_id
            or result.conversation_result.grounding_packet_sha256
            != result.projection.receipt.conversation_packet_sha256
        ):
            raise InformationResearchModeError(
                "Conversation result does not bind the expected web grounding packet.",
                code="research_mode_binding_invalid",
            )
