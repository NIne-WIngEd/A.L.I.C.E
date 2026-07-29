"""Governed fixture research execution for Phase 4 P4.7b."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .research_execution_policy import InformationResearchExecutionPolicy

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_CLAIM_OUTCOMES = {"answerable", "conflict", "uncertain"}
_NO_EVIDENCE_RUN_OUTCOMES = {"insufficient_sources", "cancelled", "failed"}
_RUN_FAILURE_REASONS = {
    "insufficient_sources": "insufficient_sources",
    "cancelled": "research_cancelled",
    "failed": "research_failed",
}


class InformationResearchExecutionError(RuntimeError):
    """Raised when a P4.7b execution cannot be safely built or verified."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchExecutionError(
            f"{field} must be non-empty text.",
            code="research_execution_binding_invalid",
        )
    return value.strip()


def _optional_text(value: object | None, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _digest(value: object, field: str) -> str:
    text = _text(value, field).lower()
    if _SHA256.fullmatch(text) is None:
        raise InformationResearchExecutionError(
            f"{field} must be a lowercase SHA-256 digest.",
            code="research_execution_binding_invalid",
        )
    return text


def _optional_digest(value: object | None, field: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field)


def _timestamp(value: object, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InformationResearchExecutionError(
            f"{field} must be valid ISO-8601 text.",
            code="research_execution_binding_invalid",
        ) from exc
    if parsed.tzinfo is None:
        raise InformationResearchExecutionError(
            f"{field} must include a timezone offset.",
            code="research_execution_binding_invalid",
        )
    return text


def _optional_timestamp(value: object | None, field: str) -> str | None:
    if value is None:
        return None
    return _timestamp(value, field)


def _provider(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchExecutionError(
            f"{field} must use an exact lowercase provider identifier.",
            code="research_execution_provider_invalid",
        )
    text = value.strip()
    if _PROVIDER.fullmatch(text) is None:
        raise InformationResearchExecutionError(
            f"{field} must use an exact lowercase provider identifier.",
            code="research_execution_provider_invalid",
        )
    return text


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class InformationResearchExecutionPlan:
    """Explicit inputs selected before one P4.7b execution begins."""

    mode: str
    availability: str
    research_request: object | None = None
    search_provider: str | None = None
    fetch_provider: str | None = None
    reference_time: str | None = None
    evidence_outcome: str | None = None
    claim_drafts: tuple[object, ...] = ()
    created_at: str | None = None
    window_start: str | None = None
    window_end: str | None = None

    def validate(self, *, policy: InformationResearchExecutionPolicy, command: object) -> None:
        policy.validate()
        validator = getattr(command, "validate", None)
        if not callable(validator):
            raise InformationResearchExecutionError(
                "Research execution requires a valid conversation command.",
                code="research_execution_command_invalid",
            )
        validator()
        if self.mode not in policy.allowed_modes:
            raise InformationResearchExecutionError(
                "An explicit recognized research execution mode is required.",
                code="research_execution_mode_required",
            )
        if self.availability not in policy.allowed_requested_availability_states:
            raise InformationResearchExecutionError(
                "Research execution availability is not recognized.",
                code="research_execution_availability_invalid",
            )
        if not isinstance(self.claim_drafts, tuple):
            raise InformationResearchExecutionError(
                "Research claim drafts must use an immutable tuple.",
                code="research_execution_plan_invalid",
            )
        research_values = (
            self.research_request,
            self.search_provider,
            self.fetch_provider,
            self.reference_time,
            self.evidence_outcome,
            self.created_at,
            self.window_start,
            self.window_end,
        )
        if self.mode == "local_only":
            if self.availability != "not_requested" or any(
                value is not None for value in research_values
            ) or self.claim_drafts:
                raise InformationResearchExecutionError(
                    "Local-only execution cannot contain research inputs.",
                    code="local_only_research_execution_forbidden",
                )
            return
        if getattr(command, "grounding", None) is not None:
            raise InformationResearchExecutionError(
                "Research execution requires the P4 path to inject the only grounding packet.",
                code="research_execution_grounding_preinjected",
            )
        if self.availability in {"offline", "unavailable"}:
            if any(value is not None for value in research_values) or self.claim_drafts:
                raise InformationResearchExecutionError(
                    "Unavailable research preflight cannot contain execution inputs.",
                    code="research_execution_unavailable_inputs_forbidden",
                )
            return
        if self.availability != "available":
            raise InformationResearchExecutionError(
                "Research mode requires an available, offline, or unavailable state.",
                code="research_execution_availability_invalid",
            )
        required = (
            self.research_request,
            self.search_provider,
            self.fetch_provider,
            self.reference_time,
            self.evidence_outcome,
            self.created_at,
        )
        if any(value is None for value in required):
            raise InformationResearchExecutionError(
                "Available research requires a complete execution plan.",
                code="research_execution_plan_incomplete",
            )
        request_validator = getattr(self.research_request, "validate", None)
        if not callable(request_validator):
            raise InformationResearchExecutionError(
                "Available research requires a valid research request.",
                code="research_execution_plan_invalid",
            )
        request_validator()
        _provider(self.search_provider, "search_provider")
        _provider(self.fetch_provider, "fetch_provider")
        _timestamp(self.reference_time, "reference_time")
        _timestamp(self.created_at, "created_at")
        _optional_timestamp(self.window_start, "window_start")
        _optional_timestamp(self.window_end, "window_end")
        if self.evidence_outcome not in {
            "answerable",
            "conflict",
            "uncertain",
            "insufficient_sources",
        }:
            raise InformationResearchExecutionError(
                "Evidence outcome is not approved for P4.7b.",
                code="research_execution_plan_invalid",
            )
        if self.evidence_outcome == "insufficient_sources" and self.claim_drafts:
            raise InformationResearchExecutionError(
                "Insufficient-source execution cannot contain claim drafts.",
                code="research_execution_plan_invalid",
            )
        if self.evidence_outcome in _CLAIM_OUTCOMES and not self.claim_drafts:
            raise InformationResearchExecutionError(
                "Claim-bearing evidence outcomes require explicit claim drafts.",
                code="research_execution_plan_invalid",
            )
        for draft in self.claim_drafts:
            draft_validator = getattr(draft, "validate_shape", None)
            if not callable(draft_validator):
                raise InformationResearchExecutionError(
                    "Research execution received an invalid claim draft.",
                    code="research_execution_plan_invalid",
                )
            draft_validator()


def _receipt_payload(
    receipt: "InformationResearchExecutionReceipt",
    *,
    include_id: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "execution_id": receipt.execution_id,
        "policy_version": receipt.policy_version,
        "mode": receipt.mode,
        "requested_availability": receipt.requested_availability,
        "availability": receipt.availability,
        "status": receipt.status,
        "unavailable_reason": receipt.unavailable_reason,
        "session_id": receipt.session_id,
        "turn_id": receipt.turn_id,
        "request_id": receipt.request_id,
        "generation_id": receipt.generation_id,
        "research_request_id": receipt.research_request_id,
        "query_id": receipt.query_id,
        "query_sha256": receipt.query_sha256,
        "search_provider": receipt.search_provider,
        "fetch_provider": receipt.fetch_provider,
        "research_run_id": receipt.research_run_id,
        "research_receipt_sha256": receipt.research_receipt_sha256,
        "research_outcome": receipt.research_outcome,
        "research_stopping_reason": receipt.research_stopping_reason,
        "evidence_pipeline_id": receipt.evidence_pipeline_id,
        "evidence_receipt_sha256": receipt.evidence_receipt_sha256,
        "evidence_outcome": receipt.evidence_outcome,
        "mode_adapter_id": receipt.mode_adapter_id,
        "mode_receipt_sha256": receipt.mode_receipt_sha256,
        "policy_versions": list(receipt.policy_versions),
        "created_at": receipt.created_at,
    }
    if not include_id:
        payload.pop("execution_id")
    return payload


def _execution_id(receipt: "InformationResearchExecutionReceipt") -> str:
    digest = hashlib.sha256(
        _canonical_json(_receipt_payload(receipt, include_id=False)).encode("utf-8")
    ).hexdigest()
    return f"research-execution-{digest[:20]}"


def _receipt_sha256(receipt: "InformationResearchExecutionReceipt") -> str:
    return hashlib.sha256(
        _canonical_json(_receipt_payload(receipt)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class InformationResearchExecutionReceipt:
    """Raw-content-free binding for one complete P4.7b decision path."""

    execution_id: str
    policy_version: str
    mode: str
    requested_availability: str
    availability: str
    status: str
    unavailable_reason: str | None
    session_id: str
    turn_id: str
    request_id: str
    generation_id: str
    research_request_id: str | None
    query_id: str | None
    query_sha256: str | None
    search_provider: str | None
    fetch_provider: str | None
    research_run_id: str | None
    research_receipt_sha256: str | None
    research_outcome: str | None
    research_stopping_reason: str | None
    evidence_pipeline_id: str | None
    evidence_receipt_sha256: str | None
    evidence_outcome: str | None
    mode_adapter_id: str
    mode_receipt_sha256: str
    policy_versions: tuple[str, ...]
    created_at: str
    receipt_sha256: str

    @classmethod
    def create(cls, **values: object) -> "InformationResearchExecutionReceipt":
        draft = cls(
            execution_id="research-execution-pending",
            receipt_sha256="0" * 64,
            **values,
        )  # type: ignore[arg-type]
        identified = cls(**{**draft.__dict__, "execution_id": _execution_id(draft)})
        receipt = cls(
            **{
                **identified.__dict__,
                "receipt_sha256": _receipt_sha256(identified),
            }
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        _text(self.execution_id, "execution_id")
        if self.policy_version != "1.0.0":
            raise InformationResearchExecutionError(
                "Research-execution receipt must bind policy version 1.0.0.",
                code="research_execution_binding_invalid",
            )
        if self.mode not in {"local_only", "research"}:
            raise InformationResearchExecutionError(
                "Research-execution receipt mode is invalid.",
                code="research_execution_binding_invalid",
            )
        allowed_availability = {"not_requested", "available", "offline", "unavailable"}
        if (
            self.requested_availability not in allowed_availability
            or self.availability not in allowed_availability
            or self.status not in {"completed", "unavailable"}
        ):
            raise InformationResearchExecutionError(
                "Research-execution status metadata is invalid.",
                code="research_execution_binding_invalid",
            )
        for field in ("session_id", "turn_id", "request_id", "generation_id"):
            _text(getattr(self, field), field)
        for field in (
            "research_request_id",
            "query_id",
            "research_run_id",
            "research_outcome",
            "research_stopping_reason",
            "evidence_pipeline_id",
            "evidence_outcome",
        ):
            _optional_text(getattr(self, field), field)
        for field in (
            "query_sha256",
            "research_receipt_sha256",
            "evidence_receipt_sha256",
        ):
            _optional_digest(getattr(self, field), field)
        if self.search_provider is not None:
            _provider(self.search_provider, "search_provider")
        if self.fetch_provider is not None:
            _provider(self.fetch_provider, "fetch_provider")
        _text(self.mode_adapter_id, "mode_adapter_id")
        _digest(self.mode_receipt_sha256, "mode_receipt_sha256")
        if len(self.policy_versions) != 4 or any(
            not isinstance(item, str) or "@" not in item
            for item in self.policy_versions
        ):
            raise InformationResearchExecutionError(
                "Research-execution policy bindings are incomplete.",
                code="research_execution_binding_invalid",
            )
        _timestamp(self.created_at, "created_at")
        research_identity = (
            self.research_request_id,
            self.query_id,
            self.query_sha256,
            self.search_provider,
            self.fetch_provider,
        )
        run_identity = (
            self.research_run_id,
            self.research_receipt_sha256,
            self.research_outcome,
            self.research_stopping_reason,
        )
        evidence_identity = (
            self.evidence_pipeline_id,
            self.evidence_receipt_sha256,
            self.evidence_outcome,
        )
        if self.mode == "local_only":
            if (
                self.requested_availability != "not_requested"
                or self.availability != "not_requested"
                or self.status != "completed"
                or self.unavailable_reason is not None
                or any(value is not None for value in research_identity + run_identity + evidence_identity)
            ):
                raise InformationResearchExecutionError(
                    "Local-only execution receipt contains research metadata.",
                    code="research_execution_binding_invalid",
                )
        elif self.requested_availability in {"offline", "unavailable"}:
            if (
                self.availability != self.requested_availability
                or self.status != "unavailable"
                or self.unavailable_reason != self.requested_availability
                or any(value is not None for value in research_identity + run_identity + evidence_identity)
            ):
                raise InformationResearchExecutionError(
                    "Preflight-unavailable execution metadata is inconsistent.",
                    code="research_execution_binding_invalid",
                )
        elif self.status == "unavailable":
            if (
                self.requested_availability != "available"
                or self.availability != "unavailable"
                or self.unavailable_reason not in {
                    "research_cancelled",
                    "research_failed",
                    "insufficient_sources",
                    "insufficient_evidence",
                }
                or any(value is None for value in research_identity + run_identity)
            ):
                raise InformationResearchExecutionError(
                    "Post-execution unavailable metadata is inconsistent.",
                    code="research_execution_binding_invalid",
                )
            if self.unavailable_reason == "insufficient_evidence":
                if any(value is None for value in evidence_identity) or self.evidence_outcome != "insufficient_sources":
                    raise InformationResearchExecutionError(
                        "Insufficient-evidence metadata is incomplete.",
                        code="research_execution_binding_invalid",
                    )
            elif any(value is not None for value in evidence_identity):
                raise InformationResearchExecutionError(
                    "Run-level unavailability cannot contain evidence metadata.",
                    code="research_execution_binding_invalid",
                )
        else:
            if (
                self.requested_availability != "available"
                or self.availability != "available"
                or self.status != "completed"
                or self.unavailable_reason is not None
                or any(value is None for value in research_identity + run_identity + evidence_identity)
                or self.research_outcome not in {"completed", "partial"}
                or self.evidence_outcome not in _CLAIM_OUTCOMES
            ):
                raise InformationResearchExecutionError(
                    "Completed research-execution metadata is incomplete.",
                    code="research_execution_binding_invalid",
                )
        if self.execution_id != _execution_id(self):
            raise InformationResearchExecutionError(
                "Research-execution ID does not match its metadata.",
                code="research_execution_binding_invalid",
            )
        if _digest(self.receipt_sha256, "receipt_sha256") != _receipt_sha256(self):
            raise InformationResearchExecutionError(
                "Research-execution receipt digest does not match its metadata.",
                code="research_execution_binding_invalid",
            )

    def to_metadata_record(self) -> dict[str, object]:
        self.validate()
        return {**_receipt_payload(self), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class InformationResearchExecutionResult:
    """One local, completed research, or fail-clean P4.7b result."""

    mode: str
    requested_availability: str
    availability: str
    status: str
    unavailable_reason: str | None
    research_run: object | None
    evidence_result: object | None
    turn_result: object
    receipt: InformationResearchExecutionReceipt

    def validate(
        self,
        *,
        executor: "DeterministicInformationResearchExecution",
        command: object,
        plan: InformationResearchExecutionPlan,
    ) -> None:
        executor._validate_result(result=self, command=command, plan=plan)


@dataclass(frozen=True)
class DeterministicInformationResearchExecution:
    """Compose bounded research, evidence gates, and one explicit P4.7a turn."""

    research_orchestrator: object
    evidence_pipeline: object
    mode_adapter: object
    execution_policy: InformationResearchExecutionPolicy
    clock: Callable[[], str]

    def __post_init__(self) -> None:
        if not callable(getattr(self.research_orchestrator, "execute", None)):
            raise InformationResearchExecutionError(
                "P4.7b requires a research orchestrator.",
                code="research_execution_configuration_invalid",
            )
        if not callable(getattr(self.evidence_pipeline, "process", None)):
            raise InformationResearchExecutionError(
                "P4.7b requires an evidence pipeline.",
                code="research_execution_configuration_invalid",
            )
        if not callable(getattr(self.mode_adapter, "run_turn", None)):
            raise InformationResearchExecutionError(
                "P4.7b requires an explicit research-mode adapter.",
                code="research_execution_configuration_invalid",
            )
        if not callable(self.clock):
            raise InformationResearchExecutionError(
                "P4.7b clock must be callable.",
                code="research_execution_configuration_invalid",
            )
        orchestration_policy = getattr(self.research_orchestrator, "policy", None)
        pipeline_policy = getattr(self.evidence_pipeline, "orchestration_policy", None)
        if orchestration_policy is None or orchestration_policy is not pipeline_policy:
            raise InformationResearchExecutionError(
                "Research orchestration and evidence must share the exact policy object.",
                code="research_execution_configuration_invalid",
            )
        if getattr(self.mode_adapter, "evidence_pipeline", None) is not self.evidence_pipeline:
            raise InformationResearchExecutionError(
                "Research mode must share the exact selected evidence pipeline.",
                code="research_execution_configuration_invalid",
            )
        self._validate_policy()

    def _validate_policy(self) -> None:
        self.execution_policy.validate(
            orchestration_policy=getattr(self.research_orchestrator, "policy", None),
            evidence_policy=getattr(self.evidence_pipeline, "evidence_policy", None),
            mode_policy=getattr(self.mode_adapter, "mode_policy", None),
        )

    def _policy_versions(self) -> tuple[str, ...]:
        policies = (
            self.execution_policy,
            getattr(self.research_orchestrator, "policy"),
            getattr(self.evidence_pipeline, "evidence_policy"),
            getattr(self.mode_adapter, "mode_policy"),
        )
        return tuple(f"{item.policy_name}@{item.version}" for item in policies)

    def run_turn(
        self,
        command: object,
        *,
        plan: InformationResearchExecutionPlan,
        cancellation: object | None = None,
    ) -> InformationResearchExecutionResult:
        self._validate_policy()
        plan.validate(policy=self.execution_policy, command=command)
        if plan.mode == "local_only":
            turn = self.mode_adapter.run_turn(
                command,
                mode="local_only",
                availability="not_requested",
                cancellation=cancellation,
            )
            result = self._build_result(
                command=command,
                plan=plan,
                availability="not_requested",
                status="completed",
                unavailable_reason=None,
                research_run=None,
                evidence_result=None,
                turn_result=turn,
            )
            self._validate_result(result=result, command=command, plan=plan)
            return result
        if plan.availability in {"offline", "unavailable"}:
            turn = self.mode_adapter.run_turn(
                command,
                mode="research",
                availability=plan.availability,
                cancellation=cancellation,
            )
            result = self._build_result(
                command=command,
                plan=plan,
                availability=plan.availability,
                status="unavailable",
                unavailable_reason=plan.availability,
                research_run=None,
                evidence_result=None,
                turn_result=turn,
            )
            self._validate_result(result=result, command=command, plan=plan)
            return result
        research_run = self.research_orchestrator.execute(
            plan.research_request,
            search_provider=plan.search_provider,
            fetch_provider=plan.fetch_provider,
            cancellation=cancellation,
        )
        research_run.validate(policy=self.research_orchestrator.policy)
        if research_run.request != plan.research_request:
            raise InformationResearchExecutionError(
                "Research orchestrator substituted the approved request.",
                code="research_execution_binding_invalid",
            )
        if (
            research_run.receipt.search_provider != plan.search_provider
            or research_run.receipt.fetch_provider != plan.fetch_provider
        ):
            raise InformationResearchExecutionError(
                "Research orchestrator substituted an approved provider.",
                code="research_execution_provider_substituted",
            )
        if research_run.receipt.outcome in _NO_EVIDENCE_RUN_OUTCOMES:
            reason = _RUN_FAILURE_REASONS[research_run.receipt.outcome]
            turn = self.mode_adapter.run_turn(
                command,
                mode="research",
                availability="unavailable",
                cancellation=cancellation,
            )
            result = self._build_result(
                command=command,
                plan=plan,
                availability="unavailable",
                status="unavailable",
                unavailable_reason=reason,
                research_run=research_run,
                evidence_result=None,
                turn_result=turn,
            )
            self._validate_result(result=result, command=command, plan=plan)
            return result
        if research_run.receipt.outcome not in {"completed", "partial"}:
            raise InformationResearchExecutionError(
                "Research run outcome is not approved for P4.7b.",
                code="research_execution_binding_invalid",
            )
        evidence = self.evidence_pipeline.process(
            research_run=research_run,
            reference_time=plan.reference_time,
            outcome=plan.evidence_outcome,
            claim_drafts=plan.claim_drafts,
            created_at=plan.created_at,
            window_start=plan.window_start,
            window_end=plan.window_end,
        )
        evidence.validate(pipeline=self.evidence_pipeline)
        evidence_outcome = evidence.grounding.packet.outcome
        if evidence_outcome == "insufficient_sources":
            turn = self.mode_adapter.run_turn(
                command,
                mode="research",
                availability="unavailable",
                cancellation=cancellation,
            )
            result = self._build_result(
                command=command,
                plan=plan,
                availability="unavailable",
                status="unavailable",
                unavailable_reason="insufficient_evidence",
                research_run=research_run,
                evidence_result=evidence,
                turn_result=turn,
            )
            self._validate_result(result=result, command=command, plan=plan)
            return result
        if evidence_outcome not in _CLAIM_OUTCOMES:
            raise InformationResearchExecutionError(
                "Evidence outcome is not eligible for a conversation turn.",
                code="research_execution_binding_invalid",
            )
        turn = self.mode_adapter.run_turn(
            command,
            mode="research",
            availability="available",
            evidence_result=evidence,
            cancellation=cancellation,
        )
        result = self._build_result(
            command=command,
            plan=plan,
            availability="available",
            status="completed",
            unavailable_reason=None,
            research_run=research_run,
            evidence_result=evidence,
            turn_result=turn,
        )
        self._validate_result(result=result, command=command, plan=plan)
        return result

    def _build_result(
        self,
        *,
        command: object,
        plan: InformationResearchExecutionPlan,
        availability: str,
        status: str,
        unavailable_reason: str | None,
        research_run: object | None,
        evidence_result: object | None,
        turn_result: object,
    ) -> InformationResearchExecutionResult:
        run_receipt = getattr(research_run, "receipt", None)
        evidence_receipt = getattr(evidence_result, "receipt", None)
        mode_receipt = getattr(turn_result, "receipt", None)
        if mode_receipt is None:
            raise InformationResearchExecutionError(
                "Research-mode result is missing its receipt.",
                code="research_execution_binding_invalid",
            )
        request = plan.research_request
        query = getattr(request, "query", None)
        receipt = InformationResearchExecutionReceipt.create(
            policy_version=self.execution_policy.version,
            mode=plan.mode,
            requested_availability=plan.availability,
            availability=availability,
            status=status,
            unavailable_reason=unavailable_reason,
            session_id=getattr(command, "session_id"),
            turn_id=getattr(command, "turn_id"),
            request_id=getattr(command, "request_id"),
            generation_id=getattr(command, "generation_id"),
            research_request_id=getattr(request, "request_id", None),
            query_id=getattr(query, "query_id", None),
            query_sha256=getattr(query, "content_sha256", None),
            search_provider=plan.search_provider,
            fetch_provider=plan.fetch_provider,
            research_run_id=getattr(run_receipt, "run_id", None),
            research_receipt_sha256=getattr(run_receipt, "receipt_sha256", None),
            research_outcome=getattr(run_receipt, "outcome", None),
            research_stopping_reason=getattr(run_receipt, "stopping_reason", None),
            evidence_pipeline_id=getattr(evidence_receipt, "pipeline_id", None),
            evidence_receipt_sha256=getattr(evidence_receipt, "receipt_sha256", None),
            evidence_outcome=(
                getattr(getattr(getattr(evidence_result, "grounding", None), "packet", None), "outcome", None)
            ),
            mode_adapter_id=getattr(mode_receipt, "adapter_id"),
            mode_receipt_sha256=getattr(mode_receipt, "receipt_sha256"),
            policy_versions=self._policy_versions(),
            created_at=self.clock(),
        )
        return InformationResearchExecutionResult(
            mode=plan.mode,
            requested_availability=plan.availability,
            availability=availability,
            status=status,
            unavailable_reason=unavailable_reason,
            research_run=research_run,
            evidence_result=evidence_result,
            turn_result=turn_result,
            receipt=receipt,
        )

    def _validate_result(
        self,
        *,
        result: InformationResearchExecutionResult,
        command: object,
        plan: InformationResearchExecutionPlan,
    ) -> None:
        self._validate_policy()
        plan.validate(policy=self.execution_policy, command=command)
        result.receipt.validate()
        if (
            result.mode != plan.mode
            or result.requested_availability != plan.availability
            or result.mode != result.receipt.mode
            or result.requested_availability != result.receipt.requested_availability
            or result.availability != result.receipt.availability
            or result.status != result.receipt.status
            or result.unavailable_reason != result.receipt.unavailable_reason
            or result.receipt.policy_versions != self._policy_versions()
        ):
            raise InformationResearchExecutionError(
                "Research-execution result does not match its receipt and plan.",
                code="research_execution_binding_invalid",
            )
        if (
            result.receipt.session_id != getattr(command, "session_id")
            or result.receipt.turn_id != getattr(command, "turn_id")
            or result.receipt.request_id != getattr(command, "request_id")
            or result.receipt.generation_id != getattr(command, "generation_id")
        ):
            raise InformationResearchExecutionError(
                "Research-execution command binding does not match.",
                code="research_execution_binding_invalid",
            )
        mode_evidence = (
            result.evidence_result
            if result.mode == "research" and result.status == "completed"
            else None
        )
        result.turn_result.validate(
            adapter=self.mode_adapter,
            command=command,
            evidence_result=mode_evidence,
        )
        mode_receipt = result.turn_result.receipt
        if (
            result.receipt.mode_adapter_id != mode_receipt.adapter_id
            or result.receipt.mode_receipt_sha256 != mode_receipt.receipt_sha256
        ):
            raise InformationResearchExecutionError(
                "Research-mode receipt binding does not match.",
                code="research_execution_binding_invalid",
            )
        if result.research_run is None:
            if result.evidence_result is not None:
                raise InformationResearchExecutionError(
                    "Evidence cannot exist without an exact research run.",
                    code="research_execution_binding_invalid",
                )
        else:
            result.research_run.validate(policy=self.research_orchestrator.policy)
            if result.research_run.request != plan.research_request:
                raise InformationResearchExecutionError(
                    "Research run does not match the approved plan.",
                    code="research_execution_binding_invalid",
                )
            run_receipt = result.research_run.receipt
            expected_run = (
                run_receipt.run_id,
                run_receipt.receipt_sha256,
                run_receipt.outcome,
                run_receipt.stopping_reason,
            )
            supplied_run = (
                result.receipt.research_run_id,
                result.receipt.research_receipt_sha256,
                result.receipt.research_outcome,
                result.receipt.research_stopping_reason,
            )
            if supplied_run != expected_run:
                raise InformationResearchExecutionError(
                    "Research-run receipt binding does not match.",
                    code="research_execution_binding_invalid",
                )
        if result.evidence_result is not None:
            if result.evidence_result.research_run != result.research_run:
                raise InformationResearchExecutionError(
                    "Evidence result does not bind the exact research run.",
                    code="research_execution_binding_invalid",
                )
            result.evidence_result.validate(pipeline=self.evidence_pipeline)
            expected_evidence = self.evidence_pipeline.process(
                research_run=result.research_run,
                reference_time=plan.reference_time,
                outcome=plan.evidence_outcome,
                claim_drafts=plan.claim_drafts,
                created_at=plan.created_at,
                window_start=plan.window_start,
                window_end=plan.window_end,
            )
            if result.evidence_result != expected_evidence:
                raise InformationResearchExecutionError(
                    "Evidence result does not match the exact execution plan.",
                    code="research_execution_binding_invalid",
                )
            evidence_receipt = result.evidence_result.receipt
            expected = (
                evidence_receipt.pipeline_id,
                evidence_receipt.receipt_sha256,
                result.evidence_result.grounding.packet.outcome,
            )
            supplied = (
                result.receipt.evidence_pipeline_id,
                result.receipt.evidence_receipt_sha256,
                result.receipt.evidence_outcome,
            )
            if supplied != expected:
                raise InformationResearchExecutionError(
                    "Evidence receipt binding does not match.",
                    code="research_execution_binding_invalid",
                )
        request = plan.research_request
        query = getattr(request, "query", None)
        expected_request = (
            getattr(request, "request_id", None),
            getattr(query, "query_id", None),
            getattr(query, "content_sha256", None),
            plan.search_provider,
            plan.fetch_provider,
        )
        supplied_request = (
            result.receipt.research_request_id,
            result.receipt.query_id,
            result.receipt.query_sha256,
            result.receipt.search_provider,
            result.receipt.fetch_provider,
        )
        if supplied_request != expected_request:
            raise InformationResearchExecutionError(
                "Research request and provider binding does not match.",
                code="research_execution_binding_invalid",
            )
        if result.mode == "local_only":
            if result.research_run is not None or result.evidence_result is not None:
                raise InformationResearchExecutionError(
                    "Local-only result contains research artifacts.",
                    code="research_execution_binding_invalid",
                )
            if result.turn_result.mode != "local_only" or result.turn_result.status != "completed":
                raise InformationResearchExecutionError(
                    "Local-only mode result is inconsistent.",
                    code="research_execution_binding_invalid",
                )
            return
        if plan.availability in {"offline", "unavailable"}:
            if result.research_run is not None or result.evidence_result is not None:
                raise InformationResearchExecutionError(
                    "Preflight unavailable result contains execution artifacts.",
                    code="research_execution_binding_invalid",
                )
            if (
                result.turn_result.status != "unavailable"
                or result.turn_result.availability != plan.availability
            ):
                raise InformationResearchExecutionError(
                    "Preflight unavailable mode result is inconsistent.",
                    code="research_execution_binding_invalid",
                )
            return
        if result.status == "unavailable":
            if result.research_run is None or result.turn_result.status != "unavailable":
                raise InformationResearchExecutionError(
                    "Post-execution unavailability is missing verified artifacts.",
                    code="research_execution_binding_invalid",
                )
            if result.turn_result.availability != "unavailable":
                raise InformationResearchExecutionError(
                    "Post-execution mode result must be unavailable.",
                    code="research_execution_binding_invalid",
                )
            return
        if result.research_run is None or result.evidence_result is None:
            raise InformationResearchExecutionError(
                "Completed research execution is missing verified artifacts.",
                code="research_execution_binding_invalid",
            )
        if (
            result.turn_result.mode != "research"
            or result.turn_result.availability != "available"
            or result.turn_result.status != "completed"
        ):
            raise InformationResearchExecutionError(
                "Completed research-mode result is inconsistent.",
                code="research_execution_binding_invalid",
            )
