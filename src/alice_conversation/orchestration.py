"""Controlled single-turn orchestration for A.L.I.C.E. Phase 3 P3.5."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .constitutional_policy import (
    ConstitutionalDialoguePolicy,
    load_constitutional_dialogue_policy,
)
from .constitutional_prompt import (
    ConstitutionalSystemContract,
    compile_constitutional_system_contract,
)
from .contracts import (
    ConversationCapabilities,
    ConversationContractError,
    ConversationGroundingPacket,
    ConversationMessage,
    ModelRequest,
    ModelResponse,
    sha256_text,
    utc_now_text,
)
from .grounding_bridge import (
    conversation_grounding_packet_sha256,
    conversation_state_references_from_grounding,
)
from .model import (
    CancellationToken,
    ConversationModelBudgetError,
    ConversationModelCancelledError,
    ConversationModelConfigurationError,
    ConversationModelError,
    ConversationModelProtocolError,
    ConversationModelProviderError,
    ConversationModelTimeoutError,
)
from .orchestration_policy import (
    ConversationOrchestrationPolicy,
    load_conversation_orchestration_policy,
)
from .registry import ConversationModelRegistry
from .state_inspection import (
    ConversationTurnInspection,
    inspect_conversation_session,
)
from .state_service import ConversationStateError, ConversationStateService


_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


class ConversationOrchestrationError(RuntimeError):
    """Base sanitized orchestration failure."""

    def __init__(self, message: str, *, turn_id: str, failure_code: str) -> None:
        self.turn_id = turn_id
        self.failure_code = failure_code
        super().__init__(message)


class ConversationTurnCancelledError(ConversationOrchestrationError):
    """Raised after a cancelled generation is atomically recorded."""


class ConversationTurnInterruptedError(ConversationOrchestrationError):
    """Raised after an interrupted generation is atomically recorded."""


class ConversationTurnFailedError(ConversationOrchestrationError):
    """Raised after a failed turn is atomically recorded."""


class ConversationGenerationInterruptedError(RuntimeError):
    """Cooperative adapter signal used to record an interrupted attempt."""

    def __init__(self, reason_code: str = "model_interrupted") -> None:
        if not isinstance(reason_code, str) or not _ID_PATTERN.fullmatch(reason_code):
            raise ValueError("Interruption reason must be a safe non-empty code.")
        self.reason_code = reason_code
        super().__init__("Conversation generation was interrupted.")


@dataclass(frozen=True)
class ConversationTurnCommand:
    session_id: str
    turn_id: str
    user_message_id: str
    assistant_message_id: str
    request_id: str
    generation_id: str
    provider: str
    model: str
    user_content: str
    data_classification: str = "PRIVATE"
    grounding: ConversationGroundingPacket | None = None

    def validate(self) -> None:
        for field in (
            "session_id",
            "turn_id",
            "user_message_id",
            "assistant_message_id",
            "request_id",
            "generation_id",
            "provider",
            "model",
        ):
            _require_identifier(getattr(self, field), field=field)
        if len(
            {
                self.user_message_id,
                self.assistant_message_id,
                self.request_id,
                self.generation_id,
            }
        ) != 4:
            raise ConversationContractError(
                "Turn message, request, and generation IDs must be distinct."
            )
        if not isinstance(self.user_content, str) or not self.user_content.strip():
            raise ConversationContractError("user_content must be non-empty text.")
        if self.grounding is not None:
            self.grounding.validate()


@dataclass(frozen=True)
class ConversationResumeCommand:
    session_id: str
    turn_id: str
    assistant_message_id: str
    request_id: str
    generation_id: str
    provider: str
    model: str
    grounding: ConversationGroundingPacket | None = None

    def validate(self) -> None:
        for field in (
            "session_id",
            "turn_id",
            "assistant_message_id",
            "request_id",
            "generation_id",
            "provider",
            "model",
        ):
            _require_identifier(getattr(self, field), field=field)
        if len(
            {self.assistant_message_id, self.request_id, self.generation_id}
        ) != 3:
            raise ConversationContractError(
                "Resume assistant, request, and generation IDs must be distinct."
            )
        if self.grounding is not None:
            self.grounding.validate()


@dataclass(frozen=True)
class ConversationTurnResult:
    session_id: str
    turn_id: str
    request_id: str
    generation_id: str
    provider: str
    model: str
    assistant_message: ConversationMessage
    response: ModelResponse
    grounding_packet_id: str | None
    grounding_packet_sha256: str | None
    replayed: bool = False

    def validate(self) -> None:
        for field in (
            "session_id",
            "turn_id",
            "request_id",
            "generation_id",
            "provider",
            "model",
        ):
            _require_identifier(getattr(self, field), field=field)
        self.assistant_message.validate()
        self.response.validate()
        if self.assistant_message.role != "assistant":
            raise ConversationContractError(
                "Orchestration results require an assistant message."
            )
        if self.assistant_message.turn_id != self.turn_id:
            raise ConversationContractError(
                "Orchestration result assistant message belongs to another turn."
            )
        if self.assistant_message.content != self.response.content:
            raise ConversationContractError(
                "Orchestration result content does not match the model response."
            )
        if self.response.request_id != self.request_id:
            raise ConversationContractError(
                "Orchestration result request identity does not match."
            )
        if (self.grounding_packet_id is None) != (
            self.grounding_packet_sha256 is None
        ):
            raise ConversationContractError(
                "Grounding packet identity and digest must be paired."
            )


class ConversationOrchestrator:
    """Compose P3.1-P3.4 into one controlled, no-tools turn lifecycle."""

    def __init__(
        self,
        *,
        state_service: ConversationStateService,
        model_registry: ConversationModelRegistry,
        system_contract: ConstitutionalSystemContract,
        policy: ConversationOrchestrationPolicy,
        clock: Callable[[], str] = utc_now_text,
    ) -> None:
        system_contract.validate()
        if not isinstance(policy, ConversationOrchestrationPolicy):
            raise ConversationContractError(
                "ConversationOrchestrator requires a validated P3.5 policy."
            )
        if not callable(clock):
            raise ConversationContractError("Orchestration clock must be callable.")
        self.state_service = state_service
        self.model_registry = model_registry
        self.system_contract = system_contract
        self.policy = policy
        self._clock = clock
        self._verify_boundaries()

    @classmethod
    def from_repository(
        cls,
        *,
        state_service: ConversationStateService,
        model_registry: ConversationModelRegistry,
        repository_root: str | Path,
        orchestration_policy: ConversationOrchestrationPolicy | None = None,
        constitutional_policy: ConstitutionalDialoguePolicy | None = None,
        clock: Callable[[], str] = utc_now_text,
    ) -> "ConversationOrchestrator":
        root = Path(repository_root).resolve()
        selected_orchestration = (
            orchestration_policy or load_conversation_orchestration_policy()
        )
        selected_constitutional = (
            constitutional_policy or load_constitutional_dialogue_policy()
        )
        contract = compile_constitutional_system_contract(
            policy=selected_constitutional,
            repository_root=root,
        )
        return cls(
            state_service=state_service,
            model_registry=model_registry,
            system_contract=contract,
            policy=selected_orchestration,
            clock=clock,
        )

    def run_turn(
        self,
        command: ConversationTurnCommand,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ConversationTurnResult:
        command.validate()
        existing = self._find_turn(command.session_id, command.turn_id)
        if existing is not None:
            return self._replay_or_reject(
                turn=existing,
                session_id=command.session_id,
                user_message_id=command.user_message_id,
                user_content=command.user_content,
                assistant_message_id=command.assistant_message_id,
                request_id=command.request_id,
                generation_id=command.generation_id,
                provider=command.provider,
                model=command.model,
            )

        created_at = self._now()
        user_message = ConversationMessage.create(
            message_id=command.user_message_id,
            turn_id=command.turn_id,
            role="user",
            content=command.user_content,
            created_at=created_at,
            data_classification=command.data_classification,
        )
        try:
            self.state_service.start_turn(
                session_id=command.session_id,
                turn_id=command.turn_id,
                user_message=user_message,
            )
        except ConversationStateError as exc:
            raise ConversationOrchestrationError(
                "The conversation turn could not be started.",
                turn_id=command.turn_id,
                failure_code="state_transition",
            ) from exc
        try:
            grounding_id, grounding_sha256 = self._set_context(
                turn_id=command.turn_id,
                grounding=command.grounding,
            )
        except (ConversationContractError, ConversationStateError) as exc:
            self._fail(command.turn_id, "protocol")
            raise ConversationTurnFailedError(
                "The turn context violated the conversation contract.",
                turn_id=command.turn_id,
                failure_code=self.policy.failure_code("protocol"),
            ) from exc
        return self._generate(
            session_id=command.session_id,
            turn_id=command.turn_id,
            user_message=user_message,
            assistant_message_id=command.assistant_message_id,
            request_id=command.request_id,
            generation_id=command.generation_id,
            provider=command.provider,
            model=command.model,
            grounding=command.grounding,
            grounding_packet_id=grounding_id,
            grounding_packet_sha256=grounding_sha256,
            cancellation=cancellation,
        )

    def resume_turn(
        self,
        command: ConversationResumeCommand,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ConversationTurnResult:
        command.validate()
        turn = self._find_turn(command.session_id, command.turn_id)
        if turn is None:
            raise ConversationOrchestrationError(
                "Cannot resume a conversation turn that does not exist.",
                turn_id=command.turn_id,
                failure_code="turn_not_found",
            )
        if turn.status == "completed":
            user = _one_message(turn, "user")
            return self._replay_or_reject(
                turn=turn,
                session_id=command.session_id,
                user_message_id=user.message_id,
                user_content=user.content or "",
                assistant_message_id=command.assistant_message_id,
                request_id=command.request_id,
                generation_id=command.generation_id,
                provider=command.provider,
                model=command.model,
            )
        if turn.status != "interrupted":
            raise ConversationOrchestrationError(
                f"Only an interrupted turn can resume; current status is {turn.status}.",
                turn_id=command.turn_id,
                failure_code="turn_not_interrupted",
            )
        self._verify_resume_grounding(turn, command.grounding)
        user_inspection = _one_message(turn, "user")
        if user_inspection.content is None:
            raise ConversationOrchestrationError(
                "Interrupted turn inspection did not expose the private user message.",
                turn_id=command.turn_id,
                failure_code="state_content_unavailable",
            )
        user_message = ConversationMessage(
            message_id=user_inspection.message_id,
            turn_id=command.turn_id,
            role="user",
            content=user_inspection.content,
            content_sha256=user_inspection.content_sha256,
            created_at=user_inspection.created_at,
            data_classification=user_inspection.data_classification,
        )
        user_message.validate()
        try:
            self.state_service.resume_turn(
                turn_id=command.turn_id,
                resumed_at=self._now(),
            )
        except ConversationStateError as exc:
            raise ConversationOrchestrationError(
                "The interrupted turn could not be resumed.",
                turn_id=command.turn_id,
                failure_code="state_transition",
            ) from exc
        return self._generate(
            session_id=command.session_id,
            turn_id=command.turn_id,
            user_message=user_message,
            assistant_message_id=command.assistant_message_id,
            request_id=command.request_id,
            generation_id=command.generation_id,
            provider=command.provider,
            model=command.model,
            grounding=command.grounding,
            grounding_packet_id=turn.grounding_packet_id,
            grounding_packet_sha256=turn.grounding_packet_sha256,
            cancellation=cancellation,
        )

    def _set_context(
        self,
        *,
        turn_id: str,
        grounding: ConversationGroundingPacket | None,
    ) -> tuple[str | None, str | None]:
        if grounding is None:
            references = ()
            packet_id = None
            packet_sha256 = None
        else:
            grounding.validate()
            references = conversation_state_references_from_grounding(grounding)
            packet_id = grounding.packet_id
            packet_sha256 = conversation_grounding_packet_sha256(grounding)
        self.state_service.set_turn_context(
            turn_id=turn_id,
            references=references,
            updated_at=self._now(),
            grounding_packet_id=packet_id,
            grounding_packet_sha256=packet_sha256,
        )
        return packet_id, packet_sha256

    def _generate(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: ConversationMessage,
        assistant_message_id: str,
        request_id: str,
        generation_id: str,
        provider: str,
        model: str,
        grounding: ConversationGroundingPacket | None,
        grounding_packet_id: str | None,
        grounding_packet_sha256: str | None,
        cancellation: CancellationToken | None,
    ) -> ConversationTurnResult:
        try:
            adapter = self.model_registry.resolve(provider=provider, model=model)
        except ConversationModelConfigurationError as exc:
            self._fail(turn_id, "configuration")
            raise ConversationTurnFailedError(
                "The requested model adapter is not available.",
                turn_id=turn_id,
                failure_code=self.policy.failure_code("configuration"),
            ) from exc

        request = ModelRequest(
            request_id=request_id,
            session_id=session_id,
            turn_id=turn_id,
            system_contract_version=self.system_contract.version,
            system_contract=self.system_contract.content,
            messages=(user_message,),
            grounding=grounding,
            capabilities=ConversationCapabilities(),
            max_output_tokens=self.policy.max_output_tokens,
            temperature=self.policy.temperature,
        )
        try:
            request.validate()
        except ConversationContractError as exc:
            self._fail(turn_id, "protocol")
            raise ConversationTurnFailedError(
                "The model request violated the conversation contract.",
                turn_id=turn_id,
                failure_code=self.policy.failure_code("protocol"),
            ) from exc

        started_at = self._now()
        try:
            self.state_service.start_generation(
                turn_id=turn_id,
                generation_id=generation_id,
                request_id=request_id,
                provider=provider,
                model=model,
                started_at=started_at,
                reasoning_status="not_persisted",
            )
        except ConversationStateError as exc:
            self._fail(turn_id, "internal")
            raise ConversationTurnFailedError(
                "The generation attempt could not be recorded.",
                turn_id=turn_id,
                failure_code=self.policy.failure_code("internal"),
            ) from exc

        try:
            response = adapter.generate(request, cancellation=cancellation)
            response.validate()
            if response.request_id != request_id:
                raise ConversationModelProtocolError(
                    "Model response request identity does not match."
                )
            if response.provider != provider or response.model != model:
                raise ConversationModelProtocolError(
                    "Model response provider/model identity does not match."
                )
            assistant_created_at = self._now()
            assistant_message = ConversationMessage.create(
                message_id=assistant_message_id,
                turn_id=turn_id,
                role="assistant",
                content=response.content,
                created_at=assistant_created_at,
                data_classification=user_message.data_classification,
            )
            self.state_service.complete_turn(
                turn_id=turn_id,
                request_id=request_id,
                response=response,
                assistant_message=assistant_message,
                completed_at=assistant_created_at,
                validation_outcome="accepted",
            )
        except ConversationGenerationInterruptedError as exc:
            code = self.policy.failure_code("interrupted")
            self.state_service.interrupt_turn(
                turn_id=turn_id,
                request_id=request_id,
                interrupted_at=self._now(),
                reason_code=code,
            )
            raise ConversationTurnInterruptedError(
                "The model generation was interrupted and may be explicitly resumed.",
                turn_id=turn_id,
                failure_code=code,
            ) from exc
        except ConversationModelCancelledError as exc:
            code = self.policy.failure_code("cancelled")
            self._cancel(turn_id, code)
            raise ConversationTurnCancelledError(
                "The model generation was cancelled.",
                turn_id=turn_id,
                failure_code=code,
            ) from exc
        except ConversationModelTimeoutError as exc:
            raise self._recorded_failure(
                turn_id, "timeout", "The model generation timed out.", exc
            )
        except ConversationModelBudgetError as exc:
            raise self._recorded_failure(
                turn_id, "budget", "The model request exceeded its budget.", exc
            )
        except ConversationModelProviderError as exc:
            raise self._recorded_failure(
                turn_id, "provider", "The model provider failed.", exc
            )
        except ConversationModelConfigurationError as exc:
            raise self._recorded_failure(
                turn_id,
                "configuration",
                "The model adapter configuration failed.",
                exc,
            )
        except (ConversationModelProtocolError, ConversationContractError) as exc:
            raise self._recorded_failure(
                turn_id, "protocol", "The model response violated the contract.", exc
            )
        except ConversationStateError as exc:
            raise self._recorded_failure(
                turn_id,
                "internal",
                "The completed response could not be committed atomically.",
                exc,
            )
        except ConversationModelError as exc:
            raise self._recorded_failure(
                turn_id, "protocol", "The model generation failed.", exc
            )
        except Exception as exc:
            raise self._recorded_failure(
                turn_id,
                "internal",
                "An internal orchestration failure was recorded.",
                exc,
            )

        result = ConversationTurnResult(
            session_id=session_id,
            turn_id=turn_id,
            request_id=request_id,
            generation_id=generation_id,
            provider=provider,
            model=model,
            assistant_message=assistant_message,
            response=response,
            grounding_packet_id=grounding_packet_id,
            grounding_packet_sha256=grounding_packet_sha256,
            replayed=False,
        )
        result.validate()
        return result

    def _recorded_failure(
        self,
        turn_id: str,
        category: str,
        message: str,
        cause: Exception,
    ) -> ConversationTurnFailedError:
        code = self.policy.failure_code(category)
        self._fail(turn_id, category)
        return ConversationTurnFailedError(
            message,
            turn_id=turn_id,
            failure_code=code,
        )

    def _fail(self, turn_id: str, category: str) -> None:
        code = self.policy.failure_code(category)
        try:
            self.state_service.fail_turn(
                turn_id=turn_id,
                failed_at=self._now(),
                failure_code=code,
            )
        except ConversationStateError:
            pass

    def _cancel(self, turn_id: str, code: str) -> None:
        try:
            self.state_service.cancel_turn(
                turn_id=turn_id,
                cancelled_at=self._now(),
                reason_code=code,
            )
        except ConversationStateError:
            pass

    def _find_turn(
        self,
        session_id: str,
        turn_id: str,
    ) -> ConversationTurnInspection | None:
        try:
            inspection = inspect_conversation_session(
                self.state_service.store,
                session_id=session_id,
                include_content=True,
            )
        except ConversationStateError as exc:
            raise ConversationOrchestrationError(
                "Conversation session does not exist.",
                turn_id=turn_id,
                failure_code="session_not_found",
            ) from exc
        return next((turn for turn in inspection.turns if turn.turn_id == turn_id), None)

    def _replay_or_reject(
        self,
        *,
        turn: ConversationTurnInspection,
        session_id: str,
        user_message_id: str,
        user_content: str,
        assistant_message_id: str,
        request_id: str,
        generation_id: str,
        provider: str,
        model: str,
    ) -> ConversationTurnResult:
        if turn.status != "completed":
            raise ConversationOrchestrationError(
                f"Conversation turn already exists with status {turn.status}.",
                turn_id=turn.turn_id,
                failure_code=f"turn_{turn.status}",
            )
        user = _one_message(turn, "user")
        assistant = _one_message(turn, "assistant")
        completed = [item for item in turn.generations if item.status == "completed"]
        if len(completed) != 1:
            raise ConversationOrchestrationError(
                "Completed turn does not contain exactly one completed generation.",
                turn_id=turn.turn_id,
                failure_code="state_integrity",
            )
        generation = completed[0]
        expected = (
            user.message_id == user_message_id
            and user.content_sha256 == sha256_text(user_content)
            and assistant.message_id == assistant_message_id
            and generation.request_id == request_id
            and generation.generation_id == generation_id
            and generation.provider == provider
            and generation.model == model
        )
        if not expected:
            raise ConversationOrchestrationError(
                "Existing completed turn does not match the requested idempotency keys.",
                turn_id=turn.turn_id,
                failure_code="idempotency_conflict",
            )
        if assistant.content is None:
            raise ConversationOrchestrationError(
                "Completed assistant message content is unavailable.",
                turn_id=turn.turn_id,
                failure_code="state_content_unavailable",
            )
        assistant_message = ConversationMessage(
            message_id=assistant.message_id,
            turn_id=turn.turn_id,
            role="assistant",
            content=assistant.content,
            content_sha256=assistant.content_sha256,
            created_at=assistant.created_at,
            data_classification=assistant.data_classification,
        )
        response = ModelResponse(
            request_id=generation.request_id,
            provider=generation.provider,
            model=generation.model,
            content=assistant.content,
            finish_reason=generation.finish_reason or "stop",
            created_at=generation.completed_at or assistant.created_at,
        )
        result = ConversationTurnResult(
            session_id=session_id,
            turn_id=turn.turn_id,
            request_id=generation.request_id,
            generation_id=generation.generation_id,
            provider=generation.provider,
            model=generation.model,
            assistant_message=assistant_message,
            response=response,
            grounding_packet_id=turn.grounding_packet_id,
            grounding_packet_sha256=turn.grounding_packet_sha256,
            replayed=True,
        )
        result.validate()
        return result

    def _verify_resume_grounding(
        self,
        turn: ConversationTurnInspection,
        grounding: ConversationGroundingPacket | None,
    ) -> None:
        if turn.grounding_packet_id is None:
            if grounding is not None:
                raise ConversationOrchestrationError(
                    "Interrupted turn did not originally use grounding.",
                    turn_id=turn.turn_id,
                    failure_code="grounding_mismatch",
                )
            return
        if grounding is None:
            raise ConversationOrchestrationError(
                "Resuming this turn requires the original grounding packet.",
                turn_id=turn.turn_id,
                failure_code="grounding_required",
            )
        grounding.validate()
        digest = conversation_grounding_packet_sha256(grounding)
        if (
            grounding.packet_id != turn.grounding_packet_id
            or digest != turn.grounding_packet_sha256
        ):
            raise ConversationOrchestrationError(
                "Resume grounding does not match the interrupted turn.",
                turn_id=turn.turn_id,
                failure_code="grounding_mismatch",
            )

    def _verify_boundaries(self) -> None:
        enabled = [
            name
            for name, value in self.policy.boundaries
            if value is not False
        ]
        state = self.state_service.policy
        state_enabled = [
            name
            for name in (
                "memory_write_allowed",
                "external_action_allowed",
                "web_access_allowed",
                "tool_calling_allowed",
                "chain_of_thought_persistence_allowed",
            )
            if getattr(state, name) is not False
        ]
        if enabled or state_enabled:
            raise ConversationContractError(
                "P3.5 orchestration cannot start with enabled capabilities."
            )
        if self.policy.lifecycle_value("automatic_retry_count") != 0:
            raise ConversationContractError(
                "P3.5 orchestration cannot enable automatic retries."
            )
        if self.policy.lifecycle_value("provider_fallback_allowed") is not False:
            raise ConversationContractError(
                "P3.5 orchestration cannot enable provider fallback."
            )

    def _now(self) -> str:
        value = self._clock()
        if not isinstance(value, str) or not value.strip():
            raise ConversationContractError(
                "Orchestration clock must return an ISO-8601 timestamp string."
            )
        return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ConversationContractError(
            f"{field} must be a safe non-empty identifier."
        )
    return value


def _one_message(turn: ConversationTurnInspection, role: str):
    selected = [message for message in turn.messages if message.role == role]
    if len(selected) != 1:
        raise ConversationOrchestrationError(
            f"Turn does not contain exactly one {role} message.",
            turn_id=turn.turn_id,
            failure_code="state_integrity",
        )
    return selected[0]
