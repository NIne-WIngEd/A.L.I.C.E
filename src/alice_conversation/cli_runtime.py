"""Controlled private local conversational runtime for A.L.I.C.E. P3.7."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .cli_policy import ConversationCliPolicy, load_conversation_cli_policy
from .constitutional_policy import load_constitutional_dialogue_policy
from .contracts import ConversationContractError, ConversationGroundingPacket, utc_now_text
from .grounding_io import load_conversation_grounding_packet
from .model import CancellationToken, ConversationModelConfigurationError
from .model_policy import load_conversation_model_policy
from .ollama import OllamaConversationModel, OllamaModelConfig
from .orchestration import (
    ConversationOrchestrationError,
    ConversationOrchestrator,
    ConversationResumeCommand,
    ConversationTurnCancelledError,
    ConversationTurnCommand,
    ConversationTurnFailedError,
    ConversationTurnInterruptedError,
    ConversationTurnValidationError,
)
from .orchestration_policy import load_conversation_orchestration_policy
from .registry import ConversationModelRegistry
from .response_validation_policy import load_conversation_response_validation_policy
from .state_inspection import inspect_conversation_session
from .state_policy import load_conversation_state_policy
from .state_service import ConversationStateError, ConversationStateService
from .state_store import ConversationStateStore


class ConversationCliError(RuntimeError):
    """Base sanitized error for the user-facing P3.7 runtime."""


class ConversationCliTurnError(ConversationCliError):
    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


class ConversationCliValidationError(ConversationCliTurnError):
    def __init__(self, *, code: str, issue_codes: tuple[str, ...]) -> None:
        self.issue_codes = issue_codes
        super().__init__("The generated response was rejected.", code=code)


class ConversationCliInterruptedError(ConversationCliTurnError):
    pass


class ConversationCliCancelledError(ConversationCliTurnError):
    pass


@dataclass(frozen=True)
class ConversationCliTurnOutput:
    content: str
    validation_outcome: str
    citation_tokens: tuple[str, ...]
    replayed: bool


@dataclass(frozen=True)
class ConversationCliCloseSummary:
    retention: str
    purged: bool
    turn_count: int
    message_count: int
    reference_count: int
    generation_count: int


@dataclass(frozen=True)
class ConversationCliInspection:
    status: str
    retention: str
    data_classification: str
    provider: str
    model: str
    turn_count: int
    message_count: int
    reference_count: int
    generation_count: int
    turn_statuses: tuple[tuple[str, int], ...]
    last_validation_outcome: str | None
    last_failure_code: str | None
    grounding_enabled: bool
    grounding_outcome: str | None
    grounding_claim_count: int
    grounding_citation_count: int


@dataclass(frozen=True)
class ConversationGroundingStatus:
    enabled: bool
    outcome: str | None
    claim_count: int
    citation_count: int


class ConversationCliRuntime:
    """One local interactive runtime over P3.2-P3.6 controlled components."""

    def __init__(
        self,
        *,
        state_service: ConversationStateService,
        orchestrator: ConversationOrchestrator,
        provider: str,
        model: str,
        policy: ConversationCliPolicy,
        grounding: ConversationGroundingPacket | None = None,
        clock: Callable[[], str] = utc_now_text,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(policy, ConversationCliPolicy):
            raise ConversationContractError("P3.7 requires a validated CLI policy.")
        if provider not in policy.allowed_providers:
            raise ConversationCliPolicyViolation(
                "The selected provider is not approved for the user-facing runtime."
            )
        if not isinstance(model, str) or not model.strip():
            raise ConversationCliPolicyViolation("A model must be selected explicitly.")
        orchestrator.model_registry.resolve(provider=provider, model=model)
        if grounding is not None:
            grounding.validate()
        if not callable(clock):
            raise ConversationContractError("CLI clock must be callable.")
        self.state_service = state_service
        self.orchestrator = orchestrator
        self.provider = provider
        self.model = model
        self.policy = policy
        self.grounding = grounding
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid4().hex)
        self._session_id: str | None = None
        self._retention: str | None = None
        self._active_turn_id: str | None = None
        self._active_cancellation: CancellationToken | None = None

    @property
    def has_session(self) -> bool:
        return self._session_id is not None

    def new_session(self, retention: str | None = None) -> None:
        selected = retention or self.policy.default_retention
        if selected not in self.policy.allowed_retentions:
            raise ConversationCliPolicyViolation("Unsupported conversation retention.")
        if self._session_id is not None:
            self.close_session()
        session_id = self._new_id("session")
        self.state_service.create_session(
            session_id=session_id,
            created_at=self._clock(),
            retention=selected,
            data_classification="PRIVATE",
        )
        self._session_id = session_id
        self._retention = selected

    def close_session(self) -> ConversationCliCloseSummary:
        session_id = self._require_session()
        inspection = inspect_conversation_session(
            self.state_service.store,
            session_id=session_id,
            include_content=False,
        )
        counts = self._counts(inspection)
        try:
            tombstone = self.state_service.close_session(
                session_id=session_id,
                closed_at=self._clock(),
            )
        except ConversationStateError as exc:
            raise ConversationCliError(
                "The session cannot close while a turn is nonterminal."
            ) from exc
        summary = ConversationCliCloseSummary(
            retention=inspection.retention,
            purged=tombstone is not None,
            turn_count=(tombstone.turn_count if tombstone else counts[0]),
            message_count=(tombstone.message_count if tombstone else counts[1]),
            reference_count=(tombstone.reference_count if tombstone else counts[2]),
            generation_count=(tombstone.generation_count if tombstone else counts[3]),
        )
        self._session_id = None
        self._retention = None
        self._active_turn_id = None
        self._active_cancellation = None
        return summary

    def send(self, content: str) -> ConversationCliTurnOutput:
        session_id = self._require_session()
        if not isinstance(content, str) or not content.strip():
            raise ConversationCliPolicyViolation("Conversation input must be non-empty.")
        if len(content) > self.policy.max_input_chars:
            raise ConversationCliPolicyViolation(
                "Conversation input exceeds the approved character limit."
            )
        turn_id = self._new_id("turn")
        command = ConversationTurnCommand(
            session_id=session_id,
            turn_id=turn_id,
            user_message_id=self._new_id("user-message"),
            assistant_message_id=self._new_id("assistant-message"),
            request_id=self._new_id("request"),
            generation_id=self._new_id("generation"),
            provider=self.provider,
            model=self.model,
            user_content=content,
            data_classification="PRIVATE",
            grounding=self.grounding,
        )
        return self._run_command(command)

    def resume(self) -> ConversationCliTurnOutput:
        session_id = self._require_session()
        inspection = inspect_conversation_session(
            self.state_service.store,
            session_id=session_id,
            include_content=False,
        )
        interrupted = [turn for turn in inspection.turns if turn.status == "interrupted"]
        if len(interrupted) != 1:
            raise ConversationCliError("There is no single interrupted turn to resume.")
        command = ConversationResumeCommand(
            session_id=session_id,
            turn_id=interrupted[0].turn_id,
            assistant_message_id=self._new_id("assistant-message"),
            request_id=self._new_id("request"),
            generation_id=self._new_id("generation"),
            provider=self.provider,
            model=self.model,
            grounding=self.grounding,
        )
        cancellation = CancellationToken()
        self._active_turn_id = command.turn_id
        self._active_cancellation = cancellation
        try:
            result = self.orchestrator.resume_turn(command, cancellation=cancellation)
            return self._output(result)
        except KeyboardInterrupt as exc:
            cancellation.cancel()
            self._cancel_turn(command.turn_id, "cli_keyboard_interrupt")
            raise ConversationCliCancelledError(
                "The interrupted generation was cancelled.",
                code="cli_keyboard_interrupt",
            ) from exc
        except ConversationTurnValidationError as exc:
            raise self._validation_error(exc) from exc
        except ConversationTurnInterruptedError as exc:
            raise ConversationCliInterruptedError(
                "Generation was interrupted and can be resumed.",
                code=exc.failure_code,
            ) from exc
        except ConversationTurnCancelledError as exc:
            raise ConversationCliCancelledError(
                "Generation was cancelled.", code=exc.failure_code
            ) from exc
        except ConversationTurnFailedError as exc:
            raise ConversationCliTurnError(
                "Generation failed.", code=exc.failure_code
            ) from exc
        except ConversationOrchestrationError as exc:
            raise ConversationCliTurnError(
                "The turn could not be resumed.", code=exc.failure_code
            ) from exc
        finally:
            self._active_turn_id = None
            self._active_cancellation = None

    def cancel(self) -> bool:
        if self._active_cancellation is not None:
            self._active_cancellation.cancel()
        session_id = self._require_session()
        inspection = inspect_conversation_session(
            self.state_service.store,
            session_id=session_id,
            include_content=False,
        )
        nonterminal = [
            turn
            for turn in inspection.turns
            if turn.status in {"received", "context_ready", "generating", "interrupted"}
        ]
        if not nonterminal:
            return False
        self._cancel_turn(nonterminal[-1].turn_id, "cli_cancelled")
        return True

    def inspect(self) -> ConversationCliInspection:
        session_id = self._require_session()
        inspection = inspect_conversation_session(
            self.state_service.store,
            session_id=session_id,
            include_content=False,
        )
        turn_count, message_count, reference_count, generation_count = self._counts(
            inspection
        )
        statuses = Counter(turn.status for turn in inspection.turns)
        generations = [
            generation
            for turn in inspection.turns
            for generation in turn.generations
        ]
        last_generation = generations[-1] if generations else None
        grounding = self.grounding_status()
        return ConversationCliInspection(
            status=inspection.status,
            retention=inspection.retention,
            data_classification=inspection.data_classification,
            provider=self.provider,
            model=self.model,
            turn_count=turn_count,
            message_count=message_count,
            reference_count=reference_count,
            generation_count=generation_count,
            turn_statuses=tuple(sorted(statuses.items())),
            last_validation_outcome=(
                last_generation.validation_outcome if last_generation else None
            ),
            last_failure_code=(last_generation.failure_code if last_generation else None),
            grounding_enabled=grounding.enabled,
            grounding_outcome=grounding.outcome,
            grounding_claim_count=grounding.claim_count,
            grounding_citation_count=grounding.citation_count,
        )

    def grounding_status(self) -> ConversationGroundingStatus:
        if self.grounding is None:
            return ConversationGroundingStatus(False, None, 0, 0)
        return ConversationGroundingStatus(
            True,
            self.grounding.outcome,
            len(self.grounding.claims),
            sum(len(claim.citations) for claim in self.grounding.claims),
        )

    def set_grounding(self, grounding: ConversationGroundingPacket | None) -> None:
        if self._active_turn_id is not None:
            raise ConversationCliError("Grounding cannot change during an active turn.")
        if grounding is not None:
            grounding.validate()
        self.grounding = grounding

    def _run_command(self, command: ConversationTurnCommand) -> ConversationCliTurnOutput:
        cancellation = CancellationToken()
        self._active_turn_id = command.turn_id
        self._active_cancellation = cancellation
        try:
            result = self.orchestrator.run_turn(command, cancellation=cancellation)
            return self._output(result)
        except KeyboardInterrupt as exc:
            cancellation.cancel()
            self._cancel_turn(command.turn_id, "cli_keyboard_interrupt")
            raise ConversationCliCancelledError(
                "Generation was cancelled.", code="cli_keyboard_interrupt"
            ) from exc
        except ConversationTurnValidationError as exc:
            raise self._validation_error(exc) from exc
        except ConversationTurnInterruptedError as exc:
            raise ConversationCliInterruptedError(
                "Generation was interrupted and can be resumed.",
                code=exc.failure_code,
            ) from exc
        except ConversationTurnCancelledError as exc:
            raise ConversationCliCancelledError(
                "Generation was cancelled.", code=exc.failure_code
            ) from exc
        except ConversationTurnFailedError as exc:
            raise ConversationCliTurnError(
                "Generation failed.", code=exc.failure_code
            ) from exc
        except ConversationOrchestrationError as exc:
            raise ConversationCliTurnError(
                "The turn could not be completed.", code=exc.failure_code
            ) from exc
        finally:
            self._active_turn_id = None
            self._active_cancellation = None

    def _output(self, result) -> ConversationCliTurnOutput:
        citations: set[str] = set()
        if self.grounding is not None:
            for claim in self.grounding.claims:
                for citation in claim.citations:
                    if citation.token in result.response.content:
                        citations.add(citation.token)
        return ConversationCliTurnOutput(
            content=result.response.content,
            validation_outcome=result.validation_outcome,
            citation_tokens=tuple(sorted(citations)),
            replayed=result.replayed,
        )

    @staticmethod
    def _validation_error(exc: ConversationTurnValidationError) -> ConversationCliValidationError:
        issue_codes = tuple(sorted({issue.code for issue in exc.report.issues}))
        return ConversationCliValidationError(
            code=exc.failure_code,
            issue_codes=issue_codes,
        )

    def _cancel_turn(self, turn_id: str, code: str) -> None:
        try:
            self.state_service.cancel_turn(
                turn_id=turn_id,
                cancelled_at=self._clock(),
                reason_code=code,
            )
        except ConversationStateError:
            pass

    def _new_id(self, prefix: str) -> str:
        value = self._id_factory()
        if not isinstance(value, str) or not value.strip():
            raise ConversationCliError("The runtime ID factory returned invalid text.")
        return f"{prefix}-{value.strip()}"

    def _require_session(self) -> str:
        if self._session_id is None:
            raise ConversationCliError("No active local conversation session exists.")
        return self._session_id

    @staticmethod
    def _counts(inspection) -> tuple[int, int, int, int]:
        return (
            len(inspection.turns),
            sum(len(turn.messages) for turn in inspection.turns),
            sum(len(turn.references) for turn in inspection.turns),
            sum(len(turn.generations) for turn in inspection.turns),
        )


class ConversationCliPolicyViolation(ConversationCliError):
    """Raised when a user-facing CLI request exceeds the P3.7 policy."""


def build_local_conversation_runtime(
    *,
    repository_root: str | Path,
    vault_root: str | Path,
    provider: str,
    model: str,
    retention: str = "session_only",
    grounding_file: str | Path | None = None,
) -> ConversationCliRuntime:
    """Build the first local-only user runtime from the repository policies."""
    root = Path(repository_root).expanduser().resolve()
    vault = Path(vault_root).expanduser().resolve(strict=False)
    cli_policy = load_conversation_cli_policy(
        root / "policies" / "conversation_cli_policy.json"
    )
    if provider not in cli_policy.allowed_providers:
        raise ConversationCliPolicyViolation(
            "The selected provider is not available in the P3.7 local runtime."
        )
    model_policy = load_conversation_model_policy(
        root / "policies" / "conversation_model_policy.json"
    )
    provider_policy = model_policy.provider(provider)
    if model not in provider_policy.allowed_models:
        raise ConversationModelConfigurationError(
            "The selected model is not approved by model policy."
        )
    registry = ConversationModelRegistry()
    registry.register(
        OllamaConversationModel(
            OllamaModelConfig.from_policy(provider_policy, model=model)
        )
    )
    state_policy = load_conversation_state_policy(
        root / "policies" / "conversation_state_policy.json"
    )
    store = ConversationStateStore.for_vault(
        vault_root=vault,
        repository_root=root,
        policy=state_policy,
    )
    store.initialize()
    state_service = ConversationStateService(store)
    orchestrator = ConversationOrchestrator.from_repository(
        state_service=state_service,
        model_registry=registry,
        repository_root=root,
        orchestration_policy=load_conversation_orchestration_policy(
            root / "policies" / "conversation_orchestration_policy.json"
        ),
        constitutional_policy=load_constitutional_dialogue_policy(
            root / "policies" / "conversation_constitutional_policy.json"
        ),
        response_validation_policy=load_conversation_response_validation_policy(
            root / "policies" / "conversation_response_validation_policy.json"
        ),
    )
    grounding = (
        load_conversation_grounding_packet(
            grounding_file,
            policy=cli_policy,
            repository_root=root,
        )
        if grounding_file is not None
        else None
    )
    runtime = ConversationCliRuntime(
        state_service=state_service,
        orchestrator=orchestrator,
        provider=provider,
        model=model,
        policy=cli_policy,
        grounding=grounding,
    )
    runtime.new_session(retention)
    return runtime
