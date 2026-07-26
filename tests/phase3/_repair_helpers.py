from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from alice_conversation.constitutional_prompt import (
    ConstitutionalSourceSnapshot,
    ConstitutionalSystemContract,
)
from alice_conversation.context_policy import load_conversation_context_policy
from alice_conversation.contracts import ModelRequest, ModelResponse, sha256_text
from alice_conversation.model import (
    CancellationToken,
    ConversationModelCancelledError,
    ConversationModelError,
    ConversationModelTimeoutError,
)
from alice_conversation.orchestration import ConversationOrchestrator, ConversationTurnCommand
from alice_conversation.orchestration_policy import load_conversation_orchestration_policy
from alice_conversation.registry import ConversationModelRegistry
from alice_conversation.repair_policy import load_conversation_response_repair_policy
from alice_conversation.response_validation_policy import load_conversation_response_validation_policy
from alice_conversation.state_policy import load_conversation_state_policy
from alice_conversation.state_service import ConversationStateService
from alice_conversation.state_store import ConversationStateStore

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ROOT / "policies"


@dataclass
class SequenceModel:
    responses: list[object]
    provider: str = "test-provider"
    model: str = "test-model"
    requests: list[ModelRequest] = field(default_factory=list)

    def generate(self, request: ModelRequest, cancellation: CancellationToken | None = None) -> ModelResponse:
        request.validate()
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ModelResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            content=str(item),
            finish_reason="stop",
            created_at="2026-07-26T00:00:00Z",
        )


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"2026-07-26T00:00:{self.value:02d}Z"


class SequenceMonotonic:
    def __init__(self, values: list[float] | None = None) -> None:
        self.values = list(values or [0.0] * 50)

    def __call__(self) -> float:
        if not self.values:
            return 0.0
        return self.values.pop(0)


def contract() -> ConstitutionalSystemContract:
    content = "You are A.L.I.C.E. Answer truthfully and obey all capability boundaries."
    return ConstitutionalSystemContract(
        version="contract-v1",
        policy_version="policy-v1",
        constitution_version="constitution-v1",
        content=content,
        content_sha256=sha256_text(content),
        sources=(
            ConstitutionalSourceSnapshot(
                path="docs/ALICE_CONSTITUTION.md",
                version="0.1.0",
                normalized_sha256="0" * 64,
            ),
        ),
    )


def enabled_repair_policy(**changes):
    policy = load_conversation_response_repair_policy(
        POLICIES / "conversation_response_repair_policy.json"
    )
    return replace(policy, enabled=True, **changes)


def build_runtime(tmp_path: Path, responses: list[object], *, repair_policy=None, monotonic=None):
    repo_root = ROOT
    vault = tmp_path / "vault"
    store = ConversationStateStore.from_paths(
        database_path=vault / "conversation.sqlite3",
        allowed_root=vault,
        repository_root=repo_root,
        policy=load_conversation_state_policy(POLICIES / "conversation_state_policy.json"),
    )
    store.initialize()
    events = iter(f"event-{i}" for i in range(1000))
    service = ConversationStateService(store, event_id_factory=lambda: next(events))
    service.create_session(
        session_id="session-1",
        created_at="2026-07-26T00:00:00Z",
        retention="retained",
    )
    adapter = SequenceModel(list(responses))
    registry = ConversationModelRegistry()
    registry.register(adapter)
    orchestrator = ConversationOrchestrator(
        state_service=service,
        model_registry=registry,
        system_contract=contract(),
        policy=load_conversation_orchestration_policy(
            POLICIES / "conversation_orchestration_policy.json"
        ),
        response_validation_policy=load_conversation_response_validation_policy(
            POLICIES / "conversation_response_validation_policy.json"
        ),
        context_policy=load_conversation_context_policy(
            POLICIES / "conversation_context_policy.json"
        ),
        repair_policy=repair_policy or enabled_repair_policy(),
        clock=IncrementingClock(),
        monotonic_clock=monotonic or SequenceMonotonic(),
    )
    return store, service, adapter, orchestrator


def command(*, turn_id="turn-1", request_id="request-1", generation_id="generation-1"):
    return ConversationTurnCommand(
        session_id="session-1",
        turn_id=turn_id,
        user_message_id=f"user-{turn_id}",
        assistant_message_id=f"assistant-{turn_id}",
        request_id=request_id,
        generation_id=generation_id,
        provider="test-provider",
        model="test-model",
        user_content="Please answer this request.",
    )
