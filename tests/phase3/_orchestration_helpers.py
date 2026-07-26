from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

from alice_conversation.constitutional_policy import load_constitutional_dialogue_policy
from alice_conversation.constitutional_prompt import compile_constitutional_system_contract
from alice_conversation.contracts import (
    ConversationCitation,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    ModelRequest,
    ModelResponse,
    sha256_text,
)
from alice_conversation.model import CancellationToken
from alice_conversation.orchestration import ConversationOrchestrator
from alice_conversation.orchestration_policy import load_conversation_orchestration_policy
from alice_conversation.registry import ConversationModelRegistry
from alice_conversation.state_policy import load_conversation_state_policy
from alice_conversation.state_service import ConversationStateService
from alice_conversation.state_store import ConversationStateStore


class DeterministicClock:
    def __init__(self, start: str = "2026-07-26T05:00:00Z") -> None:
        self.current = datetime.fromisoformat(start.replace("Z", "+00:00"))

    def __call__(self) -> str:
        value = self.current.astimezone(timezone.utc).replace(microsecond=0)
        self.current += timedelta(seconds=1)
        return value.isoformat().replace("+00:00", "Z")


@dataclass
class RecordingModel:
    response_text: str | None = None
    provider: str = "deterministic-test"
    model: str = "orchestration-v1"
    finish_reason: str = "stop"
    requests: list[ModelRequest] = field(default_factory=list)
    calls: int = 0

    def generate(
        self,
        request: ModelRequest,
        cancellation: CancellationToken | None = None,
    ) -> ModelResponse:
        request.validate()
        self.calls += 1
        self.requests.append(request)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        content = self.response_text
        if content is None:
            if request.grounding is None:
                content = "Grounded response."
            else:
                claim = request.grounding.claims[0]
                content = f"{claim.text} {claim.citations[0].token}"
        response = ModelResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            content=content,
            finish_reason=self.finish_reason,
            created_at="2026-07-26T05:00:30Z",
        )
        response.validate()
        return response


def make_orchestrator(tmp_path: Path, model=None):
    repository = tmp_path / "repository"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    source_root = Path(__file__).resolve().parents[2]
    for relative in (
        "docs/ALICE_CONSTITUTION.md",
        "docs/EVALUATION_CHARTER.md",
        "docs/PERMISSION_MODEL.md",
        "docs/THREAT_MODEL.md",
        "policies/conversation_policy.json",
    ):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((source_root / relative).read_text(encoding="utf-8"), encoding="utf-8")

    state_policy = load_conversation_state_policy(source_root / "policies/conversation_state_policy.json")
    store = ConversationStateStore.for_vault(
        vault_root=vault,
        repository_root=repository,
        policy=state_policy,
    )
    store.initialize()
    sequence = count()
    service = ConversationStateService(
        store,
        event_id_factory=lambda: f"event-{next(sequence)}",
    )
    selected_model = model or RecordingModel()
    registry = ConversationModelRegistry()
    registry.register(selected_model)
    constitutional_policy = load_constitutional_dialogue_policy(
        source_root / "policies/conversation_constitutional_policy.json"
    )
    contract = compile_constitutional_system_contract(
        policy=constitutional_policy,
        repository_root=repository,
    )
    policy = load_conversation_orchestration_policy(
        source_root / "policies/conversation_orchestration_policy.json"
    )
    clock = DeterministicClock()
    orchestrator = ConversationOrchestrator(
        state_service=service,
        model_registry=registry,
        system_contract=contract,
        policy=policy,
        clock=clock,
    )
    service.create_session(
        session_id="session-1",
        created_at=clock(),
        retention="retained",
        data_classification="PRIVATE",
    )
    return orchestrator, store, service, selected_model, registry, repository


def grounding_packet() -> ConversationGroundingPacket:
    citation = ConversationCitation(
        citation_id="citation-1",
        source_kind="memory_source",
        source_ref="memory-1",
        token="[memory:memory-1]",
        data_classification="PRIVATE",
    )
    claim_text = "Rayan prefers exact deterministic workflows."
    claim = ConversationGroundingClaim(
        claim_id="claim-1",
        text=claim_text,
        content_sha256=sha256_text(claim_text),
        knowledge_status="verified_fact",
        confidence=0.95,
        data_classification="PRIVATE",
        citations=(citation,),
    )
    packet = ConversationGroundingPacket(
        packet_id="packet-1",
        outcome="answerable",
        claims=(claim,),
        created_at="2026-07-26T04:59:00Z",
        max_classification="PRIVATE",
    )
    packet.validate()
    return packet
