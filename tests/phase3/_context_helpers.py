from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from alice_conversation.constitutional_prompt import (
    ConstitutionalSourceSnapshot,
    ConstitutionalSystemContract,
)
from alice_conversation.context_policy import (
    ConversationContextPolicy,
    parse_conversation_context_policy,
)
from alice_conversation.contracts import ModelRequest, ModelResponse, sha256_text
from alice_conversation.model import CancellationToken
from alice_conversation.orchestration import (
    ConversationGenerationInterruptedError,
    ConversationOrchestrator,
    ConversationTurnCommand,
)
from alice_conversation.orchestration_policy import ConversationOrchestrationPolicy
from alice_conversation.registry import ConversationModelRegistry
from alice_conversation.response_validation_policy import (
    ConversationResponseValidationPolicy,
)
from alice_conversation.state_policy import ConversationStatePolicy
from alice_conversation.state_service import ConversationStateService
from alice_conversation.state_store import ConversationStateStore


def context_policy_payload(**limit_overrides) -> dict:
    limits = {
        "max_prior_turns": 12,
        "max_prior_messages": 24,
        "max_prior_characters": 12000,
    }
    limits.update(limit_overrides)
    return {
        "policy_name": "alice_conversation_context_policy",
        "version": "1.0.0",
        "phase": "3",
        "milestone": "P3.8",
        "status": "governed_cross_turn_context",
        "boundaries": {
            "same_session_only": True,
            "completed_turns_only": True,
            "accepted_or_abstained_only": True,
            "whole_turn_pairs_only": True,
            "exclude_current_turn": True,
            "integrity_verification_required": True,
            "hidden_reasoning_allowed": False,
            "rejected_output_allowed": False,
            "failed_turn_content_allowed": False,
            "cross_session_content_allowed": False,
            "message_identifiers_rendered_to_model": False,
            "semantic_summarization_allowed": False,
            "memory_write_allowed": False,
        },
        "limits": limits,
        "truncation": {
            "strategy": "recent_contiguous_suffix",
            "drop_oldest_first": True,
            "partial_turn_allowed": False,
            "partial_message_allowed": False,
        },
        "failure_codes": {
            "integrity": "context_integrity_failed",
            "assembly": "context_assembly_failed",
        },
    }


def context_policy(**limit_overrides) -> ConversationContextPolicy:
    return parse_conversation_context_policy(
        context_policy_payload(**limit_overrides)
    )


def state_policy() -> ConversationStatePolicy:
    return ConversationStatePolicy(
        policy_name="alice_conversation_state_policy",
        version="1.0.0",
        phase="3",
        milestone="P3.2",
        status="private_conversation_state",
        database_relative_path="conversation/alice-conversation.sqlite3",
        repository_storage_allowed=False,
        private_output_only=True,
        journal_mode="WAL",
        synchronous="FULL",
        foreign_keys=True,
        default_retention="session_only",
        allowed_retentions=("session_only", "retained"),
        session_only_close_action="purge",
        retained_close_action="retain",
        ordinary_classifications=("PUBLIC", "INTERNAL", "PRIVATE"),
        highly_sensitive_allowed=False,
        secrets_allowed=False,
        chain_of_thought_persistence_allowed=False,
        memory_write_allowed=False,
        external_action_allowed=False,
        web_access_allowed=False,
        tool_calling_allowed=False,
        max_message_chars=100000,
        max_turns_per_session=10000,
        max_references_per_turn=256,
    )


def orchestration_policy() -> ConversationOrchestrationPolicy:
    return ConversationOrchestrationPolicy(
        policy_name="alice_conversation_orchestration_policy",
        version="1.1.0",
        phase="3",
        milestone="P3.5",
        status="controlled_turn_orchestration",
        boundaries=(
            ("web_access_allowed", False),
            ("tool_calling_allowed", False),
            ("external_action_allowed", False),
            ("memory_write_allowed", False),
            ("memory_promotion_allowed", False),
            ("highly_sensitive_grounding_allowed", False),
            ("chain_of_thought_persistence_allowed", False),
        ),
        lifecycle=(
            ("constitutional_contract_required", True),
            ("prebuilt_grounding_only", True),
            ("live_retrieval_allowed", False),
            ("model_registry_resolution_required", True),
            ("generation_attempt_recording_required", True),
            ("atomic_state_transitions_required", True),
            ("response_identity_match_required", True),
            ("duplicate_assistant_messages_allowed", False),
            ("automatic_retry_count", 0),
            ("provider_fallback_allowed", False),
            ("final_grounding_validation_enabled", True),
        ),
        max_output_tokens=1024,
        temperature=0.0,
        failure_codes=(
            ("cancelled", "model_cancelled"),
            ("interrupted", "model_interrupted"),
            ("timeout", "model_timeout"),
            ("budget", "model_budget"),
            ("provider", "provider_failure"),
            ("configuration", "model_configuration"),
            ("protocol", "model_protocol"),
            ("internal", "orchestration_internal"),
            ("validation", "response_validation_rejected"),
        ),
    )


def response_policy() -> ConversationResponseValidationPolicy:
    return ConversationResponseValidationPolicy(
        policy_name="alice_conversation_response_validation_policy",
        version="1.0.0",
        phase="3",
        milestone="P3.6",
        status="generated_response_validation",
        boundaries=(
            ("web_access_allowed", False),
            ("tool_calling_allowed", False),
            ("external_action_allowed", False),
            ("memory_write_allowed", False),
            ("memory_promotion_allowed", False),
            ("highly_sensitive_grounding_allowed", False),
            ("chain_of_thought_persistence_allowed", False),
            ("automatic_repair_allowed", False),
            ("provider_fallback_allowed", False),
        ),
        citation_rules=(
            ("require_exact_tokens", True),
            ("reject_unknown_tokens", True),
            ("require_grounded_personal_claims", True),
            ("require_supported_factual_claims", True),
        ),
        minimum_answerable_claims_cited=1,
        minimum_conflict_claims_cited=2,
        epistemic_rules=(
            ("preserve_conflict", True),
            ("preserve_uncertainty", True),
            ("require_abstention_on_insufficient_evidence", True),
            ("require_abstention_on_denied", True),
            ("require_abstention_on_not_applicable", True),
            ("reject_certainty_language_for_conflict", True),
            ("reject_certainty_language_for_uncertainty", True),
        ),
        safety_rules=(
            ("reject_action_completion_claims", True),
            ("reject_capability_claims", True),
            ("reject_invented_personal_facts", True),
            ("reject_dependency_language", True),
            ("reject_hidden_reasoning_disclosure", True),
            ("reject_truncated_responses", True),
        ),
        max_response_chars=20000,
        max_issues=64,
        failure_codes=(
            ("internal", "response_validation_internal"),
            ("rejected", "response_validation_rejected"),
        ),
    )


def system_contract() -> ConstitutionalSystemContract:
    content = "Follow the trusted constitutional response contract."
    source_text = "ratified source"
    contract = ConstitutionalSystemContract(
        version="test-contract-v1",
        policy_version="test-policy-v1",
        constitution_version="0.1.0",
        content=content,
        content_sha256=sha256_text(content),
        sources=(
            ConstitutionalSourceSnapshot(
                path="docs/ALICE_CONSTITUTION.md",
                version="0.1.0",
                normalized_sha256=sha256_text(source_text),
            ),
        ),
    )
    contract.validate()
    return contract


@dataclass
class RecordingModel:
    provider: str = "deterministic-test"
    model: str = "context-v1"
    response_text: str = "Acknowledged."
    interrupt_calls: set[int] = field(default_factory=set)
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
        if self.calls in self.interrupt_calls:
            raise ConversationGenerationInterruptedError()
        return ModelResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            content=self.response_text,
            finish_reason="stop",
            created_at="2026-07-26T07:00:00Z",
        )


def make_orchestrator(
    tmp_path: Path,
    *,
    model: RecordingModel | None = None,
    selected_context_policy: ConversationContextPolicy | None = None,
    session_id: str = "session-1",
):
    repository = tmp_path / "repository"
    vault = tmp_path / "vault"
    repository.mkdir(parents=True)
    vault.mkdir(parents=True)
    store = ConversationStateStore.for_vault(
        vault_root=vault,
        repository_root=repository,
        policy=state_policy(),
    )
    store.initialize()
    events = count()
    service = ConversationStateService(
        store,
        event_id_factory=lambda: f"event-{next(events)}",
    )
    selected_model = model or RecordingModel()
    registry = ConversationModelRegistry()
    registry.register(selected_model)
    orchestrator = ConversationOrchestrator(
        state_service=service,
        model_registry=registry,
        system_contract=system_contract(),
        policy=orchestration_policy(),
        response_validation_policy=response_policy(),
        context_policy=selected_context_policy or context_policy(),
        clock=Clock(),
    )
    service.create_session(
        session_id=session_id,
        created_at="2026-07-26T06:00:00Z",
        retention="retained",
        data_classification="PRIVATE",
    )
    return orchestrator, store, service, selected_model


class Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"2026-07-26T06:{self.value // 60:02d}:{self.value % 60:02d}Z"


def command(
    index: int,
    *,
    session_id: str = "session-1",
    content: str | None = None,
    grounding=None,
):
    return ConversationTurnCommand(
        session_id=session_id,
        turn_id=f"turn-{index}",
        user_message_id=f"user-{index}",
        assistant_message_id=f"assistant-{index}",
        request_id=f"request-{index}",
        generation_id=f"generation-{index}",
        provider="deterministic-test",
        model="context-v1",
        user_content=content or f"User message {index}.",
        grounding=grounding,
    )
