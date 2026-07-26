from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alice_conversation.cli_policy import parse_conversation_cli_policy
from alice_conversation.cli_runtime import ConversationCliRuntime
from alice_conversation.constitutional_prompt import (
    ConstitutionalSourceSnapshot,
    ConstitutionalSystemContract,
)
from alice_conversation.contracts import ModelRequest, ModelResponse, sha256_text, utc_now_text
from alice_conversation.model import CancellationToken
from alice_conversation.orchestration import (
    ConversationGenerationInterruptedError,
    ConversationOrchestrator,
)
from alice_conversation.orchestration_policy import ConversationOrchestrationPolicy
from alice_conversation.registry import ConversationModelRegistry
from alice_conversation.response_validation_policy import ConversationResponseValidationPolicy
from alice_conversation.state_policy import ConversationStatePolicy
from alice_conversation.state_service import ConversationStateService
from alice_conversation.state_store import ConversationStateStore


def cli_policy_payload() -> dict:
    return {
        "policy_name": "alice_conversation_cli_policy",
        "version": "1.0.0",
        "phase": "3",
        "milestone": "P3.7",
        "status": "local_conversational_runtime",
        "boundaries": {
            "local_only": True,
            "private_vault_required": True,
            "repository_state_allowed": False,
            "web_access_allowed": False,
            "tool_calling_allowed": False,
            "external_action_allowed": False,
            "memory_write_allowed": False,
            "memory_promotion_allowed": False,
            "live_retrieval_allowed": False,
            "hidden_reasoning_display_allowed": False,
            "raw_database_identifiers_display_allowed": False,
            "automatic_response_repair_allowed": False,
            "provider_fallback_allowed": False,
        },
        "runtime": {
            "allowed_retentions": ["session_only", "retained"],
            "default_retention": "session_only",
            "allowed_providers": ["ollama-local"],
            "explicit_provider_required": True,
            "explicit_model_required": True,
            "prebuilt_grounding_file_allowed": True,
        },
        "commands": [
            ":help",
            ":new",
            ":close",
            ":inspect",
            ":cancel",
            ":resume",
            ":grounding",
            ":exit",
        ],
        "limits": {
            "max_input_chars": 100000,
            "max_grounding_file_bytes": 1048576,
        },
    }


def cli_policy():
    return parse_conversation_cli_policy(cli_policy_payload())


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
        max_response_chars=100000,
        max_issues=64,
        failure_codes=(
            ("rejected", "response_validation_rejected"),
            ("internal", "response_validation_internal"),
        ),
    )


def system_contract() -> ConstitutionalSystemContract:
    content = "Be truthful. Do not use tools, web access, external actions, or memory writes."
    source_text = "source"
    return ConstitutionalSystemContract(
        version="alice-constitution-0.1.0",
        policy_version="1.0.0",
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


@dataclass
class SequenceModel:
    outputs: list[object]
    provider: str = "ollama-local"
    model: str = "qwen3:8b"
    requests: list[ModelRequest] = field(default_factory=list)

    def generate(self, request, cancellation: CancellationToken | None = None):
        request.validate()
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("No fake output remains.")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return ModelResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            content=str(output),
            finish_reason="stop",
            created_at=utc_now_text(),
        )


def make_runtime(tmp_path: Path, outputs: list[object], *, grounding=None, retention="session_only"):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    policy = state_policy()
    store = ConversationStateStore.from_paths(
        database_path=vault / "conversation" / "state.sqlite3",
        allowed_root=vault,
        repository_root=repository,
        policy=policy,
    )
    store.initialize()
    service = ConversationStateService(store)
    registry = ConversationModelRegistry()
    model = SequenceModel(list(outputs))
    registry.register(model)
    orchestrator = ConversationOrchestrator(
        state_service=service,
        model_registry=registry,
        system_contract=system_contract(),
        policy=orchestration_policy(),
        response_validation_policy=response_policy(),
    )
    counter = iter(range(1, 1000))
    runtime = ConversationCliRuntime(
        state_service=service,
        orchestrator=orchestrator,
        provider="ollama-local",
        model="qwen3:8b",
        policy=cli_policy(),
        grounding=grounding,
        id_factory=lambda: str(next(counter)),
    )
    runtime.new_session(retention)
    return runtime, model, store


INTERRUPT = ConversationGenerationInterruptedError("model_interrupted")
