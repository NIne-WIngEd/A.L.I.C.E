"""Versioned private conversation-state policy for A.L.I.C.E. P3.2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError

DEFAULT_STATE_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_state_policy.json"
)


class ConversationStatePolicyError(ConversationContractError):
    """Raised when the public P3.2 conversation-state policy is invalid."""


@dataclass(frozen=True)
class ConversationStatePolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    database_relative_path: str
    repository_storage_allowed: bool
    private_output_only: bool
    journal_mode: str
    synchronous: str
    foreign_keys: bool
    default_retention: str
    allowed_retentions: tuple[str, ...]
    session_only_close_action: str
    retained_close_action: str
    ordinary_classifications: tuple[str, ...]
    highly_sensitive_allowed: bool
    secrets_allowed: bool
    chain_of_thought_persistence_allowed: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    web_access_allowed: bool
    tool_calling_allowed: bool
    max_message_chars: int
    max_turns_per_session: int
    max_references_per_turn: int


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationStatePolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationStatePolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConversationStatePolicyError(
            f"{field} must remain {str(expected).lower()} in P3.2."
        )
    return expected


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationStatePolicyError(f"{field} must be a positive integer.")
    return value


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConversationStatePolicyError(f"{field} must be a non-empty list.")
    values = tuple(_text(item, field=f"{field} item") for item in value)
    if len(values) != len(set(values)):
        raise ConversationStatePolicyError(f"{field} cannot contain duplicates.")
    return values


def _relative_database_path(value: Any) -> str:
    text = _text(value, field="storage.database_relative_path")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ConversationStatePolicyError(
            "storage.database_relative_path must stay relative to the private vault."
        )
    if path.name in {"", ".", ".."}:
        raise ConversationStatePolicyError(
            "storage.database_relative_path must name a database file."
        )
    return path.as_posix()


def parse_conversation_state_policy(
    payload: dict[str, Any],
) -> ConversationStatePolicy:
    storage = _mapping(payload.get("storage"), field="storage")
    retention = _mapping(payload.get("retention"), field="retention")
    boundaries = _mapping(payload.get("boundaries"), field="boundaries")
    limits = _mapping(payload.get("limits"), field="limits")

    allowed_retentions = _string_list(
        retention.get("allowed"),
        field="retention.allowed",
    )
    if allowed_retentions != ("session_only", "retained"):
        raise ConversationStatePolicyError(
            "retention.allowed must be exactly session_only and retained."
        )

    default_retention = _text(
        retention.get("default"),
        field="retention.default",
    )
    if default_retention != "session_only":
        raise ConversationStatePolicyError(
            "retention.default must remain session_only in P3.2."
        )

    ordinary_classifications = _string_list(
        boundaries.get("ordinary_classifications"),
        field="boundaries.ordinary_classifications",
    )
    if ordinary_classifications != ("PUBLIC", "INTERNAL", "PRIVATE"):
        raise ConversationStatePolicyError(
            "P3.2 ordinary classifications must remain PUBLIC, INTERNAL, PRIVATE."
        )

    journal_mode = _text(storage.get("journal_mode"), field="storage.journal_mode")
    if journal_mode != "WAL":
        raise ConversationStatePolicyError("storage.journal_mode must remain WAL.")

    synchronous = _text(storage.get("synchronous"), field="storage.synchronous")
    if synchronous != "FULL":
        raise ConversationStatePolicyError("storage.synchronous must remain FULL.")

    session_only_action = _text(
        retention.get("session_only_close_action"),
        field="retention.session_only_close_action",
    )
    if session_only_action != "purge":
        raise ConversationStatePolicyError(
            "Session-only conversations must be purged when closed."
        )

    retained_action = _text(
        retention.get("retained_close_action"),
        field="retention.retained_close_action",
    )
    if retained_action != "retain":
        raise ConversationStatePolicyError(
            "Retained conversations must remain available after close."
        )

    policy = ConversationStatePolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        database_relative_path=_relative_database_path(
            storage.get("database_relative_path")
        ),
        repository_storage_allowed=_strict_bool(
            storage.get("repository_storage_allowed"),
            expected=False,
            field="storage.repository_storage_allowed",
        ),
        private_output_only=_strict_bool(
            storage.get("private_output_only"),
            expected=True,
            field="storage.private_output_only",
        ),
        journal_mode=journal_mode,
        synchronous=synchronous,
        foreign_keys=_strict_bool(
            storage.get("foreign_keys"),
            expected=True,
            field="storage.foreign_keys",
        ),
        default_retention=default_retention,
        allowed_retentions=allowed_retentions,
        session_only_close_action=session_only_action,
        retained_close_action=retained_action,
        ordinary_classifications=ordinary_classifications,
        highly_sensitive_allowed=_strict_bool(
            boundaries.get("highly_sensitive_allowed"),
            expected=False,
            field="boundaries.highly_sensitive_allowed",
        ),
        secrets_allowed=_strict_bool(
            boundaries.get("secrets_allowed"),
            expected=False,
            field="boundaries.secrets_allowed",
        ),
        chain_of_thought_persistence_allowed=_strict_bool(
            boundaries.get("chain_of_thought_persistence_allowed"),
            expected=False,
            field="boundaries.chain_of_thought_persistence_allowed",
        ),
        memory_write_allowed=_strict_bool(
            boundaries.get("memory_write_allowed"),
            expected=False,
            field="boundaries.memory_write_allowed",
        ),
        external_action_allowed=_strict_bool(
            boundaries.get("external_action_allowed"),
            expected=False,
            field="boundaries.external_action_allowed",
        ),
        web_access_allowed=_strict_bool(
            boundaries.get("web_access_allowed"),
            expected=False,
            field="boundaries.web_access_allowed",
        ),
        tool_calling_allowed=_strict_bool(
            boundaries.get("tool_calling_allowed"),
            expected=False,
            field="boundaries.tool_calling_allowed",
        ),
        max_message_chars=_positive_int(
            limits.get("max_message_chars"),
            field="limits.max_message_chars",
        ),
        max_turns_per_session=_positive_int(
            limits.get("max_turns_per_session"),
            field="limits.max_turns_per_session",
        ),
        max_references_per_turn=_positive_int(
            limits.get("max_references_per_turn"),
            field="limits.max_references_per_turn",
        ),
    )

    if policy.phase != "3" or policy.milestone != "P3.2":
        raise ConversationStatePolicyError(
            "Conversation-state policy must identify Phase 3 milestone P3.2."
        )
    return policy


def load_conversation_state_policy(
    path: str | Path = DEFAULT_STATE_POLICY_PATH,
) -> ConversationStatePolicy:
    policy_path = Path(path)
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConversationStatePolicyError(
            f"Unable to read conversation-state policy: {policy_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationStatePolicyError(
            f"Conversation-state policy is not valid JSON: {policy_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationStatePolicyError(
            "Conversation-state policy JSON root must be an object."
        )
    return parse_conversation_state_policy(payload)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_conversation_state_database_path(
    *,
    policy: ConversationStatePolicy,
    vault_root: str | Path,
    repository_root: str | Path,
) -> Path:
    vault = Path(vault_root).expanduser().resolve(strict=False)
    repository = Path(repository_root).expanduser().resolve(strict=False)
    if _is_within(vault, repository):
        raise ConversationStatePolicyError(
            "The private vault cannot be located inside the public repository."
        )
    database = (vault / policy.database_relative_path).resolve(strict=False)
    if not _is_within(database, vault):
        raise ConversationStatePolicyError(
            "Conversation-state database escaped the approved vault root."
        )
    if _is_within(database, repository):
        raise ConversationStatePolicyError(
            "Conversation-state database cannot be stored in the repository."
        )
    return database
