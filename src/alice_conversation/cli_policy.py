"""Versioned local conversational runtime policy for A.L.I.C.E. P3.7."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError

DEFAULT_CLI_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_cli_policy.json"
)

_EXPECTED_BOUNDARIES = {
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
}
_EXPECTED_COMMANDS = (
    ":help",
    ":new",
    ":close",
    ":inspect",
    ":cancel",
    ":resume",
    ":grounding",
    ":exit",
)


class ConversationCliPolicyError(ConversationContractError):
    """Raised when the P3.7 local-runtime policy is missing or weakened."""


@dataclass(frozen=True)
class ConversationCliPolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    boundaries: tuple[tuple[str, bool], ...]
    allowed_retentions: tuple[str, ...]
    default_retention: str
    allowed_providers: tuple[str, ...]
    explicit_provider_required: bool
    explicit_model_required: bool
    prebuilt_grounding_file_allowed: bool
    commands: tuple[str, ...]
    max_input_chars: int
    max_grounding_file_bytes: int

    def boundary(self, name: str) -> bool:
        return dict(self.boundaries)[name]


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationCliPolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationCliPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConversationCliPolicyError(
            f"{field} must remain {str(expected).lower()} in P3.7."
        )
    return expected


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationCliPolicyError(f"{field} must be a positive integer.")
    return value


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConversationCliPolicyError(f"{field} must be a non-empty array.")
    items = tuple(_text(item, field=f"{field} item") for item in value)
    if len(items) != len(set(items)):
        raise ConversationCliPolicyError(f"{field} cannot contain duplicates.")
    return items


def parse_conversation_cli_policy(payload: dict[str, Any]) -> ConversationCliPolicy:
    root = _mapping(payload, field="CLI policy")
    expected_root = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "boundaries",
        "runtime",
        "commands",
        "limits",
    }
    if set(root) != expected_root:
        raise ConversationCliPolicyError(
            "CLI policy fields do not match the P3.7 contract."
        )
    if _text(root["policy_name"], field="policy_name") != (
        "alice_conversation_cli_policy"
    ):
        raise ConversationCliPolicyError("Unexpected CLI policy name.")
    if _text(root["phase"], field="phase") != "3":
        raise ConversationCliPolicyError("CLI policy phase must be 3.")
    if _text(root["milestone"], field="milestone") != "P3.7":
        raise ConversationCliPolicyError("CLI policy milestone must be P3.7.")
    if _text(root["status"], field="status") != "local_conversational_runtime":
        raise ConversationCliPolicyError("Unexpected CLI policy status.")

    boundaries = _mapping(root["boundaries"], field="boundaries")
    if set(boundaries) != set(_EXPECTED_BOUNDARIES):
        raise ConversationCliPolicyError(
            "CLI boundaries do not match the P3.7 contract."
        )
    parsed_boundaries = tuple(
        (
            name,
            _strict_bool(
                boundaries[name], expected=expected, field=f"boundaries.{name}"
            ),
        )
        for name, expected in _EXPECTED_BOUNDARIES.items()
    )

    runtime = _mapping(root["runtime"], field="runtime")
    expected_runtime = {
        "allowed_retentions",
        "default_retention",
        "allowed_providers",
        "explicit_provider_required",
        "explicit_model_required",
        "prebuilt_grounding_file_allowed",
    }
    if set(runtime) != expected_runtime:
        raise ConversationCliPolicyError(
            "CLI runtime fields do not match the P3.7 contract."
        )
    allowed_retentions = _string_tuple(
        runtime["allowed_retentions"], field="runtime.allowed_retentions"
    )
    if allowed_retentions != ("session_only", "retained"):
        raise ConversationCliPolicyError(
            "P3.7 retentions must be exactly session_only and retained."
        )
    default_retention = _text(
        runtime["default_retention"], field="runtime.default_retention"
    )
    if default_retention != "session_only":
        raise ConversationCliPolicyError(
            "P3.7 default retention must remain session_only."
        )
    allowed_providers = _string_tuple(
        runtime["allowed_providers"], field="runtime.allowed_providers"
    )
    if allowed_providers != ("ollama-local",):
        raise ConversationCliPolicyError(
            "The user-facing P3.7 runtime may expose only ollama-local."
        )

    commands = _string_tuple(root["commands"], field="commands")
    if commands != _EXPECTED_COMMANDS:
        raise ConversationCliPolicyError(
            "P3.7 commands must match the approved local command surface."
        )

    limits = _mapping(root["limits"], field="limits")
    if set(limits) != {"max_input_chars", "max_grounding_file_bytes"}:
        raise ConversationCliPolicyError(
            "CLI limits do not match the P3.7 contract."
        )
    max_input_chars = _positive_int(
        limits["max_input_chars"], field="limits.max_input_chars"
    )
    if not 1_024 <= max_input_chars <= 100_000:
        raise ConversationCliPolicyError(
            "limits.max_input_chars must remain between 1024 and 100000."
        )
    max_grounding_file_bytes = _positive_int(
        limits["max_grounding_file_bytes"],
        field="limits.max_grounding_file_bytes",
    )
    if not 1_024 <= max_grounding_file_bytes <= 10_485_760:
        raise ConversationCliPolicyError(
            "Grounding file limit must remain between 1 KiB and 10 MiB."
        )

    return ConversationCliPolicy(
        policy_name="alice_conversation_cli_policy",
        version=_text(root["version"], field="version"),
        phase="3",
        milestone="P3.7",
        status="local_conversational_runtime",
        boundaries=parsed_boundaries,
        allowed_retentions=allowed_retentions,
        default_retention=default_retention,
        allowed_providers=allowed_providers,
        explicit_provider_required=_strict_bool(
            runtime["explicit_provider_required"],
            expected=True,
            field="runtime.explicit_provider_required",
        ),
        explicit_model_required=_strict_bool(
            runtime["explicit_model_required"],
            expected=True,
            field="runtime.explicit_model_required",
        ),
        prebuilt_grounding_file_allowed=_strict_bool(
            runtime["prebuilt_grounding_file_allowed"],
            expected=True,
            field="runtime.prebuilt_grounding_file_allowed",
        ),
        commands=commands,
        max_input_chars=max_input_chars,
        max_grounding_file_bytes=max_grounding_file_bytes,
    )


def load_conversation_cli_policy(
    path: str | Path = DEFAULT_CLI_POLICY_PATH,
) -> ConversationCliPolicy:
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConversationCliPolicyError(
            f"Unable to read conversation CLI policy: {selected}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationCliPolicyError(
            f"Conversation CLI policy is not valid JSON: {selected}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationCliPolicyError("Conversation CLI policy root must be an object.")
    return parse_conversation_cli_policy(payload)
