"""Metadata-safe inspection for governed P3.8 cross-turn context."""

from __future__ import annotations

from dataclasses import dataclass

from .context_assembly import ConversationContextAssembly
from .context_policy import ConversationContextPolicy


@dataclass(frozen=True)
class ConversationContextInspection:
    policy_version: str
    context_sha256: str
    eligible_turn_count: int
    included_turn_count: int
    included_message_count: int
    omitted_turn_count: int
    excluded_turn_count: int
    included_character_count: int
    truncated: bool
    roles: tuple[str, ...]


def inspect_conversation_context(
    assembly: ConversationContextAssembly,
    *,
    policy: ConversationContextPolicy,
) -> ConversationContextInspection:
    """Return context metadata without returning message text or identifiers."""

    assembly.validate(policy=policy)
    return ConversationContextInspection(
        policy_version=assembly.policy_version,
        context_sha256=assembly.context_sha256,
        eligible_turn_count=assembly.eligible_turn_count,
        included_turn_count=assembly.included_turn_count,
        included_message_count=len(assembly.messages),
        omitted_turn_count=assembly.omitted_turn_count,
        excluded_turn_count=assembly.excluded_turn_count,
        included_character_count=assembly.included_character_count,
        truncated=assembly.truncated,
        roles=tuple(message.role for message in assembly.messages),
    )


def render_conversation_context_inspection(
    inspection: ConversationContextInspection,
) -> str:
    """Render deterministic metadata-only context diagnostics."""

    return "\n".join(
        (
            "A.L.I.C.E. conversation context inspection",
            f"policy_version={inspection.policy_version}",
            f"context_sha256={inspection.context_sha256}",
            f"eligible_turn_count={inspection.eligible_turn_count}",
            f"included_turn_count={inspection.included_turn_count}",
            f"included_message_count={inspection.included_message_count}",
            f"omitted_turn_count={inspection.omitted_turn_count}",
            f"excluded_turn_count={inspection.excluded_turn_count}",
            f"included_character_count={inspection.included_character_count}",
            f"truncated={str(inspection.truncated).lower()}",
            "roles=" + ",".join(inspection.roles),
        )
    )
