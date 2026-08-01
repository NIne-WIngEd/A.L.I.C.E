"""Loader for P5.0d Attention and Workspace Projection policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .attention import (
    ATTENTION_HOST_OVERRIDES,
    ATTENTION_PRIORITY_CLASSES,
    ATTENTION_SUBJECT_TYPES,
    FOCUS_MODES,
    HOST_OVERRIDE_COMMANDS,
    HOST_OVERRIDE_STATUSES,
    INTERRUPTION_PREFERENCES,
    PROTECTED_INTERRUPT_REASONS,
)
from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    require_identifier,
    require_schema_version,
    require_text,
)
from .mission_policy import load_cognitive_kernel_mission_graph_policy
from .policy import load_cognitive_kernel_foundation_policy
from .workspace import (
    WORKSPACE_AUDIENCES,
    WORKSPACE_ITEM_TYPES,
    WORKSPACE_LAYOUT_MODES,
    WORKSPACE_PRIVACY_CLASSES,
    WORKSPACE_PROJECTION_STATES,
    WORKSPACE_REDACTION_STATES,
    WORKSPACE_ROLES,
)

_POLICY_KEYS = {
    "cognitive_kernel_attention_workspace_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "required_capabilities",
    "allowed_attention_subject_types",
    "allowed_attention_priority_classes",
    "allowed_protected_interrupt_reasons",
    "allowed_attention_host_overrides",
    "allowed_interruption_preferences",
    "allowed_focus_modes",
    "allowed_host_override_commands",
    "allowed_host_override_statuses",
    "allowed_workspace_item_types",
    "allowed_workspace_roles",
    "allowed_workspace_layout_modes",
    "allowed_workspace_audiences",
    "allowed_workspace_privacy_classes",
    "allowed_workspace_redaction_states",
    "allowed_workspace_projection_states",
    "invariants",
    "capability_ceiling",
}
_INVARIANT_KEYS = {
    "product_host_scope_required",
    "host_override_precedes_learned_ranking",
    "protected_interrupts_cannot_be_suppressed",
    "commercial_attention_priority_forbidden",
    "ranking_explainable",
    "canonical_state_in_frontend",
    "no_empty_fixed_slots",
    "projection_metadata_only",
    "cross_host_projection_forbidden",
    "non_host_sensitive_redaction_required",
    "remote_attention_manipulation_allowed",
    "persistent_store_implemented",
    "complete_ui_implemented",
    "autonomous_attention_execution_implemented",
    "speaker_guest_authority_implemented",
}
_REQUIRED_CAPABILITIES = (
    "attention_policy.v1",
    "workspace_projection.v1",
    "adaptive_compositor.v1",
    "host_window_override.v1",
)


@dataclass(frozen=True)
class CognitiveKernelAttentionWorkspacePolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_capabilities: tuple[str, ...]
    allowed_attention_subject_types: tuple[str, ...]
    allowed_attention_priority_classes: tuple[str, ...]
    allowed_protected_interrupt_reasons: tuple[str, ...]
    allowed_attention_host_overrides: tuple[str, ...]
    allowed_interruption_preferences: tuple[str, ...]
    allowed_focus_modes: tuple[str, ...]
    allowed_host_override_commands: tuple[str, ...]
    allowed_host_override_statuses: tuple[str, ...]
    allowed_workspace_item_types: tuple[str, ...]
    allowed_workspace_roles: tuple[str, ...]
    allowed_workspace_layout_modes: tuple[str, ...]
    allowed_workspace_audiences: tuple[str, ...]
    allowed_workspace_privacy_classes: tuple[str, ...]
    allowed_workspace_redaction_states: tuple[str, ...]
    allowed_workspace_projection_states: tuple[str, ...]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "attention-workspace policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        require_schema_version(self.version, "version")
        if self.version != "0.3.0":
            raise CognitiveKernelContractError(
                "attention-workspace contract version changed"
            )
        if self.phase != "5" or self.milestone != "P5.0d":
            raise CognitiveKernelContractError(
                "attention-workspace policy milestone changed"
            )
        if self.status != "contract_foundation":
            raise CognitiveKernelContractError(
                "attention-workspace policy status is invalid"
            )
        if self.required_capabilities != _REQUIRED_CAPABILITIES:
            raise CognitiveKernelContractError(
                "attention-workspace capabilities changed"
            )
        comparisons = (
            (set(self.allowed_attention_subject_types), ATTENTION_SUBJECT_TYPES, "attention subject types"),
            (tuple(self.allowed_attention_priority_classes), ATTENTION_PRIORITY_CLASSES, "attention priority order"),
            (set(self.allowed_protected_interrupt_reasons), PROTECTED_INTERRUPT_REASONS, "protected interrupt reasons"),
            (set(self.allowed_attention_host_overrides), ATTENTION_HOST_OVERRIDES, "attention host overrides"),
            (set(self.allowed_interruption_preferences), INTERRUPTION_PREFERENCES, "interruption preferences"),
            (set(self.allowed_focus_modes), FOCUS_MODES, "focus modes"),
            (set(self.allowed_host_override_commands), HOST_OVERRIDE_COMMANDS, "host override commands"),
            (set(self.allowed_host_override_statuses), HOST_OVERRIDE_STATUSES, "host override statuses"),
            (set(self.allowed_workspace_item_types), WORKSPACE_ITEM_TYPES, "workspace item types"),
            (set(self.allowed_workspace_roles), WORKSPACE_ROLES, "workspace roles"),
            (set(self.allowed_workspace_layout_modes), WORKSPACE_LAYOUT_MODES, "workspace layout modes"),
            (set(self.allowed_workspace_audiences), WORKSPACE_AUDIENCES, "workspace audiences"),
            (set(self.allowed_workspace_privacy_classes), WORKSPACE_PRIVACY_CLASSES, "workspace privacy classes"),
            (set(self.allowed_workspace_redaction_states), WORKSPACE_REDACTION_STATES, "workspace redaction states"),
            (set(self.allowed_workspace_projection_states), WORKSPACE_PROJECTION_STATES, "workspace projection states"),
        )
        for actual, expected, label in comparisons:
            if actual != expected:
                raise CognitiveKernelContractError(f"{label} changed")
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError(
                "attention-workspace invariants changed"
            )
        required_true = {
            "product_host_scope_required",
            "host_override_precedes_learned_ranking",
            "protected_interrupts_cannot_be_suppressed",
            "commercial_attention_priority_forbidden",
            "ranking_explainable",
            "no_empty_fixed_slots",
            "projection_metadata_only",
            "cross_host_projection_forbidden",
            "non_host_sensitive_redaction_required",
        }
        required_false = {
            "canonical_state_in_frontend",
            "remote_attention_manipulation_allowed",
            "persistent_store_implemented",
            "complete_ui_implemented",
            "autonomous_attention_execution_implemented",
            "speaker_guest_authority_implemented",
        }
        for key in required_true:
            if self.invariants.get(key) is not True:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be true"
                )
        for key in required_false:
            if self.invariants.get(key) is not False:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be false"
                )
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError(
                "attention-workspace policy may not be a capability ceiling"
            )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CognitiveKernelContractError(
            f"could not load JSON policy: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(
            f"JSON policy must be an object: {path}"
        )
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise CognitiveKernelContractError(
            f"{field} keys changed; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CognitiveKernelContractError(
            f"{field} must be a non-empty list"
        )
    result = tuple(
        require_text(item, field, maximum=256) for item in value
    )
    if len(set(result)) != len(result):
        raise CognitiveKernelContractError(
            f"{field} may not contain duplicates"
        )
    return result


def _contains_string(value: object, target: str) -> bool:
    if value == target:
        return True
    if isinstance(value, dict):
        return target in value or any(
            _contains_string(item, target) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_string(item, target) for item in value)
    return False


def load_cognitive_kernel_attention_workspace_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelAttentionWorkspacePolicy:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = (
        Path(path).resolve()
        if path is not None
        else root
        / "policies"
        / "cognitive_kernel_attention_workspace_policy.json"
    )
    payload = _read_json(source)
    _exact_keys(payload, _POLICY_KEYS, "attention-workspace policy")
    invariants = payload["invariants"]
    if not isinstance(invariants, dict):
        raise CognitiveKernelContractError(
            "attention-workspace invariants must be an object"
        )
    _exact_keys(
        invariants,
        _INVARIANT_KEYS,
        "attention-workspace invariants",
    )
    if any(not isinstance(value, bool) for value in invariants.values()):
        raise CognitiveKernelContractError(
            "attention-workspace invariants must be booleans"
        )
    policy = CognitiveKernelAttentionWorkspacePolicy(
        schema_version=int(
            payload[
                "cognitive_kernel_attention_workspace_policy_schema_version"
            ]
        ),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(
            payload["milestone"], "milestone", maximum=32
        ),
        status=require_identifier(payload["status"], "status"),
        required_capabilities=_sequence(
            payload["required_capabilities"], "required_capabilities"
        ),
        allowed_attention_subject_types=_sequence(
            payload["allowed_attention_subject_types"],
            "allowed_attention_subject_types",
        ),
        allowed_attention_priority_classes=_sequence(
            payload["allowed_attention_priority_classes"],
            "allowed_attention_priority_classes",
        ),
        allowed_protected_interrupt_reasons=_sequence(
            payload["allowed_protected_interrupt_reasons"],
            "allowed_protected_interrupt_reasons",
        ),
        allowed_attention_host_overrides=_sequence(
            payload["allowed_attention_host_overrides"],
            "allowed_attention_host_overrides",
        ),
        allowed_interruption_preferences=_sequence(
            payload["allowed_interruption_preferences"],
            "allowed_interruption_preferences",
        ),
        allowed_focus_modes=_sequence(
            payload["allowed_focus_modes"], "allowed_focus_modes"
        ),
        allowed_host_override_commands=_sequence(
            payload["allowed_host_override_commands"],
            "allowed_host_override_commands",
        ),
        allowed_host_override_statuses=_sequence(
            payload["allowed_host_override_statuses"],
            "allowed_host_override_statuses",
        ),
        allowed_workspace_item_types=_sequence(
            payload["allowed_workspace_item_types"],
            "allowed_workspace_item_types",
        ),
        allowed_workspace_roles=_sequence(
            payload["allowed_workspace_roles"],
            "allowed_workspace_roles",
        ),
        allowed_workspace_layout_modes=_sequence(
            payload["allowed_workspace_layout_modes"],
            "allowed_workspace_layout_modes",
        ),
        allowed_workspace_audiences=_sequence(
            payload["allowed_workspace_audiences"],
            "allowed_workspace_audiences",
        ),
        allowed_workspace_privacy_classes=_sequence(
            payload["allowed_workspace_privacy_classes"],
            "allowed_workspace_privacy_classes",
        ),
        allowed_workspace_redaction_states=_sequence(
            payload["allowed_workspace_redaction_states"],
            "allowed_workspace_redaction_states",
        ),
        allowed_workspace_projection_states=_sequence(
            payload["allowed_workspace_projection_states"],
            "allowed_workspace_projection_states",
        ),
        invariants={str(key): bool(value) for key, value in invariants.items()},
        capability_ceiling=bool(payload["capability_ceiling"]),
        digest=canonical_sha256(payload),
        source_path=source,
    )
    policy.validate()

    foundation = load_cognitive_kernel_foundation_policy(
        repository_root=root
    )
    if foundation.version != "0.1.0":
        raise CognitiveKernelContractError(
            "unexpected foundation contract version"
        )
    mission = load_cognitive_kernel_mission_graph_policy(
        repository_root=root
    )
    if mission.version != "0.2.0":
        raise CognitiveKernelContractError(
            "unexpected Mission Graph contract version"
        )

    parity = _read_json(
        root / "policies" / "capability_parity_ledger.json"
    )
    for capability in policy.required_capabilities:
        if not _contains_string(parity, capability):
            raise CognitiveKernelContractError(
                f"parity ledger is missing {capability}"
            )

    product_lines = _read_json(root / "policies" / "product_lines.json")
    shared_kernel = product_lines.get("shared_kernel")
    if not isinstance(shared_kernel, dict):
        raise CognitiveKernelContractError(
            "product manifest shared_kernel is invalid"
        )
    phase5_contracts = shared_kernel.get("phase5_contracts")
    if not isinstance(phase5_contracts, list):
        raise CognitiveKernelContractError(
            "product manifest phase5_contracts is invalid"
        )
    for contract in ("attention_decision", "workspace_projection"):
        if contract not in phase5_contracts:
            raise CognitiveKernelContractError(
                f"product manifest is missing {contract}"
            )
    return policy
