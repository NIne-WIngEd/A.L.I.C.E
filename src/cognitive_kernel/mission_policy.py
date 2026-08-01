"""Loader for P5.0c Mission Graph and Result Capsule policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .canonical import CognitiveKernelContractError, canonical_sha256, require_identifier, require_schema_version, require_text
from .mission import EDGE_STATES, EXECUTION_STATES, MISSION_EDGE_TYPES, MISSION_NODE_TYPES, NODE_STATUSES, VISIBILITY_STATES
from .routing import ROUTING_ACTIONS
from .results import RESULT_STATUSES, TRACEBACK_ACTIONS, TRACEBACK_STATUSES
from .policy import load_cognitive_kernel_foundation_policy

_POLICY_KEYS = {
    "cognitive_kernel_mission_graph_policy_schema_version",
    "policy_id", "version", "phase", "milestone", "status",
    "required_capabilities", "allowed_node_types", "allowed_node_statuses",
    "allowed_execution_states", "allowed_visibility_states",
    "allowed_edge_types", "allowed_edge_states", "allowed_routing_actions",
    "allowed_result_statuses", "allowed_traceback_actions",
    "allowed_traceback_statuses", "invariants", "capability_ceiling",
}
_INVARIANT_KEYS = {
    "immutable_node_identity", "product_host_scope_required",
    "metadata_only_labels", "parent_child_acyclic",
    "single_parent_for_non_root", "cross_host_links_forbidden",
    "canonical_state_in_frontend", "persistence_implemented",
    "complete_ui_implemented", "attention_engine_implemented",
    "speaker_guest_implemented",
}
_REQUIRED_CAPABILITIES = (
    "mission_graph.v1", "semantic_router.v1", "result_capsule.v1", "traceback_engine.v1"
)

@dataclass(frozen=True)
class CognitiveKernelMissionGraphPolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_capabilities: tuple[str, ...]
    allowed_node_types: tuple[str, ...]
    allowed_node_statuses: tuple[str, ...]
    allowed_execution_states: tuple[str, ...]
    allowed_visibility_states: tuple[str, ...]
    allowed_edge_types: tuple[str, ...]
    allowed_edge_states: tuple[str, ...]
    allowed_routing_actions: tuple[str, ...]
    allowed_result_statuses: tuple[str, ...]
    allowed_traceback_actions: tuple[str, ...]
    allowed_traceback_statuses: tuple[str, ...]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError("mission graph policy schema version must be 1")
        require_identifier(self.policy_id, "policy_id")
        require_schema_version(self.version, "version")
        if self.phase != "5" or self.milestone != "P5.0c":
            raise CognitiveKernelContractError("mission graph policy milestone changed")
        if self.status != "contract_foundation":
            raise CognitiveKernelContractError("mission graph policy status is invalid")
        if self.required_capabilities != _REQUIRED_CAPABILITIES:
            raise CognitiveKernelContractError("mission graph capabilities changed")
        comparisons = (
            (set(self.allowed_node_types), MISSION_NODE_TYPES, "node types"),
            (set(self.allowed_node_statuses), NODE_STATUSES, "node statuses"),
            (set(self.allowed_execution_states), EXECUTION_STATES, "execution states"),
            (set(self.allowed_visibility_states), VISIBILITY_STATES, "visibility states"),
            (set(self.allowed_edge_types), MISSION_EDGE_TYPES, "edge types"),
            (set(self.allowed_edge_states), EDGE_STATES, "edge states"),
            (set(self.allowed_routing_actions), ROUTING_ACTIONS, "routing actions"),
            (set(self.allowed_result_statuses), RESULT_STATUSES, "result statuses"),
            (set(self.allowed_traceback_actions), TRACEBACK_ACTIONS, "traceback actions"),
            (set(self.allowed_traceback_statuses), TRACEBACK_STATUSES, "traceback statuses"),
        )
        for actual, expected, label in comparisons:
            if actual != expected:
                raise CognitiveKernelContractError(f"{label} changed")
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError("mission graph invariants changed")
        required_true = {
            "immutable_node_identity", "product_host_scope_required",
            "metadata_only_labels", "parent_child_acyclic",
            "single_parent_for_non_root", "cross_host_links_forbidden",
        }
        required_false = {
            "canonical_state_in_frontend", "persistence_implemented",
            "complete_ui_implemented", "attention_engine_implemented",
            "speaker_guest_implemented",
        }
        for key in required_true:
            if self.invariants.get(key) is not True:
                raise CognitiveKernelContractError(f"invariant {key} must be true")
        for key in required_false:
            if self.invariants.get(key) is not False:
                raise CognitiveKernelContractError(f"invariant {key} must be false")
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError("mission graph policy may not be a capability ceiling")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CognitiveKernelContractError(f"could not load JSON policy: {path}") from exc
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(f"JSON policy must be an object: {path}")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CognitiveKernelContractError(f"{field} keys changed; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def _sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CognitiveKernelContractError(f"{field} must be a non-empty list")
    result = tuple(require_text(item, field, maximum=256) for item in value)
    if len(set(result)) != len(result):
        raise CognitiveKernelContractError(f"{field} may not contain duplicates")
    return result


def _contains_string(value: object, target: str) -> bool:
    if value == target:
        return True
    if isinstance(value, dict):
        return target in value or any(_contains_string(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_string(item, target) for item in value)
    return False


def load_cognitive_kernel_mission_graph_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelMissionGraphPolicy:
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[2]
    source = Path(path).resolve() if path is not None else root / "policies" / "cognitive_kernel_mission_graph_policy.json"
    payload = _read_json(source)
    _exact_keys(payload, _POLICY_KEYS, "mission graph policy")
    invariants = payload["invariants"]
    if not isinstance(invariants, dict):
        raise CognitiveKernelContractError("mission graph invariants must be an object")
    _exact_keys(invariants, _INVARIANT_KEYS, "mission graph invariants")
    if any(not isinstance(value, bool) for value in invariants.values()):
        raise CognitiveKernelContractError("mission graph invariants must be booleans")
    policy = CognitiveKernelMissionGraphPolicy(
        schema_version=int(payload["cognitive_kernel_mission_graph_policy_schema_version"]),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(payload["milestone"], "milestone", maximum=32),
        status=require_identifier(payload["status"], "status"),
        required_capabilities=_sequence(payload["required_capabilities"], "required_capabilities"),
        allowed_node_types=_sequence(payload["allowed_node_types"], "allowed_node_types"),
        allowed_node_statuses=_sequence(payload["allowed_node_statuses"], "allowed_node_statuses"),
        allowed_execution_states=_sequence(payload["allowed_execution_states"], "allowed_execution_states"),
        allowed_visibility_states=_sequence(payload["allowed_visibility_states"], "allowed_visibility_states"),
        allowed_edge_types=_sequence(payload["allowed_edge_types"], "allowed_edge_types"),
        allowed_edge_states=_sequence(payload["allowed_edge_states"], "allowed_edge_states"),
        allowed_routing_actions=_sequence(payload["allowed_routing_actions"], "allowed_routing_actions"),
        allowed_result_statuses=_sequence(payload["allowed_result_statuses"], "allowed_result_statuses"),
        allowed_traceback_actions=_sequence(payload["allowed_traceback_actions"], "allowed_traceback_actions"),
        allowed_traceback_statuses=_sequence(payload["allowed_traceback_statuses"], "allowed_traceback_statuses"),
        invariants={str(key): bool(value) for key, value in invariants.items()},
        capability_ceiling=bool(payload["capability_ceiling"]),
        digest=canonical_sha256(payload),
        source_path=source,
    )
    policy.validate()
    foundation = load_cognitive_kernel_foundation_policy(repository_root=root)
    if foundation.version != "0.1.0":
        raise CognitiveKernelContractError("unexpected foundation contract version")
    parity = _read_json(root / "policies" / "capability_parity_ledger.json")
    for capability in policy.required_capabilities:
        if not _contains_string(parity, capability):
            raise CognitiveKernelContractError(f"parity ledger is missing {capability}")
    return policy
