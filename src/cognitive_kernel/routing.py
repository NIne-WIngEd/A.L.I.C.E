"""Semantic routing decision contracts for Mission Graph operations."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_confidence,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import ProductHostScope, ProvenanceReference

ROUTING_ACTIONS = frozenset(
    {
        "continue_current",
        "create_child",
        "create_sibling",
        "reattach",
        "create_mission",
        "control_command",
    }
)


@dataclass(frozen=True)
class RoutingDecision:
    schema_version: str
    decision_id: str
    decision_key: str
    scope: ProductHostScope
    action: str
    decided_at: str
    current_mission_id: str | None
    current_node_id: str | None
    target_mission_id: str | None
    target_node_id: str | None
    control_command_id: str | None
    rationale_digest: str
    confidence: float
    provenance: ProvenanceReference
    policy_bindings: tuple[str, ...]
    decision_sha256: str

    @classmethod
    def create(
        cls,
        *,
        decision_key: object,
        scope: ProductHostScope,
        action: object,
        decided_at: object,
        rationale_digest: object,
        confidence: object,
        provenance: ProvenanceReference,
        current_mission_id: object | None = None,
        current_node_id: object | None = None,
        target_mission_id: object | None = None,
        target_node_id: object | None = None,
        control_command_id: object | None = None,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "RoutingDecision":
        scope.validate()
        provenance.validate()
        normalized_action = require_identifier(action, "action")
        if normalized_action not in ROUTING_ACTIONS:
            raise CognitiveKernelContractError("routing action is not approved")
        identity_material = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "decision_key": require_identifier(decision_key, "decision_key"),
            "decided_at": normalize_timestamp(decided_at, "decided_at"),
        }
        decision_id = f"routing-decision-{canonical_sha256(identity_material)[:32]}"
        normalized_confidence = require_confidence(confidence)
        if normalized_confidence is None:
            raise CognitiveKernelContractError("routing confidence is required")
        provisional = cls(
            schema_version=identity_material["schema_version"],
            decision_id=decision_id,
            decision_key=identity_material["decision_key"],
            scope=scope,
            action=normalized_action,
            decided_at=identity_material["decided_at"],
            current_mission_id=(require_identifier(current_mission_id, "current_mission_id") if current_mission_id is not None else None),
            current_node_id=(require_identifier(current_node_id, "current_node_id") if current_node_id is not None else None),
            target_mission_id=(require_identifier(target_mission_id, "target_mission_id") if target_mission_id is not None else None),
            target_node_id=(require_identifier(target_node_id, "target_node_id") if target_node_id is not None else None),
            control_command_id=(require_identifier(control_command_id, "control_command_id") if control_command_id is not None else None),
            rationale_digest=require_sha256(rationale_digest, "rationale_digest"),
            confidence=normalized_confidence,
            provenance=provenance,
            policy_bindings=normalize_identifier_sequence(policy_bindings, "policy_bindings"),
            decision_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        decision = cls(**{**provisional.__dict__, "decision_sha256": digest})
        decision.validate()
        return decision

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "decision_key": self.decision_key,
            "scope": self.scope.metadata_record(),
            "action": self.action,
            "decided_at": self.decided_at,
            "current_mission_id": self.current_mission_id,
            "current_node_id": self.current_node_id,
            "target_mission_id": self.target_mission_id,
            "target_node_id": self.target_node_id,
            "control_command_id": self.control_command_id,
            "rationale_digest": self.rationale_digest,
            "confidence": self.confidence,
            "provenance": self.provenance.metadata_record(),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["decision_sha256"] = self.decision_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        self.provenance.validate()
        require_identifier(self.decision_key, "decision_key")
        if require_identifier(self.action, "action") not in ROUTING_ACTIONS:
            raise CognitiveKernelContractError("routing action is not approved")
        normalize_timestamp(self.decided_at, "decided_at")
        require_sha256(self.rationale_digest, "rationale_digest")
        if require_confidence(self.confidence) is None:
            raise CognitiveKernelContractError("routing confidence is required")
        normalize_identifier_sequence(self.policy_bindings, "policy_bindings")
        current_pair = self.current_mission_id is not None and self.current_node_id is not None
        target_pair = self.target_mission_id is not None and self.target_node_id is not None
        if (self.current_mission_id is None) != (self.current_node_id is None):
            raise CognitiveKernelContractError("current mission and node must be paired")
        if (self.target_mission_id is None) != (self.target_node_id is None):
            raise CognitiveKernelContractError("target mission and node must be paired")
        if self.action == "continue_current":
            if not current_pair or target_pair or self.control_command_id is not None:
                raise CognitiveKernelContractError("continue_current fields are invalid")
        elif self.action in {"create_child", "create_sibling"}:
            if not current_pair or not target_pair or self.control_command_id is not None:
                raise CognitiveKernelContractError(f"{self.action} fields are invalid")
        elif self.action == "reattach":
            if not current_pair or not target_pair or self.control_command_id is not None:
                raise CognitiveKernelContractError("reattach fields are invalid")
            if self.current_node_id == self.target_node_id:
                raise CognitiveKernelContractError("reattach target must differ")
        elif self.action == "create_mission":
            if target_pair or self.control_command_id is not None:
                raise CognitiveKernelContractError("create_mission fields are invalid")
        elif self.action == "control_command":
            if self.control_command_id is None or target_pair:
                raise CognitiveKernelContractError("control_command fields are invalid")
        expected_id = f"routing-decision-{canonical_sha256({'schema_version': self.schema_version, 'scope': self.scope.metadata_record(), 'decision_key': self.decision_key, 'decided_at': self.decided_at})[:32]}"
        if self.decision_id != expected_id:
            raise CognitiveKernelContractError("routing decision identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.decision_sha256, "decision_sha256") != expected_digest:
            raise CognitiveKernelContractError("routing decision digest mismatch")
