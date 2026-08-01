"""Result Capsule and ordered traceback contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import (
    ProductHostScope,
    ProvenanceReference,
    RETENTION_CLASSES,
    STORAGE_TIERS,
)

RESULT_STATUSES = frozenset(
    {"succeeded", "partial", "failed", "cancelled", "blocked"}
)
TRACEBACK_ACTIONS = frozenset(
    {
        "propagate_to_parent",
        "reopen_node",
        "create_followup",
        "mark_blocker",
        "resolve_conflict",
        "stop_at_mission_root",
    }
)
TRACEBACK_STATUSES = frozenset({"planned", "applied", "blocked", "cancelled"})


def _enum(value: object, field: str, allowed: frozenset[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


@dataclass(frozen=True)
class ResultCapsule:
    schema_version: str
    capsule_id: str
    result_key: str
    scope: ProductHostScope
    mission_id: str
    node_id: str
    produced_at: str
    status: str
    summary_digest: str
    output_reference_ids: tuple[str, ...]
    evidence_reference_ids: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    provenance: ProvenanceReference
    retention_class: str
    storage_tier: str
    deletion_lineage: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    capsule_sha256: str

    @classmethod
    def create(
        cls,
        *,
        result_key: object,
        scope: ProductHostScope,
        mission_id: object,
        node_id: object,
        produced_at: object,
        status: object,
        summary_digest: object,
        provenance: ProvenanceReference,
        output_reference_ids: tuple[object, ...] | list[object] = (),
        evidence_reference_ids: tuple[object, ...] | list[object] = (),
        source_event_ids: tuple[object, ...] | list[object] = (),
        retention_class: object = "active_project",
        storage_tier: object = "ledger",
        deletion_lineage: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "ResultCapsule":
        scope.validate()
        provenance.validate()
        identity_material = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "result_key": require_identifier(result_key, "result_key"),
            "mission_id": require_identifier(mission_id, "mission_id"),
            "node_id": require_identifier(node_id, "node_id"),
            "produced_at": normalize_timestamp(produced_at, "produced_at"),
        }
        capsule_id = f"result-capsule-{canonical_sha256(identity_material)[:32]}"
        provisional = cls(
            schema_version=identity_material["schema_version"],
            capsule_id=capsule_id,
            result_key=identity_material["result_key"],
            scope=scope,
            mission_id=identity_material["mission_id"],
            node_id=identity_material["node_id"],
            produced_at=identity_material["produced_at"],
            status=_enum(status, "status", RESULT_STATUSES),
            summary_digest=require_sha256(summary_digest, "summary_digest"),
            output_reference_ids=normalize_identifier_sequence(output_reference_ids, "output_reference_ids"),
            evidence_reference_ids=normalize_identifier_sequence(evidence_reference_ids, "evidence_reference_ids"),
            source_event_ids=normalize_identifier_sequence(source_event_ids, "source_event_ids"),
            provenance=provenance,
            retention_class=_enum(retention_class, "retention_class", RETENTION_CLASSES),
            storage_tier=_enum(storage_tier, "storage_tier", STORAGE_TIERS),
            deletion_lineage=normalize_identifier_sequence(deletion_lineage, "deletion_lineage"),
            policy_bindings=normalize_identifier_sequence(policy_bindings, "policy_bindings"),
            capsule_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        capsule = cls(**{**provisional.__dict__, "capsule_sha256": digest})
        capsule.validate()
        return capsule

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "result_key": self.result_key,
            "mission_id": self.mission_id,
            "node_id": self.node_id,
            "produced_at": self.produced_at,
        }

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "capsule_id": self.capsule_id,
            "status": self.status,
            "summary_digest": self.summary_digest,
            "output_reference_ids": list(self.output_reference_ids),
            "evidence_reference_ids": list(self.evidence_reference_ids),
            "source_event_ids": list(self.source_event_ids),
            "provenance": self.provenance.metadata_record(),
            "retention_class": self.retention_class,
            "storage_tier": self.storage_tier,
            "deletion_lineage": list(self.deletion_lineage),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["capsule_sha256"] = self.capsule_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        self.provenance.validate()
        require_identifier(self.result_key, "result_key")
        require_identifier(self.mission_id, "mission_id")
        require_identifier(self.node_id, "node_id")
        normalize_timestamp(self.produced_at, "produced_at")
        status = _enum(self.status, "status", RESULT_STATUSES)
        require_sha256(self.summary_digest, "summary_digest")
        normalize_identifier_sequence(self.output_reference_ids, "output_reference_ids")
        normalize_identifier_sequence(self.evidence_reference_ids, "evidence_reference_ids")
        normalize_identifier_sequence(self.source_event_ids, "source_event_ids")
        _enum(self.retention_class, "retention_class", RETENTION_CLASSES)
        _enum(self.storage_tier, "storage_tier", STORAGE_TIERS)
        normalize_identifier_sequence(self.deletion_lineage, "deletion_lineage")
        normalize_identifier_sequence(self.policy_bindings, "policy_bindings")
        if not (self.output_reference_ids or self.evidence_reference_ids or self.source_event_ids):
            raise CognitiveKernelContractError("result capsule requires lineage references")
        if status == "succeeded" and not self.output_reference_ids:
            raise CognitiveKernelContractError("successful result requires output references")
        if self.storage_tier == "deleted" and self.output_reference_ids:
            raise CognitiveKernelContractError("deleted result may not retain outputs")
        expected_id = f"result-capsule-{canonical_sha256(self.identity_record())[:32]}"
        if self.capsule_id != expected_id:
            raise CognitiveKernelContractError("result capsule identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.capsule_sha256, "capsule_sha256") != expected_digest:
            raise CognitiveKernelContractError("result capsule digest mismatch")


@dataclass(frozen=True)
class TracebackTransition:
    schema_version: str
    transition_id: str
    transition_key: str
    scope: ProductHostScope
    capsule_id: str
    mission_id: str
    sequence: int
    source_node_id: str
    target_node_id: str | None
    action: str
    created_at: str
    rationale_digest: str
    evidence_reference_ids: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    transition_sha256: str

    @classmethod
    def create(
        cls,
        *,
        transition_key: object,
        scope: ProductHostScope,
        capsule_id: object,
        mission_id: object,
        sequence: object,
        source_node_id: object,
        target_node_id: object | None,
        action: object,
        created_at: object,
        rationale_digest: object,
        evidence_reference_ids: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "TracebackTransition":
        scope.validate()
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise CognitiveKernelContractError("traceback sequence must be non-negative")
        identity_material = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "transition_key": require_identifier(transition_key, "transition_key"),
            "capsule_id": require_identifier(capsule_id, "capsule_id"),
            "sequence": sequence,
            "created_at": normalize_timestamp(created_at, "created_at"),
        }
        transition_id = f"traceback-transition-{canonical_sha256(identity_material)[:32]}"
        provisional = cls(
            schema_version=identity_material["schema_version"],
            transition_id=transition_id,
            transition_key=identity_material["transition_key"],
            scope=scope,
            capsule_id=identity_material["capsule_id"],
            mission_id=require_identifier(mission_id, "mission_id"),
            sequence=sequence,
            source_node_id=require_identifier(source_node_id, "source_node_id"),
            target_node_id=(require_identifier(target_node_id, "target_node_id") if target_node_id is not None else None),
            action=_enum(action, "action", TRACEBACK_ACTIONS),
            created_at=identity_material["created_at"],
            rationale_digest=require_sha256(rationale_digest, "rationale_digest"),
            evidence_reference_ids=normalize_identifier_sequence(evidence_reference_ids, "evidence_reference_ids"),
            policy_bindings=normalize_identifier_sequence(policy_bindings, "policy_bindings"),
            transition_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        transition = cls(**{**provisional.__dict__, "transition_sha256": digest})
        transition.validate()
        return transition

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "transition_key": self.transition_key,
            "capsule_id": self.capsule_id,
            "sequence": self.sequence,
            "created_at": self.created_at,
        }

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "transition_id": self.transition_id,
            "mission_id": self.mission_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "action": self.action,
            "rationale_digest": self.rationale_digest,
            "evidence_reference_ids": list(self.evidence_reference_ids),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["transition_sha256"] = self.transition_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.transition_key, "transition_key")
        require_identifier(self.capsule_id, "capsule_id")
        require_identifier(self.mission_id, "mission_id")
        require_identifier(self.source_node_id, "source_node_id")
        if self.target_node_id is not None:
            require_identifier(self.target_node_id, "target_node_id")
        action = _enum(self.action, "action", TRACEBACK_ACTIONS)
        if self.sequence < 0:
            raise CognitiveKernelContractError("traceback sequence must be non-negative")
        normalize_timestamp(self.created_at, "created_at")
        require_sha256(self.rationale_digest, "rationale_digest")
        normalize_identifier_sequence(self.evidence_reference_ids, "evidence_reference_ids")
        normalize_identifier_sequence(self.policy_bindings, "policy_bindings")
        if action == "stop_at_mission_root":
            if self.target_node_id is not None:
                raise CognitiveKernelContractError("stop-at-root may not have a target")
        elif self.target_node_id is None:
            raise CognitiveKernelContractError("traceback action requires a target")
        if self.target_node_id == self.source_node_id:
            raise CognitiveKernelContractError("traceback may not self-target")
        expected_id = f"traceback-transition-{canonical_sha256(self.identity_record())[:32]}"
        if self.transition_id != expected_id:
            raise CognitiveKernelContractError("traceback transition identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.transition_sha256, "transition_sha256") != expected_digest:
            raise CognitiveKernelContractError("traceback transition digest mismatch")


@dataclass(frozen=True)
class TracebackChain:
    schema_version: str
    chain_id: str
    chain_key: str
    scope: ProductHostScope
    capsule_id: str
    mission_id: str
    created_at: str
    status: str
    transitions: tuple[TracebackTransition, ...]
    chain_sha256: str

    @classmethod
    def create(
        cls,
        *,
        chain_key: object,
        scope: ProductHostScope,
        capsule_id: object,
        mission_id: object,
        created_at: object,
        status: object,
        transitions: tuple[TracebackTransition, ...] | list[TracebackTransition],
        schema_version: object = "1.0.0",
    ) -> "TracebackChain":
        scope.validate()
        identity_material = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "chain_key": require_identifier(chain_key, "chain_key"),
            "capsule_id": require_identifier(capsule_id, "capsule_id"),
            "created_at": normalize_timestamp(created_at, "created_at"),
        }
        chain_id = f"traceback-chain-{canonical_sha256(identity_material)[:32]}"
        provisional = cls(
            schema_version=identity_material["schema_version"],
            chain_id=chain_id,
            chain_key=identity_material["chain_key"],
            scope=scope,
            capsule_id=identity_material["capsule_id"],
            mission_id=require_identifier(mission_id, "mission_id"),
            created_at=identity_material["created_at"],
            status=_enum(status, "status", TRACEBACK_STATUSES),
            transitions=tuple(transitions),
            chain_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        chain = cls(**{**provisional.__dict__, "chain_sha256": digest})
        chain.validate()
        return chain

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "chain_key": self.chain_key,
            "capsule_id": self.capsule_id,
            "created_at": self.created_at,
        }

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "chain_id": self.chain_id,
            "mission_id": self.mission_id,
            "status": self.status,
            "transitions": [transition.metadata_record() for transition in self.transitions],
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["chain_sha256"] = self.chain_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.chain_key, "chain_key")
        require_identifier(self.capsule_id, "capsule_id")
        require_identifier(self.mission_id, "mission_id")
        normalize_timestamp(self.created_at, "created_at")
        _enum(self.status, "status", TRACEBACK_STATUSES)
        if not self.transitions:
            raise CognitiveKernelContractError("traceback chain requires transitions")
        seen_ids: set[str] = set()
        expected_sequence = 0
        previous_target: str | None = None
        visited_nodes: set[str] = set()
        for transition in self.transitions:
            transition.validate()
            if transition.transition_id in seen_ids:
                raise CognitiveKernelContractError("duplicate traceback transition")
            if transition.scope != self.scope or transition.capsule_id != self.capsule_id or transition.mission_id != self.mission_id:
                raise CognitiveKernelContractError("traceback transition crossed chain scope")
            if transition.sequence != expected_sequence:
                raise CognitiveKernelContractError("traceback sequence is not contiguous")
            if previous_target is not None and transition.source_node_id != previous_target:
                raise CognitiveKernelContractError("traceback chain is not linked")
            if transition.source_node_id in visited_nodes:
                raise CognitiveKernelContractError("traceback chain contains a cycle")
            visited_nodes.add(transition.source_node_id)
            seen_ids.add(transition.transition_id)
            previous_target = transition.target_node_id
            expected_sequence += 1
        for transition in self.transitions[:-1]:
            if transition.action == "stop_at_mission_root":
                raise CognitiveKernelContractError("stop-at-root must be final")
        expected_id = f"traceback-chain-{canonical_sha256(self.identity_record())[:32]}"
        if self.chain_id != expected_id:
            raise CognitiveKernelContractError("traceback chain identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.chain_sha256, "chain_sha256") != expected_digest:
            raise CognitiveKernelContractError("traceback chain digest mismatch")
