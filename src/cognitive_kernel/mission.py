"""Mission Graph identity, node, edge, and snapshot contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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

MISSION_NODE_TYPES = frozenset(
    {"mission", "task", "decision", "monitor", "reference", "control"}
)
NODE_STATUSES = frozenset(
    {
        "planned",
        "ready",
        "active",
        "blocked",
        "waiting",
        "completed",
        "failed",
        "cancelled",
        "archived",
    }
)
EXECUTION_STATES = frozenset(
    {
        "idle",
        "queued",
        "running",
        "paused",
        "awaiting_input",
        "verifying",
        "succeeded",
        "failed",
        "cancelled",
    }
)
VISIBILITY_STATES = frozenset(
    {"foreground", "supporting", "background", "hidden", "archived"}
)
MISSION_EDGE_TYPES = frozenset(
    {
        "parent_child",
        "depends_on",
        "blocks",
        "related",
        "derived_from",
        "supersedes",
        "result_for",
    }
)
EDGE_STATES = frozenset({"active", "superseded", "deleted"})

NODE_STATUS_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "planned": frozenset({"ready", "cancelled"}),
    "ready": frozenset({"active", "blocked", "cancelled"}),
    "active": frozenset(
        {"blocked", "waiting", "completed", "failed", "cancelled"}
    ),
    "blocked": frozenset({"ready", "active", "failed", "cancelled"}),
    "waiting": frozenset({"ready", "active", "failed", "cancelled"}),
    "completed": frozenset({"active", "archived"}),
    "failed": frozenset({"active", "archived"}),
    "cancelled": frozenset({"active", "archived"}),
    "archived": frozenset(),
}

_STATUS_EXECUTION_COMPATIBILITY: Mapping[str, frozenset[str]] = {
    "planned": frozenset({"idle"}),
    "ready": frozenset({"idle", "queued"}),
    "active": frozenset({"queued", "running", "paused", "awaiting_input", "verifying"}),
    "blocked": frozenset({"idle", "paused", "awaiting_input"}),
    "waiting": frozenset({"idle", "paused", "awaiting_input"}),
    "completed": frozenset({"succeeded"}),
    "failed": frozenset({"failed"}),
    "cancelled": frozenset({"cancelled"}),
    "archived": frozenset({"idle", "succeeded", "failed", "cancelled"}),
}


def _enum(value: object, field: str, allowed: frozenset[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


def _validate_lifecycle(retention_class: str, storage_tier: str) -> None:
    if retention_class not in RETENTION_CLASSES:
        raise CognitiveKernelContractError("retention_class is not approved")
    if storage_tier not in STORAGE_TIERS:
        raise CognitiveKernelContractError("storage_tier is not approved")
    if retention_class == "quarantine" and storage_tier != "quarantine":
        raise CognitiveKernelContractError(
            "quarantine retention must use the quarantine tier"
        )


@dataclass(frozen=True)
class Mission:
    schema_version: str
    mission_id: str
    mission_key: str
    scope: ProductHostScope
    created_at: str
    title_digest: str
    provenance: ProvenanceReference
    retention_class: str
    storage_tier: str
    deletion_lineage: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    mission_sha256: str

    @classmethod
    def create(
        cls,
        *,
        mission_key: object,
        scope: ProductHostScope,
        created_at: object,
        title_digest: object,
        provenance: ProvenanceReference,
        retention_class: object = "active_project",
        storage_tier: object = "ledger",
        deletion_lineage: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "Mission":
        scope.validate()
        provenance.validate()
        normalized_key = require_identifier(mission_key, "mission_key")
        identity_material = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "mission_key": normalized_key,
            "created_at": normalize_timestamp(created_at, "created_at"),
        }
        mission_id = f"mission-{canonical_sha256(identity_material)[:32]}"
        provisional = cls(
            schema_version=identity_material["schema_version"],
            mission_id=mission_id,
            mission_key=normalized_key,
            scope=scope,
            created_at=identity_material["created_at"],
            title_digest=require_sha256(title_digest, "title_digest"),
            provenance=provenance,
            retention_class=_enum(
                retention_class, "retention_class", RETENTION_CLASSES
            ),
            storage_tier=_enum(storage_tier, "storage_tier", STORAGE_TIERS),
            deletion_lineage=normalize_identifier_sequence(
                deletion_lineage, "deletion_lineage"
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            mission_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        mission = cls(**{**provisional.__dict__, "mission_sha256": digest})
        mission.validate()
        return mission

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "mission_key": self.mission_key,
            "created_at": self.created_at,
        }

    def material_record(self) -> dict[str, object]:
        self.scope.validate()
        self.provenance.validate()
        _validate_lifecycle(self.retention_class, self.storage_tier)
        return {
            **self.identity_record(),
            "mission_id": self.mission_id,
            "title_digest": self.title_digest,
            "provenance": self.provenance.metadata_record(),
            "retention_class": self.retention_class,
            "storage_tier": self.storage_tier,
            "deletion_lineage": list(self.deletion_lineage),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["mission_sha256"] = self.mission_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        require_identifier(self.mission_key, "mission_key")
        normalize_timestamp(self.created_at, "created_at")
        require_sha256(self.title_digest, "title_digest")
        normalize_identifier_sequence(self.deletion_lineage, "deletion_lineage")
        normalize_identifier_sequence(self.policy_bindings, "policy_bindings")
        expected_id = f"mission-{canonical_sha256(self.identity_record())[:32]}"
        if self.mission_id != expected_id:
            raise CognitiveKernelContractError("mission identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.mission_sha256, "mission_sha256") != expected_digest:
            raise CognitiveKernelContractError("mission digest mismatch")


@dataclass(frozen=True)
class MissionNode:
    schema_version: str
    node_id: str
    node_key: str
    mission_id: str
    scope: ProductHostScope
    node_type: str
    status: str
    execution_state: str
    visibility_state: str
    created_at: str
    updated_at: str
    title_digest: str
    provenance: ProvenanceReference
    retention_class: str
    storage_tier: str
    deletion_lineage: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    node_sha256: str

    @classmethod
    def create(
        cls,
        *,
        node_key: object,
        mission_id: object,
        scope: ProductHostScope,
        node_type: object,
        status: object,
        execution_state: object,
        visibility_state: object,
        created_at: object,
        updated_at: object,
        title_digest: object,
        provenance: ProvenanceReference,
        retention_class: object = "active_project",
        storage_tier: object = "ledger",
        deletion_lineage: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "MissionNode":
        scope.validate()
        provenance.validate()
        normalized_key = require_identifier(node_key, "node_key")
        normalized_mission = require_identifier(mission_id, "mission_id")
        normalized_created = normalize_timestamp(created_at, "created_at")
        identity_material = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "mission_id": normalized_mission,
            "node_key": normalized_key,
            "created_at": normalized_created,
        }
        node_id = f"mission-node-{canonical_sha256(identity_material)[:32]}"
        provisional = cls(
            schema_version=identity_material["schema_version"],
            node_id=node_id,
            node_key=normalized_key,
            mission_id=normalized_mission,
            scope=scope,
            node_type=_enum(node_type, "node_type", MISSION_NODE_TYPES),
            status=_enum(status, "status", NODE_STATUSES),
            execution_state=_enum(
                execution_state, "execution_state", EXECUTION_STATES
            ),
            visibility_state=_enum(
                visibility_state, "visibility_state", VISIBILITY_STATES
            ),
            created_at=normalized_created,
            updated_at=normalize_timestamp(updated_at, "updated_at"),
            title_digest=require_sha256(title_digest, "title_digest"),
            provenance=provenance,
            retention_class=_enum(
                retention_class, "retention_class", RETENTION_CLASSES
            ),
            storage_tier=_enum(storage_tier, "storage_tier", STORAGE_TIERS),
            deletion_lineage=normalize_identifier_sequence(
                deletion_lineage, "deletion_lineage"
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            node_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        node = cls(**{**provisional.__dict__, "node_sha256": digest})
        node.validate()
        return node

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "mission_id": self.mission_id,
            "node_key": self.node_key,
            "created_at": self.created_at,
        }

    def material_record(self) -> dict[str, object]:
        self.scope.validate()
        self.provenance.validate()
        _validate_lifecycle(self.retention_class, self.storage_tier)
        return {
            **self.identity_record(),
            "node_id": self.node_id,
            "node_type": self.node_type,
            "status": self.status,
            "execution_state": self.execution_state,
            "visibility_state": self.visibility_state,
            "updated_at": self.updated_at,
            "title_digest": self.title_digest,
            "provenance": self.provenance.metadata_record(),
            "retention_class": self.retention_class,
            "storage_tier": self.storage_tier,
            "deletion_lineage": list(self.deletion_lineage),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["node_sha256"] = self.node_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        require_identifier(self.node_key, "node_key")
        require_identifier(self.mission_id, "mission_id")
        _enum(self.node_type, "node_type", MISSION_NODE_TYPES)
        status = _enum(self.status, "status", NODE_STATUSES)
        execution = _enum(
            self.execution_state, "execution_state", EXECUTION_STATES
        )
        _enum(self.visibility_state, "visibility_state", VISIBILITY_STATES)
        if execution not in _STATUS_EXECUTION_COMPATIBILITY[status]:
            raise CognitiveKernelContractError(
                "node status and execution state are incompatible"
            )
        created = normalize_timestamp(self.created_at, "created_at")
        updated = normalize_timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise CognitiveKernelContractError("updated_at precedes created_at")
        require_sha256(self.title_digest, "title_digest")
        normalize_identifier_sequence(self.deletion_lineage, "deletion_lineage")
        normalize_identifier_sequence(self.policy_bindings, "policy_bindings")
        expected_id = f"mission-node-{canonical_sha256(self.identity_record())[:32]}"
        if self.node_id != expected_id:
            raise CognitiveKernelContractError("mission node identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.node_sha256, "node_sha256") != expected_digest:
            raise CognitiveKernelContractError("mission node digest mismatch")

    def assert_valid_successor(self, successor: "MissionNode") -> None:
        self.validate()
        successor.validate()
        if self.node_id != successor.node_id:
            raise CognitiveKernelContractError("node successor changed identity")
        if self.scope != successor.scope or self.mission_id != successor.mission_id:
            raise CognitiveKernelContractError("node successor changed scope")
        if self.node_type != successor.node_type or self.created_at != successor.created_at:
            raise CognitiveKernelContractError("node successor changed immutable fields")
        if successor.updated_at <= self.updated_at:
            raise CognitiveKernelContractError("node successor timestamp did not advance")
        if successor.status != self.status and successor.status not in NODE_STATUS_TRANSITIONS[self.status]:
            raise CognitiveKernelContractError(
                f"invalid node status transition: {self.status} -> {successor.status}"
            )


@dataclass(frozen=True)
class MissionEdge:
    schema_version: str
    edge_id: str
    edge_key: str
    mission_id: str
    scope: ProductHostScope
    source_node_id: str
    target_node_id: str
    edge_type: str
    edge_state: str
    created_at: str
    provenance: ProvenanceReference
    policy_bindings: tuple[str, ...]
    edge_sha256: str

    @classmethod
    def create(
        cls,
        *,
        edge_key: object,
        mission_id: object,
        scope: ProductHostScope,
        source_node_id: object,
        target_node_id: object,
        edge_type: object,
        edge_state: object = "active",
        created_at: object,
        provenance: ProvenanceReference,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "MissionEdge":
        scope.validate()
        provenance.validate()
        normalized_source = require_identifier(source_node_id, "source_node_id")
        normalized_target = require_identifier(target_node_id, "target_node_id")
        if normalized_source == normalized_target:
            raise CognitiveKernelContractError("mission edge may not self-reference")
        identity_material = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "mission_id": require_identifier(mission_id, "mission_id"),
            "edge_key": require_identifier(edge_key, "edge_key"),
            "created_at": normalize_timestamp(created_at, "created_at"),
        }
        edge_id = f"mission-edge-{canonical_sha256(identity_material)[:32]}"
        provisional = cls(
            schema_version=identity_material["schema_version"],
            edge_id=edge_id,
            edge_key=identity_material["edge_key"],
            mission_id=identity_material["mission_id"],
            scope=scope,
            source_node_id=normalized_source,
            target_node_id=normalized_target,
            edge_type=_enum(edge_type, "edge_type", MISSION_EDGE_TYPES),
            edge_state=_enum(edge_state, "edge_state", EDGE_STATES),
            created_at=identity_material["created_at"],
            provenance=provenance,
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            edge_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        edge = cls(**{**provisional.__dict__, "edge_sha256": digest})
        edge.validate()
        return edge

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "mission_id": self.mission_id,
            "edge_key": self.edge_key,
            "created_at": self.created_at,
        }

    def material_record(self) -> dict[str, object]:
        self.scope.validate()
        self.provenance.validate()
        return {
            **self.identity_record(),
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type,
            "edge_state": self.edge_state,
            "provenance": self.provenance.metadata_record(),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["edge_sha256"] = self.edge_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        require_identifier(self.edge_key, "edge_key")
        require_identifier(self.mission_id, "mission_id")
        source = require_identifier(self.source_node_id, "source_node_id")
        target = require_identifier(self.target_node_id, "target_node_id")
        if source == target:
            raise CognitiveKernelContractError("mission edge may not self-reference")
        _enum(self.edge_type, "edge_type", MISSION_EDGE_TYPES)
        _enum(self.edge_state, "edge_state", EDGE_STATES)
        normalize_timestamp(self.created_at, "created_at")
        normalize_identifier_sequence(self.policy_bindings, "policy_bindings")
        expected_id = f"mission-edge-{canonical_sha256(self.identity_record())[:32]}"
        if self.edge_id != expected_id:
            raise CognitiveKernelContractError("mission edge identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.edge_sha256, "edge_sha256") != expected_digest:
            raise CognitiveKernelContractError("mission edge digest mismatch")


@dataclass(frozen=True)
class MissionGraphSnapshot:
    schema_version: str
    snapshot_id: str
    mission: Mission
    root_node_id: str
    nodes: tuple[MissionNode, ...]
    edges: tuple[MissionEdge, ...]
    revision: int
    snapshot_at: str
    previous_snapshot_sha256: str | None
    snapshot_sha256: str

    @classmethod
    def create(
        cls,
        *,
        mission: Mission,
        root_node_id: object,
        nodes: tuple[MissionNode, ...] | list[MissionNode],
        edges: tuple[MissionEdge, ...] | list[MissionEdge],
        revision: object,
        snapshot_at: object,
        previous_snapshot_sha256: object | None = None,
        schema_version: object = "1.0.0",
    ) -> "MissionGraphSnapshot":
        mission.validate()
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise CognitiveKernelContractError("revision must be a positive integer")
        provisional = cls(
            schema_version=require_schema_version(schema_version),
            snapshot_id="pending",
            mission=mission,
            root_node_id=require_identifier(root_node_id, "root_node_id"),
            nodes=tuple(nodes),
            edges=tuple(edges),
            revision=revision,
            snapshot_at=normalize_timestamp(snapshot_at, "snapshot_at"),
            previous_snapshot_sha256=(
                require_sha256(previous_snapshot_sha256, "previous_snapshot_sha256")
                if previous_snapshot_sha256 is not None
                else None
            ),
            snapshot_sha256="0" * 64,
        )
        digest = canonical_sha256(provisional.material_record())
        snapshot = cls(
            **{
                **provisional.__dict__,
                "snapshot_id": f"mission-snapshot-{digest[:32]}",
                "snapshot_sha256": digest,
            }
        )
        snapshot.validate()
        return snapshot

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mission": self.mission.metadata_record(),
            "root_node_id": self.root_node_id,
            "nodes": [node.metadata_record() for node in self.nodes],
            "edges": [edge.metadata_record() for edge in self.edges],
            "revision": self.revision,
            "snapshot_at": self.snapshot_at,
            "previous_snapshot_sha256": self.previous_snapshot_sha256,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["snapshot_id"] = self.snapshot_id
        record["snapshot_sha256"] = self.snapshot_sha256
        return record

    def validate(self) -> None:
        require_schema_version(self.schema_version)
        self.mission.validate()
        if not self.nodes:
            raise CognitiveKernelContractError("mission graph requires nodes")
        if self.revision < 1:
            raise CognitiveKernelContractError("revision must be positive")
        normalize_timestamp(self.snapshot_at, "snapshot_at")
        if self.previous_snapshot_sha256 is not None:
            require_sha256(
                self.previous_snapshot_sha256, "previous_snapshot_sha256"
            )
        node_ids: set[str] = set()
        node_by_id: dict[str, MissionNode] = {}
        for node in self.nodes:
            node.validate()
            if node.node_id in node_ids:
                raise CognitiveKernelContractError("duplicate mission node identity")
            if node.scope != self.mission.scope or node.mission_id != self.mission.mission_id:
                raise CognitiveKernelContractError("mission node crossed graph scope")
            node_ids.add(node.node_id)
            node_by_id[node.node_id] = node
        root = require_identifier(self.root_node_id, "root_node_id")
        if root not in node_ids:
            raise CognitiveKernelContractError("root node is missing")
        if node_by_id[root].node_type != "mission":
            raise CognitiveKernelContractError("root node must use mission node type")
        edge_ids: set[str] = set()
        active_parent_edges: list[MissionEdge] = []
        relationships: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            edge.validate()
            if edge.edge_id in edge_ids:
                raise CognitiveKernelContractError("duplicate mission edge identity")
            if edge.scope != self.mission.scope or edge.mission_id != self.mission.mission_id:
                raise CognitiveKernelContractError("mission edge crossed graph scope")
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                raise CognitiveKernelContractError("mission edge references unknown node")
            relation = (edge.source_node_id, edge.target_node_id, edge.edge_type)
            if relation in relationships and edge.edge_state == "active":
                raise CognitiveKernelContractError("duplicate active graph relationship")
            relationships.add(relation)
            edge_ids.add(edge.edge_id)
            if edge.edge_type == "parent_child" and edge.edge_state == "active":
                active_parent_edges.append(edge)
        parent_of: dict[str, str] = {}
        children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in active_parent_edges:
            child = edge.target_node_id
            if child == root:
                raise CognitiveKernelContractError("root node may not have a parent")
            if child in parent_of:
                raise CognitiveKernelContractError("mission node has multiple parents")
            parent_of[child] = edge.source_node_id
            children[edge.source_node_id].append(child)
        for node_id in node_ids - {root}:
            if node_id not in parent_of:
                raise CognitiveKernelContractError("non-root mission node is detached")
        visiting: set[str] = set()
        visited: set[str] = set()
        def walk(node_id: str) -> None:
            if node_id in visiting:
                raise CognitiveKernelContractError("parent-child graph contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for child in children[node_id]:
                walk(child)
            visiting.remove(node_id)
            visited.add(node_id)
        walk(root)
        if visited != node_ids:
            raise CognitiveKernelContractError("mission graph is not rooted and connected")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.snapshot_sha256, "snapshot_sha256") != expected_digest:
            raise CognitiveKernelContractError("mission graph snapshot digest mismatch")
        if self.snapshot_id != f"mission-snapshot-{expected_digest[:32]}":
            raise CognitiveKernelContractError("mission graph snapshot identity mismatch")
