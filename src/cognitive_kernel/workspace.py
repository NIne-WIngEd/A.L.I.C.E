"""Deterministic, metadata-only Cognitive Workspace projection contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .attention import AttentionDecision
from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import ProductHostScope

WORKSPACE_ITEM_TYPES = frozenset(
    {
        "mission_node",
        "result_capsule",
        "permission_state",
        "trust_state",
        "monitor",
    }
)
WORKSPACE_ROLES = frozenset(
    {"primary", "secondary", "pane", "compact_card", "background_panel"}
)
WORKSPACE_LAYOUT_MODES = frozenset(
    {
        "empty",
        "full_workspace",
        "focus_support_split",
        "primary_two_secondary",
        "adaptive_grid",
        "panels_live_cards",
        "command_center",
        "highest_value_set",
    }
)
WORKSPACE_AUDIENCES = frozenset({"host", "non_host"})
WORKSPACE_PRIVACY_CLASSES = frozenset(
    {"ordinary", "sensitive", "restricted"}
)
WORKSPACE_REDACTION_STATES = frozenset(
    {"none", "metadata_only", "title_hidden"}
)
WORKSPACE_PROJECTION_STATES = frozenset(
    {"active", "superseded", "expired"}
)


def _enum(value: object, field: str, allowed: Iterable[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_integer(value: object, field: str, *, maximum: int | None = None) -> int:
    normalized = _nonnegative_integer(value, field)
    if normalized < 1:
        raise CognitiveKernelContractError(f"{field} must be positive")
    if maximum is not None and normalized > maximum:
        raise CognitiveKernelContractError(f"{field} exceeds its maximum")
    return normalized


def _optional_identifier(value: object | None, field: str) -> str | None:
    return require_identifier(value, field) if value is not None else None


def default_workspace_layout_mode(
    *,
    visible_count: object,
    total_candidate_count: object,
) -> str:
    """Return the ratified generic composition for the visible work count."""

    visible = _nonnegative_integer(visible_count, "visible_count")
    total = _nonnegative_integer(
        total_candidate_count, "total_candidate_count"
    )
    if visible > total:
        raise CognitiveKernelContractError(
            "visible_count may not exceed total_candidate_count"
        )
    if visible > 10:
        raise CognitiveKernelContractError(
            "workspace projections may expose at most ten items"
        )
    if visible == 0:
        return "empty"
    if total > 10:
        return "highest_value_set"
    if visible == 1:
        return "full_workspace"
    if visible == 2:
        return "focus_support_split"
    if visible == 3:
        return "primary_two_secondary"
    if visible == 4:
        return "adaptive_grid"
    if visible <= 6:
        return "panels_live_cards"
    return "command_center"


@dataclass(frozen=True)
class WorkspaceItemProjection:
    """One sanitized projection over canonical kernel state."""

    schema_version: str
    item_id: str
    item_key: str
    scope: ProductHostScope
    reference_id: str
    item_type: str
    mission_id: str | None
    node_id: str | None
    capsule_id: str | None
    attention_entry_id: str
    attention_rank: int
    role: str
    privacy_class: str
    redaction_state: str
    state_digest: str
    projected_metadata_digest: str
    policy_bindings: tuple[str, ...]
    item_sha256: str

    @classmethod
    def create(
        cls,
        *,
        item_key: object,
        scope: ProductHostScope,
        reference_id: object,
        item_type: object,
        attention_entry_id: object,
        attention_rank: object,
        role: object,
        privacy_class: object,
        redaction_state: object,
        state_digest: object,
        projected_metadata_digest: object,
        mission_id: object | None = None,
        node_id: object | None = None,
        capsule_id: object | None = None,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "WorkspaceItemProjection":
        scope.validate()
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "item_key": require_identifier(item_key, "item_key"),
            "reference_id": require_identifier(reference_id, "reference_id"),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            item_id=f"workspace-item-{canonical_sha256(identity)[:32]}",
            item_key=identity["item_key"],
            scope=scope,
            reference_id=identity["reference_id"],
            item_type=_enum(item_type, "item_type", WORKSPACE_ITEM_TYPES),
            mission_id=_optional_identifier(mission_id, "mission_id"),
            node_id=_optional_identifier(node_id, "node_id"),
            capsule_id=_optional_identifier(capsule_id, "capsule_id"),
            attention_entry_id=require_identifier(
                attention_entry_id, "attention_entry_id"
            ),
            attention_rank=_positive_integer(
                attention_rank, "attention_rank"
            ),
            role=_enum(role, "role", WORKSPACE_ROLES),
            privacy_class=_enum(
                privacy_class,
                "privacy_class",
                WORKSPACE_PRIVACY_CLASSES,
            ),
            redaction_state=_enum(
                redaction_state,
                "redaction_state",
                WORKSPACE_REDACTION_STATES,
            ),
            state_digest=require_sha256(state_digest, "state_digest"),
            projected_metadata_digest=require_sha256(
                projected_metadata_digest,
                "projected_metadata_digest",
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            item_sha256="0" * 64,
        )
        provisional._validate_material()
        digest = canonical_sha256(provisional.material_record())
        item = cls(**{**provisional.__dict__, "item_sha256": digest})
        item.validate()
        return item

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "item_key": self.item_key,
            "reference_id": self.reference_id,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.item_key, "item_key")
        require_identifier(self.reference_id, "reference_id")
        item_type = _enum(
            self.item_type, "item_type", WORKSPACE_ITEM_TYPES
        )
        for field, value in (
            ("mission_id", self.mission_id),
            ("node_id", self.node_id),
            ("capsule_id", self.capsule_id),
        ):
            if value is not None:
                require_identifier(value, field)
        require_identifier(
            self.attention_entry_id, "attention_entry_id"
        )
        _positive_integer(self.attention_rank, "attention_rank")
        _enum(self.role, "role", WORKSPACE_ROLES)
        _enum(
            self.privacy_class,
            "privacy_class",
            WORKSPACE_PRIVACY_CLASSES,
        )
        _enum(
            self.redaction_state,
            "redaction_state",
            WORKSPACE_REDACTION_STATES,
        )
        require_sha256(self.state_digest, "state_digest")
        require_sha256(
            self.projected_metadata_digest,
            "projected_metadata_digest",
        )
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )
        if item_type == "mission_node":
            if self.mission_id is None or self.node_id is None:
                raise CognitiveKernelContractError(
                    "mission_node projections require mission_id and node_id"
                )
            if self.capsule_id is not None:
                raise CognitiveKernelContractError(
                    "mission_node projections may not carry capsule_id"
                )
        elif item_type == "result_capsule":
            if self.mission_id is None or self.capsule_id is None:
                raise CognitiveKernelContractError(
                    "result_capsule projections require mission_id and capsule_id"
                )
        elif self.capsule_id is not None:
            raise CognitiveKernelContractError(
                "only result_capsule projections may carry capsule_id"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            **self.identity_record(),
            "item_id": self.item_id,
            "item_type": self.item_type,
            "mission_id": self.mission_id,
            "node_id": self.node_id,
            "capsule_id": self.capsule_id,
            "attention_entry_id": self.attention_entry_id,
            "attention_rank": self.attention_rank,
            "role": self.role,
            "privacy_class": self.privacy_class,
            "redaction_state": self.redaction_state,
            "state_digest": self.state_digest,
            "projected_metadata_digest": self.projected_metadata_digest,
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["item_sha256"] = self.item_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"workspace-item-{canonical_sha256(self.identity_record())[:32]}"
        if self.item_id != expected_id:
            raise CognitiveKernelContractError(
                "workspace item identity mismatch"
            )
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.item_sha256, "item_sha256") != expected_digest:
            raise CognitiveKernelContractError(
                "workspace item digest mismatch"
            )


@dataclass(frozen=True)
class WorkspaceLayout:
    """Deterministic adaptive composition metadata with no empty slots."""

    schema_version: str
    layout_id: str
    layout_key: str
    scope: ProductHostScope
    created_at: str
    layout_mode: str
    visible_count: int
    total_candidate_count: int
    max_visible: int
    layout_locked: bool
    stability_anchor_digest: str
    item_order: tuple[str, ...]
    omitted_reference_digests: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    layout_sha256: str

    @classmethod
    def create(
        cls,
        *,
        layout_key: object,
        scope: ProductHostScope,
        created_at: object,
        visible_count: object,
        total_candidate_count: object,
        max_visible: object,
        layout_locked: object,
        stability_anchor_digest: object,
        item_order: tuple[object, ...] | list[object],
        omitted_reference_digests: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "WorkspaceLayout":
        scope.validate()
        if not isinstance(layout_locked, bool):
            raise CognitiveKernelContractError(
                "layout_locked must be boolean"
            )
        normalized_visible = _nonnegative_integer(
            visible_count, "visible_count"
        )
        normalized_total = _nonnegative_integer(
            total_candidate_count, "total_candidate_count"
        )
        mode = default_workspace_layout_mode(
            visible_count=normalized_visible,
            total_candidate_count=normalized_total,
        )
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "layout_key": require_identifier(layout_key, "layout_key"),
            "created_at": normalize_timestamp(created_at, "created_at"),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            layout_id=f"workspace-layout-{canonical_sha256(identity)[:32]}",
            layout_key=identity["layout_key"],
            scope=scope,
            created_at=identity["created_at"],
            layout_mode=mode,
            visible_count=normalized_visible,
            total_candidate_count=normalized_total,
            max_visible=_positive_integer(
                max_visible, "max_visible", maximum=10
            ),
            layout_locked=layout_locked,
            stability_anchor_digest=require_sha256(
                stability_anchor_digest,
                "stability_anchor_digest",
            ),
            item_order=normalize_identifier_sequence(
                item_order, "item_order"
            ),
            omitted_reference_digests=tuple(
                require_sha256(value, "omitted_reference_digests")
                for value in omitted_reference_digests
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            layout_sha256="0" * 64,
        )
        provisional._validate_material()
        digest = canonical_sha256(provisional.material_record())
        layout = cls(**{**provisional.__dict__, "layout_sha256": digest})
        layout.validate()
        return layout

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "layout_key": self.layout_key,
            "created_at": self.created_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.layout_key, "layout_key")
        normalize_timestamp(self.created_at, "created_at")
        _enum(
            self.layout_mode,
            "layout_mode",
            WORKSPACE_LAYOUT_MODES,
        )
        visible = _nonnegative_integer(
            self.visible_count, "visible_count"
        )
        total = _nonnegative_integer(
            self.total_candidate_count,
            "total_candidate_count",
        )
        maximum = _positive_integer(
            self.max_visible, "max_visible", maximum=10
        )
        if visible > total:
            raise CognitiveKernelContractError(
                "visible_count may not exceed total_candidate_count"
            )
        if visible > maximum:
            raise CognitiveKernelContractError(
                "visible_count may not exceed max_visible"
            )
        if not isinstance(self.layout_locked, bool):
            raise CognitiveKernelContractError(
                "layout_locked must be boolean"
            )
        require_sha256(
            self.stability_anchor_digest,
            "stability_anchor_digest",
        )
        normalized_order = normalize_identifier_sequence(
            self.item_order, "item_order"
        )
        if normalized_order != self.item_order:
            raise CognitiveKernelContractError(
                "item_order is not canonical"
            )
        for item_id in self.item_order:
            if "placeholder" in item_id or item_id == "empty":
                raise CognitiveKernelContractError(
                    "workspace layouts may not contain empty placeholders"
                )
        if len(self.item_order) != visible:
            raise CognitiveKernelContractError(
                "item_order must contain exactly visible_count items"
            )
        omitted = tuple(
            require_sha256(value, "omitted_reference_digests")
            for value in self.omitted_reference_digests
        )
        if omitted != self.omitted_reference_digests:
            raise CognitiveKernelContractError(
                "omitted_reference_digests are not canonical"
            )
        if len(set(omitted)) != len(omitted):
            raise CognitiveKernelContractError(
                "omitted_reference_digests may not contain duplicates"
            )
        if len(omitted) != total - visible:
            raise CognitiveKernelContractError(
                "omitted-reference count must equal total minus visible"
            )
        expected_mode = default_workspace_layout_mode(
            visible_count=visible,
            total_candidate_count=total,
        )
        if self.layout_mode != expected_mode:
            raise CognitiveKernelContractError(
                "workspace layout mode does not match composition contract"
            )
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            **self.identity_record(),
            "layout_id": self.layout_id,
            "layout_mode": self.layout_mode,
            "visible_count": self.visible_count,
            "total_candidate_count": self.total_candidate_count,
            "max_visible": self.max_visible,
            "layout_locked": self.layout_locked,
            "stability_anchor_digest": self.stability_anchor_digest,
            "item_order": list(self.item_order),
            "omitted_reference_digests": list(
                self.omitted_reference_digests
            ),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["layout_sha256"] = self.layout_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"workspace-layout-{canonical_sha256(self.identity_record())[:32]}"
        if self.layout_id != expected_id:
            raise CognitiveKernelContractError(
                "workspace layout identity mismatch"
            )
        expected_digest = canonical_sha256(self.material_record())
        if (
            require_sha256(self.layout_sha256, "layout_sha256")
            != expected_digest
        ):
            raise CognitiveKernelContractError(
                "workspace layout digest mismatch"
            )


@dataclass(frozen=True)
class WorkspaceProjection:
    """A product-neutral frontend projection over canonical kernel state."""

    schema_version: str
    projection_id: str
    projection_key: str
    scope: ProductHostScope
    attention_decision_id: str
    attention_decision_sha256: str
    projected_at: str
    audience: str
    mission_filter_id: str | None
    layout: WorkspaceLayout
    items: tuple[WorkspaceItemProjection, ...]
    state: str
    policy_bindings: tuple[str, ...]
    projection_sha256: str

    @classmethod
    def create(
        cls,
        *,
        projection_key: object,
        scope: ProductHostScope,
        attention_decision: AttentionDecision,
        projected_at: object,
        audience: object,
        layout: WorkspaceLayout,
        items: tuple[WorkspaceItemProjection, ...] | list[WorkspaceItemProjection],
        state: object = "active",
        mission_filter_id: object | None = None,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "WorkspaceProjection":
        scope.validate()
        attention_decision.validate()
        layout.validate()
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "projection_key": require_identifier(
                projection_key, "projection_key"
            ),
            "projected_at": normalize_timestamp(
                projected_at, "projected_at"
            ),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            projection_id=f"workspace-projection-{canonical_sha256(identity)[:32]}",
            projection_key=identity["projection_key"],
            scope=scope,
            attention_decision_id=attention_decision.decision_id,
            attention_decision_sha256=attention_decision.decision_sha256,
            projected_at=identity["projected_at"],
            audience=_enum(
                audience, "audience", WORKSPACE_AUDIENCES
            ),
            mission_filter_id=_optional_identifier(
                mission_filter_id, "mission_filter_id"
            ),
            layout=layout,
            items=tuple(items),
            state=_enum(
                state, "state", WORKSPACE_PROJECTION_STATES
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            projection_sha256="0" * 64,
        )
        provisional._validate_material(attention_decision)
        digest = canonical_sha256(provisional.material_record())
        projection = cls(
            **{**provisional.__dict__, "projection_sha256": digest}
        )
        projection.validate(attention_decision=attention_decision)
        return projection

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "projection_key": self.projection_key,
            "projected_at": self.projected_at,
        }

    def _validate_material(
        self,
        attention_decision: AttentionDecision | None = None,
    ) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.projection_key, "projection_key")
        require_identifier(
            self.attention_decision_id, "attention_decision_id"
        )
        require_sha256(
            self.attention_decision_sha256,
            "attention_decision_sha256",
        )
        normalize_timestamp(self.projected_at, "projected_at")
        audience = _enum(
            self.audience, "audience", WORKSPACE_AUDIENCES
        )
        if self.mission_filter_id is not None:
            require_identifier(
                self.mission_filter_id, "mission_filter_id"
            )
        _enum(
            self.state,
            "state",
            WORKSPACE_PROJECTION_STATES,
        )
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )
        self.layout.validate()
        if self.layout.scope != self.scope:
            raise CognitiveKernelContractError(
                "cross-host workspace layouts are forbidden"
            )

        item_ids: set[str] = set()
        reference_ids: set[str] = set()
        attention_entry_ids: set[str] = set()
        primary_count = 0
        for item in self.items:
            item.validate()
            if item.scope != self.scope:
                raise CognitiveKernelContractError(
                    "cross-host workspace items are forbidden"
                )
            if item.item_id in item_ids or item.reference_id in reference_ids:
                raise CognitiveKernelContractError(
                    "workspace projection items must be unique"
                )
            if item.attention_entry_id in attention_entry_ids:
                raise CognitiveKernelContractError(
                    "one attention entry may project at most once"
                )
            item_ids.add(item.item_id)
            reference_ids.add(item.reference_id)
            attention_entry_ids.add(item.attention_entry_id)
            if item.role == "primary":
                primary_count += 1
            if audience == "non_host":
                if item.privacy_class == "restricted":
                    raise CognitiveKernelContractError(
                        "restricted items may not enter non-host projections"
                    )
                if (
                    item.privacy_class == "sensitive"
                    and item.redaction_state == "none"
                ):
                    raise CognitiveKernelContractError(
                        "sensitive non-host projections require redaction"
                    )

        if self.items and primary_count != 1:
            raise CognitiveKernelContractError(
                "non-empty workspace projections require exactly one primary item"
            )
        if not self.items and primary_count != 0:
            raise CognitiveKernelContractError(
                "empty workspace projections may not contain a primary item"
            )
        if tuple(item.item_id for item in self.items) != self.layout.item_order:
            raise CognitiveKernelContractError(
                "workspace item order must match the layout contract"
            )
        if len(self.items) != self.layout.visible_count:
            raise CognitiveKernelContractError(
                "workspace item count must match layout visible_count"
            )

        if attention_decision is not None:
            attention_decision.validate()
            if attention_decision.scope != self.scope:
                raise CognitiveKernelContractError(
                    "cross-host attention decisions are forbidden"
                )
            if attention_decision.decision_id != self.attention_decision_id:
                raise CognitiveKernelContractError(
                    "workspace projection references the wrong attention decision"
                )
            if (
                attention_decision.decision_sha256
                != self.attention_decision_sha256
            ):
                raise CognitiveKernelContractError(
                    "workspace projection attention digest mismatch"
                )
            selected = {
                entry.entry_id: entry
                for entry in attention_decision.selected_entries()
            }
            for item in self.items:
                entry = selected.get(item.attention_entry_id)
                if entry is None:
                    raise CognitiveKernelContractError(
                        "workspace items must come from selected attention entries"
                    )
                if item.reference_id != entry.reference_id:
                    raise CognitiveKernelContractError(
                        "workspace item reference differs from attention entry"
                    )
                if item.attention_rank != entry.rank:
                    raise CognitiveKernelContractError(
                        "workspace item rank differs from attention entry"
                    )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            **self.identity_record(),
            "projection_id": self.projection_id,
            "attention_decision_id": self.attention_decision_id,
            "attention_decision_sha256": self.attention_decision_sha256,
            "audience": self.audience,
            "mission_filter_id": self.mission_filter_id,
            "layout": self.layout.metadata_record(),
            "items": [item.metadata_record() for item in self.items],
            "state": self.state,
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["projection_sha256"] = self.projection_sha256
        return record

    def validate(
        self,
        *,
        attention_decision: AttentionDecision | None = None,
    ) -> None:
        self._validate_material(attention_decision)
        expected_id = f"workspace-projection-{canonical_sha256(self.identity_record())[:32]}"
        if self.projection_id != expected_id:
            raise CognitiveKernelContractError(
                "workspace projection identity mismatch"
            )
        expected_digest = canonical_sha256(self.material_record())
        if (
            require_sha256(
                self.projection_sha256,
                "projection_sha256",
            )
            != expected_digest
        ):
            raise CognitiveKernelContractError(
                "workspace projection digest mismatch"
            )
