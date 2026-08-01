"""Explainable attention and explicit host-workspace override contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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

ATTENTION_SUBJECT_TYPES = frozenset(
    {
        "mission_node",
        "result_capsule",
        "security_interrupt",
        "authority_request",
        "monitor",
    }
)
ATTENTION_PRIORITY_CLASSES = (
    "host_decision_required",
    "protected_interrupt",
    "critical_path_blocker",
    "imminent_deadline",
    "result_ready",
    "host_engaged",
    "significant_state_change",
    "active_execution",
    "recent_support",
    "monitor_update",
    "low_attention",
)
PROTECTED_INTERRUPT_REASONS = frozenset(
    {
        "device_or_data_threat",
        "uncertain_identity_privileged_action",
        "destructive_action_authorization",
        "security_breach",
        "failed_migration",
        "imminent_deadline_intervention",
        "active_mission_integrity_conflict",
    }
)
ATTENTION_HOST_OVERRIDES = frozenset(
    {"none", "pinned", "foreground", "background", "suppressed"}
)
INTERRUPTION_PREFERENCES = frozenset(
    {"allow", "minimize", "focus_only"}
)
FOCUS_MODES = frozenset(
    {"automatic", "mission_focus", "quiet", "security_only"}
)
FORBIDDEN_ATTENTION_REASON_CODES = frozenset(
    {
        "advertising",
        "ad_priority",
        "engagement_maximization",
        "commercial_prioritization",
        "sponsored_priority",
    }
)
HOST_OVERRIDE_COMMANDS = frozenset(
    {
        "pin",
        "unpin",
        "foreground",
        "background",
        "set_visibility_limit",
        "lock_layout",
        "unlock_layout",
        "set_focus_mission",
        "clear_focus",
        "set_interruption_preference",
        "keep_security_visible",
        "restore_automatic_layout",
    }
)
HOST_OVERRIDE_STATUSES = frozenset(
    {"requested", "applied", "rejected", "revoked", "expired"}
)

_PRIORITY_INDEX = {
    name: index for index, name in enumerate(ATTENTION_PRIORITY_CLASSES)
}
_FORBIDDEN_REASON_FRAGMENTS = (
    "advertis",
    "engagement",
    "commercial",
    "sponsor",
)


def _enum(value: object, field: str, allowed: Iterable[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


def _positive_integer(value: object, field: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CognitiveKernelContractError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise CognitiveKernelContractError(f"{field} exceeds its maximum")
    return value


def _optional_identifier(value: object | None, field: str) -> str | None:
    return require_identifier(value, field) if value is not None else None


def _validate_reason_codes(values: tuple[str, ...]) -> None:
    normalized = normalize_identifier_sequence(values, "reason_codes")
    if normalized != values or not normalized:
        raise CognitiveKernelContractError("reason_codes must be canonical and non-empty")
    for reason in normalized:
        if reason in FORBIDDEN_ATTENTION_REASON_CODES or any(
            fragment in reason for fragment in _FORBIDDEN_REASON_FRAGMENTS
        ):
            raise CognitiveKernelContractError(
                "commercial or engagement-manipulation attention reasons are forbidden"
            )


@dataclass(frozen=True)
class AttentionRankEntry:
    """One metadata-only, explainable ranked attention candidate."""

    schema_version: str
    entry_id: str
    entry_key: str
    scope: ProductHostScope
    reference_id: str
    subject_type: str
    mission_id: str | None
    node_id: str | None
    capsule_id: str | None
    observed_at: str
    state_digest: str
    priority_class: str
    rank: int
    score: float
    interruption_cost: float
    protected_interrupt_reason: str | None
    host_override: str
    selected: bool
    reason_codes: tuple[str, ...]
    suppression_reason: str | None
    policy_bindings: tuple[str, ...]
    entry_sha256: str

    @classmethod
    def create(
        cls,
        *,
        entry_key: object,
        scope: ProductHostScope,
        reference_id: object,
        subject_type: object,
        observed_at: object,
        state_digest: object,
        priority_class: object,
        rank: object,
        score: object,
        interruption_cost: object,
        reason_codes: tuple[object, ...] | list[object],
        selected: object,
        mission_id: object | None = None,
        node_id: object | None = None,
        capsule_id: object | None = None,
        protected_interrupt_reason: object | None = None,
        host_override: object = "none",
        suppression_reason: object | None = None,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "AttentionRankEntry":
        scope.validate()
        normalized_score = require_confidence(score, "score")
        normalized_cost = require_confidence(
            interruption_cost, "interruption_cost"
        )
        if normalized_score is None or normalized_cost is None:
            raise CognitiveKernelContractError(
                "score and interruption_cost are required"
            )
        if not isinstance(selected, bool):
            raise CognitiveKernelContractError("selected must be boolean")
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "entry_key": require_identifier(entry_key, "entry_key"),
            "observed_at": normalize_timestamp(observed_at, "observed_at"),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            entry_id=f"attention-entry-{canonical_sha256(identity)[:32]}",
            entry_key=identity["entry_key"],
            scope=scope,
            reference_id=require_identifier(reference_id, "reference_id"),
            subject_type=_enum(
                subject_type, "subject_type", ATTENTION_SUBJECT_TYPES
            ),
            mission_id=_optional_identifier(mission_id, "mission_id"),
            node_id=_optional_identifier(node_id, "node_id"),
            capsule_id=_optional_identifier(capsule_id, "capsule_id"),
            observed_at=identity["observed_at"],
            state_digest=require_sha256(state_digest, "state_digest"),
            priority_class=_enum(
                priority_class,
                "priority_class",
                ATTENTION_PRIORITY_CLASSES,
            ),
            rank=_positive_integer(rank, "rank"),
            score=normalized_score,
            interruption_cost=normalized_cost,
            protected_interrupt_reason=(
                _enum(
                    protected_interrupt_reason,
                    "protected_interrupt_reason",
                    PROTECTED_INTERRUPT_REASONS,
                )
                if protected_interrupt_reason is not None
                else None
            ),
            host_override=_enum(
                host_override,
                "host_override",
                ATTENTION_HOST_OVERRIDES,
            ),
            selected=selected,
            reason_codes=normalize_identifier_sequence(
                reason_codes, "reason_codes"
            ),
            suppression_reason=_optional_identifier(
                suppression_reason, "suppression_reason"
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            entry_sha256="0" * 64,
        )
        provisional._validate_material()
        digest = canonical_sha256(provisional.material_record())
        entry = cls(**{**provisional.__dict__, "entry_sha256": digest})
        entry.validate()
        return entry

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "entry_key": self.entry_key,
            "observed_at": self.observed_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.entry_key, "entry_key")
        require_identifier(self.reference_id, "reference_id")
        _enum(self.subject_type, "subject_type", ATTENTION_SUBJECT_TYPES)
        normalize_timestamp(self.observed_at, "observed_at")
        require_sha256(self.state_digest, "state_digest")
        _enum(
            self.priority_class,
            "priority_class",
            ATTENTION_PRIORITY_CLASSES,
        )
        _positive_integer(self.rank, "rank")
        if require_confidence(self.score, "score") != self.score:
            raise CognitiveKernelContractError("score is not canonical")
        if (
            require_confidence(
                self.interruption_cost, "interruption_cost"
            )
            != self.interruption_cost
        ):
            raise CognitiveKernelContractError(
                "interruption_cost is not canonical"
            )
        if not isinstance(self.selected, bool):
            raise CognitiveKernelContractError("selected must be boolean")
        _enum(
            self.host_override,
            "host_override",
            ATTENTION_HOST_OVERRIDES,
        )
        _validate_reason_codes(self.reason_codes)
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )
        for field, value in (
            ("mission_id", self.mission_id),
            ("node_id", self.node_id),
            ("capsule_id", self.capsule_id),
            ("suppression_reason", self.suppression_reason),
        ):
            if value is not None:
                require_identifier(value, field)

        if self.subject_type == "mission_node":
            if self.mission_id is None or self.node_id is None:
                raise CognitiveKernelContractError(
                    "mission_node attention entries require mission_id and node_id"
                )
            if self.capsule_id is not None:
                raise CognitiveKernelContractError(
                    "mission_node attention entries may not carry capsule_id"
                )
        elif self.subject_type == "result_capsule":
            if self.mission_id is None or self.capsule_id is None:
                raise CognitiveKernelContractError(
                    "result_capsule entries require mission_id and capsule_id"
                )
        elif self.capsule_id is not None:
            raise CognitiveKernelContractError(
                "only result_capsule entries may carry capsule_id"
            )

        if self.priority_class == "protected_interrupt":
            if self.protected_interrupt_reason is None:
                raise CognitiveKernelContractError(
                    "protected interrupts require an approved reason"
                )
        elif self.protected_interrupt_reason is not None:
            raise CognitiveKernelContractError(
                "protected_interrupt_reason requires protected_interrupt priority"
            )

        if self.protected_interrupt_reason is not None:
            _enum(
                self.protected_interrupt_reason,
                "protected_interrupt_reason",
                PROTECTED_INTERRUPT_REASONS,
            )
            if not self.selected:
                raise CognitiveKernelContractError(
                    "protected interrupts cannot be suppressed"
                )
            if self.host_override in {"background", "suppressed"}:
                raise CognitiveKernelContractError(
                    "host overrides cannot hide protected interrupts"
                )

        if self.host_override in {"pinned", "foreground"} and not self.selected:
            raise CognitiveKernelContractError(
                "pinned or foreground entries must be selected"
            )
        if self.host_override in {"background", "suppressed"} and self.selected:
            raise CognitiveKernelContractError(
                "background or suppressed entries may not be selected"
            )
        if self.selected and self.suppression_reason is not None:
            raise CognitiveKernelContractError(
                "selected entries may not carry suppression_reason"
            )
        if not self.selected and self.suppression_reason is None:
            raise CognitiveKernelContractError(
                "unselected entries require suppression_reason"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            **self.identity_record(),
            "entry_id": self.entry_id,
            "reference_id": self.reference_id,
            "subject_type": self.subject_type,
            "mission_id": self.mission_id,
            "node_id": self.node_id,
            "capsule_id": self.capsule_id,
            "state_digest": self.state_digest,
            "priority_class": self.priority_class,
            "priority_order": _PRIORITY_INDEX[self.priority_class],
            "rank": self.rank,
            "score": self.score,
            "interruption_cost": self.interruption_cost,
            "protected_interrupt_reason": self.protected_interrupt_reason,
            "host_override": self.host_override,
            "selected": self.selected,
            "reason_codes": list(self.reason_codes),
            "suppression_reason": self.suppression_reason,
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["entry_sha256"] = self.entry_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"attention-entry-{canonical_sha256(self.identity_record())[:32]}"
        if self.entry_id != expected_id:
            raise CognitiveKernelContractError("attention entry identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if require_sha256(self.entry_sha256, "entry_sha256") != expected_digest:
            raise CognitiveKernelContractError("attention entry digest mismatch")


@dataclass(frozen=True)
class AttentionDecision:
    """One explainable ranking and selection receipt."""

    schema_version: str
    decision_id: str
    decision_key: str
    scope: ProductHostScope
    decided_at: str
    visibility_limit: int
    interruption_preference: str
    focus_mode: str
    layout_stability_weight: float
    entries: tuple[AttentionRankEntry, ...]
    candidate_snapshot_sha256: str
    provenance: ProvenanceReference
    policy_bindings: tuple[str, ...]
    decision_sha256: str

    @classmethod
    def create(
        cls,
        *,
        decision_key: object,
        scope: ProductHostScope,
        decided_at: object,
        visibility_limit: object,
        interruption_preference: object,
        focus_mode: object,
        layout_stability_weight: object,
        entries: tuple[AttentionRankEntry, ...] | list[AttentionRankEntry],
        provenance: ProvenanceReference,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "AttentionDecision":
        scope.validate()
        provenance.validate()
        stability = require_confidence(
            layout_stability_weight, "layout_stability_weight"
        )
        if stability is None:
            raise CognitiveKernelContractError(
                "layout_stability_weight is required"
            )
        normalized_entries = tuple(sorted(entries, key=lambda entry: entry.rank))
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "decision_key": require_identifier(decision_key, "decision_key"),
            "decided_at": normalize_timestamp(decided_at, "decided_at"),
        }
        candidate_digest = canonical_sha256(
            [entry.metadata_record() for entry in normalized_entries]
        )
        provisional = cls(
            schema_version=identity["schema_version"],
            decision_id=f"attention-decision-{canonical_sha256(identity)[:32]}",
            decision_key=identity["decision_key"],
            scope=scope,
            decided_at=identity["decided_at"],
            visibility_limit=_positive_integer(
                visibility_limit, "visibility_limit", maximum=10
            ),
            interruption_preference=_enum(
                interruption_preference,
                "interruption_preference",
                INTERRUPTION_PREFERENCES,
            ),
            focus_mode=_enum(focus_mode, "focus_mode", FOCUS_MODES),
            layout_stability_weight=stability,
            entries=normalized_entries,
            candidate_snapshot_sha256=candidate_digest,
            provenance=provenance,
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            decision_sha256="0" * 64,
        )
        provisional._validate_material()
        digest = canonical_sha256(provisional.material_record())
        decision = cls(**{**provisional.__dict__, "decision_sha256": digest})
        decision.validate()
        return decision

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "decision_key": self.decision_key,
            "decided_at": self.decided_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        self.provenance.validate()
        require_identifier(self.decision_key, "decision_key")
        normalize_timestamp(self.decided_at, "decided_at")
        _positive_integer(
            self.visibility_limit, "visibility_limit", maximum=10
        )
        _enum(
            self.interruption_preference,
            "interruption_preference",
            INTERRUPTION_PREFERENCES,
        )
        _enum(self.focus_mode, "focus_mode", FOCUS_MODES)
        if (
            require_confidence(
                self.layout_stability_weight,
                "layout_stability_weight",
            )
            != self.layout_stability_weight
        ):
            raise CognitiveKernelContractError(
                "layout_stability_weight is not canonical"
            )
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )

        expected_ranks = tuple(range(1, len(self.entries) + 1))
        actual_ranks = tuple(entry.rank for entry in self.entries)
        if actual_ranks != expected_ranks:
            raise CognitiveKernelContractError(
                "attention ranks must be contiguous and one-based"
            )
        entry_ids: set[str] = set()
        reference_ids: set[str] = set()
        selected_nonprotected = 0
        for entry in self.entries:
            entry.validate()
            if entry.scope != self.scope:
                raise CognitiveKernelContractError(
                    "cross-host attention entries are forbidden"
                )
            if entry.entry_id in entry_ids or entry.reference_id in reference_ids:
                raise CognitiveKernelContractError(
                    "attention decision entries must be unique"
                )
            entry_ids.add(entry.entry_id)
            reference_ids.add(entry.reference_id)
            if entry.selected and entry.protected_interrupt_reason is None:
                selected_nonprotected += 1
        if selected_nonprotected > self.visibility_limit:
            raise CognitiveKernelContractError(
                "selected non-protected entries exceed visibility_limit"
            )
        expected_snapshot = canonical_sha256(
            [entry.metadata_record() for entry in self.entries]
        )
        if (
            require_sha256(
                self.candidate_snapshot_sha256,
                "candidate_snapshot_sha256",
            )
            != expected_snapshot
        ):
            raise CognitiveKernelContractError(
                "attention candidate snapshot digest mismatch"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            **self.identity_record(),
            "decision_id": self.decision_id,
            "visibility_limit": self.visibility_limit,
            "interruption_preference": self.interruption_preference,
            "focus_mode": self.focus_mode,
            "layout_stability_weight": self.layout_stability_weight,
            "entries": [entry.metadata_record() for entry in self.entries],
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "provenance": self.provenance.metadata_record(),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["decision_sha256"] = self.decision_sha256
        return record

    def selected_entries(self) -> tuple[AttentionRankEntry, ...]:
        self.validate()
        return tuple(entry for entry in self.entries if entry.selected)

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"attention-decision-{canonical_sha256(self.identity_record())[:32]}"
        if self.decision_id != expected_id:
            raise CognitiveKernelContractError("attention decision identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if (
            require_sha256(self.decision_sha256, "decision_sha256")
            != expected_digest
        ):
            raise CognitiveKernelContractError(
                "attention decision digest mismatch"
            )


@dataclass(frozen=True)
class HostWorkspaceOverride:
    """Auditable host command receipt; it does not execute UI behavior."""

    schema_version: str
    override_id: str
    override_key: str
    scope: ProductHostScope
    command: str
    status: str
    issued_at: str
    target_reference_id: str | None
    numeric_value: int | None
    setting_value: str | None
    expires_at: str | None
    reason_digest: str
    provenance: ProvenanceReference
    policy_bindings: tuple[str, ...]
    override_sha256: str

    @classmethod
    def create(
        cls,
        *,
        override_key: object,
        scope: ProductHostScope,
        command: object,
        status: object,
        issued_at: object,
        reason_digest: object,
        provenance: ProvenanceReference,
        target_reference_id: object | None = None,
        numeric_value: object | None = None,
        setting_value: object | None = None,
        expires_at: object | None = None,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "HostWorkspaceOverride":
        scope.validate()
        provenance.validate()
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "override_key": require_identifier(
                override_key, "override_key"
            ),
            "issued_at": normalize_timestamp(issued_at, "issued_at"),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            override_id=f"host-override-{canonical_sha256(identity)[:32]}",
            override_key=identity["override_key"],
            scope=scope,
            command=_enum(command, "command", HOST_OVERRIDE_COMMANDS),
            status=_enum(status, "status", HOST_OVERRIDE_STATUSES),
            issued_at=identity["issued_at"],
            target_reference_id=_optional_identifier(
                target_reference_id, "target_reference_id"
            ),
            numeric_value=(
                _positive_integer(
                    numeric_value, "numeric_value", maximum=10
                )
                if numeric_value is not None
                else None
            ),
            setting_value=(
                require_identifier(setting_value, "setting_value")
                if setting_value is not None
                else None
            ),
            expires_at=(
                normalize_timestamp(expires_at, "expires_at")
                if expires_at is not None
                else None
            ),
            reason_digest=require_sha256(reason_digest, "reason_digest"),
            provenance=provenance,
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            override_sha256="0" * 64,
        )
        provisional._validate_material()
        digest = canonical_sha256(provisional.material_record())
        receipt = cls(**{**provisional.__dict__, "override_sha256": digest})
        receipt.validate()
        return receipt

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "override_key": self.override_key,
            "issued_at": self.issued_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        self.provenance.validate()
        require_identifier(self.override_key, "override_key")
        command = _enum(self.command, "command", HOST_OVERRIDE_COMMANDS)
        status = _enum(self.status, "status", HOST_OVERRIDE_STATUSES)
        normalize_timestamp(self.issued_at, "issued_at")
        require_sha256(self.reason_digest, "reason_digest")
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )
        if self.target_reference_id is not None:
            require_identifier(
                self.target_reference_id, "target_reference_id"
            )
        if self.setting_value is not None:
            require_identifier(self.setting_value, "setting_value")
        if self.expires_at is not None:
            normalized_expiry = normalize_timestamp(
                self.expires_at, "expires_at"
            )
            if normalized_expiry <= self.issued_at:
                raise CognitiveKernelContractError(
                    "expires_at must be later than issued_at"
                )
        if status == "expired" and self.expires_at is None:
            raise CognitiveKernelContractError(
                "expired overrides require expires_at"
            )

        target_commands = {
            "pin",
            "unpin",
            "foreground",
            "background",
            "set_focus_mission",
        }
        no_argument_commands = {
            "lock_layout",
            "unlock_layout",
            "clear_focus",
            "keep_security_visible",
            "restore_automatic_layout",
        }
        if command in target_commands:
            if self.target_reference_id is None:
                raise CognitiveKernelContractError(
                    f"{command} requires target_reference_id"
                )
            if self.numeric_value is not None or self.setting_value is not None:
                raise CognitiveKernelContractError(
                    f"{command} accepts only target_reference_id"
                )
        elif command == "set_visibility_limit":
            if self.numeric_value is None:
                raise CognitiveKernelContractError(
                    "set_visibility_limit requires numeric_value"
                )
            _positive_integer(
                self.numeric_value, "numeric_value", maximum=10
            )
            if self.target_reference_id is not None or self.setting_value is not None:
                raise CognitiveKernelContractError(
                    "set_visibility_limit accepts only numeric_value"
                )
        elif command == "set_interruption_preference":
            if self.setting_value not in INTERRUPTION_PREFERENCES:
                raise CognitiveKernelContractError(
                    "set_interruption_preference requires an approved setting"
                )
            if self.target_reference_id is not None or self.numeric_value is not None:
                raise CognitiveKernelContractError(
                    "set_interruption_preference accepts only setting_value"
                )
        elif command in no_argument_commands:
            if any(
                value is not None
                for value in (
                    self.target_reference_id,
                    self.numeric_value,
                    self.setting_value,
                )
            ):
                raise CognitiveKernelContractError(
                    f"{command} does not accept command arguments"
                )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            **self.identity_record(),
            "override_id": self.override_id,
            "command": self.command,
            "status": self.status,
            "target_reference_id": self.target_reference_id,
            "numeric_value": self.numeric_value,
            "setting_value": self.setting_value,
            "expires_at": self.expires_at,
            "reason_digest": self.reason_digest,
            "provenance": self.provenance.metadata_record(),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["override_sha256"] = self.override_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"host-override-{canonical_sha256(self.identity_record())[:32]}"
        if self.override_id != expected_id:
            raise CognitiveKernelContractError("host override identity mismatch")
        expected_digest = canonical_sha256(self.material_record())
        if (
            require_sha256(self.override_sha256, "override_sha256")
            != expected_digest
        ):
            raise CognitiveKernelContractError("host override digest mismatch")
