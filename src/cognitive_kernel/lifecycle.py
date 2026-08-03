"""Deterministic retention-lifecycle decisions, blockers, and journal receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import ProductHostScope, RETENTION_CLASSES, STORAGE_TIERS

LIFECYCLE_SCHEMA_VERSION = "1.0.0"
LIFECYCLE_RECORD_KINDS = frozenset({"decision", "blocker"})
LIFECYCLE_DECISION_TYPES = frozenset(
    {"retain", "transition", "quarantine", "delete_eligible", "override"}
)
LIFECYCLE_DECISION_OUTCOMES = frozenset(
    {"approved", "denied", "blocked", "recorded"}
)
RETENTION_BLOCKER_STATES = frozenset({"open", "resolved"})
LIFECYCLE_AUTHORITY_LEVELS = (
    "none",
    "guest_scoped",
    "host_context",
    "host_verified",
    "owner_verified",
)
RETENTION_BLOCKER_TYPES = frozenset(
    {
        "authoritative_provenance",
        "unresolved_correction_or_deletion",
        "active_project",
        "owner_hold",
        "active_training_or_replay_manifest",
        "evaluation_reproducibility",
        "champion_challenger_comparison",
        "rollback_or_disaster_recovery",
    }
)

ALLOWED_LIFECYCLE_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "ledger": frozenset(),
    "raw_buffer": frozenset(
        {"hot", "warm", "cold", "quarantine", "deleted"}
    ),
    "hot": frozenset({"warm", "cold", "quarantine", "deleted"}),
    "warm": frozenset({"hot", "cold", "quarantine", "deleted"}),
    "cold": frozenset({"hot", "warm", "quarantine", "deleted"}),
    "quarantine": frozenset({"hot", "warm", "cold", "deleted"}),
    "deleted": frozenset(),
}

LIFECYCLE_AUTHORITY_REQUIREMENTS: Mapping[str, str] = {
    "ordinary_transition": "host_context",
    "enter_quarantine": "host_context",
    "leave_quarantine": "host_verified",
    "delete_eligibility": "host_verified",
    "override": "owner_verified",
    "blocker_open": "host_context",
    "blocker_resolution": "host_verified",
    "owner_hold": "owner_verified",
}

_AUTHORITY_RANK = {
    value: index for index, value in enumerate(LIFECYCLE_AUTHORITY_LEVELS)
}
_ZERO_SHA256 = "0" * 64


def _enum(value: object, field: str, allowed: Iterable[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


def _required_sequence(
    values: tuple[object, ...] | list[object],
    field: str,
) -> tuple[str, ...]:
    normalized = normalize_identifier_sequence(values, field)
    if not normalized:
        raise CognitiveKernelContractError(f"{field} may not be empty")
    return normalized


def _authority_at_least(actual: str, required: str) -> bool:
    return _AUTHORITY_RANK[actual] >= _AUTHORITY_RANK[required]


def _normalize_optional_identifier(value: object | None, field: str) -> str | None:
    return require_identifier(value, field) if value is not None else None


def _validate_transition_shape(
    *,
    decision_type: str,
    current_tier: str,
    proposed_tier: str,
    outcome: str,
    authority_level: str,
    authority_decision_id: str | None,
    parent_decision_id: str | None,
) -> None:
    if current_tier not in STORAGE_TIERS:
        raise CognitiveKernelContractError("current_tier is not approved")
    if proposed_tier not in STORAGE_TIERS:
        raise CognitiveKernelContractError("proposed_tier is not approved")

    if decision_type == "retain":
        if current_tier != proposed_tier:
            raise CognitiveKernelContractError(
                "retain decisions must preserve the logical tier"
            )
        if outcome != "recorded":
            raise CognitiveKernelContractError(
                "retain decisions must use the recorded outcome"
            )
        return

    if current_tier == proposed_tier:
        raise CognitiveKernelContractError(
            "lifecycle transitions may not be no-op transitions"
        )
    if proposed_tier not in ALLOWED_LIFECYCLE_TRANSITIONS[current_tier]:
        raise CognitiveKernelContractError(
            "logical lifecycle transition is not authorized by the matrix"
        )

    if decision_type == "transition":
        if proposed_tier in {"quarantine", "deleted"}:
            raise CognitiveKernelContractError(
                "quarantine and deletion eligibility require dedicated decision types"
            )
        if outcome not in {"approved", "denied", "blocked"}:
            raise CognitiveKernelContractError(
                "transition outcome is not approved"
            )
        if outcome == "approved":
            required = (
                LIFECYCLE_AUTHORITY_REQUIREMENTS["leave_quarantine"]
                if current_tier == "quarantine"
                else LIFECYCLE_AUTHORITY_REQUIREMENTS["ordinary_transition"]
            )
            if not _authority_at_least(authority_level, required):
                raise CognitiveKernelContractError(
                    "approved lifecycle transition lacks required authority"
                )
            if authority_decision_id is None:
                raise CognitiveKernelContractError(
                    "approved lifecycle transition requires authority lineage"
                )
        return

    if decision_type == "quarantine":
        if proposed_tier != "quarantine":
            raise CognitiveKernelContractError(
                "quarantine decisions must target quarantine"
            )
        if outcome not in {"approved", "denied", "blocked"}:
            raise CognitiveKernelContractError(
                "quarantine outcome is not approved"
            )
        if outcome == "approved":
            required = LIFECYCLE_AUTHORITY_REQUIREMENTS["enter_quarantine"]
            if not _authority_at_least(authority_level, required):
                raise CognitiveKernelContractError(
                    "approved quarantine decision lacks required authority"
                )
            if authority_decision_id is None:
                raise CognitiveKernelContractError(
                    "approved quarantine decision requires authority lineage"
                )
        return

    if decision_type == "delete_eligible":
        if proposed_tier != "deleted":
            raise CognitiveKernelContractError(
                "delete-eligibility decisions must target deleted"
            )
        if outcome not in {"recorded", "blocked"}:
            raise CognitiveKernelContractError(
                "delete-eligibility outcome is not approved"
            )
        if outcome == "recorded":
            required = LIFECYCLE_AUTHORITY_REQUIREMENTS["delete_eligibility"]
            if not _authority_at_least(authority_level, required):
                raise CognitiveKernelContractError(
                    "delete eligibility lacks required authority"
                )
            if authority_decision_id is None:
                raise CognitiveKernelContractError(
                    "delete eligibility requires authority lineage"
                )
        return

    if decision_type == "override":
        if proposed_tier == "deleted":
            raise CognitiveKernelContractError(
                "P5.1c overrides may not authorize payload deletion"
            )
        if outcome != "approved":
            raise CognitiveKernelContractError(
                "override decisions must be approved"
            )
        if parent_decision_id is None:
            raise CognitiveKernelContractError(
                "override decisions require prior-decision lineage"
            )
        required = LIFECYCLE_AUTHORITY_REQUIREMENTS["override"]
        if not _authority_at_least(authority_level, required):
            raise CognitiveKernelContractError(
                "override decision requires owner-verified authority"
            )
        if authority_decision_id is None:
            raise CognitiveKernelContractError(
                "override decision requires authority lineage"
            )
        return

    raise CognitiveKernelContractError("decision_type is not approved")


@dataclass(frozen=True)
class LifecycleDecision:
    """Metadata-only decision describing a logical retention outcome."""

    schema_version: str
    decision_id: str
    decision_key: str
    scope: ProductHostScope
    subject_reference: str
    content_digest: str
    decision_type: str
    current_tier: str
    proposed_tier: str
    retention_class: str
    decided_at: str
    actor_id: str
    authority_level: str
    authority_decision_id: str | None
    provenance_reference_id: str
    parent_decision_id: str | None
    reason_codes: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    outcome: str
    decision_sha256: str

    @classmethod
    def create(
        cls,
        *,
        decision_key: object,
        scope: ProductHostScope,
        subject_reference: object,
        content_digest: object,
        decision_type: object,
        current_tier: object,
        proposed_tier: object,
        retention_class: object,
        decided_at: object,
        actor_id: object,
        authority_level: object,
        provenance_reference_id: object,
        reason_codes: tuple[object, ...] | list[object],
        policy_bindings: tuple[object, ...] | list[object],
        outcome: object,
        authority_decision_id: object | None = None,
        parent_decision_id: object | None = None,
        schema_version: object = LIFECYCLE_SCHEMA_VERSION,
    ) -> "LifecycleDecision":
        scope.validate()
        normalized_schema = require_schema_version(schema_version)
        if normalized_schema != LIFECYCLE_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "lifecycle decision schema version changed"
            )
        normalized_key = require_identifier(decision_key, "decision_key")
        normalized_subject = require_identifier(
            subject_reference, "subject_reference"
        )
        normalized_time = normalize_timestamp(decided_at, "decided_at")
        normalized_type = _enum(
            decision_type, "decision_type", LIFECYCLE_DECISION_TYPES
        )
        normalized_current = _enum(
            current_tier, "current_tier", STORAGE_TIERS
        )
        normalized_proposed = _enum(
            proposed_tier, "proposed_tier", STORAGE_TIERS
        )
        normalized_retention = _enum(
            retention_class, "retention_class", RETENTION_CLASSES
        )
        normalized_authority = _enum(
            authority_level,
            "authority_level",
            LIFECYCLE_AUTHORITY_LEVELS,
        )
        normalized_outcome = _enum(
            outcome, "outcome", LIFECYCLE_DECISION_OUTCOMES
        )
        normalized_authority_decision = _normalize_optional_identifier(
            authority_decision_id, "authority_decision_id"
        )
        normalized_parent = _normalize_optional_identifier(
            parent_decision_id, "parent_decision_id"
        )
        _validate_transition_shape(
            decision_type=normalized_type,
            current_tier=normalized_current,
            proposed_tier=normalized_proposed,
            outcome=normalized_outcome,
            authority_level=normalized_authority,
            authority_decision_id=normalized_authority_decision,
            parent_decision_id=normalized_parent,
        )
        identity = {
            "schema_version": normalized_schema,
            "scope": scope.metadata_record(),
            "decision_key": normalized_key,
            "subject_reference": normalized_subject,
            "decided_at": normalized_time,
        }
        provisional = cls(
            schema_version=normalized_schema,
            decision_id=(
                "lifecycle-decision-" + canonical_sha256(identity)[:32]
            ),
            decision_key=normalized_key,
            scope=scope,
            subject_reference=normalized_subject,
            content_digest=require_sha256(
                content_digest, "content_digest"
            ),
            decision_type=normalized_type,
            current_tier=normalized_current,
            proposed_tier=normalized_proposed,
            retention_class=normalized_retention,
            decided_at=normalized_time,
            actor_id=require_identifier(actor_id, "actor_id"),
            authority_level=normalized_authority,
            authority_decision_id=normalized_authority_decision,
            provenance_reference_id=require_identifier(
                provenance_reference_id, "provenance_reference_id"
            ),
            parent_decision_id=normalized_parent,
            reason_codes=_required_sequence(reason_codes, "reason_codes"),
            policy_bindings=_required_sequence(
                policy_bindings, "policy_bindings"
            ),
            outcome=normalized_outcome,
            decision_sha256=_ZERO_SHA256,
        )
        provisional._validate_material()
        decision = cls(
            **{
                **provisional.__dict__,
                "decision_sha256": canonical_sha256(
                    provisional.material_record()
                ),
            }
        )
        decision.validate()
        return decision

    @property
    def record_id(self) -> str:
        return self.decision_id

    @property
    def record_sha256(self) -> str:
        return self.decision_sha256

    @property
    def record_kind(self) -> str:
        return "decision"

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "decision_key": self.decision_key,
            "subject_reference": self.subject_reference,
            "decided_at": self.decided_at,
        }

    def _validate_material(self) -> None:
        if require_schema_version(self.schema_version) != LIFECYCLE_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "lifecycle decision schema version changed"
            )
        self.scope.validate()
        require_identifier(self.decision_key, "decision_key")
        require_identifier(self.subject_reference, "subject_reference")
        require_sha256(self.content_digest, "content_digest")
        decision_type = _enum(
            self.decision_type,
            "decision_type",
            LIFECYCLE_DECISION_TYPES,
        )
        current = _enum(self.current_tier, "current_tier", STORAGE_TIERS)
        proposed = _enum(self.proposed_tier, "proposed_tier", STORAGE_TIERS)
        _enum(
            self.retention_class,
            "retention_class",
            RETENTION_CLASSES,
        )
        normalize_timestamp(self.decided_at, "decided_at")
        require_identifier(self.actor_id, "actor_id")
        authority = _enum(
            self.authority_level,
            "authority_level",
            LIFECYCLE_AUTHORITY_LEVELS,
        )
        if self.authority_decision_id is not None:
            require_identifier(
                self.authority_decision_id, "authority_decision_id"
            )
        require_identifier(
            self.provenance_reference_id, "provenance_reference_id"
        )
        if self.parent_decision_id is not None:
            require_identifier(self.parent_decision_id, "parent_decision_id")
        _required_sequence(self.reason_codes, "reason_codes")
        _required_sequence(self.policy_bindings, "policy_bindings")
        outcome = _enum(
            self.outcome, "outcome", LIFECYCLE_DECISION_OUTCOMES
        )
        _validate_transition_shape(
            decision_type=decision_type,
            current_tier=current,
            proposed_tier=proposed,
            outcome=outcome,
            authority_level=authority,
            authority_decision_id=self.authority_decision_id,
            parent_decision_id=self.parent_decision_id,
        )

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "decision_id": self.decision_id,
            "content_digest": self.content_digest,
            "decision_type": self.decision_type,
            "current_tier": self.current_tier,
            "proposed_tier": self.proposed_tier,
            "retention_class": self.retention_class,
            "actor_id": self.actor_id,
            "authority_level": self.authority_level,
            "authority_decision_id": self.authority_decision_id,
            "provenance_reference_id": self.provenance_reference_id,
            "parent_decision_id": self.parent_decision_id,
            "reason_codes": list(self.reason_codes),
            "policy_bindings": list(self.policy_bindings),
            "outcome": self.outcome,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.material_record(),
            "decision_sha256": self.decision_sha256,
        }

    def validate(self) -> None:
        self._validate_material()
        expected_id = (
            "lifecycle-decision-"
            + canonical_sha256(self.identity_record())[:32]
        )
        if self.decision_id != expected_id:
            raise CognitiveKernelContractError(
                "lifecycle decision identity is invalid"
            )
        require_sha256(self.decision_sha256, "decision_sha256")
        if self.decision_sha256 != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError(
                "lifecycle decision digest mismatch"
            )

    @classmethod
    def from_metadata_record(
        cls, value: Mapping[str, object]
    ) -> "LifecycleDecision":
        expected = {
            "schema_version",
            "decision_id",
            "decision_key",
            "scope",
            "subject_reference",
            "content_digest",
            "decision_type",
            "current_tier",
            "proposed_tier",
            "retention_class",
            "decided_at",
            "actor_id",
            "authority_level",
            "authority_decision_id",
            "provenance_reference_id",
            "parent_decision_id",
            "reason_codes",
            "policy_bindings",
            "outcome",
            "decision_sha256",
        }
        if set(value) != expected:
            raise CognitiveKernelContractError(
                "lifecycle decision record keys changed"
            )
        scope_value = value["scope"]
        if not isinstance(scope_value, Mapping):
            raise CognitiveKernelContractError(
                "lifecycle decision scope must be an object"
            )
        scope = ProductHostScope.create(
            product_id=scope_value.get("product_id"),
            host_instance_id=scope_value.get("host_instance_id"),
            schema_version=scope_value.get("schema_version"),
            encryption_domain=scope_value.get("encryption_domain"),
        )
        reasons = value["reason_codes"]
        bindings = value["policy_bindings"]
        if not isinstance(reasons, list) or not isinstance(bindings, list):
            raise CognitiveKernelContractError(
                "lifecycle decision reason and policy fields must be lists"
            )
        decision = cls.create(
            schema_version=value["schema_version"],
            decision_key=value["decision_key"],
            scope=scope,
            subject_reference=value["subject_reference"],
            content_digest=value["content_digest"],
            decision_type=value["decision_type"],
            current_tier=value["current_tier"],
            proposed_tier=value["proposed_tier"],
            retention_class=value["retention_class"],
            decided_at=value["decided_at"],
            actor_id=value["actor_id"],
            authority_level=value["authority_level"],
            authority_decision_id=value["authority_decision_id"],
            provenance_reference_id=value["provenance_reference_id"],
            parent_decision_id=value["parent_decision_id"],
            reason_codes=reasons,
            policy_bindings=bindings,
            outcome=value["outcome"],
        )
        if decision.decision_id != value["decision_id"]:
            raise CognitiveKernelContractError(
                "stored lifecycle decision identity changed"
            )
        if decision.decision_sha256 != value["decision_sha256"]:
            raise CognitiveKernelContractError(
                "stored lifecycle decision digest changed"
            )
        return decision


@dataclass(frozen=True)
class RetentionBlockerRecord:
    """Append-only open or resolution record for one retention blocker."""

    schema_version: str
    blocker_id: str
    blocker_record_id: str
    blocker_key: str
    scope: ProductHostScope
    subject_reference: str
    content_digest: str
    blocker_type: str
    state: str
    opened_at: str
    recorded_at: str
    actor_id: str
    authority_level: str
    authority_decision_id: str
    evidence_reference_id: str
    parent_record_id: str | None
    reason_codes: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    blocker_record_sha256: str

    @classmethod
    def create(
        cls,
        *,
        blocker_key: object,
        scope: ProductHostScope,
        subject_reference: object,
        content_digest: object,
        blocker_type: object,
        state: object,
        opened_at: object,
        recorded_at: object,
        actor_id: object,
        authority_level: object,
        authority_decision_id: object,
        evidence_reference_id: object,
        reason_codes: tuple[object, ...] | list[object],
        policy_bindings: tuple[object, ...] | list[object],
        parent_record_id: object | None = None,
        schema_version: object = LIFECYCLE_SCHEMA_VERSION,
    ) -> "RetentionBlockerRecord":
        scope.validate()
        normalized_schema = require_schema_version(schema_version)
        if normalized_schema != LIFECYCLE_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "retention blocker schema version changed"
            )
        normalized_key = require_identifier(blocker_key, "blocker_key")
        normalized_subject = require_identifier(
            subject_reference, "subject_reference"
        )
        normalized_type = _enum(
            blocker_type, "blocker_type", RETENTION_BLOCKER_TYPES
        )
        normalized_state = _enum(
            state, "state", RETENTION_BLOCKER_STATES
        )
        normalized_opened = normalize_timestamp(opened_at, "opened_at")
        normalized_recorded = normalize_timestamp(recorded_at, "recorded_at")
        normalized_authority = _enum(
            authority_level,
            "authority_level",
            LIFECYCLE_AUTHORITY_LEVELS,
        )
        normalized_parent = _normalize_optional_identifier(
            parent_record_id, "parent_record_id"
        )
        identity = {
            "schema_version": normalized_schema,
            "scope": scope.metadata_record(),
            "blocker_key": normalized_key,
            "subject_reference": normalized_subject,
        }
        blocker_id = "retention-blocker-" + canonical_sha256(identity)[:32]
        record_identity = {
            "schema_version": normalized_schema,
            "blocker_id": blocker_id,
            "state": normalized_state,
            "recorded_at": normalized_recorded,
            "parent_record_id": normalized_parent,
        }
        provisional = cls(
            schema_version=normalized_schema,
            blocker_id=blocker_id,
            blocker_record_id=(
                "retention-blocker-record-"
                + canonical_sha256(record_identity)[:32]
            ),
            blocker_key=normalized_key,
            scope=scope,
            subject_reference=normalized_subject,
            content_digest=require_sha256(
                content_digest, "content_digest"
            ),
            blocker_type=normalized_type,
            state=normalized_state,
            opened_at=normalized_opened,
            recorded_at=normalized_recorded,
            actor_id=require_identifier(actor_id, "actor_id"),
            authority_level=normalized_authority,
            authority_decision_id=require_identifier(
                authority_decision_id, "authority_decision_id"
            ),
            evidence_reference_id=require_identifier(
                evidence_reference_id, "evidence_reference_id"
            ),
            parent_record_id=normalized_parent,
            reason_codes=_required_sequence(reason_codes, "reason_codes"),
            policy_bindings=_required_sequence(
                policy_bindings, "policy_bindings"
            ),
            blocker_record_sha256=_ZERO_SHA256,
        )
        provisional._validate_material()
        record = cls(
            **{
                **provisional.__dict__,
                "blocker_record_sha256": canonical_sha256(
                    provisional.material_record()
                ),
            }
        )
        record.validate()
        return record

    @property
    def record_id(self) -> str:
        return self.blocker_record_id

    @property
    def record_sha256(self) -> str:
        return self.blocker_record_sha256

    @property
    def record_kind(self) -> str:
        return "blocker"

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "blocker_key": self.blocker_key,
            "subject_reference": self.subject_reference,
        }

    def record_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "blocker_id": self.blocker_id,
            "state": self.state,
            "recorded_at": self.recorded_at,
            "parent_record_id": self.parent_record_id,
        }

    def _validate_material(self) -> None:
        if require_schema_version(self.schema_version) != LIFECYCLE_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "retention blocker schema version changed"
            )
        self.scope.validate()
        require_identifier(self.blocker_key, "blocker_key")
        require_identifier(self.subject_reference, "subject_reference")
        require_sha256(self.content_digest, "content_digest")
        blocker_type = _enum(
            self.blocker_type, "blocker_type", RETENTION_BLOCKER_TYPES
        )
        state = _enum(self.state, "state", RETENTION_BLOCKER_STATES)
        opened = normalize_timestamp(self.opened_at, "opened_at")
        recorded = normalize_timestamp(self.recorded_at, "recorded_at")
        if recorded < opened:
            raise CognitiveKernelContractError(
                "retention blocker record predates its opening"
            )
        require_identifier(self.actor_id, "actor_id")
        authority = _enum(
            self.authority_level,
            "authority_level",
            LIFECYCLE_AUTHORITY_LEVELS,
        )
        require_identifier(
            self.authority_decision_id, "authority_decision_id"
        )
        require_identifier(
            self.evidence_reference_id, "evidence_reference_id"
        )
        if state == "open":
            if self.parent_record_id is not None:
                raise CognitiveKernelContractError(
                    "open blocker record may not have a parent"
                )
            if opened != recorded:
                raise CognitiveKernelContractError(
                    "open blocker record must establish opened_at"
                )
            requirement = LIFECYCLE_AUTHORITY_REQUIREMENTS["blocker_open"]
        else:
            if self.parent_record_id is None:
                raise CognitiveKernelContractError(
                    "resolved blocker record requires open-record lineage"
                )
            require_identifier(self.parent_record_id, "parent_record_id")
            requirement = LIFECYCLE_AUTHORITY_REQUIREMENTS[
                "blocker_resolution"
            ]
        if blocker_type == "owner_hold":
            requirement = LIFECYCLE_AUTHORITY_REQUIREMENTS["owner_hold"]
        if not _authority_at_least(authority, requirement):
            raise CognitiveKernelContractError(
                "retention blocker action lacks required authority"
            )
        _required_sequence(self.reason_codes, "reason_codes")
        _required_sequence(self.policy_bindings, "policy_bindings")

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            **self.record_identity(),
            "blocker_record_id": self.blocker_record_id,
            "content_digest": self.content_digest,
            "blocker_type": self.blocker_type,
            "opened_at": self.opened_at,
            "actor_id": self.actor_id,
            "authority_level": self.authority_level,
            "authority_decision_id": self.authority_decision_id,
            "evidence_reference_id": self.evidence_reference_id,
            "reason_codes": list(self.reason_codes),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.material_record(),
            "blocker_record_sha256": self.blocker_record_sha256,
        }

    def validate(self) -> None:
        self._validate_material()
        expected_blocker = (
            "retention-blocker-"
            + canonical_sha256(self.identity_record())[:32]
        )
        if self.blocker_id != expected_blocker:
            raise CognitiveKernelContractError(
                "retention blocker identity is invalid"
            )
        expected_record = (
            "retention-blocker-record-"
            + canonical_sha256(self.record_identity())[:32]
        )
        if self.blocker_record_id != expected_record:
            raise CognitiveKernelContractError(
                "retention blocker record identity is invalid"
            )
        require_sha256(
            self.blocker_record_sha256, "blocker_record_sha256"
        )
        if self.blocker_record_sha256 != canonical_sha256(
            self.material_record()
        ):
            raise CognitiveKernelContractError(
                "retention blocker record digest mismatch"
            )

    @classmethod
    def from_metadata_record(
        cls, value: Mapping[str, object]
    ) -> "RetentionBlockerRecord":
        expected = {
            "schema_version",
            "blocker_id",
            "blocker_record_id",
            "blocker_key",
            "scope",
            "subject_reference",
            "content_digest",
            "blocker_type",
            "state",
            "opened_at",
            "recorded_at",
            "actor_id",
            "authority_level",
            "authority_decision_id",
            "evidence_reference_id",
            "parent_record_id",
            "reason_codes",
            "policy_bindings",
            "blocker_record_sha256",
        }
        if set(value) != expected:
            raise CognitiveKernelContractError(
                "retention blocker record keys changed"
            )
        scope_value = value["scope"]
        if not isinstance(scope_value, Mapping):
            raise CognitiveKernelContractError(
                "retention blocker scope must be an object"
            )
        scope = ProductHostScope.create(
            product_id=scope_value.get("product_id"),
            host_instance_id=scope_value.get("host_instance_id"),
            schema_version=scope_value.get("schema_version"),
            encryption_domain=scope_value.get("encryption_domain"),
        )
        reasons = value["reason_codes"]
        bindings = value["policy_bindings"]
        if not isinstance(reasons, list) or not isinstance(bindings, list):
            raise CognitiveKernelContractError(
                "retention blocker reason and policy fields must be lists"
            )
        record = cls.create(
            schema_version=value["schema_version"],
            blocker_key=value["blocker_key"],
            scope=scope,
            subject_reference=value["subject_reference"],
            content_digest=value["content_digest"],
            blocker_type=value["blocker_type"],
            state=value["state"],
            opened_at=value["opened_at"],
            recorded_at=value["recorded_at"],
            actor_id=value["actor_id"],
            authority_level=value["authority_level"],
            authority_decision_id=value["authority_decision_id"],
            evidence_reference_id=value["evidence_reference_id"],
            parent_record_id=value["parent_record_id"],
            reason_codes=reasons,
            policy_bindings=bindings,
        )
        if record.blocker_id != value["blocker_id"]:
            raise CognitiveKernelContractError(
                "stored retention blocker identity changed"
            )
        if record.blocker_record_id != value["blocker_record_id"]:
            raise CognitiveKernelContractError(
                "stored retention blocker record identity changed"
            )
        if record.blocker_record_sha256 != value["blocker_record_sha256"]:
            raise CognitiveKernelContractError(
                "stored retention blocker record digest changed"
            )
        return record


LifecycleJournalValue = LifecycleDecision | RetentionBlockerRecord


def lifecycle_value_from_metadata(
    record_kind: object,
    value: Mapping[str, object],
) -> LifecycleJournalValue:
    normalized = _enum(
        record_kind, "record_kind", LIFECYCLE_RECORD_KINDS
    )
    if normalized == "decision":
        return LifecycleDecision.from_metadata_record(value)
    return RetentionBlockerRecord.from_metadata_record(value)


def lifecycle_scope_record(
    *,
    product_id: str,
    host_instance_id: str,
    encryption_domain: str,
) -> dict[str, str]:
    return {
        "product_id": require_identifier(product_id, "product_id"),
        "host_instance_id": require_identifier(
            host_instance_id, "host_instance_id"
        ),
        "encryption_domain": require_identifier(
            encryption_domain, "encryption_domain"
        ),
    }


def lifecycle_scope_digest(
    *,
    product_id: str,
    host_instance_id: str,
    encryption_domain: str,
) -> str:
    return canonical_sha256(
        lifecycle_scope_record(
            product_id=product_id,
            host_instance_id=host_instance_id,
            encryption_domain=encryption_domain,
        )
    )


@dataclass(frozen=True)
class LifecycleJournalEntryReceipt:
    sequence: int
    record_id: str
    record_sha256: str
    previous_entry_sha256: str
    entry_sha256: str
    committed_at: str

    @classmethod
    def create(
        cls,
        *,
        sequence: object,
        record_id: object,
        record_sha256: object,
        previous_entry_sha256: object,
        committed_at: object,
    ) -> "LifecycleJournalEntryReceipt":
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise CognitiveKernelContractError(
                "lifecycle journal sequence must be an integer"
            )
        if sequence < 1:
            raise CognitiveKernelContractError(
                "lifecycle journal sequence must be positive"
            )
        normalized_id = require_identifier(record_id, "record_id")
        normalized_record_digest = require_sha256(
            record_sha256, "record_sha256"
        )
        normalized_previous = require_sha256(
            previous_entry_sha256, "previous_entry_sha256"
        )
        normalized_time = normalize_timestamp(
            committed_at, "committed_at"
        )
        material = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "sequence": sequence,
            "record_id": normalized_id,
            "record_sha256": normalized_record_digest,
            "previous_entry_sha256": normalized_previous,
            "committed_at": normalized_time,
        }
        return cls(
            sequence=sequence,
            record_id=normalized_id,
            record_sha256=normalized_record_digest,
            previous_entry_sha256=normalized_previous,
            entry_sha256=canonical_sha256(material),
            committed_at=normalized_time,
        )

    def validate(self) -> None:
        expected = self.create(
            sequence=self.sequence,
            record_id=self.record_id,
            record_sha256=self.record_sha256,
            previous_entry_sha256=self.previous_entry_sha256,
            committed_at=self.committed_at,
        )
        if self.entry_sha256 != expected.entry_sha256:
            raise CognitiveKernelContractError(
                "lifecycle journal entry digest mismatch"
            )

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "record_id": self.record_id,
            "record_sha256": self.record_sha256,
            "previous_entry_sha256": self.previous_entry_sha256,
            "entry_sha256": self.entry_sha256,
            "committed_at": self.committed_at,
        }


@dataclass(frozen=True)
class LifecycleJournalTransactionReceipt:
    schema_version: str
    transaction_id: str
    journal_id: str
    committed_at: str
    entries: tuple[LifecycleJournalEntryReceipt, ...]
    transaction_sha256: str

    @classmethod
    def create(
        cls,
        *,
        journal_id: object,
        committed_at: object,
        entries: tuple[LifecycleJournalEntryReceipt, ...],
        schema_version: object = LIFECYCLE_SCHEMA_VERSION,
    ) -> "LifecycleJournalTransactionReceipt":
        normalized_schema = require_schema_version(schema_version)
        if normalized_schema != LIFECYCLE_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "lifecycle journal receipt schema version changed"
            )
        normalized_journal = require_identifier(journal_id, "journal_id")
        normalized_time = normalize_timestamp(
            committed_at, "committed_at"
        )
        if not isinstance(entries, tuple) or not entries:
            raise CognitiveKernelContractError(
                "lifecycle journal transaction requires entries"
            )
        prior: int | None = None
        for entry in entries:
            if not isinstance(entry, LifecycleJournalEntryReceipt):
                raise CognitiveKernelContractError(
                    "lifecycle journal transaction entries are invalid"
                )
            entry.validate()
            if entry.committed_at != normalized_time:
                raise CognitiveKernelContractError(
                    "lifecycle journal transaction timestamps must match"
                )
            if prior is not None and entry.sequence != prior + 1:
                raise CognitiveKernelContractError(
                    "lifecycle journal transaction sequences are not contiguous"
                )
            prior = entry.sequence
        material = {
            "schema_version": normalized_schema,
            "journal_id": normalized_journal,
            "committed_at": normalized_time,
            "entries": [entry.record() for entry in entries],
        }
        digest = canonical_sha256(material)
        receipt = cls(
            schema_version=normalized_schema,
            transaction_id=(
                "lifecycle-transaction-" + digest[:32]
            ),
            journal_id=normalized_journal,
            committed_at=normalized_time,
            entries=entries,
            transaction_sha256=digest,
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        if require_schema_version(self.schema_version) != LIFECYCLE_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "lifecycle journal receipt schema version changed"
            )
        require_identifier(self.journal_id, "journal_id")
        normalized_time = normalize_timestamp(
            self.committed_at, "committed_at"
        )
        if normalized_time != self.committed_at or not self.entries:
            raise CognitiveKernelContractError(
                "lifecycle journal transaction is invalid"
            )
        prior: int | None = None
        for entry in self.entries:
            entry.validate()
            if entry.committed_at != self.committed_at:
                raise CognitiveKernelContractError(
                    "lifecycle journal transaction timestamps must match"
                )
            if prior is not None and entry.sequence != prior + 1:
                raise CognitiveKernelContractError(
                    "lifecycle journal transaction sequences are not contiguous"
                )
            prior = entry.sequence
        material = {
            "schema_version": self.schema_version,
            "journal_id": self.journal_id,
            "committed_at": self.committed_at,
            "entries": [entry.record() for entry in self.entries],
        }
        digest = canonical_sha256(material)
        if self.transaction_id != "lifecycle-transaction-" + digest[:32]:
            raise CognitiveKernelContractError(
                "lifecycle journal transaction identity mismatch"
            )
        if self.transaction_sha256 != digest:
            raise CognitiveKernelContractError(
                "lifecycle journal transaction digest mismatch"
            )

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "journal_id": self.journal_id,
            "committed_at": self.committed_at,
            "entries": [entry.record() for entry in self.entries],
            "transaction_sha256": self.transaction_sha256,
        }


@dataclass(frozen=True)
class LifecycleJournalRecord:
    """Sanitized metadata projection of one journal row."""

    sequence: int
    record_id: str
    record_kind: str
    subject_reference: str
    content_digest: str
    decision_type: str | None
    current_tier: str | None
    proposed_tier: str | None
    retention_class: str | None
    outcome: str | None
    blocker_type: str | None
    blocker_state: str | None
    record_sha256: str
    previous_entry_sha256: str
    entry_sha256: str
    committed_at: str

    def record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LifecycleJournalIntegrityReport:
    schema_version: str
    journal_id: str
    entry_count: int
    first_sequence: int | None
    last_sequence: int | None
    head_entry_sha256: str
    valid: bool
    report_sha256: str

    @classmethod
    def create(
        cls,
        *,
        journal_id: object,
        entry_count: object,
        first_sequence: object | None,
        last_sequence: object | None,
        head_entry_sha256: object,
        valid: object,
    ) -> "LifecycleJournalIntegrityReport":
        normalized_id = require_identifier(journal_id, "journal_id")
        if isinstance(entry_count, bool) or not isinstance(entry_count, int):
            raise CognitiveKernelContractError(
                "entry_count must be an integer"
            )
        if entry_count < 0:
            raise CognitiveKernelContractError(
                "entry_count may not be negative"
            )
        if not isinstance(valid, bool):
            raise CognitiveKernelContractError("valid must be boolean")
        if entry_count == 0:
            if first_sequence is not None or last_sequence is not None:
                raise CognitiveKernelContractError(
                    "empty lifecycle journal may not contain sequences"
                )
        else:
            if not isinstance(first_sequence, int) or not isinstance(
                last_sequence, int
            ):
                raise CognitiveKernelContractError(
                    "non-empty lifecycle journal requires sequences"
                )
            if first_sequence != 1 or last_sequence != entry_count:
                raise CognitiveKernelContractError(
                    "lifecycle journal sequences are not contiguous"
                )
        normalized_head = require_sha256(
            head_entry_sha256, "head_entry_sha256"
        )
        material = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "journal_id": normalized_id,
            "entry_count": entry_count,
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "head_entry_sha256": normalized_head,
            "valid": valid,
        }
        digest = canonical_sha256(material)
        report = cls(
            schema_version=LIFECYCLE_SCHEMA_VERSION,
            journal_id=normalized_id,
            entry_count=entry_count,
            first_sequence=(
                first_sequence if isinstance(first_sequence, int) else None
            ),
            last_sequence=(
                last_sequence if isinstance(last_sequence, int) else None
            ),
            head_entry_sha256=normalized_head,
            valid=valid,
            report_sha256=digest,
        )
        report.validate()
        return report

    def validate(self) -> None:
        if self.schema_version != LIFECYCLE_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "lifecycle integrity schema version changed"
            )
        require_identifier(self.journal_id, "journal_id")
        if isinstance(self.entry_count, bool) or not isinstance(
            self.entry_count, int
        ):
            raise CognitiveKernelContractError(
                "entry_count must be an integer"
            )
        if self.entry_count < 0 or not isinstance(self.valid, bool):
            raise CognitiveKernelContractError(
                "lifecycle integrity report is invalid"
            )
        if self.entry_count == 0:
            if self.first_sequence is not None or self.last_sequence is not None:
                raise CognitiveKernelContractError(
                    "empty lifecycle journal may not contain sequences"
                )
        elif (
            self.first_sequence != 1
            or self.last_sequence != self.entry_count
        ):
            raise CognitiveKernelContractError(
                "lifecycle journal sequences are not contiguous"
            )
        require_sha256(self.head_entry_sha256, "head_entry_sha256")
        material = {
            "schema_version": self.schema_version,
            "journal_id": self.journal_id,
            "entry_count": self.entry_count,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "head_entry_sha256": self.head_entry_sha256,
            "valid": self.valid,
        }
        if self.report_sha256 != canonical_sha256(material):
            raise CognitiveKernelContractError(
                "lifecycle integrity report digest mismatch"
            )

    def record(self) -> dict[str, object]:
        return dict(self.__dict__)
