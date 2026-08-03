"""Governed, non-destructive physical tier-transition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
    require_text,
)
from .contracts import ProductHostScope, RETENTION_CLASSES, STORAGE_TIERS

TIER_TRANSITION_SCHEMA_VERSION = "1.0.0"
EXECUTABLE_SOURCE_TIERS = frozenset(
    {"raw_buffer", "hot", "warm", "cold", "quarantine"}
)
EXECUTABLE_TARGET_TIERS = frozenset({"hot", "warm", "cold", "quarantine"})
TIER_TRANSITION_DECISION_TYPES = frozenset(
    {"transition", "quarantine", "override"}
)
TIER_TRANSITION_STATES = frozenset({"prepared", "published"})
_ZERO_SHA256 = "0" * 64


def _scope_material(scope: ProductHostScope) -> dict[str, str]:
    scope.validate()
    return {
        "product_id": scope.product_id,
        "host_instance_id": scope.host_instance_id,
        "encryption_domain": scope.encryption_domain,
    }


def tier_transition_scope_digest(scope: ProductHostScope) -> str:
    return canonical_sha256(_scope_material(scope))


@dataclass(frozen=True)
class TierPayloadReference:
    """Logical reference to one host-sealed payload in a durable tier."""

    schema_version: str
    reference_id: str
    transition_id: str
    lifecycle_decision_id: str
    subject_reference: str
    source_reference_id: str
    scope: ProductHostScope
    content_digest: str
    byte_length: int
    media_type: str
    sensitivity_class: str
    retention_class: str
    storage_tier: str
    published_at: str
    host_sealed: bool
    reference_sha256: str

    @classmethod
    def create(
        cls,
        *,
        transition_id: object,
        lifecycle_decision_id: object,
        subject_reference: object,
        source_reference_id: object,
        scope: ProductHostScope,
        content_digest: object,
        byte_length: object,
        media_type: object,
        sensitivity_class: object,
        retention_class: object,
        storage_tier: object,
        published_at: object,
        host_sealed: object = True,
        schema_version: object = TIER_TRANSITION_SCHEMA_VERSION,
    ) -> "TierPayloadReference":
        scope.validate()
        normalized_schema = require_schema_version(schema_version)
        if normalized_schema != TIER_TRANSITION_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "tier payload reference schema version changed"
            )
        if isinstance(byte_length, bool) or not isinstance(byte_length, int):
            raise CognitiveKernelContractError("byte_length must be an integer")
        if byte_length < 1:
            raise CognitiveKernelContractError("byte_length must be positive")
        tier = require_identifier(storage_tier, "storage_tier")
        if tier not in EXECUTABLE_TARGET_TIERS:
            raise CognitiveKernelContractError(
                "tier payload reference target tier is not executable"
            )
        retention = require_identifier(retention_class, "retention_class")
        if retention not in RETENTION_CLASSES:
            raise CognitiveKernelContractError("retention_class is not approved")
        sensitivity = require_identifier(
            sensitivity_class, "sensitivity_class"
        )
        if host_sealed is not True:
            raise CognitiveKernelContractError(
                "tier payloads must remain host-sealed opaque bytes"
            )
        material = {
            "schema_version": normalized_schema,
            "transition_id": require_identifier(
                transition_id, "transition_id"
            ),
            "lifecycle_decision_id": require_identifier(
                lifecycle_decision_id, "lifecycle_decision_id"
            ),
            "subject_reference": require_identifier(
                subject_reference, "subject_reference"
            ),
            "source_reference_id": require_identifier(
                source_reference_id, "source_reference_id"
            ),
            "scope": scope.metadata_record(),
            "content_digest": require_sha256(
                content_digest, "content_digest"
            ),
            "byte_length": byte_length,
            "media_type": require_text(media_type, "media_type", maximum=255),
            "sensitivity_class": sensitivity,
            "retention_class": retention,
            "storage_tier": tier,
            "published_at": normalize_timestamp(
                published_at, "published_at"
            ),
            "host_sealed": True,
        }
        identity = {
            "transition_id": material["transition_id"],
            "subject_reference": material["subject_reference"],
            "content_digest": material["content_digest"],
            "target_tier": material["storage_tier"],
        }
        reference_id = (
            "tier-reference-" + canonical_sha256(identity)[:32]
        )
        digest = canonical_sha256(material)
        reference = cls(
            schema_version=normalized_schema,
            reference_id=reference_id,
            transition_id=material["transition_id"],
            lifecycle_decision_id=material["lifecycle_decision_id"],
            subject_reference=material["subject_reference"],
            source_reference_id=material["source_reference_id"],
            scope=scope,
            content_digest=material["content_digest"],
            byte_length=byte_length,
            media_type=material["media_type"],
            sensitivity_class=sensitivity,
            retention_class=retention,
            storage_tier=tier,
            published_at=material["published_at"],
            host_sealed=True,
            reference_sha256=digest,
        )
        reference.validate()
        return reference

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "lifecycle_decision_id": self.lifecycle_decision_id,
            "subject_reference": self.subject_reference,
            "source_reference_id": self.source_reference_id,
            "scope": self.scope.metadata_record(),
            "content_digest": self.content_digest,
            "byte_length": self.byte_length,
            "media_type": self.media_type,
            "sensitivity_class": self.sensitivity_class,
            "retention_class": self.retention_class,
            "storage_tier": self.storage_tier,
            "published_at": self.published_at,
            "host_sealed": self.host_sealed,
        }

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            **self.material_record(),
            "reference_id": self.reference_id,
            "reference_sha256": self.reference_sha256,
        }

    def validate(self) -> None:
        if self.schema_version != TIER_TRANSITION_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "tier payload reference schema version changed"
            )
        self.scope.validate()
        require_identifier(self.transition_id, "transition_id")
        require_identifier(
            self.lifecycle_decision_id, "lifecycle_decision_id"
        )
        require_identifier(self.subject_reference, "subject_reference")
        require_identifier(self.source_reference_id, "source_reference_id")
        require_sha256(self.content_digest, "content_digest")
        if isinstance(self.byte_length, bool) or not isinstance(
            self.byte_length, int
        ) or self.byte_length < 1:
            raise CognitiveKernelContractError(
                "byte_length must be a positive integer"
            )
        require_text(self.media_type, "media_type", maximum=255)
        require_identifier(self.sensitivity_class, "sensitivity_class")
        if self.retention_class not in RETENTION_CLASSES:
            raise CognitiveKernelContractError("retention_class is not approved")
        if self.storage_tier not in EXECUTABLE_TARGET_TIERS:
            raise CognitiveKernelContractError(
                "tier payload reference target tier is not executable"
            )
        if self.host_sealed is not True:
            raise CognitiveKernelContractError(
                "tier payloads must remain host-sealed opaque bytes"
            )
        if normalize_timestamp(
            self.published_at, "published_at"
        ) != self.published_at:
            raise CognitiveKernelContractError(
                "published_at is not canonical"
            )
        identity = {
            "transition_id": self.transition_id,
            "subject_reference": self.subject_reference,
            "content_digest": self.content_digest,
            "target_tier": self.storage_tier,
        }
        expected_reference = (
            "tier-reference-" + canonical_sha256(identity)[:32]
        )
        if self.reference_id != expected_reference:
            raise CognitiveKernelContractError(
                "tier payload reference identity mismatch"
            )
        digest = canonical_sha256(self.material_record())
        if require_sha256(
            self.reference_sha256, "reference_sha256"
        ) != digest:
            raise CognitiveKernelContractError(
                "tier payload reference digest mismatch"
            )

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "TierPayloadReference":
        expected = {
            "schema_version",
            "reference_id",
            "transition_id",
            "lifecycle_decision_id",
            "subject_reference",
            "source_reference_id",
            "scope",
            "content_digest",
            "byte_length",
            "media_type",
            "sensitivity_class",
            "retention_class",
            "storage_tier",
            "published_at",
            "host_sealed",
            "reference_sha256",
        }
        if set(value) != expected:
            raise CognitiveKernelContractError(
                "tier payload reference record keys changed"
            )
        scope_value = value["scope"]
        if not isinstance(scope_value, Mapping):
            raise CognitiveKernelContractError("scope must be an object")
        reference = cls.create(
            schema_version=value["schema_version"],
            transition_id=value["transition_id"],
            lifecycle_decision_id=value["lifecycle_decision_id"],
            subject_reference=value["subject_reference"],
            source_reference_id=value["source_reference_id"],
            scope=ProductHostScope.create(
                product_id=scope_value.get("product_id"),
                host_instance_id=scope_value.get("host_instance_id"),
                schema_version=scope_value.get("schema_version"),
                encryption_domain=scope_value.get("encryption_domain"),
            ),
            content_digest=value["content_digest"],
            byte_length=value["byte_length"],
            media_type=value["media_type"],
            sensitivity_class=value["sensitivity_class"],
            retention_class=value["retention_class"],
            storage_tier=value["storage_tier"],
            published_at=value["published_at"],
            host_sealed=value["host_sealed"],
        )
        if reference.reference_id != value["reference_id"]:
            raise CognitiveKernelContractError(
                "stored tier payload reference identity changed"
            )
        if reference.reference_sha256 != value["reference_sha256"]:
            raise CognitiveKernelContractError(
                "stored tier payload reference digest changed"
            )
        return reference


@dataclass(frozen=True)
class TierTransitionIntent:
    """Immutable prepared intent bound to an approved lifecycle decision."""

    schema_version: str
    transition_id: str
    lifecycle_decision_id: str
    lifecycle_decision_sha256: str
    scope: ProductHostScope
    subject_reference: str
    content_digest: str
    source_tier: str
    target_tier: str
    source_reference_id: str
    target_reference_id: str
    byte_length: int
    media_type: str
    sensitivity_class: str
    retention_class: str
    prepared_at: str
    intent_sha256: str

    @classmethod
    def create(
        cls,
        *,
        lifecycle_decision_id: object,
        lifecycle_decision_sha256: object,
        scope: ProductHostScope,
        subject_reference: object,
        content_digest: object,
        source_tier: object,
        target_tier: object,
        source_reference_id: object,
        byte_length: object,
        media_type: object,
        sensitivity_class: object,
        retention_class: object,
        prepared_at: object,
        schema_version: object = TIER_TRANSITION_SCHEMA_VERSION,
    ) -> "TierTransitionIntent":
        scope.validate()
        normalized_schema = require_schema_version(schema_version)
        if normalized_schema != TIER_TRANSITION_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "tier transition intent schema version changed"
            )
        source = require_identifier(source_tier, "source_tier")
        target = require_identifier(target_tier, "target_tier")
        if source not in EXECUTABLE_SOURCE_TIERS:
            raise CognitiveKernelContractError(
                "source_tier is not executable"
            )
        if target not in EXECUTABLE_TARGET_TIERS:
            raise CognitiveKernelContractError(
                "target_tier is not executable"
            )
        if source == target:
            raise CognitiveKernelContractError(
                "tier transition intent may not be a no-op"
            )
        if isinstance(byte_length, bool) or not isinstance(byte_length, int):
            raise CognitiveKernelContractError("byte_length must be an integer")
        if byte_length < 1:
            raise CognitiveKernelContractError("byte_length must be positive")
        retention = require_identifier(retention_class, "retention_class")
        if retention not in RETENTION_CLASSES:
            raise CognitiveKernelContractError("retention_class is not approved")
        decision_id = require_identifier(
            lifecycle_decision_id, "lifecycle_decision_id"
        )
        subject = require_identifier(
            subject_reference, "subject_reference"
        )
        source_reference = require_identifier(
            source_reference_id, "source_reference_id"
        )
        transition_identity = {
            "schema_version": normalized_schema,
            "scope": scope.metadata_record(),
            "lifecycle_decision_id": decision_id,
            "source_reference_id": source_reference,
            "target_tier": target,
        }
        transition_id = (
            "tier-transition-" + canonical_sha256(transition_identity)[:32]
        )
        reference_identity = {
            "transition_id": transition_id,
            "subject_reference": subject,
            "content_digest": require_sha256(
                content_digest, "content_digest"
            ),
            "target_tier": target,
        }
        target_reference_id = (
            "tier-reference-" + canonical_sha256(reference_identity)[:32]
        )
        provisional = cls(
            schema_version=normalized_schema,
            transition_id=transition_id,
            lifecycle_decision_id=decision_id,
            lifecycle_decision_sha256=require_sha256(
                lifecycle_decision_sha256, "lifecycle_decision_sha256"
            ),
            scope=scope,
            subject_reference=subject,
            content_digest=reference_identity["content_digest"],
            source_tier=source,
            target_tier=target,
            source_reference_id=source_reference,
            target_reference_id=target_reference_id,
            byte_length=byte_length,
            media_type=require_text(media_type, "media_type", maximum=255),
            sensitivity_class=require_identifier(
                sensitivity_class, "sensitivity_class"
            ),
            retention_class=retention,
            prepared_at=normalize_timestamp(prepared_at, "prepared_at"),
            intent_sha256=_ZERO_SHA256,
        )
        digest = canonical_sha256(provisional.material_record())
        intent = cls(**{**provisional.__dict__, "intent_sha256": digest})
        intent.validate()
        return intent

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "lifecycle_decision_id": self.lifecycle_decision_id,
            "lifecycle_decision_sha256": self.lifecycle_decision_sha256,
            "scope": self.scope.metadata_record(),
            "subject_reference": self.subject_reference,
            "content_digest": self.content_digest,
            "source_tier": self.source_tier,
            "target_tier": self.target_tier,
            "source_reference_id": self.source_reference_id,
            "target_reference_id": self.target_reference_id,
            "byte_length": self.byte_length,
            "media_type": self.media_type,
            "sensitivity_class": self.sensitivity_class,
            "retention_class": self.retention_class,
            "prepared_at": self.prepared_at,
        }

    def record(self) -> dict[str, object]:
        self.validate()
        return {**self.material_record(), "intent_sha256": self.intent_sha256}

    def validate(self) -> None:
        if self.schema_version != TIER_TRANSITION_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "tier transition intent schema version changed"
            )
        self.scope.validate()
        require_identifier(self.transition_id, "transition_id")
        require_identifier(
            self.lifecycle_decision_id, "lifecycle_decision_id"
        )
        require_sha256(
            self.lifecycle_decision_sha256, "lifecycle_decision_sha256"
        )
        require_identifier(self.subject_reference, "subject_reference")
        require_sha256(self.content_digest, "content_digest")
        if self.source_tier not in EXECUTABLE_SOURCE_TIERS:
            raise CognitiveKernelContractError("source_tier is not executable")
        if self.target_tier not in EXECUTABLE_TARGET_TIERS:
            raise CognitiveKernelContractError("target_tier is not executable")
        if self.source_tier == self.target_tier:
            raise CognitiveKernelContractError(
                "tier transition intent may not be a no-op"
            )
        require_identifier(self.source_reference_id, "source_reference_id")
        require_identifier(self.target_reference_id, "target_reference_id")
        if isinstance(self.byte_length, bool) or not isinstance(
            self.byte_length, int
        ) or self.byte_length < 1:
            raise CognitiveKernelContractError(
                "byte_length must be a positive integer"
            )
        require_text(self.media_type, "media_type", maximum=255)
        require_identifier(self.sensitivity_class, "sensitivity_class")
        if self.retention_class not in RETENTION_CLASSES:
            raise CognitiveKernelContractError("retention_class is not approved")
        if normalize_timestamp(
            self.prepared_at, "prepared_at"
        ) != self.prepared_at:
            raise CognitiveKernelContractError("prepared_at is not canonical")
        identity = {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "lifecycle_decision_id": self.lifecycle_decision_id,
            "source_reference_id": self.source_reference_id,
            "target_tier": self.target_tier,
        }
        expected_transition = (
            "tier-transition-" + canonical_sha256(identity)[:32]
        )
        if self.transition_id != expected_transition:
            raise CognitiveKernelContractError(
                "tier transition intent identity mismatch"
            )
        reference_identity = {
            "transition_id": self.transition_id,
            "subject_reference": self.subject_reference,
            "content_digest": self.content_digest,
            "target_tier": self.target_tier,
        }
        expected_reference = (
            "tier-reference-" + canonical_sha256(reference_identity)[:32]
        )
        if self.target_reference_id != expected_reference:
            raise CognitiveKernelContractError(
                "tier transition target reference identity mismatch"
            )
        if require_sha256(
            self.intent_sha256, "intent_sha256"
        ) != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError(
                "tier transition intent digest mismatch"
            )

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "TierTransitionIntent":
        scope_value = value.get("scope")
        if not isinstance(scope_value, Mapping):
            raise CognitiveKernelContractError("scope must be an object")
        intent = cls.create(
            schema_version=value.get("schema_version"),
            lifecycle_decision_id=value.get("lifecycle_decision_id"),
            lifecycle_decision_sha256=value.get(
                "lifecycle_decision_sha256"
            ),
            scope=ProductHostScope.create(
                product_id=scope_value.get("product_id"),
                host_instance_id=scope_value.get("host_instance_id"),
                schema_version=scope_value.get("schema_version"),
                encryption_domain=scope_value.get("encryption_domain"),
            ),
            subject_reference=value.get("subject_reference"),
            content_digest=value.get("content_digest"),
            source_tier=value.get("source_tier"),
            target_tier=value.get("target_tier"),
            source_reference_id=value.get("source_reference_id"),
            byte_length=value.get("byte_length"),
            media_type=value.get("media_type"),
            sensitivity_class=value.get("sensitivity_class"),
            retention_class=value.get("retention_class"),
            prepared_at=value.get("prepared_at"),
        )
        for key in ("transition_id", "target_reference_id", "intent_sha256"):
            if getattr(intent, key) != value.get(key):
                raise CognitiveKernelContractError(
                    f"stored tier transition intent {key} changed"
                )
        return intent


@dataclass(frozen=True)
class TierTransitionReceipt:
    """Final publication receipt; the source payload is always preserved."""

    transition_id: str
    lifecycle_decision_id: str
    source_tier: str
    target_tier: str
    source_reference_id: str
    target_reference: TierPayloadReference
    source_preserved: bool
    physical_object_created: bool
    recovered_from_prepared_intent: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        transition_id: object,
        lifecycle_decision_id: object,
        source_tier: object,
        target_tier: object,
        source_reference_id: object,
        target_reference: TierPayloadReference,
        source_preserved: object = True,
        physical_object_created: object,
        recovered_from_prepared_intent: object = False,
    ) -> "TierTransitionReceipt":
        target_reference.validate()
        if source_preserved is not True:
            raise CognitiveKernelContractError(
                "P5.1d tier transitions must preserve the source payload"
            )
        if not isinstance(physical_object_created, bool):
            raise CognitiveKernelContractError(
                "physical_object_created must be boolean"
            )
        if not isinstance(recovered_from_prepared_intent, bool):
            raise CognitiveKernelContractError(
                "recovered_from_prepared_intent must be boolean"
            )
        material = {
            "transition_id": require_identifier(
                transition_id, "transition_id"
            ),
            "lifecycle_decision_id": require_identifier(
                lifecycle_decision_id, "lifecycle_decision_id"
            ),
            "source_tier": require_identifier(source_tier, "source_tier"),
            "target_tier": require_identifier(target_tier, "target_tier"),
            "source_reference_id": require_identifier(
                source_reference_id, "source_reference_id"
            ),
            "target_reference": target_reference.record(),
            "source_preserved": True,
            "physical_object_created": physical_object_created,
            "recovered_from_prepared_intent": (
                recovered_from_prepared_intent
            ),
        }
        receipt = cls(
            transition_id=material["transition_id"],
            lifecycle_decision_id=material["lifecycle_decision_id"],
            source_tier=material["source_tier"],
            target_tier=material["target_tier"],
            source_reference_id=material["source_reference_id"],
            target_reference=target_reference,
            source_preserved=True,
            physical_object_created=physical_object_created,
            recovered_from_prepared_intent=recovered_from_prepared_intent,
            receipt_sha256=canonical_sha256(material),
        )
        receipt.validate()
        return receipt

    def material_record(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "lifecycle_decision_id": self.lifecycle_decision_id,
            "source_tier": self.source_tier,
            "target_tier": self.target_tier,
            "source_reference_id": self.source_reference_id,
            "target_reference": self.target_reference.record(),
            "source_preserved": self.source_preserved,
            "physical_object_created": self.physical_object_created,
            "recovered_from_prepared_intent": (
                self.recovered_from_prepared_intent
            ),
        }

    def record(self) -> dict[str, object]:
        self.validate()
        return {**self.material_record(), "receipt_sha256": self.receipt_sha256}

    def validate(self) -> None:
        require_identifier(self.transition_id, "transition_id")
        require_identifier(
            self.lifecycle_decision_id, "lifecycle_decision_id"
        )
        if self.source_tier not in EXECUTABLE_SOURCE_TIERS:
            raise CognitiveKernelContractError("source_tier is not executable")
        if self.target_tier not in EXECUTABLE_TARGET_TIERS:
            raise CognitiveKernelContractError("target_tier is not executable")
        require_identifier(self.source_reference_id, "source_reference_id")
        self.target_reference.validate()
        if self.target_reference.transition_id != self.transition_id:
            raise CognitiveKernelContractError(
                "tier transition receipt target lineage changed"
            )
        if (
            self.target_reference.lifecycle_decision_id
            != self.lifecycle_decision_id
        ):
            raise CognitiveKernelContractError(
                "tier transition receipt decision lineage changed"
            )
        if self.target_reference.storage_tier != self.target_tier:
            raise CognitiveKernelContractError(
                "tier transition receipt target tier changed"
            )
        if self.source_preserved is not True:
            raise CognitiveKernelContractError(
                "P5.1d tier transitions must preserve the source payload"
            )
        if not isinstance(self.physical_object_created, bool):
            raise CognitiveKernelContractError(
                "physical_object_created must be boolean"
            )
        if not isinstance(self.recovered_from_prepared_intent, bool):
            raise CognitiveKernelContractError(
                "recovered_from_prepared_intent must be boolean"
            )
        if require_sha256(
            self.receipt_sha256, "receipt_sha256"
        ) != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError(
                "tier transition receipt digest mismatch"
            )

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "TierTransitionReceipt":
        target = value.get("target_reference")
        if not isinstance(target, Mapping):
            raise CognitiveKernelContractError(
                "target_reference must be an object"
            )
        receipt = cls.create(
            transition_id=value.get("transition_id"),
            lifecycle_decision_id=value.get("lifecycle_decision_id"),
            source_tier=value.get("source_tier"),
            target_tier=value.get("target_tier"),
            source_reference_id=value.get("source_reference_id"),
            target_reference=TierPayloadReference.from_record(target),
            source_preserved=value.get("source_preserved"),
            physical_object_created=value.get("physical_object_created"),
            recovered_from_prepared_intent=value.get(
                "recovered_from_prepared_intent"
            ),
        )
        if receipt.receipt_sha256 != value.get("receipt_sha256"):
            raise CognitiveKernelContractError(
                "stored tier transition receipt digest changed"
            )
        return receipt


@dataclass(frozen=True)
class TierTransitionInspectionRecord:
    sequence: int
    transition_id: str
    lifecycle_decision_id: str
    subject_reference: str
    content_digest: str
    source_tier: str
    target_tier: str
    source_reference_id: str
    target_reference_id: str
    state: str
    prepared_at: str
    published_at: str | None
    source_preserved: bool | None

    def record(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "transition_id": self.transition_id,
            "lifecycle_decision_id": self.lifecycle_decision_id,
            "subject_reference": self.subject_reference,
            "content_digest": self.content_digest,
            "source_tier": self.source_tier,
            "target_tier": self.target_tier,
            "source_reference_id": self.source_reference_id,
            "target_reference_id": self.target_reference_id,
            "state": self.state,
            "prepared_at": self.prepared_at,
            "published_at": self.published_at,
            "source_preserved": self.source_preserved,
        }


@dataclass(frozen=True)
class TierTransitionIntegrityReport:
    store_id: str
    intent_count: int
    published_count: int
    pending_count: int
    physical_object_count: int
    head_sha256: str
    valid: bool

    def record(self) -> dict[str, object]:
        return {
            "store_id": self.store_id,
            "intent_count": self.intent_count,
            "published_count": self.published_count,
            "pending_count": self.pending_count,
            "physical_object_count": self.physical_object_count,
            "head_sha256": self.head_sha256,
            "valid": self.valid,
        }
