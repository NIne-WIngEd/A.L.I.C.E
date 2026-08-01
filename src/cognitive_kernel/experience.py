"""Tamper-evident metadata-only Experience Event envelope."""

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

_EXPERIENCE_EVENT_KEYS = {
    "schema_version",
    "event_id",
    "event_type",
    "scope",
    "occurred_at",
    "content_digest",
    "provenance",
    "retention_class",
    "storage_tier",
    "deletion_lineage",
    "parent_event_ids",
    "outcome_reference_ids",
    "policy_bindings",
    "payload_reference",
    "event_sha256",
}
_SCOPE_KEYS = {
    "product_id",
    "host_instance_id",
    "schema_version",
    "encryption_domain",
}
_PROVENANCE_KEYS = {
    "provenance_type",
    "source_reference_ids",
    "derivation_activity_id",
    "responsible_component",
    "model_id",
    "confidence",
    "supersedes_record_ids",
}


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CognitiveKernelContractError(
            f"{field} keys are invalid; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class ExperienceEvent:
    """One append-only event envelope without raw payload content."""

    schema_version: str
    event_id: str
    event_type: str
    scope: ProductHostScope
    occurred_at: str
    content_digest: str
    provenance: ProvenanceReference
    retention_class: str
    storage_tier: str
    deletion_lineage: tuple[str, ...]
    parent_event_ids: tuple[str, ...]
    outcome_reference_ids: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    payload_reference: str | None
    event_sha256: str

    @classmethod
    def create(
        cls,
        *,
        event_type: object,
        scope: ProductHostScope,
        occurred_at: object,
        content_digest: object,
        provenance: ProvenanceReference,
        retention_class: object,
        storage_tier: object,
        deletion_lineage: tuple[object, ...] | list[object] = (),
        parent_event_ids: tuple[object, ...] | list[object] = (),
        outcome_reference_ids: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        payload_reference: object | None = None,
        schema_version: object = "1.0.0",
    ) -> "ExperienceEvent":
        draft = cls(
            schema_version=require_schema_version(schema_version),
            event_id="experience-pending",
            event_type=require_identifier(event_type, "event_type"),
            scope=scope,
            occurred_at=normalize_timestamp(occurred_at, "occurred_at"),
            content_digest=require_sha256(
                content_digest, "content_digest"
            ),
            provenance=provenance,
            retention_class=require_identifier(
                retention_class, "retention_class"
            ),
            storage_tier=require_identifier(
                storage_tier, "storage_tier"
            ),
            deletion_lineage=normalize_identifier_sequence(
                deletion_lineage, "deletion_lineage"
            ),
            parent_event_ids=normalize_identifier_sequence(
                parent_event_ids, "parent_event_ids"
            ),
            outcome_reference_ids=normalize_identifier_sequence(
                outcome_reference_ids, "outcome_reference_ids"
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            payload_reference=(
                require_identifier(
                    payload_reference, "payload_reference"
                )
                if payload_reference is not None
                else None
            ),
            event_sha256="0" * 64,
        )
        draft._validate_material()
        digest = canonical_sha256(draft.material_record())
        event = cls(
            schema_version=draft.schema_version,
            event_id=f"experience-{digest[:32]}",
            event_type=draft.event_type,
            scope=draft.scope,
            occurred_at=draft.occurred_at,
            content_digest=draft.content_digest,
            provenance=draft.provenance,
            retention_class=draft.retention_class,
            storage_tier=draft.storage_tier,
            deletion_lineage=draft.deletion_lineage,
            parent_event_ids=draft.parent_event_ids,
            outcome_reference_ids=draft.outcome_reference_ids,
            policy_bindings=draft.policy_bindings,
            payload_reference=draft.payload_reference,
            event_sha256=digest,
        )
        event.validate()
        return event

    def _validate_material(self) -> None:
        if require_schema_version(self.schema_version) != self.schema_version:
            raise CognitiveKernelContractError(
                "schema_version is not canonical"
            )
        if require_identifier(
            self.event_type, "event_type"
        ) != self.event_type:
            raise CognitiveKernelContractError(
                "event_type is not canonical"
            )
        self.scope.validate()
        if normalize_timestamp(
            self.occurred_at, "occurred_at"
        ) != self.occurred_at:
            raise CognitiveKernelContractError(
                "occurred_at is not canonical"
            )
        if require_sha256(
            self.content_digest, "content_digest"
        ) != self.content_digest:
            raise CognitiveKernelContractError(
                "content_digest is not canonical"
            )
        self.provenance.validate()

        retention_class = require_identifier(
            self.retention_class, "retention_class"
        )
        if retention_class != self.retention_class:
            raise CognitiveKernelContractError(
                "retention_class is not canonical"
            )
        if retention_class not in RETENTION_CLASSES:
            raise CognitiveKernelContractError(
                "retention_class is not approved"
            )
        storage_tier = require_identifier(
            self.storage_tier, "storage_tier"
        )
        if storage_tier != self.storage_tier:
            raise CognitiveKernelContractError(
                "storage_tier is not canonical"
            )
        if storage_tier not in STORAGE_TIERS:
            raise CognitiveKernelContractError(
                "storage_tier is not approved"
            )

        if normalize_identifier_sequence(
            self.deletion_lineage, "deletion_lineage"
        ) != self.deletion_lineage:
            raise CognitiveKernelContractError(
                "deletion_lineage is not canonical"
            )
        if normalize_identifier_sequence(
            self.parent_event_ids, "parent_event_ids"
        ) != self.parent_event_ids:
            raise CognitiveKernelContractError(
                "parent_event_ids is not canonical"
            )
        if normalize_identifier_sequence(
            self.outcome_reference_ids, "outcome_reference_ids"
        ) != self.outcome_reference_ids:
            raise CognitiveKernelContractError(
                "outcome_reference_ids is not canonical"
            )
        if normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        ) != self.policy_bindings:
            raise CognitiveKernelContractError(
                "policy_bindings is not canonical"
            )
        if self.payload_reference is not None:
            if require_identifier(
                self.payload_reference, "payload_reference"
            ) != self.payload_reference:
                raise CognitiveKernelContractError(
                    "payload_reference is not canonical"
                )
        if storage_tier == "deleted" and self.payload_reference is not None:
            raise CognitiveKernelContractError(
                "deleted events may not retain a payload reference"
            )
        if (
            retention_class == "quarantine"
            and storage_tier != "quarantine"
        ):
            raise CognitiveKernelContractError(
                "quarantine retention must use the quarantine tier"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "scope": self.scope.metadata_record(),
            "occurred_at": self.occurred_at,
            "content_digest": self.content_digest,
            "provenance": self.provenance.metadata_record(),
            "retention_class": self.retention_class,
            "storage_tier": self.storage_tier,
            "deletion_lineage": list(self.deletion_lineage),
            "parent_event_ids": list(self.parent_event_ids),
            "outcome_reference_ids": list(
                self.outcome_reference_ids
            ),
            "policy_bindings": list(self.policy_bindings),
            "payload_reference": self.payload_reference,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["event_id"] = self.event_id
        record["event_sha256"] = self.event_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        expected_digest = canonical_sha256(self.material_record())
        require_sha256(self.event_sha256, "event_sha256")
        if self.event_sha256 != expected_digest:
            raise CognitiveKernelContractError(
                "experience event digest mismatch"
            )
        if self.event_id != f"experience-{expected_digest[:32]}":
            raise CognitiveKernelContractError(
                "experience event identity mismatch"
            )

    @classmethod
    def from_metadata_record(
        cls,
        value: Mapping[str, object],
    ) -> "ExperienceEvent":
        if not isinstance(value, Mapping):
            raise CognitiveKernelContractError(
                "experience event record must be a mapping"
            )
        _exact_keys(value, _EXPERIENCE_EVENT_KEYS, field="event")

        scope_value = value["scope"]
        if not isinstance(scope_value, Mapping):
            raise CognitiveKernelContractError(
                "scope must be a mapping"
            )
        _exact_keys(scope_value, _SCOPE_KEYS, field="scope")
        scope = ProductHostScope.create(
            product_id=scope_value["product_id"],
            host_instance_id=scope_value["host_instance_id"],
            schema_version=scope_value["schema_version"],
            encryption_domain=scope_value["encryption_domain"],
        )

        provenance_value = value["provenance"]
        if not isinstance(provenance_value, Mapping):
            raise CognitiveKernelContractError(
                "provenance must be a mapping"
            )
        _exact_keys(
            provenance_value,
            _PROVENANCE_KEYS,
            field="provenance",
        )
        provenance = ProvenanceReference.create(
            provenance_type=provenance_value["provenance_type"],
            source_reference_ids=tuple(
                provenance_value["source_reference_ids"]  # type: ignore[arg-type]
            ),
            derivation_activity_id=provenance_value[
                "derivation_activity_id"
            ],
            responsible_component=provenance_value[
                "responsible_component"
            ],
            model_id=provenance_value["model_id"],
            confidence=provenance_value["confidence"],
            supersedes_record_ids=tuple(
                provenance_value["supersedes_record_ids"]  # type: ignore[arg-type]
            ),
        )

        event = cls(
            schema_version=require_schema_version(
                value["schema_version"]
            ),
            event_id=require_identifier(
                value["event_id"], "event_id"
            ),
            event_type=require_identifier(
                value["event_type"], "event_type"
            ),
            scope=scope,
            occurred_at=normalize_timestamp(
                value["occurred_at"], "occurred_at"
            ),
            content_digest=require_sha256(
                value["content_digest"], "content_digest"
            ),
            provenance=provenance,
            retention_class=require_identifier(
                value["retention_class"], "retention_class"
            ),
            storage_tier=require_identifier(
                value["storage_tier"], "storage_tier"
            ),
            deletion_lineage=normalize_identifier_sequence(
                tuple(value["deletion_lineage"]),  # type: ignore[arg-type]
                "deletion_lineage",
            ),
            parent_event_ids=normalize_identifier_sequence(
                tuple(value["parent_event_ids"]),  # type: ignore[arg-type]
                "parent_event_ids",
            ),
            outcome_reference_ids=normalize_identifier_sequence(
                tuple(value["outcome_reference_ids"]),  # type: ignore[arg-type]
                "outcome_reference_ids",
            ),
            policy_bindings=normalize_identifier_sequence(
                tuple(value["policy_bindings"]),  # type: ignore[arg-type]
                "policy_bindings",
            ),
            payload_reference=(
                require_identifier(
                    value["payload_reference"],
                    "payload_reference",
                )
                if value["payload_reference"] is not None
                else None
            ),
            event_sha256=require_sha256(
                value["event_sha256"], "event_sha256"
            ),
        )
        event.validate()
        return event
