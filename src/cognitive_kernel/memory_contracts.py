"""Memory M2 registration, envelope, and evidence contract artifacts.

These public records carry metadata and content references instead of private payload bytes. That artifact boundary does not limit full-memory runtime, research, prototypes, learning, or destination capability.
"""

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
    require_scope_identifier,
    require_sha256,
    require_text,
)
from .contracts import ProductHostScope, RETENTION_CLASSES

MEMORY_CONTRACT_SCHEMA_VERSION = "1.0.0"

AUTHORITY_ROLES = frozenset(
    {
        "evidence_authority",
        "claim_authority",
        "operational_workflow_state",
        "registered_projection",
        "cache",
        "replica",
        "archive",
        "candidate",
        "model_artifact",
        "evaluation_artifact",
    }
)

CAPABILITY_STATES = frozenset(
    {
        "destination",
        "research_active",
        "prototype_operational",
        "shadow_evaluated",
        "canary_enabled",
        "production_profile_enabled",
        "degraded",
        "superseded",
        "retired",
        "compatibility_only",
    }
)

EVIDENCE_RELATION_TYPES = frozenset(
    {
        "support",
        "contradiction",
        "correction",
        "context",
        "derivation",
        "evaluation",
        "deletion_cause",
    }
)


def _require_non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _optional_identifier(value: object | None, field: str) -> str | None:
    return require_identifier(value, field) if value is not None else None


def _optional_sha256(value: object | None, field: str) -> str | None:
    return require_sha256(value, field) if value is not None else None


def _optional_text(
    value: object | None,
    field: str,
    *,
    maximum: int = 512,
) -> str | None:
    return (
        require_text(value, field, maximum=maximum)
        if value is not None
        else None
    )


@dataclass(frozen=True)
class StoreRegistration:
    """Metadata registration for one authority, projection, replica, or service."""

    scope: ProductHostScope
    registration_id: str
    component_id: str
    authority_namespace_id: str
    authority_role: str
    capability_ids: tuple[str, ...]
    backend_type: str
    backend_version: str
    deployment_profile: str
    capability_state: str
    consistency_model: str
    availability_profile: str
    encryption_profile: str
    region_or_device_scope: str
    health_state: str
    performance_profile: str
    cost_profile: str
    deletion_endpoint: str
    rollback_endpoint: str
    backup_profile: str
    derives_from: tuple[str, ...]
    replicates: tuple[str, ...]
    synchronizes_with: tuple[str, ...]
    successor_component_id: str | None
    created_at: str
    registration_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        registration_id: object,
        component_id: object,
        authority_namespace_id: object,
        authority_role: object,
        capability_ids: tuple[object, ...] | list[object],
        backend_type: object,
        backend_version: object,
        deployment_profile: object,
        capability_state: object,
        consistency_model: object,
        availability_profile: object,
        encryption_profile: object,
        region_or_device_scope: object,
        health_state: object,
        performance_profile: object,
        cost_profile: object,
        deletion_endpoint: object,
        rollback_endpoint: object,
        backup_profile: object,
        derives_from: tuple[object, ...] | list[object] = (),
        replicates: tuple[object, ...] | list[object] = (),
        synchronizes_with: tuple[object, ...] | list[object] = (),
        successor_component_id: object | None = None,
        created_at: object,
    ) -> "StoreRegistration":
        draft = cls(
            scope=scope,
            registration_id=require_identifier(
                registration_id, "registration_id"
            ),
            component_id=require_identifier(component_id, "component_id"),
            authority_namespace_id=require_scope_identifier(
                authority_namespace_id, "authority_namespace_id"
            ),
            authority_role=require_identifier(
                authority_role, "authority_role"
            ),
            capability_ids=normalize_identifier_sequence(
                capability_ids, "capability_ids"
            ),
            backend_type=require_identifier(backend_type, "backend_type"),
            backend_version=require_schema_version(
                backend_version, "backend_version"
            ),
            deployment_profile=require_identifier(
                deployment_profile, "deployment_profile"
            ),
            capability_state=require_identifier(
                capability_state, "capability_state"
            ),
            consistency_model=require_identifier(
                consistency_model, "consistency_model"
            ),
            availability_profile=require_identifier(
                availability_profile, "availability_profile"
            ),
            encryption_profile=require_identifier(
                encryption_profile, "encryption_profile"
            ),
            region_or_device_scope=require_identifier(
                region_or_device_scope, "region_or_device_scope"
            ),
            health_state=require_identifier(health_state, "health_state"),
            performance_profile=require_identifier(
                performance_profile, "performance_profile"
            ),
            cost_profile=require_identifier(cost_profile, "cost_profile"),
            deletion_endpoint=require_text(
                deletion_endpoint, "deletion_endpoint", maximum=512
            ),
            rollback_endpoint=require_text(
                rollback_endpoint, "rollback_endpoint", maximum=512
            ),
            backup_profile=require_identifier(
                backup_profile, "backup_profile"
            ),
            derives_from=normalize_identifier_sequence(
                derives_from, "derives_from"
            ),
            replicates=normalize_identifier_sequence(
                replicates, "replicates"
            ),
            synchronizes_with=normalize_identifier_sequence(
                synchronizes_with, "synchronizes_with"
            ),
            successor_component_id=_optional_identifier(
                successor_component_id, "successor_component_id"
            ),
            created_at=normalize_timestamp(created_at, "created_at"),
            registration_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "registration_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        self.scope.validate()
        if require_identifier(
            self.registration_id, "registration_id"
        ) != self.registration_id:
            raise CognitiveKernelContractError(
                "registration_id is not canonical"
            )
        if require_identifier(
            self.component_id, "component_id"
        ) != self.component_id:
            raise CognitiveKernelContractError(
                "component_id is not canonical"
            )
        if require_scope_identifier(
            self.authority_namespace_id, "authority_namespace_id"
        ) != self.authority_namespace_id:
            raise CognitiveKernelContractError(
                "authority_namespace_id is not canonical"
            )
        if self.authority_role not in AUTHORITY_ROLES:
            raise CognitiveKernelContractError(
                "authority_role is not ratified"
            )
        if not self.capability_ids:
            raise CognitiveKernelContractError(
                "capability_ids may not be empty"
            )
        if normalize_identifier_sequence(
            self.capability_ids, "capability_ids"
        ) != self.capability_ids:
            raise CognitiveKernelContractError(
                "capability_ids are not canonical"
            )
        for value, field in (
            (self.backend_type, "backend_type"),
            (self.deployment_profile, "deployment_profile"),
            (self.capability_state, "capability_state"),
            (self.consistency_model, "consistency_model"),
            (self.availability_profile, "availability_profile"),
            (self.encryption_profile, "encryption_profile"),
            (self.region_or_device_scope, "region_or_device_scope"),
            (self.health_state, "health_state"),
            (self.performance_profile, "performance_profile"),
            (self.cost_profile, "cost_profile"),
            (self.backup_profile, "backup_profile"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if require_schema_version(
            self.backend_version, "backend_version"
        ) != self.backend_version:
            raise CognitiveKernelContractError(
                "backend_version is not canonical"
            )
        if self.capability_state not in CAPABILITY_STATES:
            raise CognitiveKernelContractError(
                "capability_state is not ratified"
            )
        for value, field in (
            (self.deletion_endpoint, "deletion_endpoint"),
            (self.rollback_endpoint, "rollback_endpoint"),
        ):
            if require_text(value, field, maximum=512) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        for values, field in (
            (self.derives_from, "derives_from"),
            (self.replicates, "replicates"),
            (self.synchronizes_with, "synchronizes_with"),
        ):
            if normalize_identifier_sequence(values, field) != values:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if _optional_identifier(
            self.successor_component_id, "successor_component_id"
        ) != self.successor_component_id:
            raise CognitiveKernelContractError(
                "successor_component_id is not canonical"
            )
        if normalize_timestamp(
            self.created_at, "created_at"
        ) != self.created_at:
            raise CognitiveKernelContractError(
                "created_at is not canonical"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "scope": self.scope.metadata_record(),
            "registration_id": self.registration_id,
            "component_id": self.component_id,
            "authority_namespace_id": self.authority_namespace_id,
            "authority_role": self.authority_role,
            "capability_ids": list(self.capability_ids),
            "backend_type": self.backend_type,
            "backend_version": self.backend_version,
            "deployment_profile": self.deployment_profile,
            "capability_state": self.capability_state,
            "consistency_model": self.consistency_model,
            "availability_profile": self.availability_profile,
            "encryption_profile": self.encryption_profile,
            "region_or_device_scope": self.region_or_device_scope,
            "health_state": self.health_state,
            "performance_profile": self.performance_profile,
            "cost_profile": self.cost_profile,
            "deletion_endpoint": self.deletion_endpoint,
            "rollback_endpoint": self.rollback_endpoint,
            "backup_profile": self.backup_profile,
            "derives_from": list(self.derives_from),
            "replicates": list(self.replicates),
            "synchronizes_with": list(self.synchronizes_with),
            "successor_component_id": self.successor_component_id,
            "created_at": self.created_at,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["registration_sha256"] = self.registration_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(
            self.registration_sha256, "registration_sha256"
        )
        if canonical_sha256(
            self.material_record()
        ) != self.registration_sha256:
            raise CognitiveKernelContractError(
                "store registration digest mismatch"
            )


@dataclass(frozen=True)
class MemoryUnitEnvelope:
    """Common metadata envelope for authority and registered derivative records."""

    scope: ProductHostScope
    record_id: str
    record_type: str
    authority_namespace_id: str
    host_or_cluster_id: str
    authority_role: str
    deployment_profile: str
    created_at: str
    valid_from: str
    valid_to: str | None
    transaction_time: str
    logical_clock: int
    causal_parents: tuple[str, ...]
    source_records: tuple[str, ...]
    generation: int
    state: str
    data_classification: str
    retention_class: str
    deletion_state: str
    provenance_digest: str
    content_digest: str
    writer: str
    workflow_or_request_id: str
    idempotency_namespace: str
    idempotency_key: str
    supersedes: tuple[str, ...]
    superseded_by: tuple[str, ...]
    rollback_reference: str | None
    envelope_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        record_id: object,
        record_type: object,
        authority_namespace_id: object,
        host_or_cluster_id: object,
        authority_role: object,
        deployment_profile: object,
        created_at: object,
        valid_from: object,
        valid_to: object | None,
        transaction_time: object,
        logical_clock: object,
        causal_parents: tuple[object, ...] | list[object] = (),
        source_records: tuple[object, ...] | list[object] = (),
        generation: object,
        state: object,
        data_classification: object,
        retention_class: object,
        deletion_state: object,
        provenance_digest: object,
        content_digest: object,
        writer: object,
        workflow_or_request_id: object,
        idempotency_namespace: object,
        idempotency_key: object,
        supersedes: tuple[object, ...] | list[object] = (),
        superseded_by: tuple[object, ...] | list[object] = (),
        rollback_reference: object | None = None,
    ) -> "MemoryUnitEnvelope":
        draft = cls(
            scope=scope,
            record_id=require_identifier(record_id, "record_id"),
            record_type=require_identifier(record_type, "record_type"),
            authority_namespace_id=require_scope_identifier(
                authority_namespace_id, "authority_namespace_id"
            ),
            host_or_cluster_id=require_scope_identifier(
                host_or_cluster_id, "host_or_cluster_id"
            ),
            authority_role=require_identifier(
                authority_role, "authority_role"
            ),
            deployment_profile=require_identifier(
                deployment_profile, "deployment_profile"
            ),
            created_at=normalize_timestamp(created_at, "created_at"),
            valid_from=normalize_timestamp(valid_from, "valid_from"),
            valid_to=(
                normalize_timestamp(valid_to, "valid_to")
                if valid_to is not None
                else None
            ),
            transaction_time=normalize_timestamp(
                transaction_time, "transaction_time"
            ),
            logical_clock=_require_non_negative_integer(
                logical_clock, "logical_clock"
            ),
            causal_parents=normalize_identifier_sequence(
                causal_parents, "causal_parents"
            ),
            source_records=normalize_identifier_sequence(
                source_records, "source_records"
            ),
            generation=_require_non_negative_integer(
                generation, "generation"
            ),
            state=require_identifier(state, "state"),
            data_classification=require_identifier(
                data_classification, "data_classification"
            ),
            retention_class=require_identifier(
                retention_class, "retention_class"
            ),
            deletion_state=require_identifier(
                deletion_state, "deletion_state"
            ),
            provenance_digest=require_sha256(
                provenance_digest, "provenance_digest"
            ),
            content_digest=require_sha256(
                content_digest, "content_digest"
            ),
            writer=require_identifier(writer, "writer"),
            workflow_or_request_id=require_identifier(
                workflow_or_request_id, "workflow_or_request_id"
            ),
            idempotency_namespace=require_scope_identifier(
                idempotency_namespace, "idempotency_namespace"
            ),
            idempotency_key=require_identifier(
                idempotency_key, "idempotency_key"
            ),
            supersedes=normalize_identifier_sequence(
                supersedes, "supersedes"
            ),
            superseded_by=normalize_identifier_sequence(
                superseded_by, "superseded_by"
            ),
            rollback_reference=_optional_identifier(
                rollback_reference, "rollback_reference"
            ),
            envelope_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "envelope_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        self.scope.validate()
        for value, field in (
            (self.record_id, "record_id"),
            (self.record_type, "record_type"),
            (self.deployment_profile, "deployment_profile"),
            (self.state, "state"),
            (self.data_classification, "data_classification"),
            (self.retention_class, "retention_class"),
            (self.deletion_state, "deletion_state"),
            (self.writer, "writer"),
            (self.workflow_or_request_id, "workflow_or_request_id"),
            (self.idempotency_key, "idempotency_key"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        for value, field in (
            (self.authority_namespace_id, "authority_namespace_id"),
            (self.host_or_cluster_id, "host_or_cluster_id"),
            (self.idempotency_namespace, "idempotency_namespace"),
        ):
            if require_scope_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.authority_role not in AUTHORITY_ROLES:
            raise CognitiveKernelContractError(
                "authority_role is not ratified"
            )
        if self.retention_class not in RETENTION_CLASSES:
            raise CognitiveKernelContractError(
                "retention_class is not approved"
            )
        for value, field in (
            (self.created_at, "created_at"),
            (self.valid_from, "valid_from"),
            (self.transaction_time, "transaction_time"),
        ):
            if normalize_timestamp(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.valid_to is not None:
            if normalize_timestamp(
                self.valid_to, "valid_to"
            ) != self.valid_to:
                raise CognitiveKernelContractError(
                    "valid_to is not canonical"
                )
            if self.valid_to < self.valid_from:
                raise CognitiveKernelContractError(
                    "valid_to precedes valid_from"
                )
        _require_non_negative_integer(
            self.logical_clock, "logical_clock"
        )
        _require_non_negative_integer(self.generation, "generation")
        for values, field in (
            (self.causal_parents, "causal_parents"),
            (self.source_records, "source_records"),
            (self.supersedes, "supersedes"),
            (self.superseded_by, "superseded_by"),
        ):
            if normalize_identifier_sequence(values, field) != values:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        require_sha256(self.provenance_digest, "provenance_digest")
        require_sha256(self.content_digest, "content_digest")
        if _optional_identifier(
            self.rollback_reference, "rollback_reference"
        ) != self.rollback_reference:
            raise CognitiveKernelContractError(
                "rollback_reference is not canonical"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "scope": self.scope.metadata_record(),
            "record_id": self.record_id,
            "record_type": self.record_type,
            "authority_namespace_id": self.authority_namespace_id,
            "host_or_cluster_id": self.host_or_cluster_id,
            "authority_role": self.authority_role,
            "deployment_profile": self.deployment_profile,
            "created_at": self.created_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "transaction_time": self.transaction_time,
            "logical_clock": self.logical_clock,
            "causal_parents": list(self.causal_parents),
            "source_records": list(self.source_records),
            "generation": self.generation,
            "state": self.state,
            "data_classification": self.data_classification,
            "retention_class": self.retention_class,
            "deletion_state": self.deletion_state,
            "provenance_digest": self.provenance_digest,
            "content_digest": self.content_digest,
            "writer": self.writer,
            "workflow_or_request_id": self.workflow_or_request_id,
            "idempotency_namespace": self.idempotency_namespace,
            "idempotency_key": self.idempotency_key,
            "supersedes": list(self.supersedes),
            "superseded_by": list(self.superseded_by),
            "rollback_reference": self.rollback_reference,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["envelope_sha256"] = self.envelope_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.envelope_sha256, "envelope_sha256")
        if canonical_sha256(
            self.material_record()
        ) != self.envelope_sha256:
            raise CognitiveKernelContractError(
                "memory unit envelope digest mismatch"
            )


@dataclass(frozen=True)
class EvidenceBinding:
    """Evidence-relation contract with payload-free public references."""

    scope: ProductHostScope
    binding_id: str
    authority_namespace_id: str
    evidence_record_id: str
    target_record_id: str
    target_record_type: str
    relation_type: str
    responsible_component: str
    workflow_or_request_id: str
    created_at: str
    confidence: float | None
    notes_digest: str | None
    binding_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        binding_id: object,
        authority_namespace_id: object,
        evidence_record_id: object,
        target_record_id: object,
        target_record_type: object,
        relation_type: object,
        responsible_component: object,
        workflow_or_request_id: object,
        created_at: object,
        confidence: object | None = None,
        notes_digest: object | None = None,
    ) -> "EvidenceBinding":
        draft = cls(
            scope=scope,
            binding_id=require_identifier(binding_id, "binding_id"),
            authority_namespace_id=require_scope_identifier(
                authority_namespace_id, "authority_namespace_id"
            ),
            evidence_record_id=require_identifier(
                evidence_record_id, "evidence_record_id"
            ),
            target_record_id=require_identifier(
                target_record_id, "target_record_id"
            ),
            target_record_type=require_identifier(
                target_record_type, "target_record_type"
            ),
            relation_type=require_identifier(
                relation_type, "relation_type"
            ),
            responsible_component=require_identifier(
                responsible_component, "responsible_component"
            ),
            workflow_or_request_id=require_identifier(
                workflow_or_request_id, "workflow_or_request_id"
            ),
            created_at=normalize_timestamp(created_at, "created_at"),
            confidence=require_confidence(confidence),
            notes_digest=_optional_sha256(
                notes_digest, "notes_digest"
            ),
            binding_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "binding_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        self.scope.validate()
        for value, field in (
            (self.binding_id, "binding_id"),
            (self.evidence_record_id, "evidence_record_id"),
            (self.target_record_id, "target_record_id"),
            (self.target_record_type, "target_record_type"),
            (self.relation_type, "relation_type"),
            (self.responsible_component, "responsible_component"),
            (self.workflow_or_request_id, "workflow_or_request_id"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if require_scope_identifier(
            self.authority_namespace_id, "authority_namespace_id"
        ) != self.authority_namespace_id:
            raise CognitiveKernelContractError(
                "authority_namespace_id is not canonical"
            )
        if self.relation_type not in EVIDENCE_RELATION_TYPES:
            raise CognitiveKernelContractError(
                "relation_type is not ratified"
            )
        if normalize_timestamp(
            self.created_at, "created_at"
        ) != self.created_at:
            raise CognitiveKernelContractError(
                "created_at is not canonical"
            )
        require_confidence(self.confidence)
        if _optional_sha256(
            self.notes_digest, "notes_digest"
        ) != self.notes_digest:
            raise CognitiveKernelContractError(
                "notes_digest is not canonical"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "scope": self.scope.metadata_record(),
            "binding_id": self.binding_id,
            "authority_namespace_id": self.authority_namespace_id,
            "evidence_record_id": self.evidence_record_id,
            "target_record_id": self.target_record_id,
            "target_record_type": self.target_record_type,
            "relation_type": self.relation_type,
            "responsible_component": self.responsible_component,
            "workflow_or_request_id": self.workflow_or_request_id,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "notes_digest": self.notes_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["binding_sha256"] = self.binding_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.binding_sha256, "binding_sha256")
        if canonical_sha256(
            self.material_record()
        ) != self.binding_sha256:
            raise CognitiveKernelContractError(
                "evidence binding digest mismatch"
            )
