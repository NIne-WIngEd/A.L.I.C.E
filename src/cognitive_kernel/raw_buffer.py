"""Metadata-only contracts for host-sealed raw-buffer payloads."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
    require_sha256,
    require_text,
)
from .contracts import ProductHostScope, RETENTION_CLASSES

RAW_BUFFER_SCHEMA_VERSION = "1.0.0"
RAW_BUFFER_STORAGE_TIER = "raw_buffer"
RAW_BUFFER_SENSITIVITY_CLASSES = frozenset(
    {"public", "internal", "private", "highly_sensitive"}
)


def raw_payload_object_id(scope: ProductHostScope, content_digest: object) -> str:
    scope.validate()
    digest = require_sha256(content_digest, "content_digest")
    return f"raw-object-{scope.scope_sha256()[:16]}-{digest[:32]}"


@dataclass(frozen=True)
class RawBufferReference:
    """One logical reference to opaque, host-sealed payload bytes."""

    schema_version: str
    reference_id: str
    logical_record_id: str
    payload_object_id: str
    scope: ProductHostScope
    content_digest: str
    byte_length: int
    media_type: str
    sensitivity_class: str
    retention_class: str
    storage_tier: str
    captured_at: str
    host_sealed: bool
    reference_sha256: str

    @classmethod
    def create(
        cls,
        *,
        logical_record_id: object,
        scope: ProductHostScope,
        content_digest: object,
        byte_length: object,
        media_type: object,
        sensitivity_class: object,
        retention_class: object,
        captured_at: object,
        host_sealed: object = True,
    ) -> "RawBufferReference":
        if isinstance(byte_length, bool) or not isinstance(byte_length, int):
            raise CognitiveKernelContractError("byte_length must be an integer")
        if byte_length < 1:
            raise CognitiveKernelContractError("byte_length must be positive")
        if host_sealed is not True:
            raise CognitiveKernelContractError(
                "raw-buffer payloads must be host-sealed opaque bytes"
            )
        scope.validate()
        sensitivity = require_identifier(
            sensitivity_class, "sensitivity_class"
        )
        if sensitivity not in RAW_BUFFER_SENSITIVITY_CLASSES:
            raise CognitiveKernelContractError(
                "sensitivity_class is not approved"
            )
        retention = require_identifier(retention_class, "retention_class")
        if retention not in RETENTION_CLASSES:
            raise CognitiveKernelContractError(
                "retention_class is not approved"
            )
        logical_id = require_identifier(logical_record_id, "logical_record_id")
        digest = require_sha256(content_digest, "content_digest")
        material = {
            "schema_version": RAW_BUFFER_SCHEMA_VERSION,
            "logical_record_id": logical_id,
            "payload_object_id": raw_payload_object_id(scope, digest),
            "scope": scope.metadata_record(),
            "content_digest": digest,
            "byte_length": byte_length,
            "media_type": require_text(media_type, "media_type", maximum=255),
            "sensitivity_class": sensitivity,
            "retention_class": retention,
            "storage_tier": RAW_BUFFER_STORAGE_TIER,
            "captured_at": normalize_timestamp(captured_at, "captured_at"),
            "host_sealed": True,
        }
        reference_digest = canonical_sha256(material)
        reference = cls(
            schema_version=RAW_BUFFER_SCHEMA_VERSION,
            reference_id=f"raw-reference-{reference_digest[:32]}",
            logical_record_id=logical_id,
            payload_object_id=material["payload_object_id"],
            scope=scope,
            content_digest=digest,
            byte_length=byte_length,
            media_type=material["media_type"],
            sensitivity_class=sensitivity,
            retention_class=retention,
            storage_tier=RAW_BUFFER_STORAGE_TIER,
            captured_at=material["captured_at"],
            host_sealed=True,
            reference_sha256=reference_digest,
        )
        reference.validate()
        return reference

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_record_id": self.logical_record_id,
            "payload_object_id": self.payload_object_id,
            "scope": self.scope.metadata_record(),
            "content_digest": self.content_digest,
            "byte_length": self.byte_length,
            "media_type": self.media_type,
            "sensitivity_class": self.sensitivity_class,
            "retention_class": self.retention_class,
            "storage_tier": self.storage_tier,
            "captured_at": self.captured_at,
            "host_sealed": self.host_sealed,
        }

    def validate(self) -> None:
        if self.schema_version != RAW_BUFFER_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "raw-buffer reference schema version changed"
            )
        self.scope.validate()
        if require_identifier(
            self.logical_record_id, "logical_record_id"
        ) != self.logical_record_id:
            raise CognitiveKernelContractError(
                "logical_record_id is not canonical"
            )
        if isinstance(self.byte_length, bool) or not isinstance(
            self.byte_length, int
        ) or self.byte_length < 1:
            raise CognitiveKernelContractError(
                "byte_length must be a positive integer"
            )
        require_text(self.media_type, "media_type", maximum=255)
        if self.sensitivity_class not in RAW_BUFFER_SENSITIVITY_CLASSES:
            raise CognitiveKernelContractError(
                "sensitivity_class is not approved"
            )
        if self.retention_class not in RETENTION_CLASSES:
            raise CognitiveKernelContractError(
                "retention_class is not approved"
            )
        if self.storage_tier != RAW_BUFFER_STORAGE_TIER:
            raise CognitiveKernelContractError(
                "raw-buffer reference storage tier changed"
            )
        if self.host_sealed is not True:
            raise CognitiveKernelContractError(
                "raw-buffer payloads must be host-sealed opaque bytes"
            )
        require_sha256(self.content_digest, "content_digest")
        if normalize_timestamp(
            self.captured_at, "captured_at"
        ) != self.captured_at:
            raise CognitiveKernelContractError(
                "captured_at is not canonical"
            )
        expected_object = raw_payload_object_id(
            self.scope, self.content_digest
        )
        if self.payload_object_id != expected_object:
            raise CognitiveKernelContractError(
                "raw-buffer object identity mismatch"
            )
        digest = canonical_sha256(self.material_record())
        if self.reference_id != f"raw-reference-{digest[:32]}":
            raise CognitiveKernelContractError(
                "raw-buffer reference identity mismatch"
            )
        if require_sha256(
            self.reference_sha256, "reference_sha256"
        ) != digest:
            raise CognitiveKernelContractError(
                "raw-buffer reference digest mismatch"
            )

    def record(self) -> dict[str, object]:
        self.validate()
        value = self.material_record()
        value["reference_id"] = self.reference_id
        value["reference_sha256"] = self.reference_sha256
        return value

    @classmethod
    def from_record(cls, value: dict[str, object]) -> "RawBufferReference":
        scope_value = value.get("scope")
        if not isinstance(scope_value, dict):
            raise CognitiveKernelContractError("scope must be an object")
        reference = cls(
            schema_version=require_text(
                value.get("schema_version"), "schema_version", maximum=32
            ),
            reference_id=require_identifier(
                value.get("reference_id"), "reference_id"
            ),
            logical_record_id=require_identifier(
                value.get("logical_record_id"), "logical_record_id"
            ),
            payload_object_id=require_identifier(
                value.get("payload_object_id"), "payload_object_id"
            ),
            scope=ProductHostScope.create(
                product_id=scope_value.get("product_id"),
                host_instance_id=scope_value.get("host_instance_id"),
                schema_version=scope_value.get("schema_version"),
                encryption_domain=scope_value.get("encryption_domain"),
            ),
            content_digest=require_sha256(
                value.get("content_digest"), "content_digest"
            ),
            byte_length=int(value.get("byte_length")),
            media_type=require_text(
                value.get("media_type"), "media_type", maximum=255
            ),
            sensitivity_class=require_identifier(
                value.get("sensitivity_class"), "sensitivity_class"
            ),
            retention_class=require_identifier(
                value.get("retention_class"), "retention_class"
            ),
            storage_tier=require_identifier(
                value.get("storage_tier"), "storage_tier"
            ),
            captured_at=normalize_timestamp(
                value.get("captured_at"), "captured_at"
            ),
            host_sealed=value.get("host_sealed") is True,
            reference_sha256=require_sha256(
                value.get("reference_sha256"), "reference_sha256"
            ),
        )
        reference.validate()
        return reference


@dataclass(frozen=True)
class RawBufferCaptureReceipt:
    reference: RawBufferReference
    physical_object_created: bool
    deduplicated: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        reference: RawBufferReference,
        physical_object_created: object,
        deduplicated: object,
    ) -> "RawBufferCaptureReceipt":
        reference.validate()
        if not isinstance(physical_object_created, bool):
            raise CognitiveKernelContractError(
                "physical_object_created must be boolean"
            )
        if not isinstance(deduplicated, bool):
            raise CognitiveKernelContractError("deduplicated must be boolean")
        if physical_object_created == deduplicated:
            raise CognitiveKernelContractError(
                "capture receipt creation and deduplication states conflict"
            )
        material = {
            "reference": reference.record(),
            "physical_object_created": physical_object_created,
            "deduplicated": deduplicated,
        }
        return cls(
            reference=reference,
            physical_object_created=physical_object_created,
            deduplicated=deduplicated,
            receipt_sha256=canonical_sha256(material),
        )

    def record(self) -> dict[str, object]:
        expected = self.create(
            reference=self.reference,
            physical_object_created=self.physical_object_created,
            deduplicated=self.deduplicated,
        )
        if self.receipt_sha256 != expected.receipt_sha256:
            raise CognitiveKernelContractError(
                "raw-buffer capture receipt digest mismatch"
            )
        return {
            "reference": self.reference.record(),
            "physical_object_created": self.physical_object_created,
            "deduplicated": self.deduplicated,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class PayloadStoreAccounting:
    logical_reference_count: int
    physical_object_count: int
    logical_bytes: int
    physical_bytes: int
    deduplicated_bytes: int

    def record(self) -> dict[str, int]:
        return {
            "logical_reference_count": self.logical_reference_count,
            "physical_object_count": self.physical_object_count,
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
            "deduplicated_bytes": self.deduplicated_bytes,
        }


@dataclass(frozen=True)
class PayloadStoreIntegrityReport:
    store_id: str
    logical_reference_count: int
    physical_object_count: int
    head_sha256: str
    valid: bool

    def record(self) -> dict[str, object]:
        return {
            "store_id": self.store_id,
            "logical_reference_count": self.logical_reference_count,
            "physical_object_count": self.physical_object_count,
            "head_sha256": self.head_sha256,
            "valid": self.valid,
        }
