"""Memory M2.1 claim identity, version, and current-projection contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_confidence,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope

CLAIM_CONTRACT_SCHEMA_VERSION = "1.0.0"

STANDARD_CANONICAL_VALUE_TYPES = frozenset(
    {
        "null",
        "boolean",
        "integer",
        "number",
        "text",
        "identifier",
        "timestamp",
        "sha256",
        "list",
        "map",
    }
)

CLAIM_ADJUDICATION_STATES = frozenset(
    {
        "accepted",
        "revised",
        "superseded",
        "disputed",
        "quarantined",
        "merged",
        "split",
        "rejected",
    }
)

CLAIM_VALIDITY_STATES = frozenset(
    {
        "current",
        "historical",
        "future",
        "expired",
        "unknown",
    }
)

CLAIM_CONFLICT_STATES = frozenset(
    {
        "none",
        "open",
        "resolved",
        "quarantined",
    }
)

CLAIM_DELETION_STATES = frozenset(
    {
        "active",
        "pending",
        "deleted",
        "cryptographically_erased",
        "technically_limited",
    }
)


def _require_positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CognitiveKernelContractError(
            f"{field} must be a positive integer"
        )
    return value


def _require_non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _optional_non_negative_integer(
    value: object | None,
    field: str,
) -> int | None:
    if value is None:
        return None
    return _require_non_negative_integer(value, field)


def _optional_identifier(value: object | None, field: str) -> str | None:
    if value is None:
        return None
    return require_identifier(value, field)


def _normalize_sorted_identifiers(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    return tuple(sorted(normalize_identifier_sequence(values, field)))


def _require_same_scope(
    first: ProductHostScope,
    second: ProductHostScope,
    *,
    field: str,
) -> None:
    first.validate()
    second.validate()
    if first.metadata_record() != second.metadata_record():
        raise CognitiveKernelContractError(
            f"{field} crosses product, host, schema, or encryption scope"
        )


def _json_value(canonical_json: str) -> object:
    try:
        return json.loads(canonical_json)
    except json.JSONDecodeError as exc:
        raise CognitiveKernelContractError(
            "canonical_json is not valid JSON"
        ) from exc


def _validate_type_tagged_value(type_tag: str, value: object) -> None:
    if type_tag == "null" and value is not None:
        raise CognitiveKernelContractError("null canonical value must be null")
    if type_tag == "boolean" and not isinstance(value, bool):
        raise CognitiveKernelContractError(
            "boolean canonical value must be boolean"
        )
    if type_tag == "integer" and (
        isinstance(value, bool) or not isinstance(value, int)
    ):
        raise CognitiveKernelContractError(
            "integer canonical value must be an integer"
        )
    if type_tag == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CognitiveKernelContractError(
                "number canonical value must be numeric"
            )
        if not math.isfinite(float(value)):
            raise CognitiveKernelContractError(
                "number canonical value must be finite"
            )
    if type_tag == "text" and not isinstance(value, str):
        raise CognitiveKernelContractError(
            "text canonical value must be text"
        )
    if type_tag == "identifier":
        if require_identifier(value, "canonical value") != value:
            raise CognitiveKernelContractError(
                "identifier canonical value is not canonical"
            )
    if type_tag == "timestamp":
        if normalize_timestamp(value, "canonical value") != value:
            raise CognitiveKernelContractError(
                "timestamp canonical value is not canonical"
            )
    if type_tag == "sha256":
        if require_sha256(value, "canonical value") != value:
            raise CognitiveKernelContractError(
                "SHA-256 canonical value is not canonical"
            )
    if type_tag == "list" and not isinstance(value, list):
        raise CognitiveKernelContractError(
            "list canonical value must be a JSON list"
        )
    if type_tag == "map":
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise CognitiveKernelContractError(
                "map canonical value must be a JSON object with text keys"
            )


@dataclass(frozen=True)
class CanonicalTaggedValue:
    """Immutable type-tagged canonical JSON used in semantic identity."""

    type_tag: str
    canonical_json: str
    value_sha256: str

    @classmethod
    def create(
        cls,
        *,
        type_tag: object,
        value: object,
    ) -> "CanonicalTaggedValue":
        normalized_tag = require_identifier(type_tag, "type_tag")
        _validate_type_tagged_value(normalized_tag, value)
        canonical_text = canonical_json_bytes(value).decode("utf-8")
        draft = cls(
            type_tag=normalized_tag,
            canonical_json=canonical_text,
            value_sha256="0" * 64,
        )
        result = cls(
            type_tag=draft.type_tag,
            canonical_json=draft.canonical_json,
            value_sha256=canonical_sha256(draft.material_record()),
        )
        result.validate()
        return result

    def value(self) -> object:
        self.validate()
        return _json_value(self.canonical_json)

    def material_record(self) -> dict[str, object]:
        normalized_tag = require_identifier(self.type_tag, "type_tag")
        if normalized_tag != self.type_tag:
            raise CognitiveKernelContractError("type_tag is not canonical")
        value = _json_value(self.canonical_json)
        canonical_text = canonical_json_bytes(value).decode("utf-8")
        if canonical_text != self.canonical_json:
            raise CognitiveKernelContractError(
                "canonical_json is not canonical JSON"
            )
        _validate_type_tagged_value(self.type_tag, value)
        return {
            "type_tag": self.type_tag,
            "value": value,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["value_sha256"] = self.value_sha256
        return record

    def validate(self) -> None:
        require_sha256(self.value_sha256, "value_sha256")
        if canonical_sha256(self.material_record()) != self.value_sha256:
            raise CognitiveKernelContractError(
                "canonical tagged value digest mismatch"
            )


@dataclass(frozen=True)
class ClaimQualifier:
    """One sorted semantic qualifier bound into a claim digest."""

    key: str
    value: CanonicalTaggedValue
    qualifier_sha256: str

    @classmethod
    def create(
        cls,
        *,
        key: object,
        value: CanonicalTaggedValue,
    ) -> "ClaimQualifier":
        draft = cls(
            key=require_identifier(key, "qualifier key"),
            value=value,
            qualifier_sha256="0" * 64,
        )
        draft._validate_material()
        result = cls(
            key=draft.key,
            value=draft.value,
            qualifier_sha256=canonical_sha256(draft.material_record()),
        )
        result.validate()
        return result

    def _validate_material(self) -> None:
        if require_identifier(self.key, "qualifier key") != self.key:
            raise CognitiveKernelContractError(
                "qualifier key is not canonical"
            )
        self.value.validate()

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "key": self.key,
            "value": self.value.metadata_record(),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["qualifier_sha256"] = self.qualifier_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.qualifier_sha256, "qualifier_sha256")
        if canonical_sha256(self.material_record()) != self.qualifier_sha256:
            raise CognitiveKernelContractError(
                "claim qualifier digest mismatch"
            )


def normalize_claim_qualifiers(
    values: Iterable[ClaimQualifier],
    field: str = "qualifiers",
) -> tuple[ClaimQualifier, ...]:
    if isinstance(values, (str, bytes)):
        raise CognitiveKernelContractError(f"{field} must be a sequence")
    normalized: list[ClaimQualifier] = []
    keys: set[str] = set()
    for value in values:
        if not isinstance(value, ClaimQualifier):
            raise CognitiveKernelContractError(
                f"{field} must contain ClaimQualifier records"
            )
        value.validate()
        if value.key in keys:
            raise CognitiveKernelContractError(
                f"{field} may not contain duplicate keys"
            )
        keys.add(value.key)
        normalized.append(value)
    return tuple(sorted(normalized, key=lambda item: item.key))


@dataclass(frozen=True)
class ClaimIdentity:
    """Backend-neutral semantic identity for one assertion family."""

    envelope: MemoryUnitEnvelope
    claim_id: str
    canonical_subject: CanonicalTaggedValue
    canonical_predicate: str
    canonical_value: CanonicalTaggedValue
    qualifiers: tuple[ClaimQualifier, ...]
    semantic_scope: tuple[str, ...]
    canonicalization_version: str
    semantic_digest: str
    retired_at: str | None
    retirement_reason: str | None
    identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        claim_id: object,
        canonical_subject: CanonicalTaggedValue,
        canonical_predicate: object,
        canonical_value: CanonicalTaggedValue,
        qualifiers: Iterable[ClaimQualifier] = (),
        semantic_scope: Iterable[object] = (),
        canonicalization_version: object = CLAIM_CONTRACT_SCHEMA_VERSION,
        retired_at: object | None = None,
        retirement_reason: object | None = None,
    ) -> "ClaimIdentity":
        draft = cls(
            envelope=envelope,
            claim_id=require_identifier(claim_id, "claim_id"),
            canonical_subject=canonical_subject,
            canonical_predicate=require_identifier(
                canonical_predicate, "canonical_predicate"
            ),
            canonical_value=canonical_value,
            qualifiers=normalize_claim_qualifiers(qualifiers),
            semantic_scope=_normalize_sorted_identifiers(
                semantic_scope, "semantic_scope"
            ),
            canonicalization_version=require_schema_version(
                canonicalization_version, "canonicalization_version"
            ),
            semantic_digest="0" * 64,
            retired_at=(
                normalize_timestamp(retired_at, "retired_at")
                if retired_at is not None
                else None
            ),
            retirement_reason=_optional_identifier(
                retirement_reason, "retirement_reason"
            ),
            identity_sha256="0" * 64,
        )
        draft._validate_material(check_digests=False)
        semantic_digest = canonical_sha256(draft.semantic_record())
        with_semantic_digest = cls(
            **{
                **draft.__dict__,
                "semantic_digest": semantic_digest,
            }
        )
        result = cls(
            **{
                **with_semantic_digest.__dict__,
                "identity_sha256": canonical_sha256(
                    with_semantic_digest.material_record()
                ),
            }
        )
        result.validate()
        return result

    def _validate_material(self, *, check_digests: bool = True) -> None:
        self.envelope.validate()
        if self.envelope.record_type != "claim_identity":
            raise CognitiveKernelContractError(
                "claim identity envelope must use record_type claim_identity"
            )
        if self.envelope.authority_role != "claim_authority":
            raise CognitiveKernelContractError(
                "claim identity requires claim_authority envelope"
            )
        if require_identifier(self.claim_id, "claim_id") != self.claim_id:
            raise CognitiveKernelContractError("claim_id is not canonical")
        if self.claim_id != self.envelope.record_id:
            raise CognitiveKernelContractError(
                "claim_id must equal envelope record_id"
            )
        self.canonical_subject.validate()
        self.canonical_value.validate()
        if require_identifier(
            self.canonical_predicate, "canonical_predicate"
        ) != self.canonical_predicate:
            raise CognitiveKernelContractError(
                "canonical_predicate is not canonical"
            )
        if normalize_claim_qualifiers(self.qualifiers) != self.qualifiers:
            raise CognitiveKernelContractError(
                "qualifiers are not canonical"
            )
        if _normalize_sorted_identifiers(
            self.semantic_scope, "semantic_scope"
        ) != self.semantic_scope:
            raise CognitiveKernelContractError(
                "semantic_scope is not canonical"
            )
        if require_schema_version(
            self.canonicalization_version, "canonicalization_version"
        ) != self.canonicalization_version:
            raise CognitiveKernelContractError(
                "canonicalization_version is not canonical"
            )
        if (self.retired_at is None) != (self.retirement_reason is None):
            raise CognitiveKernelContractError(
                "retired_at and retirement_reason must be set together"
            )
        if self.retired_at is not None:
            if normalize_timestamp(
                self.retired_at, "retired_at"
            ) != self.retired_at:
                raise CognitiveKernelContractError(
                    "retired_at is not canonical"
                )
            if self.retired_at < self.envelope.created_at:
                raise CognitiveKernelContractError(
                    "retired_at precedes claim creation"
                )
        if _optional_identifier(
            self.retirement_reason, "retirement_reason"
        ) != self.retirement_reason:
            raise CognitiveKernelContractError(
                "retirement_reason is not canonical"
            )
        if check_digests:
            require_sha256(self.semantic_digest, "semantic_digest")
            if canonical_sha256(self.semantic_record()) != self.semantic_digest:
                raise CognitiveKernelContractError(
                    "claim semantic digest mismatch"
                )

    def semantic_record(self) -> dict[str, object]:
        self.envelope.scope.validate()
        self.canonical_subject.validate()
        self.canonical_value.validate()
        qualifiers = normalize_claim_qualifiers(self.qualifiers)
        return {
            "product_id": self.envelope.scope.product_id,
            "authority_namespace_id": self.envelope.authority_namespace_id,
            "canonical_subject": self.canonical_subject.material_record(),
            "canonical_predicate": self.canonical_predicate,
            "canonical_value": self.canonical_value.material_record(),
            "qualifiers": [
                {
                    "key": item.key,
                    "value": item.value.material_record(),
                }
                for item in qualifiers
            ],
            "semantic_scope": list(self.semantic_scope),
            "canonicalization_version": self.canonicalization_version,
        }

    def material_record(self) -> dict[str, object]:
        self._validate_material(check_digests=True)
        return {
            "envelope": self.envelope.metadata_record(),
            "claim_id": self.claim_id,
            "canonical_subject": self.canonical_subject.metadata_record(),
            "canonical_predicate": self.canonical_predicate,
            "canonical_value": self.canonical_value.metadata_record(),
            "qualifiers": [item.metadata_record() for item in self.qualifiers],
            "semantic_scope": list(self.semantic_scope),
            "canonicalization_version": self.canonicalization_version,
            "semantic_digest": self.semantic_digest,
            "retired_at": self.retired_at,
            "retirement_reason": self.retirement_reason,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["identity_sha256"] = self.identity_sha256
        return record

    def semantically_equals(self, other: "ClaimIdentity") -> bool:
        self.validate()
        other.validate()
        return self.semantic_record() == other.semantic_record()

    def validate(self) -> None:
        self._validate_material(check_digests=True)
        require_sha256(self.identity_sha256, "identity_sha256")
        if canonical_sha256(self.material_record()) != self.identity_sha256:
            raise CognitiveKernelContractError(
                "claim identity digest mismatch"
            )


@dataclass(frozen=True)
class ClaimVersion:
    """Immutable append-only bitemporal version inside Claim Authority."""

    envelope: MemoryUnitEnvelope
    claim_version_id: str
    claim_id: str
    version_sequence: int
    store_sequence: int
    event_stream_position: int | None
    value: CanonicalTaggedValue
    qualifiers: tuple[ClaimQualifier, ...]
    authority_class: str
    confidence: float | None
    adjudication_state: str
    evidence_relation_ids: tuple[str, ...]
    conflict_set_id: str | None
    correction_of: tuple[str, ...]
    request_digest: str
    version_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        claim_version_id: object,
        claim_id: object,
        version_sequence: object,
        store_sequence: object,
        event_stream_position: object | None,
        value: CanonicalTaggedValue,
        qualifiers: Iterable[ClaimQualifier] = (),
        authority_class: object,
        confidence: object | None,
        adjudication_state: object,
        evidence_relation_ids: Iterable[object] = (),
        conflict_set_id: object | None = None,
        correction_of: Iterable[object] = (),
        request_digest: object,
    ) -> "ClaimVersion":
        draft = cls(
            envelope=envelope,
            claim_version_id=require_identifier(
                claim_version_id, "claim_version_id"
            ),
            claim_id=require_identifier(claim_id, "claim_id"),
            version_sequence=_require_positive_integer(
                version_sequence, "version_sequence"
            ),
            store_sequence=_require_positive_integer(
                store_sequence, "store_sequence"
            ),
            event_stream_position=_optional_non_negative_integer(
                event_stream_position, "event_stream_position"
            ),
            value=value,
            qualifiers=normalize_claim_qualifiers(qualifiers),
            authority_class=require_identifier(
                authority_class, "authority_class"
            ),
            confidence=require_confidence(confidence),
            adjudication_state=require_identifier(
                adjudication_state, "adjudication_state"
            ),
            evidence_relation_ids=_normalize_sorted_identifiers(
                evidence_relation_ids, "evidence_relation_ids"
            ),
            conflict_set_id=_optional_identifier(
                conflict_set_id, "conflict_set_id"
            ),
            correction_of=_normalize_sorted_identifiers(
                correction_of, "correction_of"
            ),
            request_digest=require_sha256(
                request_digest, "request_digest"
            ),
            version_sha256="0" * 64,
        )
        draft._validate_material()
        result = cls(
            **{
                **draft.__dict__,
                "version_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    def _validate_material(self) -> None:
        self.envelope.validate()
        if self.envelope.record_type != "claim_version":
            raise CognitiveKernelContractError(
                "claim version envelope must use record_type claim_version"
            )
        if self.envelope.authority_role != "claim_authority":
            raise CognitiveKernelContractError(
                "claim version requires claim_authority envelope"
            )
        for value, field in (
            (self.claim_version_id, "claim_version_id"),
            (self.claim_id, "claim_id"),
            (self.authority_class, "authority_class"),
            (self.adjudication_state, "adjudication_state"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.claim_version_id != self.envelope.record_id:
            raise CognitiveKernelContractError(
                "claim_version_id must equal envelope record_id"
            )
        if self.claim_id not in self.envelope.source_records:
            raise CognitiveKernelContractError(
                "claim version envelope must bind its claim identity"
            )
        _require_positive_integer(
            self.version_sequence, "version_sequence"
        )
        _require_positive_integer(self.store_sequence, "store_sequence")
        _optional_non_negative_integer(
            self.event_stream_position, "event_stream_position"
        )
        self.value.validate()
        if normalize_claim_qualifiers(self.qualifiers) != self.qualifiers:
            raise CognitiveKernelContractError(
                "claim version qualifiers are not canonical"
            )
        require_confidence(self.confidence)
        if self.adjudication_state not in CLAIM_ADJUDICATION_STATES:
            raise CognitiveKernelContractError(
                "adjudication_state is not ratified"
            )
        if _normalize_sorted_identifiers(
            self.evidence_relation_ids, "evidence_relation_ids"
        ) != self.evidence_relation_ids:
            raise CognitiveKernelContractError(
                "evidence_relation_ids are not canonical"
            )
        if _optional_identifier(
            self.conflict_set_id, "conflict_set_id"
        ) != self.conflict_set_id:
            raise CognitiveKernelContractError(
                "conflict_set_id is not canonical"
            )
        if self.adjudication_state in {"disputed", "quarantined"}:
            if self.conflict_set_id is None:
                raise CognitiveKernelContractError(
                    "disputed or quarantined versions require a conflict set"
                )
        if _normalize_sorted_identifiers(
            self.correction_of, "correction_of"
        ) != self.correction_of:
            raise CognitiveKernelContractError(
                "correction_of is not canonical"
            )
        if self.correction_of and not set(self.correction_of).issubset(
            set(self.envelope.supersedes)
        ):
            raise CognitiveKernelContractError(
                "correction_of must be included in envelope supersedes"
            )
        require_sha256(self.request_digest, "request_digest")

    def idempotency_record(self) -> dict[str, str]:
        self.envelope.validate()
        return {
            "idempotency_namespace": self.envelope.idempotency_namespace,
            "idempotency_key": self.envelope.idempotency_key,
            "request_digest": self.request_digest,
        }

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "envelope": self.envelope.metadata_record(),
            "claim_version_id": self.claim_version_id,
            "claim_id": self.claim_id,
            "version_sequence": self.version_sequence,
            "store_sequence": self.store_sequence,
            "event_stream_position": self.event_stream_position,
            "value": self.value.metadata_record(),
            "qualifiers": [item.metadata_record() for item in self.qualifiers],
            "authority_class": self.authority_class,
            "confidence": self.confidence,
            "adjudication_state": self.adjudication_state,
            "evidence_relation_ids": list(self.evidence_relation_ids),
            "conflict_set_id": self.conflict_set_id,
            "correction_of": list(self.correction_of),
            "request_digest": self.request_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["version_sha256"] = self.version_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.version_sha256, "version_sha256")
        if canonical_sha256(self.material_record()) != self.version_sha256:
            raise CognitiveKernelContractError(
                "claim version digest mismatch"
            )


@dataclass(frozen=True)
class CurrentClaimProjection:
    """Rebuildable bounded-read pointer to the current adjudicated version."""

    envelope: MemoryUnitEnvelope
    projection_id: str
    claim_id: str
    current_claim_version_id: str
    authority_generation: int
    projection_generation: int
    adjudication_state: str
    validity_state: str
    conflict_state: str
    deletion_state: str
    source_position: int
    projection_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        projection_id: object,
        claim_id: object,
        current_claim_version_id: object,
        authority_generation: object,
        projection_generation: object,
        adjudication_state: object,
        validity_state: object,
        conflict_state: object,
        deletion_state: object,
        source_position: object,
    ) -> "CurrentClaimProjection":
        draft = cls(
            envelope=envelope,
            projection_id=require_identifier(
                projection_id, "projection_id"
            ),
            claim_id=require_identifier(claim_id, "claim_id"),
            current_claim_version_id=require_identifier(
                current_claim_version_id, "current_claim_version_id"
            ),
            authority_generation=_require_non_negative_integer(
                authority_generation, "authority_generation"
            ),
            projection_generation=_require_non_negative_integer(
                projection_generation, "projection_generation"
            ),
            adjudication_state=require_identifier(
                adjudication_state, "adjudication_state"
            ),
            validity_state=require_identifier(
                validity_state, "validity_state"
            ),
            conflict_state=require_identifier(
                conflict_state, "conflict_state"
            ),
            deletion_state=require_identifier(
                deletion_state, "deletion_state"
            ),
            source_position=_require_non_negative_integer(
                source_position, "source_position"
            ),
            projection_sha256="0" * 64,
        )
        draft._validate_material()
        result = cls(
            **{
                **draft.__dict__,
                "projection_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    def _validate_material(self) -> None:
        self.envelope.validate()
        if self.envelope.record_type != "current_claim_projection":
            raise CognitiveKernelContractError(
                "current projection envelope has wrong record_type"
            )
        if self.envelope.authority_role != "registered_projection":
            raise CognitiveKernelContractError(
                "current claim projection requires registered_projection role"
            )
        for value, field in (
            (self.projection_id, "projection_id"),
            (self.claim_id, "claim_id"),
            (self.current_claim_version_id, "current_claim_version_id"),
            (self.adjudication_state, "adjudication_state"),
            (self.validity_state, "validity_state"),
            (self.conflict_state, "conflict_state"),
            (self.deletion_state, "deletion_state"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.projection_id != self.envelope.record_id:
            raise CognitiveKernelContractError(
                "projection_id must equal envelope record_id"
            )
        if self.claim_id not in self.envelope.source_records:
            raise CognitiveKernelContractError(
                "current projection must bind its claim identity"
            )
        if self.current_claim_version_id not in self.envelope.source_records:
            raise CognitiveKernelContractError(
                "current projection must bind its current claim version"
            )
        _require_non_negative_integer(
            self.authority_generation, "authority_generation"
        )
        _require_non_negative_integer(
            self.projection_generation, "projection_generation"
        )
        if self.adjudication_state not in CLAIM_ADJUDICATION_STATES:
            raise CognitiveKernelContractError(
                "projection adjudication_state is not ratified"
            )
        if self.validity_state not in CLAIM_VALIDITY_STATES:
            raise CognitiveKernelContractError(
                "validity_state is not ratified"
            )
        if self.conflict_state not in CLAIM_CONFLICT_STATES:
            raise CognitiveKernelContractError(
                "conflict_state is not ratified"
            )
        if self.deletion_state not in CLAIM_DELETION_STATES:
            raise CognitiveKernelContractError(
                "deletion_state is not ratified"
            )
        if self.deletion_state != self.envelope.deletion_state:
            raise CognitiveKernelContractError(
                "projection deletion state differs from its envelope"
            )
        _require_non_negative_integer(self.source_position, "source_position")

    def assert_projects(
        self,
        identity: ClaimIdentity,
        version: ClaimVersion,
    ) -> None:
        self.validate()
        identity.validate()
        version.validate()
        _require_same_scope(
            self.envelope.scope,
            identity.envelope.scope,
            field="projection identity scope",
        )
        _require_same_scope(
            self.envelope.scope,
            version.envelope.scope,
            field="projection version scope",
        )
        namespaces = {
            self.envelope.authority_namespace_id,
            identity.envelope.authority_namespace_id,
            version.envelope.authority_namespace_id,
        }
        if len(namespaces) != 1:
            raise CognitiveKernelContractError(
                "projection crosses authority namespaces"
            )
        if self.claim_id != identity.claim_id or self.claim_id != version.claim_id:
            raise CognitiveKernelContractError(
                "projection claim identity mismatch"
            )
        if self.current_claim_version_id != version.claim_version_id:
            raise CognitiveKernelContractError(
                "projection current version mismatch"
            )
        if self.source_position < version.store_sequence:
            raise CognitiveKernelContractError(
                "projection source position precedes claim version commit"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "envelope": self.envelope.metadata_record(),
            "projection_id": self.projection_id,
            "claim_id": self.claim_id,
            "current_claim_version_id": self.current_claim_version_id,
            "authority_generation": self.authority_generation,
            "projection_generation": self.projection_generation,
            "adjudication_state": self.adjudication_state,
            "validity_state": self.validity_state,
            "conflict_state": self.conflict_state,
            "deletion_state": self.deletion_state,
            "source_position": self.source_position,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["projection_sha256"] = self.projection_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.projection_sha256, "projection_sha256")
        if canonical_sha256(self.material_record()) != self.projection_sha256:
            raise CognitiveKernelContractError(
                "current claim projection digest mismatch"
            )
