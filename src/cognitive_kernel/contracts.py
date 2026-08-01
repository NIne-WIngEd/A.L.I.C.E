"""Product-neutral scope, provenance, and private-reference contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    require_confidence,
    require_identifier,
    require_schema_version,
    require_scope_identifier,
    require_sha256,
    require_text,
)

PRODUCT_IDS = frozenset({"alice", "friday"})
PROVENANCE_TYPES = frozenset(
    {
        "owner_attested_canonical",
        "derived_inference",
        "generated_reconstruction",
        "evolved_identity",
        "owner_correction",
        "conflict_or_uncertain",
    }
)
IDENTITY_LAYERS = frozenset(
    {
        "source_history",
        "source_person_model",
        "reconstruction_inference",
        "alice_continuity",
        "owner_relationship_model",
    }
)
PRIVATE_DIRECTIVE_CODES = frozenset(
    {
        "PX-ORIGIN-01",
        "PX-MOTIVE-01",
        "PX-RELATION-01",
        "PX-PERSPECTIVE-01",
        "PX-EVOLUTION-01",
        "PX-TRIBUTE-01",
        "PX-CANON-01",
        "PX-BOUNDARY-01",
        "PX-SECRECY-01",
        "PX-OWNER-01",
        "PX-CLONE-01",
        "PX-DISCLOSURE-01",
        "PX-FIDELITY-01",
    }
)
RETENTION_CLASSES = frozenset(
    {
        "authoritative_source",
        "active_project",
        "high_value_experience",
        "ordinary_experience",
        "transient_web_or_tool_cache",
        "failed_experiment",
        "training_replay",
        "quarantine",
        "owner_hold",
    }
)
STORAGE_TIERS = frozenset(
    {
        "ledger",
        "raw_buffer",
        "hot",
        "warm",
        "cold",
        "quarantine",
        "deleted",
    }
)


@dataclass(frozen=True)
class ProductHostScope:
    """Product, host, schema, and encryption boundary for one kernel record."""

    product_id: str
    host_instance_id: str
    schema_version: str
    encryption_domain: str

    @classmethod
    def create(
        cls,
        *,
        product_id: object,
        host_instance_id: object,
        schema_version: object,
        encryption_domain: object,
    ) -> "ProductHostScope":
        scope = cls(
            product_id=require_scope_identifier(product_id, "product_id"),
            host_instance_id=require_scope_identifier(
                host_instance_id, "host_instance_id"
            ),
            schema_version=require_schema_version(schema_version),
            encryption_domain=require_scope_identifier(
                encryption_domain, "encryption_domain"
            ),
        )
        scope.validate()
        return scope

    def validate(self) -> None:
        product_id = require_scope_identifier(self.product_id, "product_id")
        if product_id != self.product_id:
            raise CognitiveKernelContractError("product_id is not canonical")
        if product_id not in PRODUCT_IDS:
            raise CognitiveKernelContractError("product_id is not approved")
        if require_scope_identifier(
            self.host_instance_id, "host_instance_id"
        ) != self.host_instance_id:
            raise CognitiveKernelContractError(
                "host_instance_id is not canonical"
            )
        if require_schema_version(self.schema_version) != self.schema_version:
            raise CognitiveKernelContractError(
                "schema_version is not canonical"
            )
        if require_scope_identifier(
            self.encryption_domain, "encryption_domain"
        ) != self.encryption_domain:
            raise CognitiveKernelContractError(
                "encryption_domain is not canonical"
            )

    def storage_scope(self) -> str:
        self.validate()
        return (
            f"{self.product_id}/"
            f"{self.host_instance_id}/"
            f"{self.encryption_domain}"
        )

    def metadata_record(self) -> dict[str, str]:
        self.validate()
        return {
            "product_id": self.product_id,
            "host_instance_id": self.host_instance_id,
            "schema_version": self.schema_version,
            "encryption_domain": self.encryption_domain,
        }

    def scope_sha256(self) -> str:
        return canonical_sha256(self.metadata_record())

    def assert_isolated_from(self, other: "ProductHostScope") -> None:
        self.validate()
        other.validate()
        if self.storage_scope() == other.storage_scope():
            raise CognitiveKernelContractError(
                "two records resolve to the same product-host encryption scope"
            )


@dataclass(frozen=True)
class ProvenanceReference:
    """Metadata-only provenance for source truth, inference, and continuity."""

    provenance_type: str
    source_reference_ids: tuple[str, ...]
    derivation_activity_id: str | None
    responsible_component: str
    model_id: str | None
    confidence: float | None
    supersedes_record_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        provenance_type: object,
        source_reference_ids: tuple[object, ...] | list[object] = (),
        derivation_activity_id: object | None = None,
        responsible_component: object,
        model_id: object | None = None,
        confidence: object | None = None,
        supersedes_record_ids: tuple[object, ...] | list[object] = (),
    ) -> "ProvenanceReference":
        provenance = cls(
            provenance_type=require_identifier(
                provenance_type, "provenance_type"
            ),
            source_reference_ids=normalize_identifier_sequence(
                source_reference_ids, "source_reference_ids"
            ),
            derivation_activity_id=(
                require_identifier(
                    derivation_activity_id, "derivation_activity_id"
                )
                if derivation_activity_id is not None
                else None
            ),
            responsible_component=require_identifier(
                responsible_component, "responsible_component"
            ),
            model_id=(
                require_identifier(model_id, "model_id")
                if model_id is not None
                else None
            ),
            confidence=require_confidence(confidence),
            supersedes_record_ids=normalize_identifier_sequence(
                supersedes_record_ids, "supersedes_record_ids"
            ),
        )
        provenance.validate()
        return provenance

    def validate(self) -> None:
        provenance_type = require_identifier(
            self.provenance_type, "provenance_type"
        )
        if provenance_type != self.provenance_type:
            raise CognitiveKernelContractError(
                "provenance_type is not canonical"
            )
        if provenance_type not in PROVENANCE_TYPES:
            raise CognitiveKernelContractError(
                "provenance_type is not approved"
            )
        if normalize_identifier_sequence(
            self.source_reference_ids, "source_reference_ids"
        ) != self.source_reference_ids:
            raise CognitiveKernelContractError(
                "source_reference_ids are not canonical"
            )
        if self.derivation_activity_id is not None:
            if require_identifier(
                self.derivation_activity_id, "derivation_activity_id"
            ) != self.derivation_activity_id:
                raise CognitiveKernelContractError(
                    "derivation_activity_id is not canonical"
                )
        if require_identifier(
            self.responsible_component, "responsible_component"
        ) != self.responsible_component:
            raise CognitiveKernelContractError(
                "responsible_component is not canonical"
            )
        if self.model_id is not None:
            if require_identifier(self.model_id, "model_id") != self.model_id:
                raise CognitiveKernelContractError(
                    "model_id is not canonical"
                )
        require_confidence(self.confidence)
        if normalize_identifier_sequence(
            self.supersedes_record_ids, "supersedes_record_ids"
        ) != self.supersedes_record_ids:
            raise CognitiveKernelContractError(
                "supersedes_record_ids are not canonical"
            )

        if (
            provenance_type == "owner_attested_canonical"
            and not self.source_reference_ids
        ):
            raise CognitiveKernelContractError(
                "owner-attested canonical provenance requires a source reference"
            )
        if provenance_type in {
            "derived_inference",
            "generated_reconstruction",
        }:
            if not self.source_reference_ids:
                raise CognitiveKernelContractError(
                    f"{provenance_type} requires source references"
                )
            if self.derivation_activity_id is None:
                raise CognitiveKernelContractError(
                    f"{provenance_type} requires derivation activity"
                )
        if (
            provenance_type == "owner_correction"
            and not self.supersedes_record_ids
        ):
            raise CognitiveKernelContractError(
                "owner correction requires a superseded record"
            )
        if (
            provenance_type == "conflict_or_uncertain"
            and self.confidence is None
        ):
            raise CognitiveKernelContractError(
                "conflict or uncertainty provenance requires confidence"
            )

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "provenance_type": self.provenance_type,
            "source_reference_ids": list(self.source_reference_ids),
            "derivation_activity_id": self.derivation_activity_id,
            "responsible_component": self.responsible_component,
            "model_id": self.model_id,
            "confidence": self.confidence,
            "supersedes_record_ids": list(self.supersedes_record_ids),
        }

    def provenance_sha256(self) -> str:
        return canonical_sha256(self.metadata_record())


@dataclass(frozen=True)
class OpaquePrivateCompanionReference:
    """A.L.I.C.E.-only opaque reference with no private payload or ciphertext."""

    scope: ProductHostScope
    reference_id: str
    directive_code: str
    identity_layer: str
    provenance: ProvenanceReference
    confidentiality: str
    private_payload_included: bool
    reference_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        reference_id: object,
        directive_code: object,
        identity_layer: object,
        provenance: ProvenanceReference,
    ) -> "OpaquePrivateCompanionReference":
        draft = cls(
            scope=scope,
            reference_id=require_identifier(reference_id, "reference_id"),
            directive_code=require_text(
                directive_code, "directive_code", maximum=64
            ),
            identity_layer=require_identifier(
                identity_layer, "identity_layer"
            ),
            provenance=provenance,
            confidentiality="HIGHLY_SENSITIVE",
            private_payload_included=False,
            reference_sha256="0" * 64,
        )
        draft._validate_material()
        digest = canonical_sha256(draft.material_record())
        reference = cls(
            scope=draft.scope,
            reference_id=draft.reference_id,
            directive_code=draft.directive_code,
            identity_layer=draft.identity_layer,
            provenance=draft.provenance,
            confidentiality=draft.confidentiality,
            private_payload_included=draft.private_payload_included,
            reference_sha256=digest,
        )
        reference.validate()
        return reference

    def _validate_material(self) -> None:
        self.scope.validate()
        if self.scope.product_id != "alice":
            raise CognitiveKernelContractError(
                "private companion references are A.L.I.C.E.-only"
            )
        if require_identifier(
            self.reference_id, "reference_id"
        ) != self.reference_id:
            raise CognitiveKernelContractError(
                "reference_id is not canonical"
            )
        if self.directive_code not in PRIVATE_DIRECTIVE_CODES:
            raise CognitiveKernelContractError(
                "directive_code is not an approved opaque code"
            )
        identity_layer = require_identifier(
            self.identity_layer, "identity_layer"
        )
        if identity_layer != self.identity_layer:
            raise CognitiveKernelContractError(
                "identity_layer is not canonical"
            )
        if identity_layer not in IDENTITY_LAYERS:
            raise CognitiveKernelContractError(
                "identity_layer is not approved"
            )
        self.provenance.validate()
        if self.confidentiality != "HIGHLY_SENSITIVE":
            raise CognitiveKernelContractError(
                "private companion references must be HIGHLY_SENSITIVE"
            )
        if self.private_payload_included is not False:
            raise CognitiveKernelContractError(
                "shared contracts may not include private companion payloads"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "scope": self.scope.metadata_record(),
            "reference_id": self.reference_id,
            "directive_code": self.directive_code,
            "identity_layer": self.identity_layer,
            "provenance": self.provenance.metadata_record(),
            "confidentiality": self.confidentiality,
            "private_payload_included": self.private_payload_included,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["reference_sha256"] = self.reference_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.reference_sha256, "reference_sha256")
        if canonical_sha256(self.material_record()) != self.reference_sha256:
            raise CognitiveKernelContractError(
                "private companion reference digest mismatch"
            )
