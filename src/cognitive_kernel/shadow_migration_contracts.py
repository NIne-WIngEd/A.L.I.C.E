"""Phase 2 shadow-migration Stage A+B contracts.

These records describe source-store metadata and deterministic mapping receipts.
They do not carry private payload bytes, change canonical authority, or limit
future migration, backend, graph, vector, workflow, training, or deployment work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
    require_scope_identifier,
    require_text,
)
from .contracts import ProductHostScope


SHADOW_MIGRATION_CONTRACT_SCHEMA_VERSION = "1.0.0"
SHADOW_MIGRATION_STAGE = "stage_a_and_b"
SHADOW_MIGRATION_STATE = "implementation_started"

INVENTORY_INTEGRITY_STATES = frozenset(
    {"ok", "degraded", "unknown", "unreadable"}
)
MAPPING_OUTCOMES = frozenset(
    {"mapped", "mapped_with_ambiguity", "quarantined", "rejected"}
)
RECONCILIATION_STATES = frozenset(
    {"complete", "complete_with_ambiguity", "incomplete", "failed"}
)


def _require_non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_false(value: object, field: str) -> bool:
    if value is not False:
        raise CognitiveKernelContractError(
            f"{field} must be false in the Stage A+B read-only profile"
        )
    return False


def _normalize_counts(
    values: Iterable[tuple[object, object]],
    field: str,
) -> tuple[tuple[str, int], ...]:
    if isinstance(values, (str, bytes)):
        raise CognitiveKernelContractError(f"{field} must be a sequence")
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for name, count in values:
        table = require_identifier(name, field)
        if table in seen:
            raise CognitiveKernelContractError(
                f"{field} may not contain duplicate table names"
            )
        seen.add(table)
        normalized.append(
            (table, _require_non_negative_integer(count, field))
        )
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class Phase2SourceInventory:
    """Metadata-only inventory of one released Phase 2 source store."""

    scope: ProductHostScope
    inventory_id: str
    source_component_id: str
    authority_namespace_id: str
    source_schema_version: str
    source_generation: int
    backend_type: str
    read_profile: str
    table_record_counts: tuple[tuple[str, int], ...]
    schema_manifest_sha256: str
    integrity_state: str
    deletion_capabilities: tuple[str, ...]
    data_classifications: tuple[str, ...]
    private_payload_read: bool
    authoritative_discovery: bool
    observed_at: str
    inventory_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        inventory_id: object,
        source_component_id: object,
        authority_namespace_id: object,
        source_schema_version: object,
        source_generation: object,
        backend_type: object,
        read_profile: object,
        table_record_counts: Iterable[tuple[object, object]],
        schema_manifest_sha256: object,
        integrity_state: object,
        deletion_capabilities: Iterable[object],
        data_classifications: Iterable[object],
        private_payload_read: object = False,
        authoritative_discovery: object = False,
        observed_at: object,
    ) -> "Phase2SourceInventory":
        state = require_identifier(integrity_state, "integrity_state")
        if state not in INVENTORY_INTEGRITY_STATES:
            raise CognitiveKernelContractError(
                "integrity_state is not supported"
            )
        draft = cls(
            scope=scope,
            inventory_id=require_identifier(
                inventory_id, "inventory_id"
            ),
            source_component_id=require_identifier(
                source_component_id, "source_component_id"
            ),
            authority_namespace_id=require_scope_identifier(
                authority_namespace_id, "authority_namespace_id"
            ),
            source_schema_version=require_schema_version(
                source_schema_version, "source_schema_version"
            ),
            source_generation=_require_non_negative_integer(
                source_generation, "source_generation"
            ),
            backend_type=require_identifier(
                backend_type, "backend_type"
            ),
            read_profile=require_identifier(
                read_profile, "read_profile"
            ),
            table_record_counts=_normalize_counts(
                table_record_counts, "table_record_counts"
            ),
            schema_manifest_sha256=require_sha256(
                schema_manifest_sha256, "schema_manifest_sha256"
            ),
            integrity_state=state,
            deletion_capabilities=tuple(
                sorted(
                    normalize_identifier_sequence(
                        deletion_capabilities,
                        "deletion_capabilities",
                    )
                )
            ),
            data_classifications=tuple(
                sorted(
                    normalize_identifier_sequence(
                        data_classifications,
                        "data_classifications",
                    )
                )
            ),
            private_payload_read=_require_false(
                private_payload_read, "private_payload_read"
            ),
            authoritative_discovery=_require_false(
                authoritative_discovery, "authoritative_discovery"
            ),
            observed_at=normalize_timestamp(
                observed_at, "observed_at"
            ),
            inventory_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "inventory_sha256": canonical_sha256(
                    draft.metadata_record(include_digest=False)
                ),
            }
        )
        value.validate()
        return value

    def metadata_record(
        self, *, include_digest: bool = True
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": SHADOW_MIGRATION_CONTRACT_SCHEMA_VERSION,
            "stage": SHADOW_MIGRATION_STAGE,
            "state": SHADOW_MIGRATION_STATE,
            "scope": self.scope.metadata_record(),
            "inventory_id": self.inventory_id,
            "source_component_id": self.source_component_id,
            "authority_namespace_id": self.authority_namespace_id,
            "source_schema_version": self.source_schema_version,
            "source_generation": self.source_generation,
            "backend_type": self.backend_type,
            "read_profile": self.read_profile,
            "table_record_counts": [
                {"table": name, "count": count}
                for name, count in self.table_record_counts
            ],
            "schema_manifest_sha256": self.schema_manifest_sha256,
            "integrity_state": self.integrity_state,
            "deletion_capabilities": list(
                self.deletion_capabilities
            ),
            "data_classifications": list(self.data_classifications),
            "private_payload_read": self.private_payload_read,
            "authoritative_discovery": self.authoritative_discovery,
            "observed_at": self.observed_at,
        }
        if include_digest:
            record["inventory_sha256"] = self.inventory_sha256
        return record

    def validate(self) -> None:
        self.scope.validate()
        if self.integrity_state not in INVENTORY_INTEGRITY_STATES:
            raise CognitiveKernelContractError(
                "integrity_state is not supported"
            )
        _require_false(
            self.private_payload_read, "private_payload_read"
        )
        _require_false(
            self.authoritative_discovery, "authoritative_discovery"
        )
        expected = canonical_sha256(
            self.metadata_record(include_digest=False)
        )
        if self.inventory_sha256 != expected:
            raise CognitiveKernelContractError(
                "inventory_sha256 does not match canonical content"
            )


@dataclass(frozen=True)
class Phase2MappingReceipt:
    """Deterministic Stage B mapping result without source payload text."""

    scope: ProductHostScope
    mapping_id: str
    mapping_version: str
    source_record_type: str
    source_record_id: str
    source_record_sha256: str
    destination_record_types: tuple[str, ...]
    destination_candidate_ids: tuple[str, ...]
    authority_class: str
    mapping_outcome: str
    adjudication_hint: str
    ambiguity_codes: tuple[str, ...]
    information_loss_codes: tuple[str, ...]
    provenance_source_ids: tuple[str, ...]
    correction_lineage_ids: tuple[str, ...]
    deletion_lineage_ids: tuple[str, ...]
    private_payload_read: bool
    production_write: bool
    mapped_at: str
    mapping_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        mapping_id: object,
        mapping_version: object,
        source_record_type: object,
        source_record_id: object,
        source_record_sha256: object,
        destination_record_types: Iterable[object],
        destination_candidate_ids: Iterable[object],
        authority_class: object,
        mapping_outcome: object,
        adjudication_hint: object,
        ambiguity_codes: Iterable[object] = (),
        information_loss_codes: Iterable[object] = (),
        provenance_source_ids: Iterable[object] = (),
        correction_lineage_ids: Iterable[object] = (),
        deletion_lineage_ids: Iterable[object] = (),
        private_payload_read: object = False,
        production_write: object = False,
        mapped_at: object,
    ) -> "Phase2MappingReceipt":
        outcome = require_identifier(
            mapping_outcome, "mapping_outcome"
        )
        if outcome not in MAPPING_OUTCOMES:
            raise CognitiveKernelContractError(
                "mapping_outcome is not supported"
            )
        draft = cls(
            scope=scope,
            mapping_id=require_identifier(mapping_id, "mapping_id"),
            mapping_version=require_schema_version(
                mapping_version, "mapping_version"
            ),
            source_record_type=require_identifier(
                source_record_type, "source_record_type"
            ),
            source_record_id=require_identifier(
                source_record_id, "source_record_id"
            ),
            source_record_sha256=require_sha256(
                source_record_sha256, "source_record_sha256"
            ),
            destination_record_types=tuple(
                sorted(
                    normalize_identifier_sequence(
                        destination_record_types,
                        "destination_record_types",
                    )
                )
            ),
            destination_candidate_ids=tuple(
                sorted(
                    normalize_identifier_sequence(
                        destination_candidate_ids,
                        "destination_candidate_ids",
                    )
                )
            ),
            authority_class=require_identifier(
                authority_class, "authority_class"
            ),
            mapping_outcome=outcome,
            adjudication_hint=require_identifier(
                adjudication_hint, "adjudication_hint"
            ),
            ambiguity_codes=tuple(
                sorted(
                    normalize_identifier_sequence(
                        ambiguity_codes, "ambiguity_codes"
                    )
                )
            ),
            information_loss_codes=tuple(
                sorted(
                    normalize_identifier_sequence(
                        information_loss_codes,
                        "information_loss_codes",
                    )
                )
            ),
            provenance_source_ids=tuple(
                sorted(
                    normalize_identifier_sequence(
                        provenance_source_ids,
                        "provenance_source_ids",
                    )
                )
            ),
            correction_lineage_ids=tuple(
                sorted(
                    normalize_identifier_sequence(
                        correction_lineage_ids,
                        "correction_lineage_ids",
                    )
                )
            ),
            deletion_lineage_ids=tuple(
                sorted(
                    normalize_identifier_sequence(
                        deletion_lineage_ids,
                        "deletion_lineage_ids",
                    )
                )
            ),
            private_payload_read=_require_false(
                private_payload_read, "private_payload_read"
            ),
            production_write=_require_false(
                production_write, "production_write"
            ),
            mapped_at=normalize_timestamp(mapped_at, "mapped_at"),
            mapping_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "mapping_sha256": canonical_sha256(
                    draft.metadata_record(include_digest=False)
                ),
            }
        )
        value.validate()
        return value

    def metadata_record(
        self, *, include_digest: bool = True
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": SHADOW_MIGRATION_CONTRACT_SCHEMA_VERSION,
            "stage": SHADOW_MIGRATION_STAGE,
            "scope": self.scope.metadata_record(),
            "mapping_id": self.mapping_id,
            "mapping_version": self.mapping_version,
            "source_record_type": self.source_record_type,
            "source_record_id": self.source_record_id,
            "source_record_sha256": self.source_record_sha256,
            "destination_record_types": list(
                self.destination_record_types
            ),
            "destination_candidate_ids": list(
                self.destination_candidate_ids
            ),
            "authority_class": self.authority_class,
            "mapping_outcome": self.mapping_outcome,
            "adjudication_hint": self.adjudication_hint,
            "ambiguity_codes": list(self.ambiguity_codes),
            "information_loss_codes": list(
                self.information_loss_codes
            ),
            "provenance_source_ids": list(
                self.provenance_source_ids
            ),
            "correction_lineage_ids": list(
                self.correction_lineage_ids
            ),
            "deletion_lineage_ids": list(
                self.deletion_lineage_ids
            ),
            "private_payload_read": self.private_payload_read,
            "production_write": self.production_write,
            "mapped_at": self.mapped_at,
        }
        if include_digest:
            record["mapping_sha256"] = self.mapping_sha256
        return record

    def validate(self) -> None:
        self.scope.validate()
        if self.mapping_outcome not in MAPPING_OUTCOMES:
            raise CognitiveKernelContractError(
                "mapping_outcome is not supported"
            )
        _require_false(
            self.private_payload_read, "private_payload_read"
        )
        _require_false(self.production_write, "production_write")
        expected = canonical_sha256(
            self.metadata_record(include_digest=False)
        )
        if self.mapping_sha256 != expected:
            raise CognitiveKernelContractError(
                "mapping_sha256 does not match canonical content"
            )


@dataclass(frozen=True)
class ShadowMigrationReconciliationReceipt:
    """Batch-level accounting for deterministic Stage B mapping."""

    scope: ProductHostScope
    reconciliation_id: str
    mapping_version: str
    source_inventory_id: str
    expected_source_records: int
    mapped_records: int
    ambiguous_records: int
    quarantined_records: int
    rejected_records: int
    mapping_receipt_ids: tuple[str, ...]
    state: str
    production_write: bool
    reconciled_at: str
    reconciliation_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        reconciliation_id: object,
        mapping_version: object,
        source_inventory_id: object,
        expected_source_records: object,
        mapped_records: object,
        ambiguous_records: object,
        quarantined_records: object,
        rejected_records: object,
        mapping_receipt_ids: Iterable[object],
        state: object,
        production_write: object = False,
        reconciled_at: object,
    ) -> "ShadowMigrationReconciliationReceipt":
        normalized_state = require_identifier(state, "state")
        if normalized_state not in RECONCILIATION_STATES:
            raise CognitiveKernelContractError(
                "reconciliation state is not supported"
            )
        draft = cls(
            scope=scope,
            reconciliation_id=require_identifier(
                reconciliation_id, "reconciliation_id"
            ),
            mapping_version=require_schema_version(
                mapping_version, "mapping_version"
            ),
            source_inventory_id=require_identifier(
                source_inventory_id, "source_inventory_id"
            ),
            expected_source_records=_require_non_negative_integer(
                expected_source_records, "expected_source_records"
            ),
            mapped_records=_require_non_negative_integer(
                mapped_records, "mapped_records"
            ),
            ambiguous_records=_require_non_negative_integer(
                ambiguous_records, "ambiguous_records"
            ),
            quarantined_records=_require_non_negative_integer(
                quarantined_records, "quarantined_records"
            ),
            rejected_records=_require_non_negative_integer(
                rejected_records, "rejected_records"
            ),
            mapping_receipt_ids=tuple(
                sorted(
                    normalize_identifier_sequence(
                        mapping_receipt_ids,
                        "mapping_receipt_ids",
                    )
                )
            ),
            state=normalized_state,
            production_write=_require_false(
                production_write, "production_write"
            ),
            reconciled_at=normalize_timestamp(
                reconciled_at, "reconciled_at"
            ),
            reconciliation_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "reconciliation_sha256": canonical_sha256(
                    draft.metadata_record(include_digest=False)
                ),
            }
        )
        value.validate()
        return value

    def metadata_record(
        self, *, include_digest: bool = True
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": SHADOW_MIGRATION_CONTRACT_SCHEMA_VERSION,
            "stage": SHADOW_MIGRATION_STAGE,
            "scope": self.scope.metadata_record(),
            "reconciliation_id": self.reconciliation_id,
            "mapping_version": self.mapping_version,
            "source_inventory_id": self.source_inventory_id,
            "expected_source_records": self.expected_source_records,
            "mapped_records": self.mapped_records,
            "ambiguous_records": self.ambiguous_records,
            "quarantined_records": self.quarantined_records,
            "rejected_records": self.rejected_records,
            "mapping_receipt_ids": list(self.mapping_receipt_ids),
            "state": self.state,
            "production_write": self.production_write,
            "reconciled_at": self.reconciled_at,
        }
        if include_digest:
            record["reconciliation_sha256"] = (
                self.reconciliation_sha256
            )
        return record

    def validate(self) -> None:
        self.scope.validate()
        accounted = (
            self.mapped_records
            + self.quarantined_records
            + self.rejected_records
        )
        if accounted != self.expected_source_records:
            raise CognitiveKernelContractError(
                "reconciliation counts do not account for every source record"
            )
        if self.ambiguous_records > self.expected_source_records:
            raise CognitiveKernelContractError(
                "ambiguous_records cannot exceed expected_source_records"
            )
        if len(self.mapping_receipt_ids) != self.expected_source_records:
            raise CognitiveKernelContractError(
                "mapping_receipt_ids must account for every source record"
            )
        _require_false(self.production_write, "production_write")
        expected = canonical_sha256(
            self.metadata_record(include_digest=False)
        )
        if self.reconciliation_sha256 != expected:
            raise CognitiveKernelContractError(
                "reconciliation_sha256 does not match canonical content"
            )
