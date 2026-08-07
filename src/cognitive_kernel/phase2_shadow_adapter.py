"""Read-only Phase 2 inventory and deterministic Stage B adapters.

The database inspection path reads schema metadata, integrity state, and record
counts only. Record adapters accept caller-supplied synthetic mappings. They do
not enumerate private Phase 2 rows, write either store, or transfer authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Mapping

from alice_memory.schema import (
    MEMORY_STORABLE_CLASSIFICATIONS,
    SCHEMA_VERSION,
)
from alice_memory.store import validate_private_database_path

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    require_identifier,
    require_sha256,
)
from .contracts import ProductHostScope
from .memory_contracts import StoreRegistration
from .shadow_migration_contracts import (
    Phase2MappingReceipt,
    Phase2SourceInventory,
    ShadowMigrationReconciliationReceipt,
)


PHASE2_STAGE_AB_MAPPING_VERSION = "1.0.0"
PHASE2_EXPECTED_TABLES = frozenset(
    {
        "schema_migrations",
        "memories",
        "memory_sources",
        "memory_relations",
        "memory_derivations",
        "memory_entities",
        "memory_events",
        "memory_tombstones",
        "memory_sensitive_payloads",
        "sensitive_memory_access_events",
        "memory_candidates",
        "memory_candidate_sources",
        "memory_candidate_events",
    }
)

_AUTHORITY_CLASS_BY_STATUS = {
    "verified_fact": "evidence_backed",
    "rayan_statement": "owner_attested",
    "external_claim": "external_source",
    "alice_inference": "model_inference",
    "estimate": "uncertain_inference",
    "uncertain": "uncertain_inference",
    "disputed": "disputed",
    "historical": "historical",
    "superseded": "historical",
}

_RELATION_DESTINATIONS = {
    "supersedes": ("claim_version", "current_claim_projection"),
    "conflicts_with": ("claim_conflict_record",),
    "supports": ("claim_evidence_relation",),
    "duplicates": ("claim_evidence_relation",),
    "derived_from": ("claim_evidence_relation",),
    "corrects": ("claim_version", "claim_evidence_relation"),
}


def _row_value(
    record: Mapping[str, object],
    key: str,
    *,
    required: bool = True,
) -> object | None:
    value = record.get(key)
    if required and value is None:
        raise CognitiveKernelContractError(
            f"Phase 2 record is missing required field {key}"
        )
    return value


def _optional_identifiers(
    values: Iterable[object] | None,
) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(require_identifier(value, "lineage_id") for value in values)


def _candidate_id(kind: str, source_type: str, source_id: str) -> str:
    digest = canonical_sha256(
        {
            "mapping_version": PHASE2_STAGE_AB_MAPPING_VERSION,
            "kind": kind,
            "source_type": source_type,
            "source_id": source_id,
        }
    )
    return f"{kind}.phase2.{digest[:32]}"


@contextmanager
def open_phase2_metadata_reader(
    database_path: Path,
    *,
    repository_root: Path,
) -> Iterator[sqlite3.Connection]:
    """Open a Phase 2 SQLite store through SQLite's read-only URI mode."""

    resolved = validate_private_database_path(
        database_path,
        repository_root=repository_root,
    )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    uri = resolved.as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        yield connection
    finally:
        connection.close()


def _schema_manifest(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, tuple[tuple[str, str, int, int], ...]], ...]:
    tables = [
        str(row["name"])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    ]
    manifest: list[
        tuple[str, tuple[tuple[str, str, int, int], ...]]
    ] = []
    for table in tables:
        if table not in PHASE2_EXPECTED_TABLES:
            continue
        columns = tuple(
            (
                str(row["name"]),
                str(row["type"]),
                int(row["notnull"]),
                int(row["pk"]),
            )
            for row in connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
        )
        manifest.append((table, columns))
    return tuple(manifest)


def inspect_phase2_source(
    database_path: Path,
    *,
    repository_root: Path,
    scope: ProductHostScope,
    authority_namespace_id: str,
    observed_at: str,
) -> tuple[Phase2SourceInventory, StoreRegistration]:
    """Inventory schema/count metadata without reading memory payload columns."""

    with open_phase2_metadata_reader(
        database_path,
        repository_root=repository_root,
    ) as connection:
        manifest = _schema_manifest(connection)
        available_tables = {name for name, _ in manifest}
        missing = sorted(PHASE2_EXPECTED_TABLES - available_tables)
        if missing:
            raise CognitiveKernelContractError(
                "Phase 2 source is missing expected tables: "
                + ", ".join(missing)
            )
        version_row = connection.execute(
            "SELECT MAX(version) AS version FROM schema_migrations"
        ).fetchone()
        version = None if version_row is None else version_row["version"]
        if version != SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "Phase 2 schema version does not match the released oracle"
            )
        counts = tuple(
            (
                table,
                int(
                    connection.execute(
                        f'SELECT COUNT(*) AS count FROM "{table}"'
                    ).fetchone()["count"]
                ),
            )
            for table in sorted(available_tables)
        )
        integrity_row = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()
        integrity_state = (
            "ok"
            if integrity_row is not None
            and str(integrity_row[0]).lower() == "ok"
            else "degraded"
        )

    schema_digest = canonical_sha256(
        {
            "schema_version": version,
            "tables": [
                {
                    "name": table,
                    "columns": [
                        {
                            "name": name,
                            "type": column_type,
                            "not_null": not_null,
                            "primary_key": primary_key,
                        }
                        for name, column_type, not_null, primary_key
                        in columns
                    ],
                }
                for table, columns in manifest
            ],
        }
    )
    inventory_id = f"phase2.inventory.{schema_digest[:32]}"
    inventory = Phase2SourceInventory.create(
        scope=scope,
        inventory_id=inventory_id,
        source_component_id="phase2.memory.core",
        authority_namespace_id=authority_namespace_id,
        source_schema_version=f"{version}.0.0",
        source_generation=0,
        backend_type="sqlite",
        read_profile="sqlite.uri.read_only.metadata",
        table_record_counts=counts,
        schema_manifest_sha256=schema_digest,
        integrity_state=integrity_state,
        deletion_capabilities=(
            "tombstone",
            "retrieval_exclusion",
            "sensitive_payload_erasure",
            "event_lineage",
        ),
        data_classifications=tuple(
            classification.lower()
            for classification in MEMORY_STORABLE_CLASSIFICATIONS
        ),
        private_payload_read=False,
        authoritative_discovery=False,
        observed_at=observed_at,
    )
    registration = StoreRegistration.create(
        scope=scope,
        registration_id=f"phase2.registration.{schema_digest[:32]}",
        component_id="phase2.memory.core",
        authority_namespace_id=authority_namespace_id,
        authority_role="claim_authority",
        capability_ids=(
            "memory.phase2.compatibility",
            "memory.read",
            "memory.provenance",
            "memory.deletion",
        ),
        backend_type="sqlite",
        backend_version=f"{version}.0.0",
        deployment_profile="single_workstation",
        capability_state="compatibility_only",
        consistency_model="sqlite.transactional",
        availability_profile="single_store",
        encryption_profile="vault.custody.with.aes256gcm.payloads",
        region_or_device_scope="owner.authorized.device",
        health_state=integrity_state,
        performance_profile="phase2.released.oracle",
        cost_profile="embedded.local",
        deletion_endpoint="phase2://memory/delete",
        rollback_endpoint="phase2://memory/restore-from-backup",
        backup_profile="owner.authorized.encrypted.snapshot",
        created_at=observed_at,
    )
    return inventory, registration


def adapt_phase2_memory_record(
    record: Mapping[str, object],
    *,
    scope: ProductHostScope,
    mapped_at: str,
    provenance_source_ids: Iterable[object] = (),
    correction_lineage_ids: Iterable[object] = (),
    deletion_lineage_ids: Iterable[object] = (),
) -> Phase2MappingReceipt:
    """Map one caller-supplied synthetic memory record into neutral receipts."""

    source_id = require_identifier(
        _row_value(record, "memory_id"), "memory_id"
    )
    content_sha256 = require_sha256(
        _row_value(record, "content_sha256"),
        "content_sha256",
    )
    knowledge_status = require_identifier(
        _row_value(record, "knowledge_status"),
        "knowledge_status",
    )
    validity_state = require_identifier(
        _row_value(record, "validity_state"),
        "validity_state",
    )
    deletion_state = require_identifier(
        _row_value(record, "deletion_state"),
        "deletion_state",
    )
    owner_confirmed = bool(record.get("rayan_confirmed", False))
    authority_class = (
        "owner_attested"
        if owner_confirmed
        else _AUTHORITY_CLASS_BY_STATUS.get(
            knowledge_status, "unclassified"
        )
    )

    ambiguity: list[str] = []
    if not record.get("memory_key"):
        ambiguity.append("missing.memory.key")
    if not record.get("valid_from"):
        ambiguity.append("missing.valid.from")
    sources = _optional_identifiers(provenance_source_ids)
    if not sources:
        ambiguity.append("missing.provenance.binding")
    if authority_class == "unclassified":
        ambiguity.append("unknown.knowledge.status")

    loss: list[str] = [
        "payload.omitted.from.public.receipt",
        "category.requires.destination.ontology.mapping",
        "retention.requires.profile.mapping",
    ]
    if record.get("time_precision"):
        loss.append("time.precision.requires.destination.mapping")

    deletions = list(_optional_identifiers(deletion_lineage_ids))
    if deletion_state == "pending_deletion":
        deletions.append(f"phase2.deletion.pending.{source_id}")

    if knowledge_status == "disputed" or validity_state == "disputed":
        adjudication_hint = "quarantine"
        outcome = "quarantined"
    elif knowledge_status == "superseded":
        adjudication_hint = "supersede"
        outcome = (
            "mapped_with_ambiguity" if ambiguity else "mapped"
        )
    else:
        adjudication_hint = "candidate_only"
        outcome = (
            "mapped_with_ambiguity" if ambiguity else "mapped"
        )

    destinations = (
        "evidence_event",
        "claim_identity",
        "claim_version",
        "current_claim_projection",
    )
    candidate_ids = tuple(
        _candidate_id(kind, "memory", source_id)
        for kind in destinations
    )
    mapping_id = _candidate_id("mapping", "memory", source_id)
    return Phase2MappingReceipt.create(
        scope=scope,
        mapping_id=mapping_id,
        mapping_version=PHASE2_STAGE_AB_MAPPING_VERSION,
        source_record_type="memory",
        source_record_id=source_id,
        source_record_sha256=content_sha256,
        destination_record_types=destinations,
        destination_candidate_ids=candidate_ids,
        authority_class=authority_class,
        mapping_outcome=outcome,
        adjudication_hint=adjudication_hint,
        ambiguity_codes=ambiguity,
        information_loss_codes=loss,
        provenance_source_ids=sources,
        correction_lineage_ids=_optional_identifiers(
            correction_lineage_ids
        ),
        deletion_lineage_ids=deletions,
        private_payload_read=False,
        production_write=False,
        mapped_at=mapped_at,
    )


def adapt_phase2_relation_record(
    record: Mapping[str, object],
    *,
    scope: ProductHostScope,
    mapped_at: str,
) -> Phase2MappingReceipt:
    """Map a synthetic Phase 2 relationship into a neutral relation receipt."""

    relation_id = require_identifier(
        _row_value(record, "relation_id"), "relation_id"
    )
    relation_type = require_identifier(
        _row_value(record, "relation_type"), "relation_type"
    )
    from_id = require_identifier(
        _row_value(record, "from_memory_id"), "from_memory_id"
    )
    to_id = require_identifier(
        _row_value(record, "to_memory_id"), "to_memory_id"
    )
    destinations = _RELATION_DESTINATIONS.get(relation_type)
    ambiguity: list[str] = []
    if destinations is None:
        destinations = ("claim_evidence_relation",)
        ambiguity.append("unknown.relation.type")

    source_digest = canonical_sha256(
        {
            "relation_id": relation_id,
            "relation_type": relation_type,
            "from_memory_id": from_id,
            "to_memory_id": to_id,
        }
    )
    correction_ids = (
        (relation_id,) if relation_type == "corrects" else ()
    )
    candidate_ids = tuple(
        _candidate_id(kind, "relation", relation_id)
        for kind in destinations
    )
    return Phase2MappingReceipt.create(
        scope=scope,
        mapping_id=_candidate_id(
            "mapping", "relation", relation_id
        ),
        mapping_version=PHASE2_STAGE_AB_MAPPING_VERSION,
        source_record_type="memory_relation",
        source_record_id=relation_id,
        source_record_sha256=source_digest,
        destination_record_types=destinations,
        destination_candidate_ids=candidate_ids,
        authority_class="deterministic_derivation",
        mapping_outcome=(
            "mapped_with_ambiguity" if ambiguity else "mapped"
        ),
        adjudication_hint=(
            "conflict_review"
            if relation_type == "conflicts_with"
            else "candidate_only"
        ),
        ambiguity_codes=ambiguity,
        information_loss_codes=(),
        provenance_source_ids=(from_id, to_id),
        correction_lineage_ids=correction_ids,
        deletion_lineage_ids=(),
        private_payload_read=False,
        production_write=False,
        mapped_at=mapped_at,
    )


def adapt_phase2_tombstone_record(
    record: Mapping[str, object],
    *,
    scope: ProductHostScope,
    mapped_at: str,
) -> Phase2MappingReceipt:
    """Map a synthetic tombstone while preserving deletion lineage."""

    tombstone_id = require_identifier(
        _row_value(record, "tombstone_id"), "tombstone_id"
    )
    deleted_memory_id = require_identifier(
        _row_value(record, "deleted_memory_id"),
        "deleted_memory_id",
    )
    content_sha256 = require_sha256(
        _row_value(record, "content_sha256"),
        "content_sha256",
    )
    deleted_at = str(_row_value(record, "deleted_at"))
    deletion_scope = str(_row_value(record, "deletion_scope"))
    event_id = record.get("event_id")
    source_digest = canonical_sha256(
        {
            "tombstone_id": tombstone_id,
            "deleted_memory_id": deleted_memory_id,
            "content_sha256": content_sha256,
            "deleted_at": deleted_at,
            "deletion_scope": deletion_scope,
            "event_id": event_id,
        }
    )
    destination = "deletion_propagation_receipt"
    return Phase2MappingReceipt.create(
        scope=scope,
        mapping_id=_candidate_id(
            "mapping", "tombstone", tombstone_id
        ),
        mapping_version=PHASE2_STAGE_AB_MAPPING_VERSION,
        source_record_type="memory_tombstone",
        source_record_id=tombstone_id,
        source_record_sha256=source_digest,
        destination_record_types=(destination,),
        destination_candidate_ids=(
            _candidate_id(destination, "tombstone", tombstone_id),
        ),
        authority_class="deterministic_deletion_lineage",
        mapping_outcome="mapped",
        adjudication_hint="propagate_deletion",
        ambiguity_codes=(),
        information_loss_codes=(),
        provenance_source_ids=(deleted_memory_id,),
        correction_lineage_ids=(),
        deletion_lineage_ids=(tombstone_id,),
        private_payload_read=False,
        production_write=False,
        mapped_at=mapped_at,
    )


def reconcile_phase2_mappings(
    receipts: Iterable[Phase2MappingReceipt],
    *,
    scope: ProductHostScope,
    source_inventory_id: str,
    reconciled_at: str,
) -> ShadowMigrationReconciliationReceipt:
    """Account for every source record represented by mapping receipts."""

    values = tuple(receipts)
    if not values:
        raise CognitiveKernelContractError(
            "at least one mapping receipt is required"
        )
    mapped = sum(
        receipt.mapping_outcome
        in {"mapped", "mapped_with_ambiguity"}
        for receipt in values
    )
    ambiguous = sum(bool(receipt.ambiguity_codes) for receipt in values)
    quarantined = sum(
        receipt.mapping_outcome == "quarantined"
        for receipt in values
    )
    rejected = sum(
        receipt.mapping_outcome == "rejected"
        for receipt in values
    )
    if rejected:
        state = "incomplete"
    elif ambiguous or quarantined:
        state = "complete_with_ambiguity"
    else:
        state = "complete"
    digest = canonical_sha256(
        {
            "source_inventory_id": source_inventory_id,
            "mapping_version": PHASE2_STAGE_AB_MAPPING_VERSION,
            "receipt_ids": sorted(
                receipt.mapping_id for receipt in values
            ),
        }
    )
    return ShadowMigrationReconciliationReceipt.create(
        scope=scope,
        reconciliation_id=f"phase2.reconciliation.{digest[:32]}",
        mapping_version=PHASE2_STAGE_AB_MAPPING_VERSION,
        source_inventory_id=source_inventory_id,
        expected_source_records=len(values),
        mapped_records=mapped,
        ambiguous_records=ambiguous,
        quarantined_records=quarantined,
        rejected_records=rejected,
        mapping_receipt_ids=tuple(
            receipt.mapping_id for receipt in values
        ),
        state=state,
        production_write=False,
        reconciled_at=reconciled_at,
    )
