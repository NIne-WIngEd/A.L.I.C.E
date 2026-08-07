"""Stage A+B Phase 2 shadow-migration tests."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from alice_memory.schema import initialize_schema
from alice_memory.store import UnsafeMemoryStorePathError
from cognitive_kernel import ProductHostScope
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.phase2_shadow_adapter import (
    adapt_phase2_memory_record,
    adapt_phase2_relation_record,
    adapt_phase2_tombstone_record,
    inspect_phase2_source,
    open_phase2_metadata_reader,
    reconcile_phase2_mappings,
)


REFERENCE_TIME = "2026-08-07T03:00:00Z"
CONTENT_DIGEST = "a" * 64


def scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="owner-primary",
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def synthetic_phase2_database(
    tmp_path: Path,
) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    database = vault / "memory-core.sqlite3"
    connection = sqlite3.connect(database)
    try:
        initialize_schema(connection, applied_at=REFERENCE_TIME)
        connection.execute(
            """
            INSERT INTO memories (
                memory_id, schema_version, content, content_sha256,
                memory_key, category, knowledge_status, confidence,
                data_classification, valid_from, valid_to,
                time_precision, recorded_at, verified_at,
                rayan_confirmed, validity_state, retention_state,
                deletion_state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "memory-1", 3, "synthetic fixture", CONTENT_DIGEST,
                "project:alice", "project", "rayan_statement", 1.0,
                "PRIVATE", REFERENCE_TIME, None, "second",
                REFERENCE_TIME, REFERENCE_TIME, 1, "current",
                "durable", "active", REFERENCE_TIME, REFERENCE_TIME,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return repository, database


def synthetic_memory() -> dict[str, object]:
    return {
        "memory_id": "memory-1",
        "content": "synthetic fixture",
        "content_sha256": CONTENT_DIGEST,
        "memory_key": "project:alice",
        "category": "project",
        "knowledge_status": "rayan_statement",
        "confidence": 1.0,
        "data_classification": "PRIVATE",
        "valid_from": REFERENCE_TIME,
        "time_precision": "second",
        "rayan_confirmed": 1,
        "validity_state": "current",
        "retention_state": "durable",
        "deletion_state": "active",
    }


def test_stage_a_inventory_is_metadata_only_and_non_authoritative(
    tmp_path: Path,
) -> None:
    repository, database = synthetic_phase2_database(tmp_path)

    inventory, registration = inspect_phase2_source(
        database,
        repository_root=repository,
        scope=scope(),
        authority_namespace_id="owner-primary",
        observed_at=REFERENCE_TIME,
    )

    assert inventory.private_payload_read is False
    assert inventory.authoritative_discovery is False
    assert inventory.integrity_state == "ok"
    assert dict(inventory.table_record_counts)["memories"] == 1
    assert registration.capability_state == "compatibility_only"
    assert registration.authority_role == "claim_authority"
    assert "payload" not in inventory.metadata_record()
    assert "content" not in inventory.metadata_record()


def test_stage_a_reader_rejects_writes(tmp_path: Path) -> None:
    repository, database = synthetic_phase2_database(tmp_path)

    with open_phase2_metadata_reader(
        database,
        repository_root=repository,
    ) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute(
                "UPDATE memories SET memory_key = 'changed'"
            )


def test_stage_a_rejects_repository_local_database(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    database = repository / "memory-core.sqlite3"
    sqlite3.connect(database).close()

    with pytest.raises(UnsafeMemoryStorePathError):
        with open_phase2_metadata_reader(
            database,
            repository_root=repository,
        ):
            pass


def test_memory_adapter_is_deterministic_and_payload_free() -> None:
    first = adapt_phase2_memory_record(
        synthetic_memory(),
        scope=scope(),
        mapped_at=REFERENCE_TIME,
        provenance_source_ids=("source-1",),
    )
    second = adapt_phase2_memory_record(
        synthetic_memory(),
        scope=scope(),
        mapped_at=REFERENCE_TIME,
        provenance_source_ids=("source-1",),
    )

    assert first == second
    assert first.authority_class == "owner_attested"
    assert first.mapping_outcome == "mapped"
    assert first.private_payload_read is False
    assert first.production_write is False
    record = first.metadata_record()
    assert "content" not in record
    assert "synthetic fixture" not in str(record)


def test_missing_provenance_is_reported_not_invented() -> None:
    record = synthetic_memory()
    record["memory_key"] = None
    record["valid_from"] = None

    receipt = adapt_phase2_memory_record(
        record,
        scope=scope(),
        mapped_at=REFERENCE_TIME,
    )

    assert receipt.mapping_outcome == "mapped_with_ambiguity"
    assert set(receipt.ambiguity_codes) == {
        "missing.memory.key",
        "missing.provenance.binding",
        "missing.valid.from",
    }


def test_disputed_memory_is_quarantined() -> None:
    record = synthetic_memory()
    record["knowledge_status"] = "disputed"
    record["validity_state"] = "disputed"

    receipt = adapt_phase2_memory_record(
        record,
        scope=scope(),
        mapped_at=REFERENCE_TIME,
        provenance_source_ids=("source-1",),
    )

    assert receipt.mapping_outcome == "quarantined"
    assert receipt.adjudication_hint == "quarantine"


def test_correction_and_deletion_lineage_are_preserved() -> None:
    correction = adapt_phase2_relation_record(
        {
            "relation_id": "relation-1",
            "from_memory_id": "memory-2",
            "to_memory_id": "memory-1",
            "relation_type": "corrects",
        },
        scope=scope(),
        mapped_at=REFERENCE_TIME,
    )
    tombstone = adapt_phase2_tombstone_record(
        {
            "tombstone_id": "tombstone-1",
            "deleted_memory_id": "memory-1",
            "content_sha256": CONTENT_DIGEST,
            "deleted_at": REFERENCE_TIME,
            "deletion_scope": "all",
        },
        scope=scope(),
        mapped_at=REFERENCE_TIME,
    )

    assert correction.correction_lineage_ids == ("relation-1",)
    assert tombstone.deletion_lineage_ids == ("tombstone-1",)
    assert tombstone.adjudication_hint == "propagate_deletion"


def test_reconciliation_accounts_for_every_receipt() -> None:
    memory = adapt_phase2_memory_record(
        synthetic_memory(),
        scope=scope(),
        mapped_at=REFERENCE_TIME,
        provenance_source_ids=("source-1",),
    )
    tombstone = adapt_phase2_tombstone_record(
        {
            "tombstone_id": "tombstone-1",
            "deleted_memory_id": "memory-1",
            "content_sha256": CONTENT_DIGEST,
            "deleted_at": REFERENCE_TIME,
            "deletion_scope": "all",
        },
        scope=scope(),
        mapped_at=REFERENCE_TIME,
    )

    receipt = reconcile_phase2_mappings(
        (memory, tombstone),
        scope=scope(),
        source_inventory_id="phase2.inventory.synthetic",
        reconciled_at=REFERENCE_TIME,
    )

    assert receipt.expected_source_records == 2
    assert receipt.mapped_records == 2
    assert receipt.quarantined_records == 0
    assert receipt.rejected_records == 0
    assert receipt.production_write is False


def test_reconciliation_rejects_empty_input() -> None:
    with pytest.raises(CognitiveKernelContractError):
        reconcile_phase2_mappings(
            (),
            scope=scope(),
            source_inventory_id="phase2.inventory.synthetic",
            reconciled_at=REFERENCE_TIME,
        )


def test_stage_ab_policy_state_metadata_is_flat_and_governance_compatible() -> None:
    policy_path = (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "memory_shadow_migration_stage_a_b_policy.json"
    )
    payload = json.loads(policy_path.read_text(encoding="utf-8"))

    assert "capability_state_semantics" not in payload
    assert payload["research_status"] == "allowed"
    assert payload["prototype_status"] == "stage_a_b_prototype_operational"
    assert payload["production_status"] == "not_activated_by_this_profile"
    assert payload["successor_profile"] == "memory.shadow_migration.stage_c_e"
    assert payload["removal_criterion"] == "superseded_by_evaluated_successor_stage"
    assert payload["capability_ceiling"] is False
