from __future__ import annotations

import sqlite3

import pytest

from cognitive_kernel import (
    TierTransitionIntegrityError,
    open_tier_transition_store,
)
from tier_transition_helpers import (
    COMMITTED_AT,
    EXECUTED_AT,
    capture_raw,
    make_decision,
    make_paths,
    make_scope,
    open_journal,
    open_raw_store,
    open_tier_store,
)


def test_payload_tampering_is_detected(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            approved = make_decision(
                content_digest=captured.reference.content_digest
            )
            journal.append_record(approved, committed_at=COMMITTED_AT)
            with open_tier_store(vault, repository) as tiers:
                receipt = tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=approved.decision_id,
                    source_reference_id=captured.reference.reference_id,
                    executed_at=EXECUTED_AT,
                    raw_buffer_store=raw,
                )
                path = tiers._object_path(
                    "hot",
                    receipt.target_reference.content_digest,
                )
                path.write_bytes(b"tampered")
                with pytest.raises(
                    TierTransitionIntegrityError,
                    match="length|digest",
                ):
                    tiers.verify_integrity()


def test_metadata_tampering_is_detected_on_reopen(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    raw = open_raw_store(vault, repository)
    captured = capture_raw(raw)
    with open_journal(vault, repository) as journal:
        approved = make_decision(
            content_digest=captured.reference.content_digest
        )
        journal.append_record(approved, committed_at=COMMITTED_AT)
        tiers = open_tier_store(vault, repository)
        tiers.execute(
            lifecycle_journal=journal,
            decision_id=approved.decision_id,
            source_reference_id=captured.reference.reference_id,
            executed_at=EXECUTED_AT,
            raw_buffer_store=raw,
        )
        database = tiers.root / "tier-transition.sqlite3"
        tiers.close()
        connection = sqlite3.connect(database)
        connection.execute(
            "DROP TRIGGER tier_transition_intents_no_update"
        )
        connection.execute(
            "UPDATE tier_transition_intents SET target_tier = 'cold'"
        )
        connection.commit()
        connection.close()
        with pytest.raises(TierTransitionIntegrityError):
            open_tier_transition_store(
                vault / "tier-store",
                scope=make_scope(),
                repository_root=repository,
            )
    raw.close()


def test_append_only_tables_reject_update_and_delete(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_tier_store(vault, repository) as tiers:
        connection = tiers._connection
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE tier_transition_metadata SET store_id = 'changed'"
            )
        connection.rollback()
