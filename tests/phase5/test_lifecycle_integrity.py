from __future__ import annotations

import sqlite3

import pytest

from cognitive_kernel import (
    DuplicateLifecycleRecordError,
    LifecycleJournalIntegrityError,
    UnsafeLifecycleJournalPathError,
    open_lifecycle_journal,
)
from lifecycle_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    make_decision,
    make_open_blocker,
    make_scope,
    paths,
)


def test_append_reopen_inspection_and_hash_chain(tmp_path) -> None:
    repository, database = paths(tmp_path)
    decision = make_decision()
    blocker = make_open_blocker()
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as journal:
        receipt = journal.append_records(
            (decision, blocker), committed_at=COMMITTED_AT
        )
        assert [entry.sequence for entry in receipt.entries] == [1, 2]
        inspection = journal.inspect()
        assert len(inspection) == 2
        exposed = inspection[0].record()
        assert "authority_decision_id" not in exposed
        assert "evidence_reference_id" not in exposed
        assert "payload" not in exposed
    with open_lifecycle_journal(
        database,
        scope=make_scope(schema_version="1.1.0"),
        repository_root=repository,
    ) as reopened:
        report = reopened.verify_integrity()
        assert report.valid is True
        assert report.last_sequence == 2


def test_duplicate_rolls_back_entire_batch(tmp_path) -> None:
    repository, database = paths(tmp_path)
    existing = make_decision()
    new = make_decision(
        decision_key="decision-2",
        content_digest="b" * 64,
    )
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as journal:
        journal.append_record(existing, committed_at=COMMITTED_AT)
        with pytest.raises(DuplicateLifecycleRecordError):
            journal.append_records(
                (new, existing),
                committed_at="2026-08-03T04:32:00Z",
            )
        assert journal.verify_integrity().entry_count == 1
        with pytest.raises(KeyError):
            journal.load_record(new.record_id)


def test_out_of_band_tampering_is_detected(tmp_path) -> None:
    repository, database = paths(tmp_path)
    decision = make_decision()
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as journal:
        journal.append_record(decision, committed_at=COMMITTED_AT)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER lifecycle_journal_entries_no_update")
        connection.execute(
            "UPDATE lifecycle_journal_entries SET content_digest = ?",
            ("f" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
    ) as journal:
        with pytest.raises(
            LifecycleJournalIntegrityError,
            match="content_digest column changed",
        ):
            journal.verify_integrity()


def test_database_path_inside_repository_is_rejected(tmp_path) -> None:
    repository, _ = paths(tmp_path)
    with pytest.raises(UnsafeLifecycleJournalPathError):
        with open_lifecycle_journal(
            repository / "lifecycle.sqlite3",
            scope=make_scope(),
            repository_root=repository,
            created_at=CREATED_AT,
        ):
            pass
