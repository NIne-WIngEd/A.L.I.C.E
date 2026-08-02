from __future__ import annotations

import sqlite3

import pytest

from cognitive_kernel import (
    ExperienceLedgerIntegrityError,
    open_experience_ledger,
)
from experience_ledger_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    SHA_B,
    make_event,
    make_scope,
    paths,
)


def test_ordinary_update_and_delete_are_blocked(tmp_path) -> None:
    repository, database = paths(tmp_path)
    event = make_event()
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        ledger.append_event(event, committed_at=COMMITTED_AT)
    connection = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE experience_ledger_entries SET event_type = ?",
                ("tampered",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM experience_ledger_entries")
    finally:
        connection.close()


def test_out_of_band_tampering_is_detected(tmp_path) -> None:
    repository, database = paths(tmp_path)
    first = make_event()
    second = make_event(
        event_type="answer-produced",
        content_digest=SHA_B,
        payload_reference=None,
    )
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        ledger.append_events((first, second), committed_at=COMMITTED_AT)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER experience_ledger_entries_no_update")
        connection.execute(
            "UPDATE experience_ledger_entries "
            "SET previous_entry_sha256 = ? WHERE sequence = 2",
            ("f" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
    ) as ledger:
        with pytest.raises(
            ExperienceLedgerIntegrityError,
            match="hash chain is broken",
        ):
            ledger.verify_integrity()
