from __future__ import annotations

import pytest

from cognitive_kernel import (
    DuplicateExperienceEventError,
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


def test_append_persists_and_reopens(tmp_path) -> None:
    repository, database = paths(tmp_path)
    event = make_event()
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        receipt = ledger.append_event(event, committed_at=COMMITTED_AT)
        assert receipt.entries[0].sequence == 1
        assert receipt.entries[0].event_id == event.event_id
        assert ledger.load_event(event.event_id) == event
    with open_experience_ledger(
        database,
        scope=make_scope(schema_version="1.1.0"),
        repository_root=repository,
    ) as ledger:
        assert ledger.load_event(event.event_id) == event
        report = ledger.verify_integrity()
        assert report.valid is True
        assert report.entry_count == 1


def test_sequence_and_hash_chain_are_deterministic(tmp_path) -> None:
    repository, database = paths(tmp_path)
    first = make_event()
    second = make_event(
        event_type="answer-produced",
        content_digest=SHA_B,
        occurred_at="2026-08-02T16:00:30Z",
        payload_reference=None,
    )
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        one = ledger.append_event(first, committed_at=COMMITTED_AT)
        two = ledger.append_event(
            second,
            committed_at="2026-08-02T16:02:00Z",
        )
        assert one.entries[0].sequence == 1
        assert two.entries[0].sequence == 2
        assert (
            two.entries[0].previous_entry_sha256
            == one.entries[0].entry_sha256
        )
        assert ledger.verify_integrity().last_sequence == 2


def test_duplicate_event_is_rejected(tmp_path) -> None:
    repository, database = paths(tmp_path)
    event = make_event()
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        ledger.append_event(event, committed_at=COMMITTED_AT)
        with pytest.raises(
            DuplicateExperienceEventError,
            match="already exists",
        ):
            ledger.append_event(
                event,
                committed_at="2026-08-02T16:02:00Z",
            )
