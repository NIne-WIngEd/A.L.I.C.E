from __future__ import annotations

import pytest

from cognitive_kernel import (
    DuplicateExperienceEventError,
    ExperienceLedgerTransactionError,
    open_experience_ledger,
)
from experience_ledger_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    SHA_B,
    SHA_C,
    make_event,
    make_scope,
    paths,
)


def test_batch_append_is_atomic_and_contiguous(tmp_path) -> None:
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
        receipt = ledger.append_events(
            (first, second),
            committed_at=COMMITTED_AT,
        )
        assert [entry.sequence for entry in receipt.entries] == [1, 2]
        assert receipt.entries[1].previous_entry_sha256 == (
            receipt.entries[0].entry_sha256
        )
        assert ledger.verify_integrity().entry_count == 2


def test_late_duplicate_rolls_back_entire_batch(tmp_path) -> None:
    repository, database = paths(tmp_path)
    existing = make_event()
    new_event = make_event(
        event_type="new-result",
        content_digest=SHA_C,
        payload_reference=None,
    )
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        ledger.append_event(existing, committed_at=COMMITTED_AT)
        with pytest.raises(DuplicateExperienceEventError):
            ledger.append_events(
                (new_event, existing),
                committed_at="2026-08-02T16:03:00Z",
            )
        assert ledger.verify_integrity().entry_count == 1
        with pytest.raises(KeyError):
            ledger.load_event(new_event.event_id)


def test_empty_transaction_is_rejected(tmp_path) -> None:
    repository, database = paths(tmp_path)
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        with pytest.raises(
            ExperienceLedgerTransactionError,
            match="requires events",
        ):
            ledger.append_events((), committed_at=COMMITTED_AT)
