from __future__ import annotations

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    LifecycleJournalTransactionError,
    RetentionBlockerRecord,
    open_lifecycle_journal,
)
from lifecycle_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    make_open_blocker,
    make_resolution,
    make_scope,
    paths,
)


def test_blocker_open_and_resolution_preserve_lineage(tmp_path) -> None:
    repository, database = paths(tmp_path)
    opened = make_open_blocker()
    resolved = make_resolution(opened)
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as journal:
        journal.append_records(
            (opened, resolved),
            committed_at=COMMITTED_AT,
        )
        assert journal.load_record(opened.record_id) == opened
        assert journal.load_record(resolved.record_id) == resolved
        assert journal.verify_integrity().entry_count == 2


def test_resolution_requires_existing_open_record(tmp_path) -> None:
    repository, database = paths(tmp_path)
    opened = make_open_blocker()
    resolved = make_resolution(opened)
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as journal:
        with pytest.raises(
            LifecycleJournalTransactionError,
            match="parent does not exist",
        ):
            journal.append_record(resolved, committed_at=COMMITTED_AT)


def test_owner_hold_requires_owner_verified_authority() -> None:
    with pytest.raises(CognitiveKernelContractError, match="authority"):
        make_open_blocker(
            blocker_type="owner_hold",
            authority_level="host_verified",
        )
    owner_hold = make_open_blocker(
        blocker_type="owner_hold",
        authority_level="owner_verified",
    )
    assert owner_hold.blocker_type == "owner_hold"


def test_blocker_metadata_round_trip() -> None:
    record = make_open_blocker()
    assert RetentionBlockerRecord.from_metadata_record(
        record.metadata_record()
    ) == record
