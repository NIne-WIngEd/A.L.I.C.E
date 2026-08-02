from __future__ import annotations

import pytest

from cognitive_kernel import (
    ExperienceLedgerIsolationError,
    open_experience_ledger,
)
from experience_ledger_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    make_event,
    make_scope,
    paths,
)


def test_event_from_another_host_is_rejected(tmp_path) -> None:
    repository, database = paths(tmp_path)
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        with pytest.raises(
            ExperienceLedgerIsolationError,
            match="scope does not match",
        ):
            ledger.append_event(
                make_event(scope=make_scope(host_instance_id="other-host")),
                committed_at=COMMITTED_AT,
            )


def test_database_cannot_be_reopened_under_another_scope(tmp_path) -> None:
    repository, database = paths(tmp_path)
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ):
        pass
    with pytest.raises(
        ExperienceLedgerIsolationError,
        match="bound to another scope",
    ):
        with open_experience_ledger(
            database,
            scope=make_scope(encryption_domain="other-domain"),
            repository_root=repository,
        ):
            pass


def test_record_schema_version_does_not_collapse_storage_scope(tmp_path) -> None:
    repository, database = paths(tmp_path)
    with open_experience_ledger(
        database,
        scope=make_scope(schema_version="1.0.0"),
        repository_root=repository,
        created_at=CREATED_AT,
    ):
        pass
    with open_experience_ledger(
        database,
        scope=make_scope(schema_version="2.0.0"),
        repository_root=repository,
    ) as ledger:
        assert ledger.verify_integrity().entry_count == 0
