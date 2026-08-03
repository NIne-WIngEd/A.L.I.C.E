from __future__ import annotations

import pytest

from cognitive_kernel import (
    LifecycleJournalIsolationError,
    open_lifecycle_journal,
)
from lifecycle_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    make_decision,
    make_scope,
    paths,
)


def test_record_scope_mismatch_is_rejected(tmp_path) -> None:
    repository, database = paths(tmp_path)
    foreign = make_decision(
        scope=make_scope(host_instance_id="other-host")
    )
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as journal:
        with pytest.raises(LifecycleJournalIsolationError):
            journal.append_record(foreign, committed_at=COMMITTED_AT)


def test_existing_database_is_bound_to_product_host_and_domain(tmp_path) -> None:
    repository, database = paths(tmp_path)
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ):
        pass
    with pytest.raises(LifecycleJournalIsolationError):
        with open_lifecycle_journal(
            database,
            scope=make_scope(encryption_domain="other-domain"),
            repository_root=repository,
        ):
            pass
