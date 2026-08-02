from __future__ import annotations

import pytest

from cognitive_kernel import (
    UnsafeExperienceLedgerPathError,
    validate_experience_ledger_path,
)


def test_database_inside_repository_is_rejected(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(
        UnsafeExperienceLedgerPathError,
        match="inside the public repository",
    ):
        validate_experience_ledger_path(
            repository / "ledger.sqlite3",
            repository_root=repository,
        )


def test_database_outside_repository_is_accepted(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    database = tmp_path / "vault" / "ledger.sqlite3"
    assert validate_experience_ledger_path(
        database,
        repository_root=repository,
    ) == database.resolve(strict=False)
