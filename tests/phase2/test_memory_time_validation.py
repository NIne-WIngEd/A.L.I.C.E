"""P2.4 canonical timestamp and valid-time input validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.service import (
    MemoryCreateRequest,
    MemoryValidationError,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.store import open_memory_store


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return open_memory_store(
        vault,
        repository_root=repository,
    )


def _authorization() -> MemoryWriteAuthorization:
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="time validation test",
    )


def _request(
    *,
    valid_from: str | None,
    valid_to: str | None,
    recorded_at: str = "2026-07-21T00:00:00Z",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id="memory-1",
        content="Temporal test memory",
        memory_key="temporal.test",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at=recorded_at,
        rayan_confirmed=True,
    )


def test_create_memory_normalizes_timezone_offsets_before_persistence(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        memory = create_memory(
            connection,
            request=_request(
                valid_from="2026-01-02T00:00:00+14:00",
                valid_to="2026-01-01T23:00:00-12:00",
            ),
            authorization=_authorization(),
            created_at="2026-07-20T19:00:00-05:00",
        )

        assert memory.valid_from == "2026-01-01T10:00:00Z"
        assert memory.valid_to == "2026-01-02T11:00:00Z"
        assert memory.created_at == "2026-07-21T00:00:00Z"


def test_create_memory_rejects_actual_reverse_interval_across_offsets(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(
                    valid_from="2026-01-01T23:00:00-06:00",
                    valid_to="2026-01-02T04:00:00Z",
                ),
                authorization=_authorization(),
                created_at="2026-07-21T00:00:00Z",
            )


def test_create_memory_rejects_naive_recorded_at(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(
                    valid_from=None,
                    valid_to=None,
                    recorded_at="2026-07-21T00:00:00",
                ),
                authorization=_authorization(),
                created_at="2026-07-21T00:00:00Z",
            )
