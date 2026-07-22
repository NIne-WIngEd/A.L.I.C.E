"""P2.4 valid-time resolution tests."""

from __future__ import annotations

from pathlib import Path

from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.temporal import (
    correct_memory,
    list_current_memories_for_key,
    list_memory_history,
    mark_memory_conflict,
    resolve_memory_at,
    supersede_memory,
)


def _test_source() -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="approved_manual_entry",
        source_ref="test-suite:synthetic-memory",
        support_relation="supports",
    )


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
        reason="temporal test",
    )


def _request(
    memory_id: str,
    content: str,
    *,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        sources=(_test_source(),),
        memory_id=memory_id,
        content=content,
        memory_key="project.phase",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def test_supersession_uses_half_open_temporal_boundaries(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "phase-1",
                "Phase 1",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        supersede_memory(
            connection,
            memory_id="phase-1",
            replacement=_request(
                "phase-2",
                "Phase 2",
                valid_from="2026-06-01T00:00:00Z",
            ),
            authorization=_authorization(),
            superseded_at="2026-06-01T00:00:00Z",
        )

        before = resolve_memory_at(
            connection,
            memory_key="project.phase",
            at="2026-05-31T23:59:59Z",
        )
        boundary = resolve_memory_at(
            connection,
            memory_key="project.phase",
            at="2026-06-01T00:00:00Z",
        )

        assert [item.memory_id for item in before.memories] == [
            "phase-1"
        ]
        assert [item.memory_id for item in boundary.memories] == [
            "phase-2"
        ]


def test_temporal_resolution_handles_timezone_offsets(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "phase-1",
                "Phase 1",
                valid_from="2026-01-01T00:00:00-06:00",
                valid_to="2026-01-02T00:00:00-06:00",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T06:00:00Z",
        )

        resolution = resolve_memory_at(
            connection,
            memory_key="project.phase",
            at="2026-01-01T12:00:00Z",
        )

        assert [item.memory_id for item in resolution.memories] == [
            "phase-1"
        ]


def test_current_listing_excludes_historical_superseded_record(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "phase-1",
                "Phase 1",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        supersede_memory(
            connection,
            memory_id="phase-1",
            replacement=_request(
                "phase-2",
                "Phase 2",
                valid_from="2026-06-01T00:00:00Z",
            ),
            authorization=_authorization(),
            superseded_at="2026-06-01T00:00:00Z",
        )

        current = list_current_memories_for_key(
            connection,
            memory_key="project.phase",
        )

        assert [item.memory_id for item in current] == [
            "phase-2"
        ]


def test_history_preserves_old_and_new_records(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "phase-1",
                "Phase 1",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        supersede_memory(
            connection,
            memory_id="phase-1",
            replacement=_request(
                "phase-2",
                "Phase 2",
                valid_from="2026-06-01T00:00:00Z",
            ),
            authorization=_authorization(),
            superseded_at="2026-06-01T00:00:00Z",
        )

        history = list_memory_history(
            connection,
            memory_key="project.phase",
        )

        assert [item.memory_id for item in history] == [
            "phase-1",
            "phase-2",
        ]


def test_corrected_record_is_history_but_not_valid_time_truth(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "wrong",
                "Incorrect phase",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        correct_memory(
            connection,
            memory_id="wrong",
            replacement=_request(
                "correct",
                "Correct phase",
            ),
            authorization=_authorization(),
            corrected_at="2026-02-01T00:00:00Z",
        )

        history = list_memory_history(
            connection,
            memory_key="project.phase",
        )
        resolution = resolve_memory_at(
            connection,
            memory_key="project.phase",
            at="2026-01-15T00:00:00Z",
        )

        assert {
            item.memory_id
            for item in history
        } == {
            "wrong",
            "correct",
        }
        assert [item.memory_id for item in resolution.memories] == [
            "correct"
        ]


def test_temporal_resolution_surfaces_conflicting_candidates(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "first",
                "Candidate A",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "second",
                "Candidate B",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:01Z",
        )

        mark_memory_conflict(
            connection,
            first_memory_id="first",
            second_memory_id="second",
            authorization=_authorization(),
            disputed_at="2026-01-01T00:01:00Z",
        )

        resolution = resolve_memory_at(
            connection,
            memory_key="project.phase",
            at="2026-02-01T00:00:00Z",
        )

        assert {
            item.memory_id
            for item in resolution.memories
        } == {
            "first",
            "second",
        }
        assert resolution.has_conflict is True
        assert len(resolution.conflict_pairs) == 1


def test_temporal_resolution_returns_empty_when_no_interval_matches(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "future",
                "Future phase",
                valid_from="2027-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        resolution = resolve_memory_at(
            connection,
            memory_key="project.phase",
            at="2026-07-21T00:00:00Z",
        )

        assert resolution.memories == ()
        assert resolution.has_conflict is False
