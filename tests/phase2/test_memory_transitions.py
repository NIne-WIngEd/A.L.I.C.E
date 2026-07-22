"""P2.4 correction, supersession, and conflict transition tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.service import (
    MemoryAlreadyExistsError,
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.temporal import (
    InvalidMemoryTransitionError,
    correct_memory,
    mark_memory_conflict,
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
        reason="transition test",
    )


def _request(
    memory_id: str,
    content: str,
    *,
    memory_key: str | None = "profile.location",
    valid_from: str | None = None,
    valid_to: str | None = None,
    classification: str = "PRIVATE",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        sources=(_test_source(),),
        memory_id=memory_id,
        content=content,
        memory_key=memory_key,
        category="profile",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification=classification,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def test_correction_preserves_old_record_and_links_replacement(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Incorrect location",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        result = correct_memory(
            connection,
            memory_id="old",
            replacement=_request(
                "new",
                "Correct location",
                memory_key=None,
            ),
            authorization=_authorization(),
            corrected_at="2026-07-21T00:01:00Z",
        )

        assert result.previous.memory_id == "old"
        assert result.previous.knowledge_status == "superseded"
        assert result.previous.validity_state == "historical"
        assert result.replacement.memory_id == "new"
        assert result.replacement.memory_key == "profile.location"
        assert result.replacement.validity_state == "current"
        assert result.relation.from_memory_id == "new"
        assert result.relation.to_memory_id == "old"
        assert result.relation.relation_type == "corrects"

        count = connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        assert count == 2


def test_transition_cannot_downgrade_classification(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Private value",
                classification="PRIVATE",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(InvalidMemoryTransitionError):
            correct_memory(
                connection,
                memory_id="old",
                replacement=_request(
                    "new",
                    "Downgraded value",
                    classification="INTERNAL",
                ),
                authorization=_authorization(),
                corrected_at="2026-07-21T00:01:00Z",
            )

        old = connection.execute(
            """
            SELECT validity_state
            FROM memories
            WHERE memory_id = 'old'
            """
        ).fetchone()
        assert old["validity_state"] == "current"


def test_correction_is_atomic_when_replacement_insert_fails(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Original",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "duplicate",
                "Existing",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:01Z",
        )

        with pytest.raises(MemoryAlreadyExistsError):
            correct_memory(
                connection,
                memory_id="old",
                replacement=_request(
                    "duplicate",
                    "Replacement",
                ),
                authorization=_authorization(),
                corrected_at="2026-07-21T00:01:00Z",
            )

        row = connection.execute(
            """
            SELECT knowledge_status, validity_state
            FROM memories
            WHERE memory_id = 'old'
            """
        ).fetchone()
        assert row["knowledge_status"] == "rayan_statement"
        assert row["validity_state"] == "current"

        relation_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_relations
            WHERE relation_type = 'corrects'
            """
        ).fetchone()[0]
        assert relation_count == 0


def test_supersession_closes_previous_validity_interval(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Old status",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        result = supersede_memory(
            connection,
            memory_id="old",
            replacement=_request(
                "new",
                "New status",
                valid_from="2026-06-01T00:00:00Z",
            ),
            authorization=_authorization(),
            superseded_at="2026-06-01T00:00:00Z",
        )

        assert result.previous.knowledge_status == "historical"
        assert result.previous.validity_state == "historical"
        assert result.previous.valid_to == "2026-06-01T00:00:00Z"
        assert result.replacement.valid_from == "2026-06-01T00:00:00Z"
        assert result.relation.relation_type == "supersedes"


def test_supersession_does_not_extend_existing_valid_to(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Old status",
                valid_from="2026-01-01T00:00:00Z",
                valid_to="2026-03-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        result = supersede_memory(
            connection,
            memory_id="old",
            replacement=_request(
                "new",
                "New status",
                valid_from="2026-06-01T00:00:00Z",
            ),
            authorization=_authorization(),
            superseded_at="2026-06-01T00:00:00Z",
        )

        assert result.previous.valid_to == "2026-03-01T00:00:00Z"


def test_supersession_requires_replacement_valid_from(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Old status",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        with pytest.raises(InvalidMemoryTransitionError):
            supersede_memory(
                connection,
                memory_id="old",
                replacement=_request(
                    "new",
                    "New status",
                    valid_from=None,
                ),
                authorization=_authorization(),
                superseded_at="2026-06-01T00:00:00Z",
            )


def test_supersession_rejects_naive_timestamp(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Old status",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        with pytest.raises(InvalidMemoryTransitionError):
            supersede_memory(
                connection,
                memory_id="old",
                replacement=_request(
                    "new",
                    "New status",
                    valid_from="2026-06-01T00:00:00",
                ),
                authorization=_authorization(),
                superseded_at="2026-06-01T00:00:00Z",
            )


def test_conflict_preserves_both_memories_and_is_idempotent(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "first",
                "Value A",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "second",
                "Value B",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:01Z",
        )

        first = mark_memory_conflict(
            connection,
            first_memory_id="first",
            second_memory_id="second",
            authorization=_authorization(),
            disputed_at="2026-07-21T00:01:00Z",
        )
        second = mark_memory_conflict(
            connection,
            first_memory_id="second",
            second_memory_id="first",
            authorization=_authorization(),
            disputed_at="2026-07-21T00:02:00Z",
        )

        assert first.first.validity_state == "disputed"
        assert first.second.knowledge_status == "disputed"
        assert first.relation.relation_id == second.relation.relation_id
        assert second.first.updated_at == "2026-07-21T00:01:00Z"

        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_relations
            WHERE relation_type = 'conflicts_with'
            """
        ).fetchone()[0]
        assert count == 1


def test_correction_writes_sanitized_audit_event(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Incorrect private content",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        correct_memory(
            connection,
            memory_id="old",
            replacement=_request(
                "new",
                "Correct private content",
            ),
            authorization=_authorization(),
            corrected_at="2026-07-21T00:01:00-05:00",
        )

        event = connection.execute(
            """
            SELECT event_type, actor, details_json, created_at
            FROM memory_events
            WHERE memory_id = ?
              AND event_type = 'corrected'
            """,
            ("old",),
        ).fetchone()

        assert event["event_type"] == "corrected"
        assert event["actor"] == "test"
        assert event["created_at"] == "2026-07-21T05:01:00Z"
        assert "Incorrect private content" not in event["details_json"]
        assert "Correct private content" not in event["details_json"]
        assert '"replacement_memory_id":"new"' in event["details_json"]


def test_supersession_writes_audit_event(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "Old status",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_authorization(),
            created_at="2026-01-01T00:00:00Z",
        )

        supersede_memory(
            connection,
            memory_id="old",
            replacement=_request(
                "new",
                "New status",
                valid_from="2026-06-01T00:00:00Z",
            ),
            authorization=_authorization(),
            superseded_at="2026-06-01T00:00:00Z",
        )

        event = connection.execute(
            """
            SELECT event_type, details_json
            FROM memory_events
            WHERE memory_id = ?
              AND event_type = 'superseded'
            """,
            ("old",),
        ).fetchone()

        assert event["event_type"] == "superseded"
        assert '"replacement_memory_id":"new"' in event["details_json"]
        assert '"closed_valid_to":"2026-06-01T00:00:00Z"' in event["details_json"]


def test_conflict_writes_one_audit_event_per_memory_and_remains_idempotent(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request("first", "Candidate A"),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request("second", "Candidate B"),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:01Z",
        )

        mark_memory_conflict(
            connection,
            first_memory_id="first",
            second_memory_id="second",
            authorization=_authorization(),
            disputed_at="2026-07-21T00:01:00Z",
        )
        mark_memory_conflict(
            connection,
            first_memory_id="second",
            second_memory_id="first",
            authorization=_authorization(),
            disputed_at="2026-07-21T00:02:00Z",
        )

        rows = connection.execute(
            """
            SELECT memory_id, event_type, details_json
            FROM memory_events
            WHERE event_type = 'conflict_marked'
            ORDER BY memory_id
            """
        ).fetchall()

        assert [row["memory_id"] for row in rows] == [
            "first",
            "second",
        ]
        assert all(row["event_type"] == "conflict_marked" for row in rows)
        assert "Candidate A" not in rows[0]["details_json"]
        assert "Candidate B" not in rows[1]["details_json"]
