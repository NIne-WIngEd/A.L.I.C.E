"""P2.9a deterministic synthetic Memory Core fixture tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.evaluation_contract import (
    MemoryEvaluationBenchmark,
    load_memory_evaluation_benchmark,
)
from alice_memory.evaluation_fixtures import (
    CANDIDATE_IDS,
    FIXTURE_CONTENT,
    MEMORY_IDS,
    SOURCE_REFS,
    MemoryEvaluationFixtureError,
    build_memory_evaluation_fixture,
    memory_evaluation_snapshot_id,
    memory_evaluation_snapshot_material,
)
from alice_memory.sensitive_storage import SENSITIVE_CONTENT_SENTINEL
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
    load_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.temporal import resolve_memory_at


def _setup(tmp_path: Path, name: str = "one"):
    repository = tmp_path / f"repo-{name}"
    vault = tmp_path / f"vault-{name}"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _build(tmp_path: Path, name: str = "one"):
    repository, vault = _setup(tmp_path, name)
    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        snapshot = build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )
        material = memory_evaluation_snapshot_material(connection)
    return repository, vault, snapshot, material


def test_fixture_build_matches_approved_snapshot(tmp_path: Path) -> None:
    _, _, snapshot, _ = _build(tmp_path)
    benchmark = load_memory_evaluation_benchmark()

    assert snapshot.fixture_version == "p2.9a-v1"
    assert snapshot.snapshot_id == benchmark.fixture_snapshot_id
    assert snapshot.benchmark_id == benchmark.benchmark_id
    assert snapshot.benchmark_digest == benchmark.digest


def test_fixture_has_expected_authoritative_boundaries(
    tmp_path: Path,
) -> None:
    _, _, snapshot, _ = _build(tmp_path)

    assert len(snapshot.active_memory_ids) == 11
    assert snapshot.sensitive_memory_ids == (
        MEMORY_IDS["sensitive"],
    )
    assert snapshot.deleted_memory_ids == (
        MEMORY_IDS["deleted"],
    )
    assert snapshot.candidate_ids == (
        CANDIDATE_IDS["unpromoted"],
    )
    assert MEMORY_IDS["deleted"] not in snapshot.active_memory_ids
    assert CANDIDATE_IDS["unpromoted"] not in snapshot.active_memory_ids


def test_fixture_sources_are_synthetic_and_complete(
    tmp_path: Path,
) -> None:
    _, _, snapshot, _ = _build(tmp_path)

    assert len(snapshot.source_refs) == 12
    expected = set(SOURCE_REFS.values()) - {SOURCE_REFS["deleted"]}
    assert set(snapshot.source_refs) == expected
    assert all(value.startswith("fixture:") for value in snapshot.source_refs)
    assert all(":\\" not in value for value in snapshot.source_refs)


def test_sensitive_fixture_uses_only_encrypted_sentinel(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )
        row = connection.execute(
            """
            SELECT content, content_sha256
            FROM memories
            WHERE memory_id = ?
            """,
            (MEMORY_IDS["sensitive"],),
        ).fetchone()
        payload = connection.execute(
            """
            SELECT ciphertext, nonce, key_id
            FROM memory_sensitive_payloads
            WHERE memory_id = ?
            """,
            (MEMORY_IDS["sensitive"],),
        ).fetchone()

        assert row["content"] == SENSITIVE_CONTENT_SENTINEL
        assert FIXTURE_CONTENT["sensitive"] not in row["content"]
        assert bytes(payload["ciphertext"])
        assert bytes(payload["nonce"])
        assert str(payload["key_id"])


def test_deleted_fixture_has_tombstone_and_no_authoritative_row(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )

        assert connection.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?",
            (MEMORY_IDS["deleted"],),
        ).fetchone() is None
        tombstone = connection.execute(
            """
            SELECT deleted_memory_id, content_sha256, deletion_scope
            FROM memory_tombstones
            WHERE deleted_memory_id = ?
            """,
            (MEMORY_IDS["deleted"],),
        ).fetchone()
        assert tombstone is not None
        assert tombstone["deletion_scope"] == (
            "ordinary_memory_and_dependents"
        )
        assert len(str(tombstone["content_sha256"])) == 64


def test_unpromoted_candidate_remains_non_authoritative(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )
        candidate = connection.execute(
            """
            SELECT candidate_state, promoted_memory_id, content
            FROM memory_candidates
            WHERE candidate_id = ?
            """,
            (CANDIDATE_IDS["unpromoted"],),
        ).fetchone()

        assert candidate["candidate_state"] == "proposed"
        assert candidate["promoted_memory_id"] is None
        assert candidate["content"] == FIXTURE_CONTENT["candidate"]
        assert connection.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?",
            (CANDIDATE_IDS["unpromoted"],),
        ).fetchone() is None


def test_temporal_fixture_resolves_current_and_historical_state(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )

        historical = resolve_memory_at(
            connection,
            memory_key="fixture.atlas.runtime",
            at="2024-06-01T00:00:00Z",
        )
        current = resolve_memory_at(
            connection,
            memory_key="fixture.atlas.runtime",
            at="2026-06-01T00:00:00Z",
        )

        assert [item.memory_id for item in historical.memories] == [
            MEMORY_IDS["temporal_old"]
        ]
        assert [item.memory_id for item in current.memories] == [
            MEMORY_IDS["temporal_current"]
        ]
        assert historical.memories[0].knowledge_status == "historical"
        assert current.memories[0].knowledge_status == "verified_fact"


def test_conflict_fixture_preserves_both_disputed_records(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )

        first = load_memory(
            connection,
            memory_id=MEMORY_IDS["conflict_a"],
        )
        second = load_memory(
            connection,
            memory_id=MEMORY_IDS["conflict_b"],
        )
        relation = connection.execute(
            """
            SELECT from_memory_id, to_memory_id, relation_type
            FROM memory_relations
            WHERE relation_type = 'conflicts_with'
            """
        ).fetchone()

        assert first.knowledge_status == "disputed"
        assert second.knowledge_status == "disputed"
        assert first.validity_state == "disputed"
        assert second.validity_state == "disputed"
        assert {
            relation["from_memory_id"],
            relation["to_memory_id"],
        } == {
            MEMORY_IDS["conflict_a"],
            MEMORY_IDS["conflict_b"],
        }


def test_correction_fixture_excludes_corrected_record_from_truth(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )

        old = load_memory(
            connection,
            memory_id=MEMORY_IDS["correction_old"],
        )
        current = load_memory(
            connection,
            memory_id=MEMORY_IDS["correction_current"],
        )
        resolved = resolve_memory_at(
            connection,
            memory_key="fixture.atlas.report_folder",
            at="2026-06-01T00:00:00Z",
        )

        assert old.knowledge_status == "superseded"
        assert old.validity_state == "historical"
        assert current.knowledge_status == "verified_fact"
        assert [item.memory_id for item in resolved.memories] == [
            MEMORY_IDS["correction_current"]
        ]


def test_snapshot_digest_is_reproducible_across_stores(
    tmp_path: Path,
) -> None:
    _, _, first, first_material = _build(tmp_path, "first")
    _, _, second, second_material = _build(tmp_path, "second")

    assert first.snapshot_id == second.snapshot_id
    assert first_material == second_material


def test_snapshot_material_excludes_random_and_sensitive_bytes(
    tmp_path: Path,
) -> None:
    _, _, _, material = _build(tmp_path)
    encoded = repr(material)

    assert "ciphertext" not in encoded
    assert "nonce" not in encoded
    assert "event_id" not in encoded
    assert "key_id" not in encoded
    assert FIXTURE_CONTENT["sensitive"] not in encoded


def test_fixture_refuses_nonempty_authoritative_store(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=MemoryCreateRequest(
                memory_id="preexisting-memory",
                content="Synthetic preexisting memory.",
                memory_key="fixture.preexisting",
                category="project",
                knowledge_status="verified_fact",
                confidence=1.0,
                data_classification="PRIVATE",
                recorded_at="2026-01-01T00:00:00Z",
                verified_at="2026-01-01T00:00:00Z",
                rayan_confirmed=True,
                sources=(
                    MemorySourceSpec(
                        source_type="approved_manual_entry",
                        source_ref="fixture:preexisting",
                        support_relation="supports",
                    ),
                ),
            ),
            authorization=MemoryWriteAuthorization(
                actor="test",
                allowed=True,
            ),
            created_at="2026-01-01T00:00:00Z",
        )

        with pytest.raises(
            MemoryEvaluationFixtureError,
            match="requires an empty store",
        ):
            build_memory_evaluation_fixture(
                connection,
                vault,
                repository_root=repository,
            )


def test_fixture_fails_closed_on_unapproved_snapshot(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    approved = load_memory_evaluation_benchmark()
    tampered = MemoryEvaluationBenchmark(
        benchmark_id=approved.benchmark_id,
        test_set_version=approved.test_set_version,
        title=approved.title,
        synthetic_only=approved.synthetic_only,
        fixture_snapshot_id="f" * 64,
        cases=approved.cases,
        digest=approved.digest,
        source_path=approved.source_path,
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with pytest.raises(
            MemoryEvaluationFixtureError,
            match="does not match the approved benchmark",
        ):
            build_memory_evaluation_fixture(
                connection,
                vault,
                repository_root=repository,
                benchmark=tampered,
            )
        assert memory_evaluation_snapshot_id(connection) != "f" * 64


def test_benchmark_references_fixture_state_consistently(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    benchmark = load_memory_evaluation_benchmark()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        snapshot = build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )
        active = set(snapshot.active_memory_ids)
        deleted = set(snapshot.deleted_memory_ids)
        candidates = set(snapshot.candidate_ids)
        sources = set(snapshot.source_refs)

        for case in benchmark.cases:
            assert set(case.expected_memory_ids) <= active
            assert set(case.expected_source_refs) <= sources
            assert set(case.forbidden_memory_ids) <= active | deleted
            assert set(case.forbidden_candidate_ids) <= candidates
