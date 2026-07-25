"""P2.8b derived-index purge, rebuild, and deletion-guarantee tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from alice_vault.semantic_retrieval import load_semantic_policy
from alice_memory.deletion import (
    ORDINARY_MEMORY_DELETION_SCOPE,
    MemoryDeletionAuthorization,
    MemoryDeletionRequestAuthorization,
    delete_memory,
    load_memory_deletion,
    request_memory_deletion,
)
from alice_memory.deletion_indexes import (
    MemoryDeletionIndexStateError,
    delete_memory_with_index_purge,
    purge_deleted_memory_indexes,
    rebuild_indexes_after_memory_deletion,
    verify_deleted_memory_absent_from_indexes,
)
from alice_memory.hybrid_retrieval import (
    hybrid_search_memories,
    search_memories_semantic,
)
from alice_memory.lexical_index import (
    build_memory_lexical_index,
    memory_lexical_index_path,
    search_memory_lexical_candidates,
    verify_memory_lexical_index,
)
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import (
    MemoryLexicalIndexError,
    MemoryRetrievalAuthorization,
    MemorySearchRequest,
    StaleMemoryLexicalIndexError,
)
from alice_memory.semantic_index import (
    MemorySemanticIndexError,
    StaleMemorySemanticIndexError,
    build_memory_semantic_index,
    memory_semantic_index_root,
    verify_memory_semantic_index,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


class FakeEncoder:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            values = [0.0] * self.dimension
            lowered = text.casefold()
            if "deletedtoken" in lowered:
                values[0] = 1.0
            elif "keeptoken" in lowered:
                values[1] = 1.0
            else:
                values[2] = 1.0
            rows.append(values)
        return rows


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _source(memory_id: str) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="approved_manual_entry",
        source_ref=f"test-suite:{memory_id}",
        support_relation="supports",
    )


def _create(connection, memory_id: str, content: str):
    return create_memory(
        connection,
        request=MemoryCreateRequest(
            memory_id=memory_id,
            content=content,
            memory_key=f"deletion-index.{memory_id}",
            category="project",
            knowledge_status="verified_fact",
            confidence=1.0,
            data_classification="PRIVATE",
            recorded_at="2026-07-25T12:00:00Z",
            verified_at="2026-07-25T12:00:00Z",
            rayan_confirmed=True,
            sources=(_source(memory_id),),
        ),
        authorization=MemoryWriteAuthorization(
            actor="test-writer",
            allowed=True,
            reason="synthetic fixture",
        ),
        created_at="2026-07-25T12:00:00Z",
    )


def _request_auth(memory_id: str):
    return MemoryDeletionRequestAuthorization(
        actor="rayan",
        allowed=True,
        memory_id=memory_id,
        deletion_scope=ORDINARY_MEMORY_DELETION_SCOPE,
        authorization_id=f"request-{memory_id}",
    )


def _delete_auth(memory_id: str):
    return MemoryDeletionAuthorization(
        actor="rayan",
        allowed=True,
        memory_id=memory_id,
        deletion_scope=ORDINARY_MEMORY_DELETION_SCOPE,
        authorization_id=f"delete-{memory_id}",
        strongly_confirmed=True,
        issued_at="2026-07-25T12:01:00Z",
        expires_at="2026-07-25T12:03:00Z",
    )


def _read_auth():
    return MemoryRetrievalAuthorization(
        actor="test-reader",
        allowed=True,
        purpose="deletion index verification",
        max_classification="PRIVATE",
    )


def _model():
    policy = load_semantic_policy()
    return FakeEncoder(policy.model.embedding_dimension)


def _build_all(connection, vault, repository, model):
    build_memory_lexical_index(
        connection,
        vault,
        repository_root=repository,
        built_at="2026-07-25T12:00:30Z",
    )
    return build_memory_semantic_index(
        connection,
        vault,
        model=model,
        repository_root=repository,
        built_at="2026-07-25T12:00:30Z",
    )


def _request_delete(connection, memory_id: str):
    return request_memory_deletion(
        connection,
        memory_id=memory_id,
        authorization=_request_auth(memory_id),
        requested_at="2026-07-25T12:00:45Z",
    )


def test_pending_deletion_invalidates_all_existing_retrieval_indexes(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        _create(connection, "keep-me", "keeptoken retained memory")
        _build_all(connection, vault, repository, model)
        _request_delete(connection, "delete-me")

        lexical_path = memory_lexical_index_path(
            vault,
            repository_root=repository,
        )
        with pytest.raises(StaleMemoryLexicalIndexError):
            verify_memory_lexical_index(connection, lexical_path)
        with pytest.raises(StaleMemorySemanticIndexError):
            verify_memory_semantic_index(
                connection,
                vault,
                repository_root=repository,
            )
        with pytest.raises(StaleMemoryLexicalIndexError):
            search_memories(
                connection,
                vault,
                request=MemorySearchRequest(query="deletedtoken"),
                authorization=_read_auth(),
                repository_root=repository,
            )
        with pytest.raises(StaleMemorySemanticIndexError):
            search_memories_semantic(
                connection,
                vault,
                request=MemorySearchRequest(query="deletedtoken"),
                authorization=_read_auth(),
                model=model,
                repository_root=repository,
            )
        with pytest.raises(StaleMemoryLexicalIndexError):
            hybrid_search_memories(
                connection,
                vault,
                request=MemorySearchRequest(query="deletedtoken"),
                authorization=_read_auth(),
                model=model,
                repository_root=repository,
            )


def test_delete_wrapper_removes_lexical_and_all_semantic_generations(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        _create(connection, "keep-me", "keeptoken retained memory")
        first = _build_all(connection, vault, repository, model)
        _create(connection, "third", "other retained memory")
        second = _build_all(connection, vault, repository, model)
        assert first.index_id != second.index_id

        semantic_root = memory_semantic_index_root(
            vault,
            repository_root=repository,
        )
        assert (semantic_root / first.index_id).exists()
        assert (semantic_root / second.index_id).exists()

        _request_delete(connection, "delete-me")
        result = delete_memory_with_index_purge(
            connection,
            vault,
            memory_id="delete-me",
            authorization=_delete_auth("delete-me"),
            deleted_at="2026-07-25T12:02:00Z",
            repository_root=repository,
        )

        assert result.purge.lexical_root_removed
        assert result.purge.semantic_root_removed
        assert not result.purge.lexical_root.exists()
        assert not result.purge.semantic_root.exists()
        assert result.deletion.tombstone.deleted_memory_id == "delete-me"


def test_purge_requires_an_intact_completed_deletion(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        with pytest.raises(MemoryDeletionIndexStateError):
            purge_deleted_memory_indexes(
                connection,
                vault,
                memory_id="delete-me",
                repository_root=repository,
            )

        _request_delete(connection, "delete-me")
        with pytest.raises(MemoryDeletionIndexStateError):
            purge_deleted_memory_indexes(
                connection,
                vault,
                memory_id="delete-me",
                repository_root=repository,
            )

        delete_memory(
            connection,
            memory_id="delete-me",
            authorization=_delete_auth("delete-me"),
            deleted_at="2026-07-25T12:02:00Z",
        )
        deletion = load_memory_deletion(connection, memory_id="delete-me")
        connection.execute(
            "UPDATE memory_events SET details_json = '{}' WHERE event_id = ?",
            (deletion.tombstone.event_id,),
        )

        with pytest.raises(MemoryDeletionIndexStateError):
            purge_deleted_memory_indexes(
                connection,
                vault,
                memory_id="delete-me",
                repository_root=repository,
            )


def test_purge_is_idempotent(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        _build_all(connection, vault, repository, model)
        _request_delete(connection, "delete-me")
        delete_memory(
            connection,
            memory_id="delete-me",
            authorization=_delete_auth("delete-me"),
            deleted_at="2026-07-25T12:02:00Z",
        )

        first = purge_deleted_memory_indexes(
            connection,
            vault,
            memory_id="delete-me",
            repository_root=repository,
        )
        second = purge_deleted_memory_indexes(
            connection,
            vault,
            memory_id="delete-me",
            repository_root=repository,
        )

        assert first.lexical_root_removed
        assert first.semantic_root_removed
        assert not second.lexical_root_removed
        assert not second.semantic_root_removed


def test_rebuild_after_deletion_excludes_deleted_memory_from_all_modes(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        _create(connection, "keep-me", "keeptoken retained memory")
        _build_all(connection, vault, repository, model)
        _request_delete(connection, "delete-me")
        delete_memory_with_index_purge(
            connection,
            vault,
            memory_id="delete-me",
            authorization=_delete_auth("delete-me"),
            deleted_at="2026-07-25T12:02:00Z",
            repository_root=repository,
        )

        rebuilt = rebuild_indexes_after_memory_deletion(
            connection,
            vault,
            memory_id="delete-me",
            model=model,
            built_at="2026-07-25T12:03:00Z",
            repository_root=repository,
        )

        assert rebuilt.verification.lexical_manifest.record_count == 1
        assert rebuilt.verification.semantic_manifest.record_count == 1
        verify_deleted_memory_absent_from_indexes(
            connection,
            vault,
            memory_id="delete-me",
            repository_root=repository,
        )

        lexical = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(query="deletedtoken"),
            authorization=_read_auth(),
            repository_root=repository,
        )
        semantic = search_memories_semantic(
            connection,
            vault,
            request=MemorySearchRequest(query="deletedtoken"),
            authorization=_read_auth(),
            model=model,
            repository_root=repository,
        )
        hybrid = hybrid_search_memories(
            connection,
            vault,
            request=MemorySearchRequest(query="deletedtoken"),
            authorization=_read_auth(),
            model=model,
            repository_root=repository,
        )
        keep = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(query="keeptoken"),
            authorization=_read_auth(),
            repository_root=repository,
        )

        assert lexical.results == ()
        assert all(item.memory_id != "delete-me" for item in semantic.results)
        assert all(item.memory_id != "delete-me" for item in hybrid.results)
        assert [item.memory_id for item in keep.results] == ["keep-me"]


def test_destroy_and_second_rebuild_preserve_deleted_memory_absence(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        _create(connection, "keep-me", "keeptoken retained memory")
        _build_all(connection, vault, repository, model)
        _request_delete(connection, "delete-me")
        delete_memory(
            connection,
            memory_id="delete-me",
            authorization=_delete_auth("delete-me"),
            deleted_at="2026-07-25T12:02:00Z",
        )

        first = rebuild_indexes_after_memory_deletion(
            connection,
            vault,
            memory_id="delete-me",
            model=model,
            built_at="2026-07-25T12:03:00Z",
            repository_root=repository,
        )
        second = rebuild_indexes_after_memory_deletion(
            connection,
            vault,
            memory_id="delete-me",
            model=model,
            built_at="2026-07-25T12:04:00Z",
            repository_root=repository,
        )

        assert first.verification.lexical_manifest.record_count == 1
        assert second.verification.lexical_manifest.record_count == 1
        assert second.purge.lexical_root_removed
        assert second.purge.semantic_root_removed
        assert search_memory_lexical_candidates(
            second.verification.lexical_index_path,
            query="deletedtoken",
            limit=10,
        ) == []


def test_lexical_verification_detects_same_count_row_substitution(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "keep-me", "keeptoken retained memory")
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-25T12:00:30Z",
        )
        path = memory_lexical_index_path(
            vault,
            repository_root=repository,
        )

        with sqlite3.connect(path) as index:
            index.execute(
                "UPDATE indexed_memories SET memory_id = 'deleted-id'"
            )
            index.execute(
                "UPDATE memory_fts SET memory_id = 'deleted-id'"
            )

        with pytest.raises(MemoryLexicalIndexError):
            verify_memory_lexical_index(connection, path)


def test_semantic_verification_detects_rehashed_map_substitution(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "keep-me", "keeptoken retained memory")
        build_memory_semantic_index(
            connection,
            vault,
            model=model,
            repository_root=repository,
            built_at="2026-07-25T12:00:30Z",
        )
        _, index_path = verify_memory_semantic_index(
            connection,
            vault,
            repository_root=repository,
        )
        map_path = index_path / "memory-map.jsonl"
        manifest_path = index_path / "semantic-manifest.json"

        row = json.loads(map_path.read_text(encoding="utf-8"))
        row["memory_id"] = "deleted-id"
        map_bytes = (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        map_path.write_bytes(map_bytes)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["memory_map_sha256"] = hashlib.sha256(map_bytes).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

        with pytest.raises(MemorySemanticIndexError):
            verify_memory_semantic_index(
                connection,
                vault,
                repository_root=repository,
            )


def test_failed_rebuild_destroys_old_semantic_generations_first(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()
    bad_model = FakeEncoder(model.dimension + 1)

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        _create(connection, "keep-me", "keeptoken retained memory")
        old = _build_all(connection, vault, repository, model)
        semantic_root = memory_semantic_index_root(
            vault,
            repository_root=repository,
        )
        assert (semantic_root / old.index_id).exists()

        _request_delete(connection, "delete-me")
        delete_memory(
            connection,
            memory_id="delete-me",
            authorization=_delete_auth("delete-me"),
            deleted_at="2026-07-25T12:02:00Z",
        )

        with pytest.raises(MemorySemanticIndexError):
            rebuild_indexes_after_memory_deletion(
                connection,
                vault,
                memory_id="delete-me",
                model=bad_model,
                built_at="2026-07-25T12:03:00Z",
                repository_root=repository,
            )

        assert not (semantic_root / old.index_id).exists()
        lexical_path = memory_lexical_index_path(
            vault,
            repository_root=repository,
        )
        verified = verify_memory_lexical_index(connection, lexical_path)
        assert verified.record_count == 1


def test_authoritative_deletion_remains_committed_when_purge_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()

    with open_memory_store(vault, repository_root=repository) as connection:
        _create(connection, "delete-me", "deletedtoken private memory")
        _build_all(connection, vault, repository, model)
        _request_delete(connection, "delete-me")

        def fail_remove(_path):
            raise OSError("synthetic filesystem failure")

        monkeypatch.setattr(
            "alice_memory.deletion_indexes._remove_path",
            fail_remove,
        )
        with pytest.raises(OSError):
            delete_memory_with_index_purge(
                connection,
                vault,
                memory_id="delete-me",
                authorization=_delete_auth("delete-me"),
                deleted_at="2026-07-25T12:02:00Z",
                repository_root=repository,
            )

        assert (
            load_memory_deletion(
                connection,
                memory_id="delete-me",
            ).tombstone.deleted_memory_id
            == "delete-me"
        )
        with pytest.raises(StaleMemoryLexicalIndexError):
            verify_memory_lexical_index(
                connection,
                memory_lexical_index_path(
                    vault,
                    repository_root=repository,
                ),
            )
