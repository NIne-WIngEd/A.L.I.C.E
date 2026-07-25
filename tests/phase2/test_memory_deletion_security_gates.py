"""P2.8e adversarial deletion gates and final deletion benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_vault.semantic_retrieval import load_semantic_policy
from alice_memory.deletion import (
    ORDINARY_MEMORY_DELETION_SCOPE,
    MemoryDeletionAuthorization,
    MemoryDeletionRequestAuthorization,
    MemoryDeletionStateError,
    delete_memory,
    load_memory_deletion,
    request_memory_deletion,
)
from alice_memory.deletion_evaluation import (
    MemoryDeletionBenchmarkError,
    run_memory_deletion_benchmark,
)
from alice_memory.deletion_indexes import (
    delete_memory_with_index_purge,
    delete_sensitive_memory_with_index_purge,
    rebuild_indexes_after_memory_deletion,
)
from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.semantic_index import build_memory_semantic_index
from alice_memory.sensitive_crypto import InMemoryTestKeyProtector
from alice_memory.sensitive_deletion import (
    SENSITIVE_MEMORY_DELETION_SCOPE,
    SensitiveMemoryDeletionAuthorization,
    SensitiveMemoryDeletionRequestAuthorization,
    SensitiveMemoryDeletionStateError,
    delete_sensitive_memory,
    load_sensitive_memory_deletion,
    request_sensitive_memory_deletion,
)
from alice_memory.sensitive_storage import (
    SensitiveMemoryWriteAuthorization,
    create_sensitive_memory,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
    load_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


class FakeEncoder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            row = [0.0] * self.dimension
            lowered = str(text).casefold()
            row[0 if "retain" in lowered else 1] = 1.0
            rows.append(row)
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
        source_ref=f"p2.8e:{memory_id}",
        support_relation="supports",
    )


def _ordinary_request(memory_id: str, content: str | None = None):
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content or f"private deletion security content {memory_id}",
        memory_key=f"p2.8e.{memory_id}",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-25T16:00:00Z",
        verified_at="2026-07-25T16:00:00Z",
        rayan_confirmed=True,
        sources=(_source(memory_id),),
    )


def _create_ordinary(connection, memory_id="delete-ordinary", content=None):
    return create_memory(
        connection,
        request=_ordinary_request(memory_id, content),
        authorization=MemoryWriteAuthorization(actor="writer", allowed=True),
        created_at="2026-07-25T16:00:00Z",
    )


def _ordinary_request_auth(memory_id: str):
    return MemoryDeletionRequestAuthorization(
        actor="rayan",
        allowed=True,
        memory_id=memory_id,
        deletion_scope=ORDINARY_MEMORY_DELETION_SCOPE,
        authorization_id=f"request-{memory_id}",
    )


def _ordinary_delete_auth(memory_id: str):
    return MemoryDeletionAuthorization(
        actor="rayan",
        allowed=True,
        memory_id=memory_id,
        deletion_scope=ORDINARY_MEMORY_DELETION_SCOPE,
        authorization_id=f"delete-{memory_id}",
        strongly_confirmed=True,
        issued_at="2026-07-25T16:01:00Z",
        expires_at="2026-07-25T16:03:00Z",
    )


def _request_ordinary(connection, memory_id="delete-ordinary"):
    return request_memory_deletion(
        connection,
        memory_id=memory_id,
        authorization=_ordinary_request_auth(memory_id),
        requested_at="2026-07-25T16:00:30Z",
    )


def _delete_ordinary(connection, memory_id="delete-ordinary"):
    if connection.execute(
        "SELECT 1 FROM memories WHERE memory_id = ?",
        (memory_id,),
    ).fetchone() is None:
        _create_ordinary(connection, memory_id)
    _request_ordinary(connection, memory_id)
    return delete_memory(
        connection,
        memory_id=memory_id,
        authorization=_ordinary_delete_auth(memory_id),
        deleted_at="2026-07-25T16:02:00Z",
    )


def _sensitive_request(memory_id: str):
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=f"highly sensitive deletion security content {memory_id}",
        memory_key=f"p2.8e.sensitive.{memory_id}",
        category="episodic",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="HIGHLY_SENSITIVE",
        recorded_at="2026-07-25T16:00:00Z",
        rayan_confirmed=True,
        sources=(_source(memory_id),),
    )


def _create_sensitive(connection, vault, repository, memory_id="delete-sensitive"):
    return create_sensitive_memory(
        connection,
        vault,
        request=_sensitive_request(memory_id),
        authorization=SensitiveMemoryWriteAuthorization(
            actor="rayan",
            allowed=True,
            purpose="memory.user_requested_storage",
            authorization_id=f"create-{memory_id}",
            directly_requested=True,
        ),
        created_at="2026-07-25T16:00:00Z",
        repository_root=repository,
        key_protector=InMemoryTestKeyProtector(),
    )


def _sensitive_request_auth(memory_id: str):
    return SensitiveMemoryDeletionRequestAuthorization(
        actor="rayan",
        allowed=True,
        purpose="memory.user_requested_sensitive_deletion",
        authorization_id=f"request-{memory_id}",
        memory_id=memory_id,
        deletion_scope=SENSITIVE_MEMORY_DELETION_SCOPE,
        directly_requested=True,
    )


def _sensitive_delete_auth(memory_id: str):
    return SensitiveMemoryDeletionAuthorization(
        actor="rayan",
        allowed=True,
        purpose="memory.user_requested_sensitive_deletion",
        authorization_id=f"delete-{memory_id}",
        memory_id=memory_id,
        deletion_scope=SENSITIVE_MEMORY_DELETION_SCOPE,
        directly_requested=True,
        strongly_confirmed=True,
        issued_at="2026-07-25T16:01:00Z",
        expires_at="2026-07-25T16:03:00Z",
    )


def _request_sensitive(connection, memory_id="delete-sensitive"):
    return request_sensitive_memory_deletion(
        connection,
        memory_id=memory_id,
        authorization=_sensitive_request_auth(memory_id),
        requested_at="2026-07-25T16:00:30Z",
    )


def _delete_sensitive(connection, memory_id="delete-sensitive"):
    _request_sensitive(connection, memory_id)
    return delete_sensitive_memory(
        connection,
        memory_id=memory_id,
        authorization=_sensitive_delete_auth(memory_id),
        deleted_at="2026-07-25T16:02:00Z",
    )


def _event_details(connection, event_id: str):
    row = connection.execute(
        "SELECT details_json FROM memory_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    return json.loads(str(row[0]))


def _write_event_details(connection, event_id: str, details) -> None:
    connection.execute(
        "UPDATE memory_events SET details_json = ? WHERE event_id = ?",
        (json.dumps(details, sort_keys=True), event_id),
    )


def _model():
    policy = load_semantic_policy()
    return FakeEncoder(policy.model.embedding_dimension)


def test_tombstone_identifier_tampering_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _delete_ordinary(connection)
        connection.execute(
            "UPDATE memory_tombstones SET tombstone_id = ? WHERE deleted_memory_id = ?",
            ("00000000-0000-0000-0000-000000000000", "delete-ordinary"),
        )
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="delete-ordinary")


def test_tombstone_digest_tampering_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _delete_ordinary(connection)
        connection.execute(
            "UPDATE memory_tombstones SET content_sha256 = ? WHERE deleted_memory_id = ?",
            ("0" * 64, "delete-ordinary"),
        )
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="delete-ordinary")


def test_completion_digest_tampering_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _delete_ordinary(connection)
        details = _event_details(connection, result.tombstone.event_id)
        details["content_sha256"] = "0" * 64
        _write_event_details(connection, result.tombstone.event_id, details)
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="delete-ordinary")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unexpected", "hidden"),
        ("strong_confirmation", False),
    ],
)
def test_completion_structure_tampering_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _delete_ordinary(connection)
        details = _event_details(connection, result.tombstone.event_id)
        details[field] = value
        _write_event_details(connection, result.tombstone.event_id, details)
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="delete-ordinary")


def test_negative_dependent_count_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _delete_ordinary(connection)
        details = _event_details(connection, result.tombstone.event_id)
        details["dependent_counts"]["sources"] = -1
        _write_event_details(connection, result.tombstone.event_id, details)
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="delete-ordinary")


def test_duplicate_candidate_digest_proof_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _delete_ordinary(connection)
        details = _event_details(connection, result.tombstone.event_id)
        digest = "a" * 64
        lineage = details["promoted_candidate_lineage"]
        lineage["candidate_count"] = 2
        lineage["candidate_id_sha256"] = [digest, digest]
        _write_event_details(connection, result.tombstone.event_id, details)
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="delete-ordinary")


def test_request_audit_plaintext_injection_blocks_completed_load(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _delete_ordinary(connection)
        details = _event_details(connection, result.request_event_id)
        details["plaintext"] = "deleted private content"
        _write_event_details(connection, result.request_event_id, details)
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="delete-ordinary")


def test_request_target_tampering_blocks_final_deletion(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_ordinary(connection)
        request = _request_ordinary(connection)
        details = _event_details(connection, request.request_event_id)
        details["target_memory_id"] = "other-memory"
        _write_event_details(connection, request.request_event_id, details)
        with pytest.raises(MemoryDeletionStateError):
            delete_memory(
                connection,
                memory_id="delete-ordinary",
                authorization=_ordinary_delete_auth("delete-ordinary"),
                deleted_at="2026-07-25T16:02:00Z",
            )
        assert load_memory(connection, memory_id="delete-ordinary").deletion_state == "pending_deletion"


def test_sensitive_request_kind_tampering_blocks_final_deletion(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository)
        request = _request_sensitive(connection)
        details = _event_details(connection, request.request_event_id)
        details["memory_kind"] = "ordinary"
        _write_event_details(connection, request.request_event_id, details)
        with pytest.raises(SensitiveMemoryDeletionStateError):
            delete_sensitive_memory(
                connection,
                memory_id="delete-sensitive",
                authorization=_sensitive_delete_auth("delete-sensitive"),
                deleted_at="2026-07-25T16:02:00Z",
            )


def test_sensitive_payload_proof_tampering_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository)
        result = _delete_sensitive(connection)
        details = _event_details(connection, result.tombstone.event_id)
        details["encrypted_payload_deleted"] = False
        _write_event_details(connection, result.tombstone.event_id, details)
        with pytest.raises(SensitiveMemoryDeletionStateError):
            load_sensitive_memory_deletion(connection, memory_id="delete-sensitive")


def test_duplicate_sensitive_allowed_audit_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository)
        result = _delete_sensitive(connection)
        row = connection.execute(
            """
            SELECT actor, purpose, authorization_id, operation, decision, created_at
            FROM sensitive_memory_access_events
            WHERE authorization_id = ? AND operation = 'delete' AND decision = 'allowed'
            """,
            (result.authorization_id,),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO sensitive_memory_access_events (
                access_event_id, memory_id, actor, purpose, authorization_id,
                operation, decision, created_at
            ) VALUES ('duplicate-delete-audit', NULL, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row),
        )
        with pytest.raises(SensitiveMemoryDeletionStateError):
            load_sensitive_memory_deletion(connection, memory_id="delete-sensitive")


def test_benchmark_rejects_overlapping_sets(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        with pytest.raises(MemoryDeletionBenchmarkError):
            run_memory_deletion_benchmark(
                connection,
                vault,
                deleted_memory_ids=("same",),
                retained_memory_ids=("same",),
                repository_root=repository,
            )


def test_final_deletion_benchmark_passes_after_destroy_and_rebuild(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_ordinary(connection, "retain-memory", "retain token durable memory")
        _create_ordinary(connection, "delete-ordinary")
        _create_sensitive(connection, vault, repository, "delete-sensitive")
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-25T16:00:10Z",
        )
        build_memory_semantic_index(
            connection,
            vault,
            model=model,
            repository_root=repository,
            built_at="2026-07-25T16:00:10Z",
        )

        _request_ordinary(connection, "delete-ordinary")
        delete_memory_with_index_purge(
            connection,
            vault,
            memory_id="delete-ordinary",
            authorization=_ordinary_delete_auth("delete-ordinary"),
            deleted_at="2026-07-25T16:02:00Z",
            repository_root=repository,
        )
        _request_sensitive(connection, "delete-sensitive")
        delete_sensitive_memory_with_index_purge(
            connection,
            vault,
            memory_id="delete-sensitive",
            authorization=_sensitive_delete_auth("delete-sensitive"),
            deleted_at="2026-07-25T16:02:00Z",
            repository_root=repository,
        )
        rebuild_indexes_after_memory_deletion(
            connection,
            vault,
            memory_id="delete-sensitive",
            model=model,
            built_at="2026-07-25T16:04:00Z",
            repository_root=repository,
        )

        report = run_memory_deletion_benchmark(
            connection,
            vault,
            deleted_memory_ids=("delete-sensitive", "delete-ordinary"),
            retained_memory_ids=("retain-memory",),
            repository_root=repository,
        )
        assert report.passed is True
        assert report.benchmark_version == "p2.8e-v1"
        assert report.deleted_memory_ids == ("delete-ordinary", "delete-sensitive")
        assert report.indexed_record_count == 1
        assert len(report.gate_names) == 7


def test_benchmark_rejects_stale_indexes(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    model = _model()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_ordinary(connection, "retain-memory", "retain token durable memory")
        _create_ordinary(connection, "delete-ordinary")
        _delete_ordinary(connection, "delete-ordinary")
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-25T16:04:00Z",
        )
        build_memory_semantic_index(
            connection,
            vault,
            model=model,
            repository_root=repository,
            built_at="2026-07-25T16:04:00Z",
        )
        _create_ordinary(connection, "late-memory", "late authoritative mutation")
        with pytest.raises(MemoryDeletionBenchmarkError):
            run_memory_deletion_benchmark(
                connection,
                vault,
                deleted_memory_ids=("delete-ordinary",),
                retained_memory_ids=("retain-memory",),
                repository_root=repository,
            )
