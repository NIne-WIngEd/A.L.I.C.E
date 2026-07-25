"""P2.8d promoted-lineage and derived-state deletion integrity tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from alice_memory.candidate_assessment import (
    MemoryCandidateAssessmentAuthorization,
    assess_memory_candidate,
)
from alice_memory.deletion import (
    ORDINARY_MEMORY_DELETION_SCOPE,
    MemoryDeletionAuthorization,
    MemoryDeletionRequestAuthorization,
    MemoryDeletionStateError,
    MemoryDeletionValidationError,
    delete_memory,
    load_memory_deletion,
    request_memory_deletion,
)
from alice_memory.formation import (
    MemoryCandidateCreateRequest,
    MemoryCandidateWriteAuthorization,
    propose_memory_candidate,
)
from alice_memory.promotion import (
    MemoryCandidatePromotionAuthorization,
    promote_memory_candidate,
)
from alice_memory.sensitive_crypto import InMemoryTestKeyProtector
from alice_memory.sensitive_deletion import (
    SENSITIVE_MEMORY_DELETION_SCOPE,
    SensitiveMemoryDeletionAuthorization,
    SensitiveMemoryDeletionRequestAuthorization,
    SensitiveMemoryDeletionValidationError,
    delete_sensitive_memory,
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


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _source(ref: str) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="approved_manual_entry",
        source_ref=ref,
        support_relation="supports",
    )


def _ordinary_request(
    memory_id: str,
    *,
    content: str = "ordinary deletion integrity content",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        memory_key=f"integrity.{memory_id}",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-25T15:00:00Z",
        verified_at="2026-07-25T15:00:00Z",
        rayan_confirmed=True,
        sources=(_source(f"private-source:{memory_id}"),),
    )


def _create_ordinary(connection, memory_id="ordinary-integrity", *, content=None):
    return create_memory(
        connection,
        request=_ordinary_request(
            memory_id,
            content=(
                content
                if content is not None
                else f"ordinary deletion integrity content {memory_id}"
            ),
        ),
        authorization=MemoryWriteAuthorization(
            actor="test-writer",
            allowed=True,
        ),
        created_at="2026-07-25T15:00:00Z",
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
        issued_at="2026-07-25T15:01:00Z",
        expires_at="2026-07-25T15:03:00Z",
    )


def _delete_ordinary(connection, memory_id: str):
    request_memory_deletion(
        connection,
        memory_id=memory_id,
        authorization=_ordinary_request_auth(memory_id),
        requested_at="2026-07-25T15:00:30Z",
    )
    return delete_memory(
        connection,
        memory_id=memory_id,
        authorization=_ordinary_delete_auth(memory_id),
        deleted_at="2026-07-25T15:02:00Z",
    )


def _promote_real_candidate(connection, candidate_id="candidate-integrity"):
    candidate = propose_memory_candidate(
        connection,
        request=MemoryCandidateCreateRequest(
            candidate_id=candidate_id,
            content="promoted candidate private plaintext",
            memory_key=f"candidate.{candidate_id}",
            category="project",
            knowledge_status="rayan_statement",
            confidence=1.0,
            data_classification="PRIVATE",
            recorded_at="2026-07-25T15:00:00Z",
            sources=(
                MemorySourceSpec(
                    source_type="rayan_direct_statement",
                    source_ref="private-candidate-source-ref",
                    support_relation="supports",
                ),
            ),
            origin="explicit_user",
            rayan_confirmed=True,
        ),
        authorization=MemoryCandidateWriteAuthorization(
            actor="candidate-proposer",
            allowed=True,
        ),
        proposed_at="2026-07-25T15:00:05Z",
    )
    assess_memory_candidate(
        connection,
        candidate_id=candidate.candidate_id,
        authorization=MemoryCandidateAssessmentAuthorization(
            actor="candidate-assessor",
            allowed=True,
        ),
        assessed_at="2026-07-25T15:00:10Z",
    )
    return promote_memory_candidate(
        connection,
        candidate_id=candidate.candidate_id,
        authorization=MemoryCandidatePromotionAuthorization(
            actor="candidate-promoter",
            allowed=True,
            candidate_id=candidate.candidate_id,
            authorization_id="candidate-promotion-auth",
            user_confirmed=False,
        ),
        promoted_at="2026-07-25T15:00:20Z",
    )


def _insert_linked_candidate(
    connection,
    *,
    candidate_id: str,
    memory_id: str,
    content: str,
    content_sha256: str | None = None,
    state: str = "promoted",
) -> None:
    digest = content_sha256 or hashlib.sha256(content.encode("utf-8")).hexdigest()
    connection.execute(
        """
        INSERT INTO memory_candidates (
            candidate_id, schema_version, content, content_sha256, memory_key,
            category, knowledge_status, confidence, data_classification,
            valid_from, valid_to, time_precision, recorded_at, verified_at,
            rayan_confirmed, validity_state, retention_state, candidate_state,
            origin, proposed_by, policy_version, model, model_version,
            prompt_version, run_id, promoted_memory_id, rejection_reason,
            created_at, updated_at
        ) VALUES (
            ?, 3, ?, ?, NULL, 'project', 'rayan_statement', 1.0, 'PRIVATE',
            NULL, NULL, NULL, '2026-07-25T15:00:00Z', NULL,
            1, 'current', 'durable', ?, 'explicit_user', 'fixture',
            NULL, NULL, NULL, NULL, NULL, ?, NULL,
            '2026-07-25T15:00:00Z', '2026-07-25T15:00:00Z'
        )
        """,
        (candidate_id, content, digest, state, memory_id),
    )
    connection.execute(
        """
        INSERT INTO memory_candidate_sources (
            candidate_source_id, candidate_id, source_type, source_ref,
            support_relation, created_at
        ) VALUES (?, ?, 'approved_manual_entry', ?, 'supports', ?)
        """,
        (
            str(uuid.uuid4()),
            candidate_id,
            f"private-lineage-source:{candidate_id}",
            "2026-07-25T15:00:00Z",
        ),
    )
    connection.execute(
        """
        INSERT INTO memory_candidate_events (
            candidate_event_id, candidate_id, event_type, actor,
            details_json, created_at
        ) VALUES (?, ?, 'promoted', 'fixture', '{}', ?)
        """,
        (str(uuid.uuid4()), candidate_id, "2026-07-25T15:00:00Z"),
    )


def _sensitive_request(memory_id="sensitive-integrity"):
    return MemoryCreateRequest(
        memory_id=memory_id,
        content="sensitive integrity private plaintext",
        memory_key=f"sensitive.{memory_id}",
        category="episodic",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="HIGHLY_SENSITIVE",
        recorded_at="2026-07-25T16:00:00Z",
        rayan_confirmed=True,
        sources=(_source(f"private-sensitive-source:{memory_id}"),),
    )


def _create_sensitive(connection, vault, repository, memory_id="sensitive-integrity"):
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


def _request_sensitive_delete(connection, memory_id: str):
    return request_sensitive_memory_deletion(
        connection,
        memory_id=memory_id,
        authorization=_sensitive_request_auth(memory_id),
        requested_at="2026-07-25T16:00:30Z",
    )


def test_ordinary_deletion_purges_real_promoted_candidate_lineage(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        promotion = _promote_real_candidate(connection)
        memory_id = promotion.memory.memory_id
        candidate_id = promotion.candidate.candidate_id
        result = _delete_ordinary(connection, memory_id)

        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_sources"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_events"
        ).fetchone()[0] == 0

        details = json.loads(
            connection.execute(
                "SELECT details_json FROM memory_events WHERE event_id = ?",
                (result.tombstone.event_id,),
            ).fetchone()[0]
        )
        lineage = details["promoted_candidate_lineage"]
        assert lineage["candidate_count"] == 1
        assert lineage["source_count"] == 1
        assert lineage["event_count"] == 3
        assert lineage["candidate_id_sha256"] == [
            hashlib.sha256(candidate_id.encode()).hexdigest()
        ]
        serialized = json.dumps(details, sort_keys=True)
        assert candidate_id not in serialized
        assert "promoted candidate private plaintext" not in serialized
        assert "private-candidate-source-ref" not in serialized


def test_direct_memory_records_zero_candidate_lineage(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_ordinary(connection)
        result = _delete_ordinary(connection, "ordinary-integrity")
        details = json.loads(
            connection.execute(
                "SELECT details_json FROM memory_events WHERE event_id = ?",
                (result.tombstone.event_id,),
            ).fetchone()[0]
        )
        assert details["promoted_candidate_lineage"] == {
            "candidate_count": 0,
            "candidate_id_sha256": [],
            "event_count": 0,
            "integrity_version": "p2.8d-v1",
            "purged": True,
            "source_count": 0,
        }


def test_multiple_linked_candidates_are_all_purged(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        for candidate_id in ("candidate-a", "candidate-b"):
            _insert_linked_candidate(
                connection,
                candidate_id=candidate_id,
                memory_id=memory.memory_id,
                content=f"ordinary deletion integrity content {memory.memory_id}",
                content_sha256=memory.content_sha256,
            )
        result = _delete_ordinary(connection, memory.memory_id)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 0
        details = json.loads(
            connection.execute(
                "SELECT details_json FROM memory_events WHERE event_id = ?",
                (result.tombstone.event_id,),
            ).fetchone()[0]
        )
        assert details["promoted_candidate_lineage"]["candidate_count"] == 2
        assert details["promoted_candidate_lineage"]["source_count"] == 2
        assert details["promoted_candidate_lineage"]["event_count"] == 2


def test_linked_candidate_digest_mismatch_rolls_back(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        _insert_linked_candidate(
            connection,
            candidate_id="candidate-mismatch",
            memory_id=memory.memory_id,
            content="different plaintext",
        )
        request_memory_deletion(
            connection,
            memory_id=memory.memory_id,
            authorization=_ordinary_request_auth(memory.memory_id),
            requested_at="2026-07-25T15:00:30Z",
        )
        with pytest.raises(MemoryDeletionValidationError):
            delete_memory(
                connection,
                memory_id=memory.memory_id,
                authorization=_ordinary_delete_auth(memory.memory_id),
                deleted_at="2026-07-25T15:02:00Z",
            )
        assert load_memory(
            connection,
            memory_id=memory.memory_id,
        ).deletion_state == "pending_deletion"
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_tombstones"
        ).fetchone()[0] == 0


def test_nonpromoted_linked_candidate_rolls_back(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        _insert_linked_candidate(
            connection,
            candidate_id="candidate-invalid-state",
            memory_id=memory.memory_id,
            content=f"ordinary deletion integrity content {memory.memory_id}",
            content_sha256=memory.content_sha256,
            state="validated",
        )
        request_memory_deletion(
            connection,
            memory_id=memory.memory_id,
            authorization=_ordinary_request_auth(memory.memory_id),
            requested_at="2026-07-25T15:00:30Z",
        )
        with pytest.raises(MemoryDeletionValidationError):
            delete_memory(
                connection,
                memory_id=memory.memory_id,
                authorization=_ordinary_delete_auth(memory.memory_id),
                deleted_at="2026-07-25T15:02:00Z",
            )
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 1


def test_candidate_purge_failure_rolls_back_entire_deletion(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        promotion = _promote_real_candidate(connection)
        memory_id = promotion.memory.memory_id
        request_memory_deletion(
            connection,
            memory_id=memory_id,
            authorization=_ordinary_request_auth(memory_id),
            requested_at="2026-07-25T15:00:30Z",
        )
        connection.execute(
            """
            CREATE TRIGGER reject_candidate_event_delete
            BEFORE DELETE ON memory_candidate_events
            BEGIN
                SELECT RAISE(ABORT, 'reject candidate event deletion');
            END
            """
        )
        with pytest.raises(MemoryDeletionValidationError):
            delete_memory(
                connection,
                memory_id=memory_id,
                authorization=_ordinary_delete_auth(memory_id),
                deleted_at="2026-07-25T15:02:00Z",
            )
        assert load_memory(connection, memory_id=memory_id)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_tombstones"
        ).fetchone()[0] == 0


def test_unmanaged_cache_table_blocks_ordinary_deletion(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        connection.execute(
            "CREATE TABLE memory_cache (memory_id TEXT, content TEXT)"
        )
        connection.execute(
            "INSERT INTO memory_cache VALUES (?, ?)",
            (memory.memory_id, "cached private plaintext"),
        )
        request_memory_deletion(
            connection,
            memory_id=memory.memory_id,
            authorization=_ordinary_request_auth(memory.memory_id),
            requested_at="2026-07-25T15:00:30Z",
        )
        with pytest.raises(MemoryDeletionValidationError):
            delete_memory(
                connection,
                memory_id=memory.memory_id,
                authorization=_ordinary_delete_auth(memory.memory_id),
                deleted_at="2026-07-25T15:02:00Z",
            )
        assert load_memory(connection, memory_id=memory.memory_id)
        assert connection.execute(
            "SELECT content FROM memory_cache"
        ).fetchone()[0] == "cached private plaintext"


def test_unmanaged_summary_table_blocks_sensitive_deletion(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_sensitive(connection, vault, repository)
        connection.execute(
            "CREATE TABLE memory_summaries (memory_id TEXT, summary TEXT)"
        )
        connection.execute(
            "INSERT INTO memory_summaries VALUES (?, ?)",
            (memory.memory_id, "derived sensitive summary"),
        )
        _request_sensitive_delete(connection, memory.memory_id)
        with pytest.raises(SensitiveMemoryDeletionValidationError):
            delete_sensitive_memory(
                connection,
                memory_id=memory.memory_id,
                authorization=_sensitive_delete_auth(memory.memory_id),
                deleted_at="2026-07-25T16:02:00Z",
            )
        assert load_memory(connection, memory_id=memory.memory_id)


def test_sensitive_deletion_purges_linked_candidate_plaintext(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_sensitive(connection, vault, repository)
        candidate_id = "private candidate identifier with spaces"
        _insert_linked_candidate(
            connection,
            candidate_id=candidate_id,
            memory_id=memory.memory_id,
            content="sensitive integrity private plaintext",
            content_sha256=memory.content_sha256,
        )
        _request_sensitive_delete(connection, memory.memory_id)
        result = delete_sensitive_memory(
            connection,
            memory_id=memory.memory_id,
            authorization=_sensitive_delete_auth(memory.memory_id),
            deleted_at="2026-07-25T16:02:00Z",
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 0
        details_json = connection.execute(
            "SELECT details_json FROM memory_events WHERE event_id = ?",
            (result.tombstone.event_id,),
        ).fetchone()[0]
        assert candidate_id not in details_json
        assert "sensitive integrity private plaintext" not in details_json
        details = json.loads(details_json)
        assert details["promoted_candidate_lineage"]["candidate_count"] == 1


def test_tampered_lineage_proof_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        result = _delete_ordinary(connection, memory.memory_id)
        row = connection.execute(
            "SELECT details_json FROM memory_events WHERE event_id = ?",
            (result.tombstone.event_id,),
        ).fetchone()
        details = json.loads(row[0])
        details["promoted_candidate_lineage"]["purged"] = False
        connection.execute(
            "UPDATE memory_events SET details_json = ? WHERE event_id = ?",
            (json.dumps(details), result.tombstone.event_id),
        )
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id=memory.memory_id)


def test_plaintext_field_in_deletion_audit_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        result = _delete_ordinary(connection, memory.memory_id)
        row = connection.execute(
            "SELECT details_json FROM memory_events WHERE event_id = ?",
            (result.tombstone.event_id,),
        ).fetchone()
        details = json.loads(row[0])
        details["plaintext"] = "deleted private content"
        connection.execute(
            "UPDATE memory_events SET details_json = ? WHERE event_id = ?",
            (json.dumps(details), result.tombstone.event_id),
        )
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id=memory.memory_id)


def test_invalid_candidate_digest_proof_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        result = _delete_ordinary(connection, memory.memory_id)
        row = connection.execute(
            "SELECT details_json FROM memory_events WHERE event_id = ?",
            (result.tombstone.event_id,),
        ).fetchone()
        details = json.loads(row[0])
        lineage = details["promoted_candidate_lineage"]
        lineage["candidate_count"] = 1
        lineage["candidate_id_sha256"] = ["not-a-digest"]
        connection.execute(
            "UPDATE memory_events SET details_json = ? WHERE event_id = ?",
            (json.dumps(details), result.tombstone.event_id),
        )
        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id=memory.memory_id)


def test_completed_retry_preserves_candidate_lineage_purge(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        promotion = _promote_real_candidate(connection)
        memory_id = promotion.memory.memory_id
        first = _delete_ordinary(connection, memory_id)
        second = delete_memory(
            connection,
            memory_id=memory_id,
            authorization=_ordinary_delete_auth(memory_id),
            deleted_at="2026-07-25T15:02:30Z",
        )
        assert second == first
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 0


def test_unlinked_candidate_remains_outside_targeted_deletion(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_ordinary(connection)
        candidate = propose_memory_candidate(
            connection,
            request=MemoryCandidateCreateRequest(
                candidate_id="independent-candidate",
                content=f"ordinary deletion integrity content {memory.memory_id}",
                category="project",
                knowledge_status="rayan_statement",
                confidence=1.0,
                data_classification="PRIVATE",
                recorded_at="2026-07-25T15:00:00Z",
                sources=(
                    MemorySourceSpec(
                        source_type="rayan_direct_statement",
                        source_ref="independent-source",
                        support_relation="supports",
                    ),
                ),
                origin="explicit_user",
                rayan_confirmed=True,
            ),
            authorization=MemoryCandidateWriteAuthorization(
                actor="candidate-proposer",
                allowed=True,
            ),
            proposed_at="2026-07-25T15:00:05Z",
        )
        _delete_ordinary(connection, memory.memory_id)
        assert connection.execute(
            "SELECT content FROM memory_candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()[0] == f"ordinary deletion integrity content {memory.memory_id}"
