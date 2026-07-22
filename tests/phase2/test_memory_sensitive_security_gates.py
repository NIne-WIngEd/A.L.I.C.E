"""P2.6c adversarial security gates for HIGHLY_SENSITIVE memory isolation.

These tests intentionally add no new production capability. They prove that the
P2.6 protected-access path remains separate from ordinary P2.5 retrieval and
that local P1 sensitive-memory access cannot be confused with P4 external
transmission authorization.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alice_vault.semantic_retrieval import load_semantic_policy
from alice_memory.hybrid_retrieval import (
    hybrid_search_memories,
    search_memories_semantic,
)
from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import (
    MemoryRetrievalAuthorization,
    MemorySearchRequest,
)
from alice_memory.semantic_index import build_memory_semantic_index
from alice_memory.sensitive_access import (
    SensitiveMemoryAccessAuthorization,
    SensitiveMemoryAccessAuthorizationError,
    SensitiveMemoryMetadataSearchRequest,
    search_sensitive_memory_metadata,
)
from alice_memory.sensitive_crypto import InMemoryTestKeyProtector
from alice_memory.sensitive_storage import (
    SensitiveMemoryWriteAuthorization,
    create_sensitive_memory,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


ACCESSED_AT = "2026-07-22T12:00:00Z"
EXPIRES_AT = "2026-07-22T12:05:00Z"
SENSITIVE_ID = "sensitive-security-gate"
PRIVATE_ID = "private-security-gate"
SENSITIVE_TOKEN = "sensitiveisolationtoken"


class FakeEncoder:
    """Deterministic local encoder used only to exercise semantic isolation."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            values = [0.0] * self.dimension
            lowered = str(text).casefold()
            if SENSITIVE_TOKEN in lowered:
                values[0] = 1.0
            elif "ordinaryretrievaltoken" in lowered:
                values[1] = 1.0
            else:
                values[2] = 1.0
            rows.append(values)
        return rows


def _source(ref: str) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="approved_manual_entry",
        source_ref=ref,
        support_relation="supports",
    )


def _private_request() -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=PRIVATE_ID,
        content="ordinaryretrievaltoken ordinary private memory",
        memory_key="security.ordinary",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-22T00:00:00Z",
        sources=(_source("test-suite:p2.6c-private"),),
        rayan_confirmed=True,
    )


def _sensitive_request() -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=SENSITIVE_ID,
        content=f"{SENSITIVE_TOKEN} appears only inside encrypted sensitive content",
        memory_key="security.sensitive",
        category="episodic",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="HIGHLY_SENSITIVE",
        recorded_at="2026-07-22T00:00:00Z",
        sources=(_source("test-suite:p2.6c-sensitive"),),
        rayan_confirmed=True,
    )


def _ordinary_write_auth() -> MemoryWriteAuthorization:
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="P2.6c isolation fixture",
    )


def _sensitive_write_auth() -> SensitiveMemoryWriteAuthorization:
    return SensitiveMemoryWriteAuthorization(
        actor="rayan",
        allowed=True,
        purpose="memory.user_requested_storage",
        authorization_id="auth-p26c-sensitive-create",
        directly_requested=True,
    )


def _ordinary_retrieval_auth() -> MemoryRetrievalAuthorization:
    return MemoryRetrievalAuthorization(
        actor="test",
        allowed=True,
        purpose="p2.6c_isolation_test",
        max_classification="PRIVATE",
    )


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def test_highly_sensitive_memory_never_surfaces_through_ordinary_retrieval(
    tmp_path: Path,
) -> None:
    """Prove lexical, semantic, and hybrid retrieval cannot surface it."""
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    policy = load_semantic_policy()
    model = FakeEncoder(policy.model.embedding_dimension)

    with open_memory_store(vault, repository_root=repository) as connection:
        create_memory(
            connection,
            request=_private_request(),
            authorization=_ordinary_write_auth(),
            created_at="2026-07-22T00:00:00Z",
        )
        create_sensitive_memory(
            connection,
            vault,
            request=_sensitive_request(),
            authorization=_sensitive_write_auth(),
            created_at="2026-07-22T00:00:00Z",
            repository_root=repository,
            key_protector=protector,
        )

        lexical_manifest = build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-22T00:01:00Z",
        )
        semantic_manifest = build_memory_semantic_index(
            connection,
            vault,
            model=model,
            repository_root=repository,
            built_at="2026-07-22T00:01:00Z",
        )

        # Only the ordinary PRIVATE record is eligible for either derived index.
        assert lexical_manifest.record_count == 1
        assert semantic_manifest.record_count == 1

        request = MemorySearchRequest(query=SENSITIVE_TOKEN, limit=10)
        authorization = _ordinary_retrieval_auth()

        lexical = search_memories(
            connection,
            vault,
            request=request,
            authorization=authorization,
            repository_root=repository,
        )
        semantic = search_memories_semantic(
            connection,
            vault,
            request=request,
            authorization=authorization,
            model=model,
            repository_root=repository,
        )
        hybrid = hybrid_search_memories(
            connection,
            vault,
            request=request,
            authorization=authorization,
            model=model,
            repository_root=repository,
        )

        for response in (lexical, semantic, hybrid):
            returned_ids = {item.memory_id for item in response.results}
            assert SENSITIVE_ID not in returned_ids
            assert all(
                item.data_classification != "HIGHLY_SENSITIVE"
                for item in response.results
            )


def test_local_sensitive_authorization_cannot_grant_external_transmission(
    tmp_path: Path,
) -> None:
    """The local P2.6b authorization vocabulary excludes transmission."""
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        authorization = SensitiveMemoryAccessAuthorization(
            actor="rayan",
            allowed=True,
            purpose="memory.local_sensitive_access",
            authorization_id="auth-p26c-invalid-external-op",
            allowed_operations=("highly_sensitive.transmit_external",),
            expires_at=EXPIRES_AT,
        )

        with pytest.raises(SensitiveMemoryAccessAuthorizationError):
            search_sensitive_memory_metadata(
                connection,
                request=SensitiveMemoryMetadataSearchRequest(
                    category="episodic"
                ),
                authorization=authorization,
                accessed_at=ACCESSED_AT,
            )


def test_permission_registry_separates_local_search_from_external_transmission() -> None:
    """Lock the P1 local-read vs P4 external-disclosure policy boundary."""
    repository_root = Path(__file__).resolve().parents[2]
    registry = yaml.safe_load(
        (repository_root / "policies" / "permissions.yaml").read_text(
            encoding="utf-8"
        )
    )
    permissions = {
        item["id"]: item
        for item in registry["permissions"]
    }

    local_search = permissions["memory.search"]
    external = permissions["highly_sensitive.transmit_external"]

    assert local_search["level"] == "P1"
    assert "HIGHLY_SENSITIVE" in local_search["allowed_data_classes"]

    assert external["level"] == "P4"
    assert external["confirmation"] == "strong"
    assert external["standing_authorization_allowed"] is False
    assert external["allowed_data_classes"] == ["HIGHLY_SENSITIVE"]

    # A local read permission must never be treated as the external action.
    assert local_search["id"] != external["id"]
    assert local_search["level"] != external["level"]
