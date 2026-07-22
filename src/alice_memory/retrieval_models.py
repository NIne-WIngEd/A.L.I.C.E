"""Data contracts for deterministic Phase 2 memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass


class MemoryRetrievalError(RuntimeError):
    """Base error for Phase 2 memory retrieval."""


class MemoryRetrievalAuthorizationError(MemoryRetrievalError):
    """Raised when retrieval is not explicitly authorized."""


class MemoryRetrievalValidationError(MemoryRetrievalError):
    """Raised when a retrieval request is invalid."""


class MemoryLexicalIndexError(MemoryRetrievalError):
    """Raised when the derived lexical index cannot be used safely."""


class StaleMemoryLexicalIndexError(MemoryLexicalIndexError):
    """Raised when the derived index no longer matches authoritative memory."""


@dataclass(frozen=True)
class MemoryRetrievalAuthorization:
    """Explicit deterministic authorization for metadata-safe memory retrieval."""

    actor: str
    allowed: bool
    purpose: str
    max_classification: str = "PRIVATE"


@dataclass(frozen=True)
class MemorySearchRequest:
    """One deterministic lexical memory search request."""

    query: str
    limit: int = 10
    memory_key: str | None = None
    category: str | None = None
    at: str | None = None
    include_historical: bool = False
    include_archived: bool = False
    expand_conflicts: bool = True


@dataclass(frozen=True)
class MemorySearchResult:
    """Metadata-only memory search result.

    Plaintext content and snippets are intentionally absent.
    """

    memory_id: str
    score: float
    memory_key: str | None
    category: str
    knowledge_status: str
    confidence: float
    data_classification: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str
    validity_state: str
    retention_state: str
    conflict_memory_ids: tuple[str, ...]
    matched_by: str = "lexical"


@dataclass(frozen=True)
class MemorySearchResponse:
    """Metadata-safe retrieval response."""

    query: str
    results: tuple[MemorySearchResult, ...]
    index_id: str
    authoritative_digest: str
