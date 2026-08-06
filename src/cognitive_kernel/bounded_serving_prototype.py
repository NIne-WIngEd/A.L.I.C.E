"""Memory M2.4 reversible bounded-serving and retrieval-trace prototype."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
    require_sha256,
    require_text,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope
from .serving_contracts import (
    ContextPacket,
    ContextSelection,
    RetrievalTrace,
    RetrievalTraceStep,
)

BOUNDED_SERVING_PROTOTYPE_SCHEMA_VERSION = "1.0.0"
BOUNDED_SERVING_PROTOTYPE_STATE = "reversible_nonproduction"
INDEX_KINDS = frozenset({"lexical", "vector", "graph", "temporal"})
INDEX_STATES = frozenset({"ready", "stale", "unavailable", "rebuilding"})
FALLBACK_MODES = frozenset({"bounded_scan", "skip_unavailable", "custom"})


class BoundedServingPrototypeError(RuntimeError):
    """Base error for the M2.4 bounded-serving profile."""


class UnsafeBoundedServingPathError(BoundedServingPrototypeError):
    """Raised when prototype storage would enter the public repository."""


class BoundedServingIsolationError(BoundedServingPrototypeError):
    """Raised when a database belongs to a different product/host scope."""


class BoundedServingIntegrityError(BoundedServingPrototypeError):
    """Raised when persisted serving material fails integrity validation."""


class BoundedServingTransactionError(BoundedServingPrototypeError):
    """Raised when one bounded-serving transaction cannot complete."""


@dataclass(frozen=True)
class BoundedServingProfile:
    """Profile-selected retrieval budgets and fusion behavior."""

    scope: ProductHostScope
    authority_namespace_id: str
    profile_id: str
    item_budget: int
    byte_budget: int
    expansion_depth: int
    fusion_strategy: str
    lexical_weight: float
    vector_weight: float
    graph_weight: float
    mission_weight: float
    freshness_weight: float
    fallback_mode: str
    production_influence: bool = False

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        authority_namespace_id: object,
        profile_id: object,
        item_budget: object,
        byte_budget: object,
        expansion_depth: object,
        fusion_strategy: object = "weighted_score",
        lexical_weight: object = 1.0,
        vector_weight: object = 1.0,
        graph_weight: object = 1.0,
        mission_weight: object = 0.5,
        freshness_weight: object = 0.25,
        fallback_mode: object = "bounded_scan",
        production_influence: object = False,
    ) -> "BoundedServingProfile":
        if not isinstance(scope, ProductHostScope):
            raise CognitiveKernelContractError(
                "scope must be ProductHostScope"
            )
        scope.validate()
        if not isinstance(production_influence, bool):
            raise CognitiveKernelContractError(
                "production_influence must be boolean"
            )
        if production_influence:
            raise CognitiveKernelContractError(
                "M2.4 reversible profile may not influence production"
            )
        value = cls(
            scope=scope,
            authority_namespace_id=require_identifier(
                authority_namespace_id,
                "authority_namespace_id",
            ),
            profile_id=require_identifier(profile_id, "profile_id"),
            item_budget=_positive_integer(item_budget, "item_budget"),
            byte_budget=_positive_integer(byte_budget, "byte_budget"),
            expansion_depth=_non_negative_integer(
                expansion_depth,
                "expansion_depth",
            ),
            fusion_strategy=require_identifier(
                fusion_strategy,
                "fusion_strategy",
            ),
            lexical_weight=_non_negative_float(
                lexical_weight,
                "lexical_weight",
            ),
            vector_weight=_non_negative_float(
                vector_weight,
                "vector_weight",
            ),
            graph_weight=_non_negative_float(
                graph_weight,
                "graph_weight",
            ),
            mission_weight=_non_negative_float(
                mission_weight,
                "mission_weight",
            ),
            freshness_weight=_non_negative_float(
                freshness_weight,
                "freshness_weight",
            ),
            fallback_mode=require_identifier(
                fallback_mode,
                "fallback_mode",
            ),
            production_influence=False,
        )
        value.validate()
        return value

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": BOUNDED_SERVING_PROTOTYPE_SCHEMA_VERSION,
            "prototype_state": BOUNDED_SERVING_PROTOTYPE_STATE,
            "scope": self.scope.metadata_record(),
            "authority_namespace_id": self.authority_namespace_id,
            "profile_id": self.profile_id,
            "item_budget": self.item_budget,
            "byte_budget": self.byte_budget,
            "expansion_depth": self.expansion_depth,
            "fusion_strategy": self.fusion_strategy,
            "lexical_weight": self.lexical_weight,
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "mission_weight": self.mission_weight,
            "freshness_weight": self.freshness_weight,
            "fallback_mode": self.fallback_mode,
            "production_influence": self.production_influence,
        }

    def profile_sha256(self) -> str:
        return canonical_sha256(self.metadata_record())

    def validate(self) -> None:
        self.scope.validate()
        if self.fusion_strategy != "weighted_score":
            raise CognitiveKernelContractError(
                "M2.4 prototype currently implements weighted_score; "
                "successor fusion strategies remain contract-authorized"
            )
        if self.fallback_mode not in FALLBACK_MODES:
            raise CognitiveKernelContractError(
                "fallback_mode is not registered"
            )
        if self.fallback_mode != "bounded_scan":
            raise CognitiveKernelContractError(
                "M2.4 prototype currently implements bounded_scan; "
                "successor fallback modes remain contract-authorized"
            )
        if (
            self.lexical_weight
            + self.vector_weight
            + self.graph_weight
            + self.mission_weight
            + self.freshness_weight
        ) <= 0.0:
            raise CognitiveKernelContractError(
                "at least one fusion weight must be positive"
            )
        if self.production_influence:
            raise CognitiveKernelContractError(
                "M2.4 reversible profile may not influence production"
            )


@dataclass(frozen=True)
class ServingDocumentReceipt:
    record_id: str
    record_version_id: str
    generation: int
    content_digest: str
    document_sha256: str


@dataclass(frozen=True)
class BoundedServingReceipt:
    packet: ContextPacket
    trace: RetrievalTrace
    full_packet_content: tuple[dict[str, object], ...]
    fallback_used: bool
    stale_index_observed: bool
    receipt_sha256: str

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": BOUNDED_SERVING_PROTOTYPE_SCHEMA_VERSION,
            "packet_sha256": self.packet.packet_sha256,
            "trace_sha256": self.trace.trace_sha256,
            "record_ids": [
                str(item["record_id"])
                for item in self.full_packet_content
            ],
            "fallback_used": self.fallback_used,
            "stale_index_observed": self.stale_index_observed,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class BoundedServingIntegrityReport:
    checked_documents: int
    checked_packets: int
    checked_traces: int
    problems: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class _Candidate:
    record_id: str
    record_version_id: str
    source_kind: str
    authority_namespace_id: str
    content_digest: str
    full_content: dict[str, object]
    generation: int
    lexical_score: float
    vector_score: float
    graph_score: float
    mission_score: float
    freshness_score: float
    stale: bool
    reason_codes: tuple[str, ...]


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CognitiveKernelContractError(
            f"{field} must be a positive integer"
        )
    return value


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _non_negative_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveKernelContractError(
            f"{field} must be numeric"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise CognitiveKernelContractError(
            f"{field} must be finite and non-negative"
        )
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in "".join(
            character.lower()
            if character.isalnum()
            else " "
            for character in text
        ).split()
        if token
    )


def _cosine(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _normalized_embedding(
    values: Iterable[object] | None,
    field: str,
) -> tuple[float, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise CognitiveKernelContractError(
            f"{field} must be numeric sequence"
        )
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CognitiveKernelContractError(
                f"{field} must contain numeric values"
            )
        number = float(value)
        if not math.isfinite(number):
            raise CognitiveKernelContractError(
                f"{field} must contain finite values"
            )
        normalized.append(number)
    if not normalized:
        raise CognitiveKernelContractError(
            f"{field} may not be empty"
        )
    return tuple(normalized)


def _scope_digest(scope: ProductHostScope) -> str:
    return canonical_sha256(scope.metadata_record())


def validate_bounded_serving_path(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        raise UnsafeBoundedServingPathError(
            "bounded-serving prototype requires a database file path"
        )
    if repository_root is not None:
        repository = repository_root.expanduser().resolve()
        try:
            resolved.relative_to(repository)
        except ValueError:
            pass
        else:
            raise UnsafeBoundedServingPathError(
                "prototype database may not be stored in public Git"
            )
    if resolved.exists() and resolved.is_symlink():
        raise UnsafeBoundedServingPathError(
            "prototype database path may not be a symbolic link"
        )
    return resolved


class BoundedServingPrototypeStore:
    """Persistent M2.4 full-content bounded-serving research profile."""

    def __init__(
        self,
        *,
        path: Path,
        profile: BoundedServingProfile,
        repository_root: Path | None = None,
    ) -> None:
        profile.validate()
        self.path = validate_bounded_serving_path(
            path,
            repository_root=repository_root,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prototype_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS serving_documents (
                record_id TEXT PRIMARY KEY,
                record_version_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                authority_namespace_id TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                embedding_json TEXT,
                graph_neighbors_json TEXT NOT NULL,
                mission_node_ids_json TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                generation INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS serving_document_history (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                record_version_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                document_json TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS index_state (
                index_kind TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                state_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retrieval_traces (
                trace_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                trace_contract_json TEXT NOT NULL,
                full_trace_json TEXT NOT NULL,
                trace_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS context_packets (
                packet_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                packet_contract_json TEXT NOT NULL,
                full_packet_json TEXT NOT NULL,
                packet_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS serving_idempotency (
                idempotency_namespace TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                packet_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                PRIMARY KEY (idempotency_namespace, idempotency_key)
            );
            """
        )
        metadata = {
            "schema_version": BOUNDED_SERVING_PROTOTYPE_SCHEMA_VERSION,
            "prototype_state": BOUNDED_SERVING_PROTOTYPE_STATE,
            "scope_digest": _scope_digest(self.profile.scope),
            "profile_sha256": self.profile.profile_sha256(),
            "production_influence": "false",
        }
        current = {
            str(row["key"]): str(row["value"])
            for row in self._connection.execute(
                "SELECT key, value FROM prototype_metadata"
            )
        }
        if current:
            for key, expected in metadata.items():
                actual = current.get(key)
                if actual != expected:
                    if key in {"scope_digest", "profile_sha256"}:
                        raise BoundedServingIsolationError(
                            f"prototype metadata mismatch for {key}"
                        )
                    raise BoundedServingIntegrityError(
                        f"prototype metadata mismatch for {key}"
                    )
        else:
            with self._connection:
                self._connection.executemany(
                    "INSERT INTO prototype_metadata(key, value) VALUES (?, ?)",
                    sorted(metadata.items()),
                )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "BoundedServingPrototypeStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def remove_database(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)
        Path(f"{self.path}-wal").unlink(missing_ok=True)
        Path(f"{self.path}-shm").unlink(missing_ok=True)

    def set_index_state(
        self,
        *,
        index_kind: object,
        generation: object,
        state: object,
        updated_at: object | None = None,
    ) -> None:
        kind = require_identifier(index_kind, "index_kind")
        normalized_state = require_identifier(state, "state")
        if kind not in INDEX_KINDS:
            raise CognitiveKernelContractError(
                "index_kind is not registered"
            )
        if normalized_state not in INDEX_STATES:
            raise CognitiveKernelContractError(
                "index state is not registered"
            )
        normalized_generation = _non_negative_integer(
            generation,
            "generation",
        )
        timestamp = normalize_timestamp(
            updated_at if updated_at is not None else _utc_now(),
            "updated_at",
        )
        material = {
            "index_kind": kind,
            "generation": normalized_generation,
            "state": normalized_state,
            "updated_at": timestamp,
        }
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO index_state(
                    index_kind, generation, state, updated_at, state_sha256
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(index_kind) DO UPDATE SET
                    generation = excluded.generation,
                    state = excluded.state,
                    updated_at = excluded.updated_at,
                    state_sha256 = excluded.state_sha256
                """,
                (
                    kind,
                    normalized_generation,
                    normalized_state,
                    timestamp,
                    canonical_sha256(material),
                ),
            )

    def upsert_document(
        self,
        *,
        record_id: object,
        record_version_id: object,
        source_kind: object,
        authority_namespace_id: object,
        searchable_text: object,
        full_content: Mapping[str, object],
        embedding: Iterable[object] | None = None,
        graph_neighbors: Iterable[object] = (),
        mission_node_ids: Iterable[object] = (),
        valid_from: object,
        valid_to: object | None = None,
        generation: object,
        updated_at: object | None = None,
    ) -> ServingDocumentReceipt:
        normalized_id = require_identifier(record_id, "record_id")
        normalized_version = require_identifier(
            record_version_id,
            "record_version_id",
        )
        normalized_kind = require_identifier(source_kind, "source_kind")
        normalized_authority = require_identifier(
            authority_namespace_id,
            "authority_namespace_id",
        )
        if normalized_authority != self.profile.authority_namespace_id:
            raise BoundedServingIsolationError(
                "document authority namespace differs from profile"
            )
        normalized_text = require_text(
            searchable_text,
            "searchable_text",
            maximum=1_000_000,
            allow_newlines=True,
        )
        content_record = dict(full_content)
        content_bytes = canonical_json_bytes(content_record)
        content_digest = canonical_sha256(content_record)
        normalized_embedding = _normalized_embedding(
            embedding,
            "embedding",
        )
        neighbors = tuple(
            sorted(
                {
                    require_identifier(value, "graph_neighbors")
                    for value in graph_neighbors
                }
            )
        )
        document_missions = tuple(
            sorted(
                {
                    require_identifier(value, "mission_node_ids")
                    for value in mission_node_ids
                }
            )
        )
        from_time = normalize_timestamp(valid_from, "valid_from")
        to_time = (
            normalize_timestamp(valid_to, "valid_to")
            if valid_to is not None
            else None
        )
        if to_time is not None and to_time < from_time:
            raise CognitiveKernelContractError(
                "valid_to may not precede valid_from"
            )
        normalized_generation = _positive_integer(
            generation,
            "generation",
        )
        timestamp = normalize_timestamp(
            updated_at if updated_at is not None else _utc_now(),
            "updated_at",
        )
        material = {
            "schema_version": BOUNDED_SERVING_PROTOTYPE_SCHEMA_VERSION,
            "scope": self.profile.scope.metadata_record(),
            "record_id": normalized_id,
            "record_version_id": normalized_version,
            "source_kind": normalized_kind,
            "authority_namespace_id": normalized_authority,
            "searchable_text": normalized_text,
            "full_content": content_record,
            "embedding": (
                list(normalized_embedding)
                if normalized_embedding is not None
                else None
            ),
            "graph_neighbors": list(neighbors),
            "mission_node_ids": list(document_missions),
            "valid_from": from_time,
            "valid_to": to_time,
            "generation": normalized_generation,
            "content_digest": content_digest,
            "updated_at": timestamp,
        }
        document_sha256 = canonical_sha256(material)
        current = self._connection.execute(
            "SELECT generation, document_sha256 FROM serving_documents "
            "WHERE record_id = ?",
            (normalized_id,),
        ).fetchone()
        if current is not None:
            current_generation = int(current["generation"])
            current_digest = str(current["document_sha256"])
            if current_digest == document_sha256:
                return ServingDocumentReceipt(
                    record_id=normalized_id,
                    record_version_id=normalized_version,
                    generation=normalized_generation,
                    content_digest=content_digest,
                    document_sha256=document_sha256,
                )
            if normalized_generation <= current_generation:
                raise BoundedServingTransactionError(
                    "document generation must advance for changed content"
                )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO serving_document_history(
                        record_id, record_version_id, generation,
                        document_json, document_sha256, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_id,
                        normalized_version,
                        normalized_generation,
                        canonical_json_bytes(material).decode("utf-8"),
                        document_sha256,
                        timestamp,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO serving_documents(
                        record_id, record_version_id, source_kind,
                        authority_namespace_id, searchable_text,
                        full_content_json, embedding_json,
                        graph_neighbors_json, mission_node_ids_json,
                        valid_from, valid_to, generation, content_digest,
                        document_sha256, updated_at, deleted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(record_id) DO UPDATE SET
                        record_version_id = excluded.record_version_id,
                        source_kind = excluded.source_kind,
                        authority_namespace_id = excluded.authority_namespace_id,
                        searchable_text = excluded.searchable_text,
                        full_content_json = excluded.full_content_json,
                        embedding_json = excluded.embedding_json,
                        graph_neighbors_json = excluded.graph_neighbors_json,
                        mission_node_ids_json = excluded.mission_node_ids_json,
                        valid_from = excluded.valid_from,
                        valid_to = excluded.valid_to,
                        generation = excluded.generation,
                        content_digest = excluded.content_digest,
                        document_sha256 = excluded.document_sha256,
                        updated_at = excluded.updated_at,
                        deleted = 0
                    """,
                    (
                        normalized_id,
                        normalized_version,
                        normalized_kind,
                        normalized_authority,
                        normalized_text,
                        content_bytes.decode("utf-8"),
                        (
                            json.dumps(list(normalized_embedding))
                            if normalized_embedding is not None
                            else None
                        ),
                        json.dumps(list(neighbors)),
                        json.dumps(list(document_missions)),
                        from_time,
                        to_time,
                        normalized_generation,
                        content_digest,
                        document_sha256,
                        timestamp,
                    ),
                )
        except sqlite3.DatabaseError as exc:
            raise BoundedServingTransactionError(
                "document persistence failed"
            ) from exc
        return ServingDocumentReceipt(
            record_id=normalized_id,
            record_version_id=normalized_version,
            generation=normalized_generation,
            content_digest=content_digest,
            document_sha256=document_sha256,
        )

    def _index_status(self, kind: str) -> tuple[str, int]:
        row = self._connection.execute(
            "SELECT state, generation FROM index_state WHERE index_kind = ?",
            (kind,),
        ).fetchone()
        if row is None:
            return "unavailable", 0
        return str(row["state"]), int(row["generation"])

    def _load_documents(self) -> list[sqlite3.Row]:
        return list(
            self._connection.execute(
                "SELECT * FROM serving_documents WHERE deleted = 0"
            )
        )

    def serve(
        self,
        *,
        request_id: object,
        query_text: object,
        query_embedding: Iterable[object] | None = None,
        seed_record_ids: Iterable[object] = (),
        mission_node_ids: Iterable[object] = (),
        idempotency_namespace: object,
        idempotency_key: object,
        now: object | None = None,
    ) -> BoundedServingReceipt:
        normalized_request = require_identifier(request_id, "request_id")
        normalized_query = require_text(
            query_text,
            "query_text",
            maximum=100_000,
            allow_newlines=True,
        )
        normalized_embedding = _normalized_embedding(
            query_embedding,
            "query_embedding",
        )
        seeds = tuple(
            sorted(
                {
                    require_identifier(value, "seed_record_ids")
                    for value in seed_record_ids
                }
            )
        )
        missions = tuple(
            sorted(
                {
                    require_identifier(value, "mission_node_ids")
                    for value in mission_node_ids
                }
            )
        )
        namespace = require_identifier(
            idempotency_namespace,
            "idempotency_namespace",
        )
        key = require_identifier(idempotency_key, "idempotency_key")
        timestamp = normalize_timestamp(
            now if now is not None else _utc_now(),
            "now",
        )
        query_record = {
            "query_text": normalized_query,
            "query_embedding": (
                list(normalized_embedding)
                if normalized_embedding is not None
                else None
            ),
            "seed_record_ids": list(seeds),
            "mission_node_ids": list(missions),
            "profile_sha256": self.profile.profile_sha256(),
        }
        query_digest = canonical_sha256(query_record)
        request_digest = canonical_sha256(
            {
                "request_id": normalized_request,
                "query_digest": query_digest,
                "idempotency_namespace": namespace,
                "idempotency_key": key,
            }
        )
        existing = self._connection.execute(
            """
            SELECT request_digest, packet_id, trace_id
            FROM serving_idempotency
            WHERE idempotency_namespace = ? AND idempotency_key = ?
            """,
            (namespace, key),
        ).fetchone()
        if existing is not None:
            if str(existing["request_digest"]) != request_digest:
                raise BoundedServingTransactionError(
                    "idempotency key was reused with changed request"
                )
            return self.load_receipt(
                packet_id=str(existing["packet_id"]),
                trace_id=str(existing["trace_id"]),
            )

        rows = self._load_documents()
        query_tokens = _tokens(normalized_query)
        row_by_id = {str(row["record_id"]): row for row in rows}

        index_status = {
            kind: self._index_status(kind)
            for kind in ("lexical", "vector", "graph", "temporal")
        }
        stale_kinds = tuple(
            sorted(
                kind
                for kind, (state, _) in index_status.items()
                if state != "ready"
            )
        )
        stale_observed = bool(stale_kinds)
        fallback_used = (
            stale_observed
            and self.profile.fallback_mode == "bounded_scan"
        )

        graph_distance: dict[str, int] = {}
        frontier = list(seeds)
        for seed in seeds:
            graph_distance[seed] = 0
        while frontier:
            current = frontier.pop(0)
            distance = graph_distance[current]
            if distance >= self.profile.expansion_depth:
                continue
            row = row_by_id.get(current)
            if row is None:
                continue
            neighbors = json.loads(str(row["graph_neighbors_json"]))
            for neighbor in sorted(str(item) for item in neighbors):
                if neighbor not in graph_distance:
                    graph_distance[neighbor] = distance + 1
                    frontier.append(neighbor)

        candidates: list[_Candidate] = []
        excluded: set[str] = set()
        for row in rows:
            record_id = str(row["record_id"])
            valid_from = str(row["valid_from"])
            valid_to = (
                str(row["valid_to"])
                if row["valid_to"] is not None
                else None
            )
            if timestamp < valid_from or (
                valid_to is not None and timestamp > valid_to
            ):
                excluded.add(record_id)
                continue
            text_tokens = _tokens(str(row["searchable_text"]))
            lexical = (
                len(query_tokens & text_tokens)
                / len(query_tokens | text_tokens)
                if query_tokens and text_tokens
                else 0.0
            )
            document_embedding = (
                tuple(float(item) for item in json.loads(str(row["embedding_json"])))
                if row["embedding_json"] is not None
                else None
            )
            vector = (
                _cosine(normalized_embedding, document_embedding)
                if normalized_embedding is not None
                and document_embedding is not None
                else 0.0
            )
            distance = graph_distance.get(record_id)
            graph = (
                1.0 / (1.0 + float(distance))
                if distance is not None
                else 0.0
            )
            document_missions = set(
                str(item)
                for item in json.loads(str(row["mission_node_ids_json"]))
            )
            mission = (
                len(set(missions) & document_missions)
                / len(set(missions) | document_missions)
                if missions and document_missions
                else 0.0
            )
            generation = int(row["generation"])
            freshness = 1.0 / (1.0 + float(max(0, generation - 1)))
            reasons: list[str] = []
            if lexical > 0.0:
                reasons.append("lexical_match")
            if vector > 0.0:
                reasons.append("vector_similarity")
            if graph > 0.0:
                reasons.append("graph_expansion")
            if mission > 0.0:
                reasons.append("mission_match")
            if fallback_used:
                reasons.append("bounded_scan_fallback")
            if not reasons:
                excluded.add(record_id)
                continue
            candidates.append(
                _Candidate(
                    record_id=record_id,
                    record_version_id=str(row["record_version_id"]),
                    source_kind=str(row["source_kind"]),
                    authority_namespace_id=str(
                        row["authority_namespace_id"]
                    ),
                    content_digest=str(row["content_digest"]),
                    full_content=json.loads(str(row["full_content_json"])),
                    generation=generation,
                    lexical_score=lexical,
                    vector_score=vector,
                    graph_score=graph,
                    mission_score=mission,
                    freshness_score=freshness,
                    stale=stale_observed,
                    reason_codes=tuple(sorted(reasons)),
                )
            )

        total_weight = (
            self.profile.lexical_weight
            + self.profile.vector_weight
            + self.profile.graph_weight
            + self.profile.mission_weight
            + self.profile.freshness_weight
        )

        def fused(candidate: _Candidate) -> float:
            raw = (
                candidate.lexical_score * self.profile.lexical_weight
                + candidate.vector_score * self.profile.vector_weight
                + candidate.graph_score * self.profile.graph_weight
                + candidate.mission_score * self.profile.mission_weight
                + candidate.freshness_score * self.profile.freshness_weight
            ) / total_weight
            return max(0.0, min(1.0, raw))

        ranked = sorted(
            candidates,
            key=lambda candidate: (-fused(candidate), candidate.record_id),
        )
        selected: list[_Candidate] = []
        selected_bytes = 0
        for candidate in ranked:
            content_size = len(canonical_json_bytes(candidate.full_content))
            if len(selected) >= self.profile.item_budget:
                excluded.add(candidate.record_id)
                continue
            if selected_bytes + content_size > self.profile.byte_budget:
                excluded.add(candidate.record_id)
                continue
            selected.append(candidate)
            selected_bytes += content_size

        trace_id = f"trace-{request_digest[:24]}"
        packet_id = f"packet-{request_digest[:24]}"
        selected_ids = tuple(sorted(item.record_id for item in selected))
        excluded_ids = tuple(sorted(excluded))
        index_generations = [generation for _, generation in index_status.values()]
        selection_generation = max(
            [0, *index_generations, *(item.generation for item in selected)]
        )

        steps = self._build_trace_steps(
            trace_id=trace_id,
            timestamp=timestamp,
            seeds=seeds,
            candidates=tuple(candidates),
            selected_ids=selected_ids,
            excluded_ids=excluded_ids,
            index_status=index_status,
            fallback_used=fallback_used,
            stale_observed=stale_observed,
        )
        full_trace = {
            "request": query_record,
            "index_state": {
                kind: {"state": state, "generation": generation}
                for kind, (state, generation) in sorted(index_status.items())
            },
            "candidate_scores": [
                {
                    "record_id": item.record_id,
                    "lexical": item.lexical_score,
                    "vector": item.vector_score,
                    "graph": item.graph_score,
                    "mission": item.mission_score,
                    "freshness": item.freshness_score,
                    "fused": fused(item),
                    "reason_codes": list(item.reason_codes),
                }
                for item in ranked
            ],
            "selected_record_ids": list(selected_ids),
            "excluded_record_ids": list(excluded_ids),
        }
        full_packet = tuple(
            {
                "record_id": item.record_id,
                "record_version_id": item.record_version_id,
                "source_kind": item.source_kind,
                "authority_namespace_id": item.authority_namespace_id,
                "content_digest": item.content_digest,
                "full_content": item.full_content,
                "fused_score": fused(item),
                "reason_codes": list(item.reason_codes),
            }
            for item in selected
        )
        trace_digest = canonical_sha256(full_trace)
        packet_digest = canonical_sha256(list(full_packet))
        trace_envelope = self._envelope(
            record_id=trace_id,
            record_type="retrieval_trace",
            content_digest=trace_digest,
            timestamp=timestamp,
            request_id=normalized_request,
        )
        packet_envelope = self._envelope(
            record_id=packet_id,
            record_type="context_packet",
            content_digest=packet_digest,
            timestamp=timestamp,
            request_id=normalized_request,
        )
        trace = RetrievalTrace.create(
            envelope=trace_envelope,
            trace_id=trace_id,
            request_id=normalized_request,
            query_digest=query_digest,
            profile_id=self.profile.profile_id,
            fusion_strategy=self.profile.fusion_strategy,
            steps=steps,
            selected_record_ids=selected_ids,
            excluded_record_ids=excluded_ids,
            started_at=timestamp,
            completed_at=timestamp,
            fallback_used=fallback_used,
            stale_index_observed=stale_observed,
            trace_content_digest=trace_digest,
        )
        selections = tuple(
            ContextSelection.create(
                record_id=item.record_id,
                record_version_id=item.record_version_id,
                source_kind=item.source_kind,
                authority_namespace_id=item.authority_namespace_id,
                rank=index,
                fused_score=fused(item),
                content_digest=item.content_digest,
                reason_codes=item.reason_codes,
                stale=item.stale,
                selected_from_generation=item.generation,
            )
            for index, item in enumerate(selected, start=1)
        )
        state = (
            "stale_fallback"
            if fallback_used
            else "degraded"
            if stale_observed
            else "assembled"
        )
        packet = ContextPacket.create(
            envelope=packet_envelope,
            packet_id=packet_id,
            request_id=normalized_request,
            trace_id=trace_id,
            query_digest=query_digest,
            profile_id=self.profile.profile_id,
            packet_state=state,
            mission_node_ids=missions,
            selections=selections,
            excluded_record_ids=excluded_ids,
            assembled_at=timestamp,
            expires_at=None,
            item_budget=self.profile.item_budget,
            byte_budget=self.profile.byte_budget,
            hydrated_item_count=len(selections),
            hydrated_byte_count=selected_bytes,
            selection_generation=selection_generation,
            fallback_used=fallback_used,
            degraded=stale_observed,
            packet_content_digest=packet_digest,
        )
        receipt_digest = canonical_sha256(
            {
                "packet_sha256": packet.packet_sha256,
                "trace_sha256": trace.trace_sha256,
                "fallback_used": fallback_used,
                "stale_index_observed": stale_observed,
            }
        )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO retrieval_traces(
                        trace_id, request_id, trace_contract_json,
                        full_trace_json, trace_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace_id,
                        normalized_request,
                        json.dumps(trace.metadata_record(), sort_keys=True),
                        json.dumps(full_trace, sort_keys=True),
                        trace.trace_sha256,
                        timestamp,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO context_packets(
                        packet_id, trace_id, request_id,
                        packet_contract_json, full_packet_json,
                        packet_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        packet_id,
                        trace_id,
                        normalized_request,
                        json.dumps(packet.metadata_record(), sort_keys=True),
                        json.dumps(list(full_packet), sort_keys=True),
                        packet.packet_sha256,
                        timestamp,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO serving_idempotency(
                        idempotency_namespace, idempotency_key,
                        request_digest, packet_id, trace_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (namespace, key, request_digest, packet_id, trace_id),
                )
        except sqlite3.DatabaseError as exc:
            raise BoundedServingTransactionError(
                "bounded serving transaction failed"
            ) from exc
        return BoundedServingReceipt(
            packet=packet,
            trace=trace,
            full_packet_content=full_packet,
            fallback_used=fallback_used,
            stale_index_observed=stale_observed,
            receipt_sha256=receipt_digest,
        )

    def _build_trace_steps(
        self,
        *,
        trace_id: str,
        timestamp: str,
        seeds: tuple[str, ...],
        candidates: tuple[_Candidate, ...],
        selected_ids: tuple[str, ...],
        excluded_ids: tuple[str, ...],
        index_status: dict[str, tuple[str, int]],
        fallback_used: bool,
        stale_observed: bool,
    ) -> tuple[RetrievalTraceStep, ...]:
        candidate_ids = tuple(sorted(item.record_id for item in candidates))
        steps: list[RetrievalTraceStep] = []
        stage_specs = [
            (
                "lexical",
                "lexical_retrieval",
                tuple(sorted(
                    item.record_id
                    for item in candidates
                    if item.lexical_score > 0.0
                )),
            ),
            (
                "vector",
                "vector_retrieval",
                tuple(sorted(
                    item.record_id
                    for item in candidates
                    if item.vector_score > 0.0
                )),
            ),
            (
                "graph",
                "graph_expansion",
                tuple(sorted(
                    item.record_id
                    for item in candidates
                    if item.graph_score > 0.0
                )),
            ),
            (
                "temporal",
                "temporal_filtering",
                candidate_ids,
            ),
            (
                None,
                "mission_filtering",
                tuple(sorted(
                    item.record_id
                    for item in candidates
                    if item.mission_score > 0.0
                )),
            ),
        ]
        for sequence, (kind, stage_kind, output_ids) in enumerate(
            stage_specs,
            start=1,
        ):
            if kind is None:
                state, generation = "ready", 0
            else:
                state, generation = index_status[kind]
            step_fallback = state != "ready" and fallback_used
            outcome = (
                "fallback_used"
                if step_fallback
                else "stale"
                if state == "stale"
                else "unavailable"
                if state == "unavailable"
                else "completed"
                if output_ids
                else "empty"
            )
            steps.append(
                RetrievalTraceStep.create(
                    stage_id=f"{trace_id}-stage-{sequence}",
                    stage_kind=stage_kind,
                    outcome=outcome,
                    started_at=timestamp,
                    completed_at=timestamp,
                    input_record_ids=seeds,
                    output_record_ids=output_ids,
                    reason_codes=(
                        ("bounded_scan_fallback",)
                        if step_fallback
                        else ()
                    ),
                    index_kind=kind,
                    index_generation=(generation if kind is not None else None),
                    fallback_used=step_fallback,
                    stale_index_observed=state != "ready",
                    metrics_digest=canonical_sha256(
                        {
                            "candidate_count": len(output_ids),
                            "state": state,
                            "generation": generation,
                        }
                    ),
                )
            )
        next_sequence = len(stage_specs) + 1
        steps.append(
            RetrievalTraceStep.create(
                stage_id=f"{trace_id}-stage-{next_sequence}",
                stage_kind="fusion",
                outcome="completed" if candidate_ids else "empty",
                started_at=timestamp,
                completed_at=timestamp,
                input_record_ids=candidate_ids,
                output_record_ids=selected_ids,
                excluded_record_ids=excluded_ids,
                reason_codes=("weighted_score",),
                fallback_used=fallback_used,
                stale_index_observed=stale_observed,
                metrics_digest=canonical_sha256(
                    {
                        "candidate_count": len(candidate_ids),
                        "selected_count": len(selected_ids),
                        "excluded_count": len(excluded_ids),
                    }
                ),
            )
        )
        steps.append(
            RetrievalTraceStep.create(
                stage_id=f"{trace_id}-stage-{next_sequence + 1}",
                stage_kind="packet_assembly",
                outcome="completed" if selected_ids else "empty",
                started_at=timestamp,
                completed_at=timestamp,
                input_record_ids=selected_ids,
                output_record_ids=selected_ids,
                excluded_record_ids=excluded_ids,
                reason_codes=("profile_budget_applied",),
                fallback_used=fallback_used,
                stale_index_observed=stale_observed,
                metrics_digest=canonical_sha256(
                    {
                        "item_budget": self.profile.item_budget,
                        "byte_budget": self.profile.byte_budget,
                        "selected_count": len(selected_ids),
                    }
                ),
            )
        )
        return tuple(steps)

    def _envelope(
        self,
        *,
        record_id: str,
        record_type: str,
        content_digest: str,
        timestamp: str,
        request_id: str,
    ) -> MemoryUnitEnvelope:
        return MemoryUnitEnvelope.create(
            scope=self.profile.scope,
            record_id=record_id,
            record_type=record_type,
            authority_namespace_id=self.profile.authority_namespace_id,
            host_or_cluster_id=self.profile.scope.host_instance_id,
            authority_role="registered_projection",
            deployment_profile="reversible_prototype",
            created_at=timestamp,
            valid_from=timestamp,
            valid_to=None,
            transaction_time=timestamp,
            logical_clock=0,
            causal_parents=(),
            source_records=(),
            generation=0,
            state="committed",
            data_classification="private",
            retention_class="transient_web_or_tool_cache",
            deletion_state="active",
            provenance_digest=canonical_sha256(
                {
                    "profile_sha256": self.profile.profile_sha256(),
                    "record_type": record_type,
                }
            ),
            content_digest=require_sha256(
                content_digest,
                "content_digest",
            ),
            writer="bounded-serving-prototype",
            workflow_or_request_id=request_id,
            idempotency_namespace="bounded-serving",
            idempotency_key=record_id,
        )

    def load_receipt(
        self,
        *,
        packet_id: object,
        trace_id: object,
    ) -> BoundedServingReceipt:
        normalized_packet = require_identifier(packet_id, "packet_id")
        normalized_trace = require_identifier(trace_id, "trace_id")
        packet_row = self._connection.execute(
            "SELECT * FROM context_packets WHERE packet_id = ?",
            (normalized_packet,),
        ).fetchone()
        trace_row = self._connection.execute(
            "SELECT * FROM retrieval_traces WHERE trace_id = ?",
            (normalized_trace,),
        ).fetchone()
        if packet_row is None or trace_row is None:
            raise KeyError("serving packet or trace was not found")
        packet = _context_packet_from_record(
            json.loads(str(packet_row["packet_contract_json"])),
        )
        trace = _retrieval_trace_from_record(
            json.loads(str(trace_row["trace_contract_json"])),
        )
        full_packet = tuple(
            dict(item)
            for item in json.loads(str(packet_row["full_packet_json"]))
        )
        receipt_digest = canonical_sha256(
            {
                "packet_sha256": packet.packet_sha256,
                "trace_sha256": trace.trace_sha256,
                "fallback_used": packet.fallback_used,
                "stale_index_observed": trace.stale_index_observed,
            }
        )
        return BoundedServingReceipt(
            packet=packet,
            trace=trace,
            full_packet_content=full_packet,
            fallback_used=packet.fallback_used,
            stale_index_observed=trace.stale_index_observed,
            receipt_sha256=receipt_digest,
        )

    def list_packet_ids(self) -> tuple[str, ...]:
        return tuple(
            str(row["packet_id"])
            for row in self._connection.execute(
                "SELECT packet_id FROM context_packets ORDER BY created_at, packet_id"
            )
        )

    def verify_integrity(self) -> BoundedServingIntegrityReport:
        problems: list[str] = []
        documents = list(
            self._connection.execute(
                "SELECT * FROM serving_documents"
            )
        )
        for row in documents:
            material = {
                "schema_version": BOUNDED_SERVING_PROTOTYPE_SCHEMA_VERSION,
                "scope": self.profile.scope.metadata_record(),
                "record_id": str(row["record_id"]),
                "record_version_id": str(row["record_version_id"]),
                "source_kind": str(row["source_kind"]),
                "authority_namespace_id": str(row["authority_namespace_id"]),
                "searchable_text": str(row["searchable_text"]),
                "full_content": json.loads(str(row["full_content_json"])),
                "embedding": (
                    json.loads(str(row["embedding_json"]))
                    if row["embedding_json"] is not None
                    else None
                ),
                "graph_neighbors": json.loads(
                    str(row["graph_neighbors_json"])
                ),
                "mission_node_ids": json.loads(
                    str(row["mission_node_ids_json"])
                ),
                "valid_from": str(row["valid_from"]),
                "valid_to": (
                    str(row["valid_to"])
                    if row["valid_to"] is not None
                    else None
                ),
                "generation": int(row["generation"]),
                "content_digest": str(row["content_digest"]),
                "updated_at": str(row["updated_at"]),
            }
            if canonical_sha256(material) != str(row["document_sha256"]):
                problems.append(
                    f"document_digest:{row['record_id']}"
                )
            if canonical_sha256(material["full_content"]) != str(
                row["content_digest"]
            ):
                problems.append(
                    f"content_digest:{row['record_id']}"
                )
        packets = list(
            self._connection.execute(
                "SELECT * FROM context_packets"
            )
        )
        for row in packets:
            try:
                packet = _context_packet_from_record(
                    json.loads(str(row["packet_contract_json"])),
                )
                if packet.packet_sha256 != str(row["packet_sha256"]):
                    problems.append(
                        f"packet_digest:{row['packet_id']}"
                    )
                full_packet = json.loads(str(row["full_packet_json"]))
                if canonical_sha256(full_packet) != packet.packet_content_digest:
                    problems.append(
                        f"packet_content:{row['packet_id']}"
                    )
            except Exception:
                problems.append(f"packet_parse:{row['packet_id']}")
        traces = list(
            self._connection.execute(
                "SELECT * FROM retrieval_traces"
            )
        )
        for row in traces:
            try:
                trace = _retrieval_trace_from_record(
                    json.loads(str(row["trace_contract_json"])),
                )
                if trace.trace_sha256 != str(row["trace_sha256"]):
                    problems.append(f"trace_digest:{row['trace_id']}")
                full_trace = json.loads(str(row["full_trace_json"]))
                if canonical_sha256(full_trace) != trace.trace_content_digest:
                    problems.append(f"trace_content:{row['trace_id']}")
            except Exception:
                problems.append(f"trace_parse:{row['trace_id']}")
        return BoundedServingIntegrityReport(
            checked_documents=len(documents),
            checked_packets=len(packets),
            checked_traces=len(traces),
            problems=tuple(sorted(problems)),
        )

    def require_integrity(self) -> BoundedServingIntegrityReport:
        report = self.verify_integrity()
        if not report.healthy:
            raise BoundedServingIntegrityError(
                f"bounded-serving integrity problems: {report.problems}"
            )
        return report


def _scope_from_record(record: Mapping[str, object]) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=record["product_id"],
        host_instance_id=record["host_instance_id"],
        schema_version=record["schema_version"],
        encryption_domain=record["encryption_domain"],
    )


def _envelope_from_record(record: Mapping[str, object]) -> MemoryUnitEnvelope:
    scope = _scope_from_record(record["scope"])
    value = MemoryUnitEnvelope(
        scope=scope,
        record_id=str(record["record_id"]),
        record_type=str(record["record_type"]),
        authority_namespace_id=str(record["authority_namespace_id"]),
        host_or_cluster_id=str(record["host_or_cluster_id"]),
        authority_role=str(record["authority_role"]),
        deployment_profile=str(record["deployment_profile"]),
        created_at=str(record["created_at"]),
        valid_from=str(record["valid_from"]),
        valid_to=(
            str(record["valid_to"])
            if record["valid_to"] is not None
            else None
        ),
        transaction_time=str(record["transaction_time"]),
        logical_clock=int(record["logical_clock"]),
        causal_parents=tuple(str(item) for item in record["causal_parents"]),
        source_records=tuple(str(item) for item in record["source_records"]),
        generation=int(record["generation"]),
        state=str(record["state"]),
        data_classification=str(record["data_classification"]),
        retention_class=str(record["retention_class"]),
        deletion_state=str(record["deletion_state"]),
        provenance_digest=str(record["provenance_digest"]),
        content_digest=str(record["content_digest"]),
        writer=str(record["writer"]),
        workflow_or_request_id=str(record["workflow_or_request_id"]),
        idempotency_namespace=str(record["idempotency_namespace"]),
        idempotency_key=str(record["idempotency_key"]),
        supersedes=tuple(str(item) for item in record["supersedes"]),
        superseded_by=tuple(str(item) for item in record["superseded_by"]),
        rollback_reference=(
            str(record["rollback_reference"])
            if record["rollback_reference"] is not None
            else None
        ),
        envelope_sha256=str(record["envelope_sha256"]),
    )
    value.validate()
    return value


def _selection_from_record(record: Mapping[str, object]) -> ContextSelection:
    value = ContextSelection(
        record_id=str(record["record_id"]),
        record_version_id=str(record["record_version_id"]),
        source_kind=str(record["source_kind"]),
        authority_namespace_id=str(record["authority_namespace_id"]),
        rank=int(record["rank"]),
        fused_score=float(record["fused_score"]),
        content_digest=str(record["content_digest"]),
        reason_codes=tuple(str(item) for item in record["reason_codes"]),
        stale=bool(record["stale"]),
        selected_from_generation=int(record["selected_from_generation"]),
        selection_sha256=str(record["selection_sha256"]),
    )
    value.validate()
    return value


def _step_from_record(record: Mapping[str, object]) -> RetrievalTraceStep:
    value = RetrievalTraceStep(
        stage_id=str(record["stage_id"]),
        stage_kind=str(record["stage_kind"]),
        outcome=str(record["outcome"]),
        started_at=str(record["started_at"]),
        completed_at=str(record["completed_at"]),
        input_record_ids=tuple(str(item) for item in record["input_record_ids"]),
        output_record_ids=tuple(str(item) for item in record["output_record_ids"]),
        excluded_record_ids=tuple(str(item) for item in record["excluded_record_ids"]),
        reason_codes=tuple(str(item) for item in record["reason_codes"]),
        index_kind=(
            str(record["index_kind"])
            if record["index_kind"] is not None
            else None
        ),
        index_generation=(
            int(record["index_generation"])
            if record["index_generation"] is not None
            else None
        ),
        fallback_used=bool(record["fallback_used"]),
        stale_index_observed=bool(record["stale_index_observed"]),
        metrics_digest=str(record["metrics_digest"]),
        step_sha256=str(record["step_sha256"]),
    )
    value.validate()
    return value


def _retrieval_trace_from_record(
    record: Mapping[str, object],
) -> RetrievalTrace:
    value = RetrievalTrace(
        envelope=_envelope_from_record(record["envelope"]),
        trace_id=str(record["trace_id"]),
        request_id=str(record["request_id"]),
        query_digest=str(record["query_digest"]),
        profile_id=str(record["profile_id"]),
        fusion_strategy=str(record["fusion_strategy"]),
        steps=tuple(_step_from_record(item) for item in record["steps"]),
        selected_record_ids=tuple(str(item) for item in record["selected_record_ids"]),
        excluded_record_ids=tuple(str(item) for item in record["excluded_record_ids"]),
        started_at=str(record["started_at"]),
        completed_at=str(record["completed_at"]),
        fallback_used=bool(record["fallback_used"]),
        stale_index_observed=bool(record["stale_index_observed"]),
        trace_content_digest=str(record["trace_content_digest"]),
        trace_sha256=str(record["trace_sha256"]),
    )
    value.validate()
    return value


def _context_packet_from_record(
    record: Mapping[str, object],
) -> ContextPacket:
    value = ContextPacket(
        envelope=_envelope_from_record(record["envelope"]),
        packet_id=str(record["packet_id"]),
        request_id=str(record["request_id"]),
        trace_id=str(record["trace_id"]),
        query_digest=str(record["query_digest"]),
        profile_id=str(record["profile_id"]),
        packet_state=str(record["packet_state"]),
        mission_node_ids=tuple(str(item) for item in record["mission_node_ids"]),
        selections=tuple(_selection_from_record(item) for item in record["selections"]),
        excluded_record_ids=tuple(str(item) for item in record["excluded_record_ids"]),
        assembled_at=str(record["assembled_at"]),
        expires_at=(
            str(record["expires_at"])
            if record["expires_at"] is not None
            else None
        ),
        item_budget=int(record["item_budget"]),
        byte_budget=int(record["byte_budget"]),
        hydrated_item_count=int(record["hydrated_item_count"]),
        hydrated_byte_count=int(record["hydrated_byte_count"]),
        selection_generation=int(record["selection_generation"]),
        fallback_used=bool(record["fallback_used"]),
        degraded=bool(record["degraded"]),
        packet_content_digest=str(record["packet_content_digest"]),
        packet_sha256=str(record["packet_sha256"]),
    )
    value.validate()
    return value


def open_bounded_serving_prototype(
    *,
    path: Path,
    profile: BoundedServingProfile,
    repository_root: Path | None = None,
) -> BoundedServingPrototypeStore:
    return BoundedServingPrototypeStore(
        path=path,
        profile=profile,
        repository_root=repository_root,
    )
