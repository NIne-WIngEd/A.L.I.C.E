"""Memory M2.3 reversible full-content projection-fabric prototype.

The prototype persists episode content, immutable projection versions, graph
edges, vectors, temporal history, and owner/source/self model projections
outside public Git. It is a reversible research profile, not canonical truth
and not a destination backend decision.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Sequence

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
    require_sha256,
)
from .contracts import ProductHostScope
from .projection_contracts import (
    EpisodeRecord,
    ProjectionVersion,
)

PROJECTION_PROTOTYPE_SCHEMA_VERSION = "1.0.0"
PROJECTION_PROTOTYPE_STATE = "reversible_nonproduction"


class ProjectionPrototypeError(RuntimeError):
    """Base error for the reversible projection prototype."""


class UnsafeProjectionPrototypePathError(ProjectionPrototypeError):
    """Raised when the prototype database resolves inside public Git."""


class ProjectionPrototypeIsolationError(ProjectionPrototypeError):
    """Raised when product, host, encryption, or authority scope differs."""


class ProjectionPrototypeConflictError(ProjectionPrototypeError):
    """Raised for idempotency, immutable, or expected-current conflicts."""


class ProjectionPrototypeIntegrityError(ProjectionPrototypeError):
    """Raised when persisted projection records fail integrity checks."""


class ProjectionPrototypeTransactionError(ProjectionPrototypeError):
    """Raised for invalid or nested prototype write transactions."""


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_projection_prototype_path(
    database_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    candidate = Path(database_path).expanduser().resolve(strict=False)
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else default_repository_root().resolve(strict=True)
    )
    if _is_within(candidate, root):
        raise UnsafeProjectionPrototypePathError(
            "Refusing to create or open a projection prototype inside "
            f"the public repository: {candidate}"
        )
    return candidate


def _canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _load_json_object(value: str, *, field: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProjectionPrototypeIntegrityError(
            f"stored {field} is not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ProjectionPrototypeIntegrityError(
            f"stored {field} is not a JSON object"
        )
    return parsed


def _load_json_array(value: str, *, field: str) -> list[object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProjectionPrototypeIntegrityError(
            f"stored {field} is not valid JSON"
        ) from exc
    if not isinstance(parsed, list):
        raise ProjectionPrototypeIntegrityError(
            f"stored {field} is not a JSON array"
        )
    return parsed


def _same_scope(first: ProductHostScope, second: ProductHostScope) -> bool:
    return first.metadata_record() == second.metadata_record()


def _finite_vector(
    values: Sequence[object],
    *,
    field: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(
        values,
        Sequence,
    ):
        raise CognitiveKernelContractError(
            f"{field} must be a numeric sequence"
        )
    if not values:
        raise CognitiveKernelContractError(
            f"{field} may not be empty"
        )
    if len(values) > 16384:
        raise CognitiveKernelContractError(
            f"{field} exceeds the prototype dimension limit"
        )
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise CognitiveKernelContractError(
                f"{field} must contain only numbers"
            )
        number = float(value)
        if not math.isfinite(number):
            raise CognitiveKernelContractError(
                f"{field} must contain finite numbers"
            )
        normalized.append(number)
    if math.sqrt(sum(item * item for item in normalized)) == 0.0:
        raise CognitiveKernelContractError(
            f"{field} may not be a zero vector"
        )
    return tuple(normalized)


def _cosine(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    if len(first) != len(second):
        raise ProjectionPrototypeConflictError(
            "vector dimensions differ"
        )
    first_norm = math.sqrt(sum(item * item for item in first))
    second_norm = math.sqrt(sum(item * item for item in second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ProjectionPrototypeIntegrityError(
            "stored vector has zero norm"
        )
    return sum(a * b for a, b in zip(first, second)) / (
        first_norm * second_norm
    )


@dataclass(frozen=True)
class ProjectionPrototypeProfile:
    """Scope and nonproduction boundary for one projection store."""

    scope: ProductHostScope
    authority_namespace_id: str
    store_id: str
    backend_profile: str
    state: str
    production_influence: bool
    canonical_authority: bool
    full_content_persistence: bool
    graph_projection: bool
    vector_projection: bool
    temporal_projection: bool
    owner_source_self_projection: bool
    profile_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        authority_namespace_id: object,
        store_id: object,
        backend_profile: object = (
            "embedded_sqlite_reference_challenger"
        ),
    ) -> "ProjectionPrototypeProfile":
        draft = cls(
            scope=scope,
            authority_namespace_id=require_identifier(
                authority_namespace_id,
                "authority_namespace_id",
            ),
            store_id=require_identifier(store_id, "store_id"),
            backend_profile=require_identifier(
                backend_profile,
                "backend_profile",
            ),
            state=PROJECTION_PROTOTYPE_STATE,
            production_influence=False,
            canonical_authority=False,
            full_content_persistence=True,
            graph_projection=True,
            vector_projection=True,
            temporal_projection=True,
            owner_source_self_projection=True,
            profile_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "profile_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        self.scope.validate()
        for field, value in (
            (
                "authority_namespace_id",
                self.authority_namespace_id,
            ),
            ("store_id", self.store_id),
            ("backend_profile", self.backend_profile),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.state != PROJECTION_PROTOTYPE_STATE:
            raise CognitiveKernelContractError(
                "projection prototype state is invalid"
            )
        if self.production_influence is not False:
            raise CognitiveKernelContractError(
                "projection prototype may not influence production"
            )
        if self.canonical_authority is not False:
            raise CognitiveKernelContractError(
                "projection prototype may not claim canonical authority"
            )
        for field in (
            "full_content_persistence",
            "graph_projection",
            "vector_projection",
            "temporal_projection",
            "owner_source_self_projection",
        ):
            if getattr(self, field) is not True:
                raise CognitiveKernelContractError(
                    f"{field} must be enabled for this profile"
                )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "schema_version": PROJECTION_PROTOTYPE_SCHEMA_VERSION,
            "scope": self.scope.metadata_record(),
            "authority_namespace_id": self.authority_namespace_id,
            "store_id": self.store_id,
            "backend_profile": self.backend_profile,
            "state": self.state,
            "production_influence": self.production_influence,
            "canonical_authority": self.canonical_authority,
            "full_content_persistence": (
                self.full_content_persistence
            ),
            "graph_projection": self.graph_projection,
            "vector_projection": self.vector_projection,
            "temporal_projection": self.temporal_projection,
            "owner_source_self_projection": (
                self.owner_source_self_projection
            ),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["profile_sha256"] = self.profile_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.profile_sha256, "profile_sha256")
        if (
            canonical_sha256(self.material_record())
            != self.profile_sha256
        ):
            raise CognitiveKernelContractError(
                "projection profile digest mismatch"
            )


@dataclass(frozen=True)
class ProjectionGraphEdge:
    """Version-bound temporal graph edge for the research prototype."""

    edge_id: str
    graph_namespace_id: str
    source_node_id: str
    relation_type: str
    target_node_id: str
    valid_from: str
    valid_to: str | None
    weight: float
    source_record_ids: tuple[str, ...]
    edge_sha256: str

    @classmethod
    def create(
        cls,
        *,
        edge_id: object,
        graph_namespace_id: object,
        source_node_id: object,
        relation_type: object,
        target_node_id: object,
        valid_from: object,
        valid_to: object | None,
        weight: object,
        source_record_ids: Iterable[object],
    ) -> "ProjectionGraphEdge":
        if isinstance(weight, bool) or not isinstance(
            weight,
            (int, float),
        ):
            raise CognitiveKernelContractError(
                "weight must be numeric"
            )
        normalized_weight = float(weight)
        if (
            not math.isfinite(normalized_weight)
            or not -1.0 <= normalized_weight <= 1.0
        ):
            raise CognitiveKernelContractError(
                "weight must be finite and between -1 and 1"
            )
        normalized_sources = tuple(
            sorted(
                require_identifier(value, "source_record_ids")
                for value in source_record_ids
            )
        )
        if len(set(normalized_sources)) != len(normalized_sources):
            raise CognitiveKernelContractError(
                "source_record_ids may not contain duplicates"
            )
        draft = cls(
            edge_id=require_identifier(edge_id, "edge_id"),
            graph_namespace_id=require_identifier(
                graph_namespace_id,
                "graph_namespace_id",
            ),
            source_node_id=require_identifier(
                source_node_id,
                "source_node_id",
            ),
            relation_type=require_identifier(
                relation_type,
                "relation_type",
            ),
            target_node_id=require_identifier(
                target_node_id,
                "target_node_id",
            ),
            valid_from=normalize_timestamp(
                valid_from,
                "valid_from",
            ),
            valid_to=(
                normalize_timestamp(valid_to, "valid_to")
                if valid_to is not None
                else None
            ),
            weight=normalized_weight,
            source_record_ids=normalized_sources,
            edge_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "edge_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        for field, value in (
            ("edge_id", self.edge_id),
            ("graph_namespace_id", self.graph_namespace_id),
            ("source_node_id", self.source_node_id),
            ("relation_type", self.relation_type),
            ("target_node_id", self.target_node_id),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.source_node_id == self.target_node_id:
            raise CognitiveKernelContractError(
                "graph edge may not self-reference"
            )
        valid_from = normalize_timestamp(
            self.valid_from,
            "valid_from",
        )
        if valid_from != self.valid_from:
            raise CognitiveKernelContractError(
                "valid_from is not canonical"
            )
        if self.valid_to is not None:
            valid_to = normalize_timestamp(
                self.valid_to,
                "valid_to",
            )
            if valid_to != self.valid_to:
                raise CognitiveKernelContractError(
                    "valid_to is not canonical"
                )
            if valid_to < valid_from:
                raise CognitiveKernelContractError(
                    "valid_to may not precede valid_from"
                )
        if (
            not math.isfinite(self.weight)
            or not -1.0 <= self.weight <= 1.0
        ):
            raise CognitiveKernelContractError(
                "weight is invalid"
            )
        if tuple(sorted(self.source_record_ids)) != (
            self.source_record_ids
        ):
            raise CognitiveKernelContractError(
                "source_record_ids are not canonical"
            )
        if not self.source_record_ids:
            raise CognitiveKernelContractError(
                "graph edge requires source lineage"
            )
        for value in self.source_record_ids:
            if require_identifier(
                value,
                "source_record_ids",
            ) != value:
                raise CognitiveKernelContractError(
                    "source_record_ids are not canonical"
                )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "edge_id": self.edge_id,
            "graph_namespace_id": self.graph_namespace_id,
            "source_node_id": self.source_node_id,
            "relation_type": self.relation_type,
            "target_node_id": self.target_node_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "weight": self.weight,
            "source_record_ids": list(self.source_record_ids),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["edge_sha256"] = self.edge_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.edge_sha256, "edge_sha256")
        if (
            canonical_sha256(self.material_record())
            != self.edge_sha256
        ):
            raise CognitiveKernelContractError(
                "graph edge digest mismatch"
            )


@dataclass(frozen=True)
class ProjectionAppendReceipt:
    """Idempotent receipt for one prototype append."""

    operation: str
    record_id: str
    generation: int
    idempotency_namespace: str
    idempotency_key: str
    request_digest: str
    content_digest: str
    created_at: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        operation: object,
        record_id: object,
        generation: object,
        idempotency_namespace: object,
        idempotency_key: object,
        request_digest: object,
        content_digest: object,
        created_at: object,
    ) -> "ProjectionAppendReceipt":
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            raise CognitiveKernelContractError(
                "generation must be a positive integer"
            )
        draft = cls(
            operation=require_identifier(operation, "operation"),
            record_id=require_identifier(record_id, "record_id"),
            generation=generation,
            idempotency_namespace=require_identifier(
                idempotency_namespace,
                "idempotency_namespace",
            ),
            idempotency_key=require_identifier(
                idempotency_key,
                "idempotency_key",
            ),
            request_digest=require_sha256(
                request_digest,
                "request_digest",
            ),
            content_digest=require_sha256(
                content_digest,
                "content_digest",
            ),
            created_at=normalize_timestamp(
                created_at,
                "created_at",
            ),
            receipt_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "receipt_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def material_record(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "record_id": self.record_id,
            "generation": self.generation,
            "idempotency_namespace": self.idempotency_namespace,
            "idempotency_key": self.idempotency_key,
            "request_digest": self.request_digest,
            "content_digest": self.content_digest,
            "created_at": self.created_at,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["receipt_sha256"] = self.receipt_sha256
        return record

    def validate(self) -> None:
        for field, value in (
            ("operation", self.operation),
            ("record_id", self.record_id),
            (
                "idempotency_namespace",
                self.idempotency_namespace,
            ),
            ("idempotency_key", self.idempotency_key),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise CognitiveKernelContractError(
                "generation is invalid"
            )
        require_sha256(self.request_digest, "request_digest")
        require_sha256(self.content_digest, "content_digest")
        if normalize_timestamp(
            self.created_at,
            "created_at",
        ) != self.created_at:
            raise CognitiveKernelContractError(
                "created_at is not canonical"
            )
        require_sha256(self.receipt_sha256, "receipt_sha256")
        if (
            canonical_sha256(self.material_record())
            != self.receipt_sha256
        ):
            raise CognitiveKernelContractError(
                "append receipt digest mismatch"
            )


@dataclass(frozen=True)
class ProjectionIntegrityReport:
    """Sanitized integrity result for one projection database."""

    valid: bool
    episode_count: int
    projection_version_count: int
    current_projection_count: int
    graph_edge_count: int
    vector_count: int
    idempotency_receipt_count: int
    errors: tuple[str, ...]

    def metadata_record(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "episode_count": self.episode_count,
            "projection_version_count": (
                self.projection_version_count
            ),
            "current_projection_count": (
                self.current_projection_count
            ),
            "graph_edge_count": self.graph_edge_count,
            "vector_count": self.vector_count,
            "idempotency_receipt_count": (
                self.idempotency_receipt_count
            ),
            "errors": list(self.errors),
        }


class ProjectionPrototypeStore:
    """Persistent reversible projection-fabric research store."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        profile: ProjectionPrototypeProfile,
        repository_root: str | Path | None = None,
    ) -> None:
        profile.validate()
        self.database_path = validate_projection_prototype_path(
            database_path,
            repository_root=repository_root,
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self._connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._in_transaction = False
        self._initialize_schema()
        self._bind_profile()

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projection_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                episode_json TEXT NOT NULL,
                content_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                summary_digest TEXT NOT NULL,
                formed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projection_versions (
                version_id TEXT PRIMARY KEY,
                projection_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                projection_type TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                projection_json TEXT NOT NULL,
                content_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                produced_at TEXT NOT NULL,
                UNIQUE(projection_id, generation)
            );

            CREATE TABLE IF NOT EXISTS projection_current (
                projection_id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL UNIQUE,
                generation INTEGER NOT NULL,
                FOREIGN KEY(version_id)
                    REFERENCES projection_versions(version_id)
            );

            CREATE TABLE IF NOT EXISTS projection_graph_edges (
                edge_id TEXT PRIMARY KEY,
                projection_version_id TEXT NOT NULL,
                graph_namespace_id TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                weight REAL NOT NULL,
                edge_json TEXT NOT NULL,
                edge_digest TEXT NOT NULL,
                FOREIGN KEY(projection_version_id)
                    REFERENCES projection_versions(version_id)
            );

            CREATE TABLE IF NOT EXISTS projection_vectors (
                projection_version_id TEXT PRIMARY KEY,
                vector_space_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                vector_digest TEXT NOT NULL,
                FOREIGN KEY(projection_version_id)
                    REFERENCES projection_versions(version_id)
            );

            CREATE TABLE IF NOT EXISTS projection_idempotency (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                PRIMARY KEY(namespace, key)
            );

            CREATE INDEX IF NOT EXISTS idx_projection_versions_id
                ON projection_versions(
                    projection_id,
                    generation
                );

            CREATE INDEX IF NOT EXISTS idx_projection_subject
                ON projection_versions(
                    projection_type,
                    subject_type,
                    subject_id
                );

            CREATE INDEX IF NOT EXISTS idx_projection_edges_source
                ON projection_graph_edges(
                    graph_namespace_id,
                    source_node_id,
                    relation_type
                );

            CREATE INDEX IF NOT EXISTS idx_projection_edges_target
                ON projection_graph_edges(
                    graph_namespace_id,
                    target_node_id,
                    relation_type
                );

            CREATE INDEX IF NOT EXISTS idx_projection_vectors_space
                ON projection_vectors(vector_space_id);
            """
        )

    def _bind_profile(self) -> None:
        profile_json = _canonical_json(self.profile.metadata_record())
        existing = self._connection.execute(
            "SELECT value FROM projection_meta WHERE key = ?",
            ("profile",),
        ).fetchone()
        if existing is None:
            self._connection.execute(
                "INSERT INTO projection_meta(key, value) VALUES (?, ?)",
                ("profile", profile_json),
            )
            return
        if str(existing["value"]) != profile_json:
            raise ProjectionPrototypeIsolationError(
                "projection database is bound to a different profile"
            )

    def _assert_scope(self, scope: ProductHostScope) -> None:
        scope.validate()
        if not _same_scope(scope, self.profile.scope):
            raise ProjectionPrototypeIsolationError(
                "record scope differs from projection store scope"
            )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._in_transaction:
            raise ProjectionPrototypeTransactionError(
                "nested prototype transactions are not allowed"
            )
        self._in_transaction = True
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        finally:
            self._in_transaction = False

    def _existing_receipt(
        self,
        *,
        namespace: str,
        key: str,
        request_digest: str,
    ) -> ProjectionAppendReceipt | None:
        row = self._connection.execute(
            """
            SELECT request_digest, receipt_json
            FROM projection_idempotency
            WHERE namespace = ? AND key = ?
            """,
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        if str(row["request_digest"]) != request_digest:
            raise ProjectionPrototypeConflictError(
                "idempotency key was reused with a different request"
            )
        record = _load_json_object(
            str(row["receipt_json"]),
            field="idempotency receipt",
        )
        receipt = ProjectionAppendReceipt(
            operation=str(record["operation"]),
            record_id=str(record["record_id"]),
            generation=int(record["generation"]),
            idempotency_namespace=str(
                record["idempotency_namespace"]
            ),
            idempotency_key=str(record["idempotency_key"]),
            request_digest=str(record["request_digest"]),
            content_digest=str(record["content_digest"]),
            created_at=str(record["created_at"]),
            receipt_sha256=str(record["receipt_sha256"]),
        )
        receipt.validate()
        return receipt

    def _store_receipt(
        self,
        receipt: ProjectionAppendReceipt,
    ) -> None:
        receipt.validate()
        self._connection.execute(
            """
            INSERT INTO projection_idempotency(
                namespace,
                key,
                request_digest,
                receipt_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                receipt.idempotency_namespace,
                receipt.idempotency_key,
                receipt.request_digest,
                _canonical_json(receipt.metadata_record()),
            ),
        )

    def append_episode(
        self,
        episode: EpisodeRecord,
        *,
        full_content: dict[str, object],
        summary_content: dict[str, object],
        idempotency_namespace: object,
        idempotency_key: object,
        request_digest: object,
    ) -> ProjectionAppendReceipt:
        episode.validate()
        self._assert_scope(episode.scope)
        if (
            episode.envelope.authority_namespace_id
            != self.profile.authority_namespace_id
        ):
            raise ProjectionPrototypeIsolationError(
                "episode authority namespace differs from profile"
            )
        namespace = require_identifier(
            idempotency_namespace,
            "idempotency_namespace",
        )
        key = require_identifier(
            idempotency_key,
            "idempotency_key",
        )
        normalized_request_digest = require_sha256(
            request_digest,
            "request_digest",
        )
        full_digest = canonical_sha256(full_content)
        summary_digest = canonical_sha256(summary_content)
        if full_digest != episode.full_content_digest:
            raise ProjectionPrototypeConflictError(
                "episode full-content digest mismatch"
            )
        if summary_digest != episode.summary_content_digest:
            raise ProjectionPrototypeConflictError(
                "episode summary digest mismatch"
            )
        existing = self._existing_receipt(
            namespace=namespace,
            key=key,
            request_digest=normalized_request_digest,
        )
        if existing is not None:
            return existing
        receipt = ProjectionAppendReceipt.create(
            operation="append_episode",
            record_id=episode.episode_id,
            generation=episode.generation,
            idempotency_namespace=namespace,
            idempotency_key=key,
            request_digest=normalized_request_digest,
            content_digest=full_digest,
            created_at=episode.formed_at,
        )
        with self.transaction():
            if self._connection.execute(
                "SELECT 1 FROM episodes WHERE episode_id = ?",
                (episode.episode_id,),
            ).fetchone() is not None:
                raise ProjectionPrototypeConflictError(
                    "episode_id already exists"
                )
            self._connection.execute(
                """
                INSERT INTO episodes(
                    episode_id,
                    episode_json,
                    content_json,
                    summary_json,
                    content_digest,
                    summary_digest,
                    formed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    _canonical_json(episode.metadata_record()),
                    _canonical_json(full_content),
                    _canonical_json(summary_content),
                    full_digest,
                    summary_digest,
                    episode.formed_at,
                ),
            )
            self._store_receipt(receipt)
        return receipt

    def append_projection(
        self,
        projection: ProjectionVersion,
        *,
        full_content: dict[str, object],
        idempotency_namespace: object,
        idempotency_key: object,
        request_digest: object,
        expected_current_generation: int | None,
        vector: Sequence[object] | None = None,
        graph_edges: Iterable[ProjectionGraphEdge] = (),
    ) -> ProjectionAppendReceipt:
        projection.validate()
        self._assert_scope(projection.scope)
        if (
            projection.envelope.authority_namespace_id
            != self.profile.authority_namespace_id
        ):
            raise ProjectionPrototypeIsolationError(
                "projection authority namespace differs from profile"
            )
        namespace = require_identifier(
            idempotency_namespace,
            "idempotency_namespace",
        )
        key = require_identifier(
            idempotency_key,
            "idempotency_key",
        )
        normalized_request_digest = require_sha256(
            request_digest,
            "request_digest",
        )
        content_digest = canonical_sha256(full_content)
        if content_digest != projection.content_digest:
            raise ProjectionPrototypeConflictError(
                "projection content digest mismatch"
            )

        normalized_vector: tuple[float, ...] | None = None
        if "vector" in projection.modalities:
            if vector is None:
                raise ProjectionPrototypeConflictError(
                    "vector modality requires vector content"
                )
            normalized_vector = _finite_vector(
                vector,
                field="vector",
            )
        elif vector is not None:
            raise ProjectionPrototypeConflictError(
                "vector content requires vector modality"
            )

        normalized_edges = tuple(graph_edges)
        if "graph" in projection.modalities:
            if not normalized_edges:
                raise ProjectionPrototypeConflictError(
                    "graph modality requires graph edges"
                )
        elif normalized_edges:
            raise ProjectionPrototypeConflictError(
                "graph edges require graph modality"
            )
        edge_ids: set[str] = set()
        for edge in normalized_edges:
            edge.validate()
            if (
                edge.graph_namespace_id
                != projection.graph_namespace_id
            ):
                raise ProjectionPrototypeConflictError(
                    "graph edge namespace differs from projection"
                )
            if edge.edge_id in edge_ids:
                raise ProjectionPrototypeConflictError(
                    "graph edge IDs may not repeat"
                )
            edge_ids.add(edge.edge_id)

        existing = self._existing_receipt(
            namespace=namespace,
            key=key,
            request_digest=normalized_request_digest,
        )
        if existing is not None:
            return existing

        current = self._connection.execute(
            """
            SELECT version_id, generation
            FROM projection_current
            WHERE projection_id = ?
            """,
            (projection.projection_id,),
        ).fetchone()
        current_generation = (
            int(current["generation"])
            if current is not None
            else None
        )
        current_version_id = (
            str(current["version_id"])
            if current is not None
            else None
        )
        if expected_current_generation != current_generation:
            raise ProjectionPrototypeConflictError(
                "expected-current generation mismatch"
            )
        required_generation = (
            1 if current_generation is None else current_generation + 1
        )
        if projection.generation != required_generation:
            raise ProjectionPrototypeConflictError(
                "projection generation is not the next store generation"
            )
        if current_version_id != projection.supersedes_version_id:
            raise ProjectionPrototypeConflictError(
                "projection predecessor differs from current version"
            )

        receipt = ProjectionAppendReceipt.create(
            operation="append_projection",
            record_id=projection.version_id,
            generation=projection.generation,
            idempotency_namespace=namespace,
            idempotency_key=key,
            request_digest=normalized_request_digest,
            content_digest=content_digest,
            created_at=projection.produced_at,
        )

        with self.transaction():
            if self._connection.execute(
                """
                SELECT 1 FROM projection_versions
                WHERE version_id = ?
                """,
                (projection.version_id,),
            ).fetchone() is not None:
                raise ProjectionPrototypeConflictError(
                    "projection version already exists"
                )
            self._connection.execute(
                """
                INSERT INTO projection_versions(
                    version_id,
                    projection_id,
                    generation,
                    projection_type,
                    subject_type,
                    subject_id,
                    projection_json,
                    content_json,
                    content_digest,
                    produced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    projection.version_id,
                    projection.projection_id,
                    projection.generation,
                    projection.projection_type,
                    projection.subject_type,
                    projection.subject_id,
                    _canonical_json(projection.metadata_record()),
                    _canonical_json(full_content),
                    content_digest,
                    projection.produced_at,
                ),
            )
            if current is None:
                self._connection.execute(
                    """
                    INSERT INTO projection_current(
                        projection_id,
                        version_id,
                        generation
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        projection.projection_id,
                        projection.version_id,
                        projection.generation,
                    ),
                )
            else:
                self._connection.execute(
                    """
                    UPDATE projection_current
                    SET version_id = ?, generation = ?
                    WHERE projection_id = ?
                    """,
                    (
                        projection.version_id,
                        projection.generation,
                        projection.projection_id,
                    ),
                )

            if normalized_vector is not None:
                vector_json = _canonical_json(
                    list(normalized_vector)
                )
                vector_digest = canonical_sha256(
                    list(normalized_vector)
                )
                self._connection.execute(
                    """
                    INSERT INTO projection_vectors(
                        projection_version_id,
                        vector_space_id,
                        dimensions,
                        vector_json,
                        vector_digest
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        projection.version_id,
                        projection.vector_space_id,
                        len(normalized_vector),
                        vector_json,
                        vector_digest,
                    ),
                )

            for edge in normalized_edges:
                self._connection.execute(
                    """
                    INSERT INTO projection_graph_edges(
                        edge_id,
                        projection_version_id,
                        graph_namespace_id,
                        source_node_id,
                        relation_type,
                        target_node_id,
                        valid_from,
                        valid_to,
                        weight,
                        edge_json,
                        edge_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.edge_id,
                        projection.version_id,
                        edge.graph_namespace_id,
                        edge.source_node_id,
                        edge.relation_type,
                        edge.target_node_id,
                        edge.valid_from,
                        edge.valid_to,
                        edge.weight,
                        _canonical_json(edge.metadata_record()),
                        edge.edge_sha256,
                    ),
                )
            self._store_receipt(receipt)
        return receipt

    def get_episode(
        self,
        episode_id: object,
    ) -> dict[str, object] | None:
        normalized_id = require_identifier(
            episode_id,
            "episode_id",
        )
        row = self._connection.execute(
            """
            SELECT episode_json, content_json, summary_json
            FROM episodes
            WHERE episode_id = ?
            """,
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "episode": _load_json_object(
                str(row["episode_json"]),
                field="episode",
            ),
            "full_content": _load_json_object(
                str(row["content_json"]),
                field="episode content",
            ),
            "summary_content": _load_json_object(
                str(row["summary_json"]),
                field="episode summary",
            ),
        }

    def get_projection_version(
        self,
        version_id: object,
    ) -> dict[str, object] | None:
        normalized_id = require_identifier(
            version_id,
            "version_id",
        )
        row = self._connection.execute(
            """
            SELECT projection_json, content_json
            FROM projection_versions
            WHERE version_id = ?
            """,
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        result: dict[str, object] = {
            "projection": _load_json_object(
                str(row["projection_json"]),
                field="projection version",
            ),
            "full_content": _load_json_object(
                str(row["content_json"]),
                field="projection content",
            ),
        }
        vector_row = self._connection.execute(
            """
            SELECT vector_space_id, vector_json
            FROM projection_vectors
            WHERE projection_version_id = ?
            """,
            (normalized_id,),
        ).fetchone()
        if vector_row is not None:
            result["vector_space_id"] = str(
                vector_row["vector_space_id"]
            )
            result["vector"] = _load_json_array(
                str(vector_row["vector_json"]),
                field="projection vector",
            )
        edge_rows = self._connection.execute(
            """
            SELECT edge_json
            FROM projection_graph_edges
            WHERE projection_version_id = ?
            ORDER BY edge_id
            """,
            (normalized_id,),
        ).fetchall()
        result["graph_edges"] = [
            _load_json_object(
                str(item["edge_json"]),
                field="graph edge",
            )
            for item in edge_rows
        ]
        return result

    def get_current_projection(
        self,
        projection_id: object,
    ) -> dict[str, object] | None:
        normalized_id = require_identifier(
            projection_id,
            "projection_id",
        )
        row = self._connection.execute(
            """
            SELECT version_id
            FROM projection_current
            WHERE projection_id = ?
            """,
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        return self.get_projection_version(str(row["version_id"]))

    def projection_history(
        self,
        projection_id: object,
    ) -> tuple[dict[str, object], ...]:
        normalized_id = require_identifier(
            projection_id,
            "projection_id",
        )
        rows = self._connection.execute(
            """
            SELECT version_id
            FROM projection_versions
            WHERE projection_id = ?
            ORDER BY generation
            """,
            (normalized_id,),
        ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            value = self.get_projection_version(
                str(row["version_id"])
            )
            if value is not None:
                result.append(value)
        return tuple(result)

    def projection_as_of(
        self,
        projection_id: object,
        *,
        produced_at: object,
    ) -> dict[str, object] | None:
        normalized_id = require_identifier(
            projection_id,
            "projection_id",
        )
        timestamp = normalize_timestamp(
            produced_at,
            "produced_at",
        )
        row = self._connection.execute(
            """
            SELECT version_id
            FROM projection_versions
            WHERE projection_id = ? AND produced_at <= ?
            ORDER BY produced_at DESC, generation DESC
            LIMIT 1
            """,
            (normalized_id, timestamp),
        ).fetchone()
        if row is None:
            return None
        return self.get_projection_version(str(row["version_id"]))

    def neighbors(
        self,
        *,
        graph_namespace_id: object,
        node_id: object,
        relation_type: object | None = None,
        valid_at: object | None = None,
    ) -> tuple[dict[str, object], ...]:
        namespace = require_identifier(
            graph_namespace_id,
            "graph_namespace_id",
        )
        node = require_identifier(node_id, "node_id")
        relation = (
            require_identifier(relation_type, "relation_type")
            if relation_type is not None
            else None
        )
        timestamp = (
            normalize_timestamp(valid_at, "valid_at")
            if valid_at is not None
            else None
        )
        query = """
            SELECT edge_json
            FROM projection_graph_edges
            WHERE graph_namespace_id = ?
              AND (source_node_id = ? OR target_node_id = ?)
        """
        parameters: list[object] = [namespace, node, node]
        if relation is not None:
            query += " AND relation_type = ?"
            parameters.append(relation)
        if timestamp is not None:
            query += (
                " AND valid_from <= ?"
                " AND (valid_to IS NULL OR valid_to >= ?)"
            )
            parameters.extend([timestamp, timestamp])
        query += " ORDER BY edge_id"
        rows = self._connection.execute(
            query,
            tuple(parameters),
        ).fetchall()
        return tuple(
            _load_json_object(
                str(row["edge_json"]),
                field="graph edge",
            )
            for row in rows
        )

    def similarity_search(
        self,
        query_vector: Sequence[object],
        *,
        vector_space_id: object,
        limit: int = 10,
    ) -> tuple[dict[str, object], ...]:
        vector = _finite_vector(
            query_vector,
            field="query_vector",
        )
        space = require_identifier(
            vector_space_id,
            "vector_space_id",
        )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise CognitiveKernelContractError(
                "limit must be between 1 and 100"
            )
        rows = self._connection.execute(
            """
            SELECT
                v.projection_version_id,
                v.dimensions,
                v.vector_json,
                p.projection_id,
                p.projection_type,
                p.subject_type,
                p.subject_id
            FROM projection_vectors AS v
            JOIN projection_versions AS p
              ON p.version_id = v.projection_version_id
            WHERE v.vector_space_id = ?
            """,
            (space,),
        ).fetchall()
        scored: list[dict[str, object]] = []
        for row in rows:
            stored_raw = _load_json_array(
                str(row["vector_json"]),
                field="stored vector",
            )
            stored = _finite_vector(
                stored_raw,
                field="stored vector",
            )
            if len(stored) != len(vector):
                continue
            scored.append(
                {
                    "projection_version_id": str(
                        row["projection_version_id"]
                    ),
                    "projection_id": str(row["projection_id"]),
                    "projection_type": str(
                        row["projection_type"]
                    ),
                    "subject_type": str(row["subject_type"]),
                    "subject_id": str(row["subject_id"]),
                    "similarity": _cosine(vector, stored),
                }
            )
        scored.sort(
            key=lambda item: (
                -float(item["similarity"]),
                str(item["projection_version_id"]),
            )
        )
        return tuple(scored[:limit])

    def verify_integrity(self) -> ProjectionIntegrityReport:
        errors: list[str] = []
        profile_row = self._connection.execute(
            "SELECT value FROM projection_meta WHERE key = ?",
            ("profile",),
        ).fetchone()
        if profile_row is None:
            errors.append("profile metadata is missing")
        elif str(profile_row["value"]) != _canonical_json(
            self.profile.metadata_record()
        ):
            errors.append("profile metadata differs from active profile")

        episode_rows = self._connection.execute(
            """
            SELECT
                episode_id,
                episode_json,
                content_json,
                summary_json,
                content_digest,
                summary_digest
            FROM episodes
            ORDER BY episode_id
            """
        ).fetchall()
        for row in episode_rows:
            try:
                record = _load_json_object(
                    str(row["episode_json"]),
                    field="episode",
                )
                digest = str(record.pop("episode_sha256"))
                if canonical_sha256(record) != digest:
                    errors.append(
                        f"episode digest mismatch:{row['episode_id']}"
                    )
                content = _load_json_object(
                    str(row["content_json"]),
                    field="episode content",
                )
                summary = _load_json_object(
                    str(row["summary_json"]),
                    field="episode summary",
                )
                if canonical_sha256(content) != str(
                    row["content_digest"]
                ):
                    errors.append(
                        f"episode content mismatch:{row['episode_id']}"
                    )
                if canonical_sha256(summary) != str(
                    row["summary_digest"]
                ):
                    errors.append(
                        f"episode summary mismatch:{row['episode_id']}"
                    )
            except (KeyError, ProjectionPrototypeIntegrityError):
                errors.append(
                    f"episode record malformed:{row['episode_id']}"
                )

        projection_rows = self._connection.execute(
            """
            SELECT
                version_id,
                projection_json,
                content_json,
                content_digest
            FROM projection_versions
            ORDER BY version_id
            """
        ).fetchall()
        version_ids: set[str] = set()
        for row in projection_rows:
            version_id = str(row["version_id"])
            version_ids.add(version_id)
            try:
                record = _load_json_object(
                    str(row["projection_json"]),
                    field="projection version",
                )
                digest = str(record.pop("projection_sha256"))
                if canonical_sha256(record) != digest:
                    errors.append(
                        f"projection digest mismatch:{version_id}"
                    )
                content = _load_json_object(
                    str(row["content_json"]),
                    field="projection content",
                )
                if canonical_sha256(content) != str(
                    row["content_digest"]
                ):
                    errors.append(
                        f"projection content mismatch:{version_id}"
                    )
            except (KeyError, ProjectionPrototypeIntegrityError):
                errors.append(
                    f"projection record malformed:{version_id}"
                )

        current_rows = self._connection.execute(
            """
            SELECT projection_id, version_id, generation
            FROM projection_current
            ORDER BY projection_id
            """
        ).fetchall()
        for row in current_rows:
            version_id = str(row["version_id"])
            if version_id not in version_ids:
                errors.append(
                    "current projection references missing version:"
                    f"{version_id}"
                )

        edge_rows = self._connection.execute(
            """
            SELECT
                edge_id,
                projection_version_id,
                edge_json,
                edge_digest
            FROM projection_graph_edges
            ORDER BY edge_id
            """
        ).fetchall()
        for row in edge_rows:
            edge_id = str(row["edge_id"])
            if str(row["projection_version_id"]) not in version_ids:
                errors.append(
                    f"edge references missing projection:{edge_id}"
                )
            try:
                record = _load_json_object(
                    str(row["edge_json"]),
                    field="graph edge",
                )
                digest = str(record.pop("edge_sha256"))
                if (
                    canonical_sha256(record) != digest
                    or digest != str(row["edge_digest"])
                ):
                    errors.append(
                        f"graph edge digest mismatch:{edge_id}"
                    )
            except (KeyError, ProjectionPrototypeIntegrityError):
                errors.append(f"graph edge malformed:{edge_id}")

        vector_rows = self._connection.execute(
            """
            SELECT
                projection_version_id,
                dimensions,
                vector_json,
                vector_digest
            FROM projection_vectors
            ORDER BY projection_version_id
            """
        ).fetchall()
        for row in vector_rows:
            version_id = str(row["projection_version_id"])
            if version_id not in version_ids:
                errors.append(
                    f"vector references missing projection:{version_id}"
                )
            try:
                raw = _load_json_array(
                    str(row["vector_json"]),
                    field="projection vector",
                )
                vector = _finite_vector(
                    raw,
                    field="projection vector",
                )
                if len(vector) != int(row["dimensions"]):
                    errors.append(
                        f"vector dimension mismatch:{version_id}"
                    )
                if canonical_sha256(list(vector)) != str(
                    row["vector_digest"]
                ):
                    errors.append(
                        f"vector digest mismatch:{version_id}"
                    )
            except (
                CognitiveKernelContractError,
                ProjectionPrototypeIntegrityError,
            ):
                errors.append(f"vector malformed:{version_id}")

        receipt_rows = self._connection.execute(
            """
            SELECT namespace, key, request_digest, receipt_json
            FROM projection_idempotency
            ORDER BY namespace, key
            """
        ).fetchall()
        for row in receipt_rows:
            try:
                record = _load_json_object(
                    str(row["receipt_json"]),
                    field="idempotency receipt",
                )
                if str(record["request_digest"]) != str(
                    row["request_digest"]
                ):
                    errors.append(
                        "idempotency request digest mismatch:"
                        f"{row['namespace']}:{row['key']}"
                    )
                digest = str(record.pop("receipt_sha256"))
                if canonical_sha256(record) != digest:
                    errors.append(
                        "idempotency receipt digest mismatch:"
                        f"{row['namespace']}:{row['key']}"
                    )
            except (KeyError, ProjectionPrototypeIntegrityError):
                errors.append(
                    "idempotency receipt malformed:"
                    f"{row['namespace']}:{row['key']}"
                )

        return ProjectionIntegrityReport(
            valid=not errors,
            episode_count=len(episode_rows),
            projection_version_count=len(projection_rows),
            current_projection_count=len(current_rows),
            graph_edge_count=len(edge_rows),
            vector_count=len(vector_rows),
            idempotency_receipt_count=len(receipt_rows),
            errors=tuple(errors),
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    def delete_database(self) -> None:
        self.close()
        self.database_path.unlink(missing_ok=True)
        Path(str(self.database_path) + "-wal").unlink(missing_ok=True)
        Path(str(self.database_path) + "-shm").unlink(missing_ok=True)

    def __enter__(self) -> "ProjectionPrototypeStore":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()


def open_projection_prototype(
    database_path: str | Path,
    *,
    profile: ProjectionPrototypeProfile,
    repository_root: str | Path | None = None,
) -> ProjectionPrototypeStore:
    """Open one reversible projection-fabric prototype store."""
    return ProjectionPrototypeStore(
        database_path,
        profile=profile,
        repository_root=repository_root,
    )
