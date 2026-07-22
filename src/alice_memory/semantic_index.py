"""Private, derived semantic index for Phase 2 memory retrieval.

The semantic index is memory-specific and rebuildable from the authoritative
Memory Core. It may reuse the same local embedding-model policy as Phase 1, but
it never reuses or mutates Phase 1 evidence indexes.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from alice_vault.semantic_retrieval import (
    SemanticPolicy,
    load_semantic_policy,
    model_root,
)

from .lexical_index import (
    _eligible_rows,
    authoritative_retrieval_digest,
)
from .retrieval_models import (
    MemoryLexicalIndexError,
    StaleMemoryLexicalIndexError,
)
from .store import default_repository_root, validate_private_database_path


MEMORY_SEMANTIC_INDEX_RELATIVE_ROOT = Path(
    "memory",
    "phase2",
    "indexes",
    "semantic",
)
_SEMANTIC_INDEX_VERSION = 1


class MemorySemanticIndexError(MemoryLexicalIndexError):
    """Raised when the derived semantic index cannot be used safely."""


class StaleMemorySemanticIndexError(
    StaleMemoryLexicalIndexError,
    MemorySemanticIndexError,
):
    """Raised when semantic index state differs from authoritative memory."""


@dataclass(frozen=True)
class MemorySemanticIndexManifest:
    index_id: str
    index_version: int
    authoritative_digest: str
    record_count: int
    embedding_dimension: int
    model_id: str
    model_revision: str
    policy_digest: str
    built_at: str


def memory_semantic_index_root(
    vault_root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    vault = Path(vault_root).expanduser().resolve(strict=True)
    repository = (
        default_repository_root()
        if repository_root is None
        else Path(repository_root).expanduser().resolve(strict=True)
    )
    path = (
        vault
        / MEMORY_SEMANTIC_INDEX_RELATIVE_ROOT
    ).resolve(strict=False)
    return validate_private_database_path(
        path / "sentinel.sqlite3",
        repository_root=repository,
    ).parent


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_index_id(
    *,
    authoritative_digest: str,
    policy: SemanticPolicy,
    record_count: int,
) -> str:
    payload = {
        "version": _SEMANTIC_INDEX_VERSION,
        "authoritative_digest": authoritative_digest,
        "policy_digest": policy.digest,
        "model_id": policy.model.model_id,
        "model_revision": policy.model.revision,
        "record_count": record_count,
    }
    return hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()[:32]


def _normalize_rows(
    rows: Sequence[Sequence[float]],
    *,
    dimension: int,
) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in rows:
        values = [float(value) for value in row]
        if len(values) != dimension:
            raise MemorySemanticIndexError(
                "Encoder returned unexpected embedding dimension."
            )
        norm = math.sqrt(
            sum(value * value for value in values)
        )
        if not math.isfinite(norm) or norm <= 0.0:
            raise MemorySemanticIndexError(
                "Encoder returned an invalid zero/non-finite embedding."
            )
        normalized.append(
            [value / norm for value in values]
        )
    return normalized


def _encoded_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        raise MemorySemanticIndexError(
            "Encoder returned an unsupported embedding container."
        )

    if value and not isinstance(value[0], (list, tuple)):
        return [[float(item) for item in value]]

    return [
        [float(item) for item in row]
        for row in value
    ]


def _encode_documents(
    model: Any,
    texts: list[str],
    policy: SemanticPolicy,
) -> list[list[float]]:
    if not texts:
        return []
    value = model.encode(
        texts,
        batch_size=policy.build.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=policy.build.show_progress,
    )
    return _normalize_rows(
        _encoded_rows(value),
        dimension=policy.model.embedding_dimension,
    )


def encode_memory_query(
    model: Any,
    query: str,
    policy: SemanticPolicy,
) -> list[float]:
    value = model.encode(
        [query],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    rows = _normalize_rows(
        _encoded_rows(value),
        dimension=policy.model.embedding_dimension,
    )
    if len(rows) != 1:
        raise MemorySemanticIndexError(
            "Query encoder returned an unexpected row count."
        )
    return rows[0]


def _write_float32_matrix(
    path: Path,
    rows: Iterable[Sequence[float]],
) -> None:
    values = array.array("f")
    for row in rows:
        values.extend(float(value) for value in row)
    with path.open("wb") as handle:
        values.tofile(handle)


def _read_float32_matrix(
    path: Path,
    *,
    rows: int,
    dimension: int,
) -> list[float]:
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, rows * dimension)
    if len(values) != rows * dimension:
        raise MemorySemanticIndexError(
            "Semantic embedding binary has unexpected size."
        )
    return [float(value) for value in values]


def load_memory_embedding_model(
    vault_root: str | Path,
    *,
    policy_path: Path | None = None,
    device: str = "auto",
) -> tuple[Any, SemanticPolicy]:
    """Load the already-prepared local embedding model with no network access."""
    policy = load_semantic_policy(policy_path)
    path = model_root(
        Path(vault_root).expanduser().resolve(strict=True),
        policy,
    )
    if not path.exists():
        raise FileNotFoundError(
            "Local embedding model is missing. Prepare the Phase 1 embedding "
            "model before building the Phase 2 semantic memory index."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for semantic memory retrieval."
        ) from exc

    selected_device = None if device in {"", "auto"} else device
    model = SentenceTransformer(
        str(path),
        device=selected_device,
        prompts={
            "query": policy.model.query_prompt,
            "document": policy.model.document_prompt,
        },
        trust_remote_code=False,
        local_files_only=True,
        model_kwargs={"use_safetensors": True},
    )

    dimension = int(
        model.get_sentence_embedding_dimension()
    )
    if dimension != policy.model.embedding_dimension:
        raise MemorySemanticIndexError(
            "Loaded embedding model dimension does not match semantic policy."
        )
    return model, policy


def build_memory_semantic_index(
    connection,
    vault_root: str | Path,
    *,
    model: Any,
    policy_path: Path | None = None,
    repository_root: str | Path | None = None,
    built_at: str,
) -> MemorySemanticIndexManifest:
    """Build an immutable semantic index and atomically update its pointer."""
    policy = load_semantic_policy(policy_path)
    dimension = int(
        model.get_sentence_embedding_dimension()
    )
    if dimension != policy.model.embedding_dimension:
        raise MemorySemanticIndexError(
            "Loaded model dimension does not match semantic policy."
        )

    root = memory_semantic_index_root(
        vault_root,
        repository_root=repository_root,
    )
    root.mkdir(parents=True, exist_ok=True)

    authoritative_digest, record_count = (
        authoritative_retrieval_digest(connection)
    )
    index_id = _semantic_index_id(
        authoritative_digest=authoritative_digest,
        policy=policy,
        record_count=record_count,
    )

    rows = _eligible_rows(connection)
    texts = [str(row["content"]) for row in rows]
    embeddings = _encode_documents(
        model,
        texts,
        policy,
    )
    if len(embeddings) != len(rows):
        raise MemorySemanticIndexError(
            "Encoder returned unexpected semantic row count."
        )

    destination = root / index_id
    with tempfile.TemporaryDirectory(
        prefix="alice-memory-semantic-",
        dir=root,
    ) as temp:
        stage = Path(temp) / index_id
        stage.mkdir()

        embeddings_path = stage / "embeddings.f32"
        map_path = stage / "memory-map.jsonl"
        manifest_path = stage / "semantic-manifest.json"

        _write_float32_matrix(
            embeddings_path,
            embeddings,
        )

        with map_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "memory_id": str(row["memory_id"]),
                            "content_sha256": str(
                                row["content_sha256"]
                            ),
                            "data_classification": str(
                                row["data_classification"]
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        manifest = {
            "index_id": index_id,
            "index_version": _SEMANTIC_INDEX_VERSION,
            "authoritative_digest": authoritative_digest,
            "record_count": record_count,
            "embedding_dimension": dimension,
            "model_id": policy.model.model_id,
            "model_revision": policy.model.revision,
            "policy_digest": policy.digest,
            "built_at": built_at,
            "embeddings_sha256": _sha256_file(
                embeddings_path
            ),
            "memory_map_sha256": _sha256_file(
                map_path
            ),
        }
        manifest_path.write_bytes(
            _canonical_json(manifest)
        )

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(
            stage,
            destination,
        )

    pointer = root / "current.json"
    pointer_temp = root / (
        f"current.{os.getpid()}.tmp"
    )
    pointer_temp.write_bytes(
        _canonical_json(
            {
                "index_id": index_id,
            }
        )
    )
    os.replace(
        pointer_temp,
        pointer,
    )

    return MemorySemanticIndexManifest(
        index_id=index_id,
        index_version=_SEMANTIC_INDEX_VERSION,
        authoritative_digest=authoritative_digest,
        record_count=record_count,
        embedding_dimension=dimension,
        model_id=policy.model.model_id,
        model_revision=policy.model.revision,
        policy_digest=policy.digest,
        built_at=built_at,
    )


def _current_semantic_index_path(
    root: Path,
) -> Path:
    pointer = root / "current.json"
    if not pointer.exists():
        raise MemorySemanticIndexError(
            "Current semantic memory index pointer is missing."
        )
    data = json.loads(
        pointer.read_text(encoding="utf-8")
    )
    index_id = str(
        data.get("index_id", "")
    ).strip()
    if not index_id:
        raise MemorySemanticIndexError(
            "Current semantic memory index pointer is invalid."
        )
    path = root / index_id
    if not path.exists():
        raise MemorySemanticIndexError(
            "Current semantic memory index directory is missing."
        )
    return path


def verify_memory_semantic_index(
    connection,
    vault_root: str | Path,
    *,
    policy_path: Path | None = None,
    repository_root: str | Path | None = None,
) -> tuple[MemorySemanticIndexManifest, Path]:
    """Verify digests, model policy, and authoritative-memory freshness."""
    policy = load_semantic_policy(policy_path)
    root = memory_semantic_index_root(
        vault_root,
        repository_root=repository_root,
    )
    index_path = _current_semantic_index_path(
        root
    )
    manifest_path = (
        index_path
        / "semantic-manifest.json"
    )
    map_path = index_path / "memory-map.jsonl"
    embeddings_path = (
        index_path
        / "embeddings.f32"
    )

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MemorySemanticIndexError(
            "Semantic memory index manifest is missing or invalid."
        ) from exc

    digest, record_count = authoritative_retrieval_digest(
        connection
    )
    if (
        str(manifest.get("authoritative_digest")) != digest
        or int(manifest.get("record_count", -1)) != record_count
    ):
        raise StaleMemorySemanticIndexError(
            "Semantic memory index is stale relative to the authoritative "
            "Memory Core. Rebuild it before retrieval."
        )

    if int(
        manifest.get("index_version", -1)
    ) != _SEMANTIC_INDEX_VERSION:
        raise MemorySemanticIndexError(
            "Unsupported semantic memory index version."
        )
    if str(
        manifest.get("policy_digest")
    ) != policy.digest:
        raise MemorySemanticIndexError(
            "Semantic memory index policy digest mismatch."
        )
    if str(
        manifest.get("model_id")
    ) != policy.model.model_id:
        raise MemorySemanticIndexError(
            "Semantic memory index model ID mismatch."
        )
    if str(
        manifest.get("model_revision")
    ) != policy.model.revision:
        raise MemorySemanticIndexError(
            "Semantic memory index model revision mismatch."
        )
    if int(
        manifest.get("embedding_dimension", -1)
    ) != policy.model.embedding_dimension:
        raise MemorySemanticIndexError(
            "Semantic memory index dimension mismatch."
        )
    if str(
        manifest.get("embeddings_sha256")
    ) != _sha256_file(embeddings_path):
        raise MemorySemanticIndexError(
            "Semantic embedding binary digest mismatch."
        )
    if str(
        manifest.get("memory_map_sha256")
    ) != _sha256_file(map_path):
        raise MemorySemanticIndexError(
            "Semantic memory-map digest mismatch."
        )

    expected_bytes = (
        record_count
        * policy.model.embedding_dimension
        * 4
    )
    if embeddings_path.stat().st_size != expected_bytes:
        raise MemorySemanticIndexError(
            "Semantic embedding binary size mismatch."
        )

    return (
        MemorySemanticIndexManifest(
            index_id=str(manifest["index_id"]),
            index_version=int(
                manifest["index_version"]
            ),
            authoritative_digest=str(
                manifest["authoritative_digest"]
            ),
            record_count=int(
                manifest["record_count"]
            ),
            embedding_dimension=int(
                manifest["embedding_dimension"]
            ),
            model_id=str(
                manifest["model_id"]
            ),
            model_revision=str(
                manifest["model_revision"]
            ),
            policy_digest=str(
                manifest["policy_digest"]
            ),
            built_at=str(
                manifest["built_at"]
            ),
        ),
        index_path,
    )


def semantic_memory_candidates(
    connection,
    vault_root: str | Path,
    *,
    query: str,
    model: Any,
    policy_path: Path | None = None,
    repository_root: str | Path | None = None,
    limit: int,
) -> tuple[
    MemorySemanticIndexManifest,
    list[tuple[str, float]],
]:
    """Return semantic candidate memory IDs and cosine similarities."""
    policy = load_semantic_policy(policy_path)
    manifest, index_path = (
        verify_memory_semantic_index(
            connection,
            vault_root,
            policy_path=policy_path,
            repository_root=repository_root,
        )
    )

    dimension = int(
        model.get_sentence_embedding_dimension()
    )
    if dimension != manifest.embedding_dimension:
        raise MemorySemanticIndexError(
            "Query model dimension does not match semantic index."
        )

    query_vector = encode_memory_query(
        model,
        query,
        policy,
    )

    map_rows = [
        json.loads(line)
        for line in (
            index_path
            / "memory-map.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    values = _read_float32_matrix(
        index_path / "embeddings.f32",
        rows=manifest.record_count,
        dimension=manifest.embedding_dimension,
    )

    scored: list[tuple[str, float]] = []
    for row_index, item in enumerate(map_rows):
        start = (
            row_index
            * manifest.embedding_dimension
        )
        end = (
            start
            + manifest.embedding_dimension
        )
        vector = values[start:end]
        score = sum(
            left * right
            for left, right in zip(
                query_vector,
                vector,
                strict=True,
            )
        )
        scored.append(
            (
                str(item["memory_id"]),
                float(score),
            )
        )

    scored.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )
    return manifest, scored[:limit]
