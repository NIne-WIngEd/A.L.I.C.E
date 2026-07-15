from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import uuid
from array import array
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .retrieval import (
    SearchFilters,
    atomic_json,
    load_chunk_catalog,
    locate_chunk_set,
    search_index,
)


SEMANTIC_POLICY_SCHEMA_VERSION = 1
SEMANTIC_INDEX_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ModelPolicy:
    model_id: str
    revision: str
    license: str
    language: str
    embedding_dimension: int
    maximum_sequence_tokens: int
    query_prompt: str
    document_prompt: str
    trust_remote_code: bool
    safe_serialization_required: bool
    local_directory_name: str


@dataclass(frozen=True)
class BuildPolicy:
    batch_size: int
    normalize_embeddings: bool
    storage_dtype: str
    device: str
    show_progress: bool


@dataclass(frozen=True)
class SearchPolicy:
    semantic_candidate_k: int
    lexical_candidate_k: int
    default_limit: int
    maximum_chunks_per_source: int
    snippet_characters: int
    rrf_k: int
    lexical_weight: float
    semantic_weight: float


@dataclass(frozen=True)
class BenchmarkPolicy:
    minimum_approved_cases: int
    candidate_sources_per_question: int
    evaluation_k_values: tuple[int, ...]


@dataclass(frozen=True)
class SemanticPolicy:
    policy_id: str
    model: ModelPolicy
    build: BuildPolicy
    search: SearchPolicy
    benchmark: BenchmarkPolicy
    digest: str
    source_path: Path


def default_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "semantic_retrieval_policy.json"
    )


def load_semantic_policy(
    path: Path | None = None,
) -> SemanticPolicy:
    source = (path or default_policy_path()).expanduser().resolve(
        strict=True
    )
    data = json.loads(source.read_text(encoding="utf-8"))
    if (
        int(data.get("semantic_retrieval_policy_schema_version", -1))
        != SEMANTIC_POLICY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported semantic-retrieval policy schema")

    model_data = dict(data["model"])
    build_data = dict(data["build"])
    search_data = dict(data["search"])
    benchmark_data = dict(data["benchmark"])
    policy = SemanticPolicy(
        policy_id=str(data["policy_id"]),
        model=ModelPolicy(
            model_id=str(model_data["model_id"]),
            revision=str(model_data["revision"]),
            license=str(model_data["license"]),
            language=str(model_data["language"]),
            embedding_dimension=int(
                model_data["embedding_dimension"]
            ),
            maximum_sequence_tokens=int(
                model_data["maximum_sequence_tokens"]
            ),
            query_prompt=str(model_data["query_prompt"]),
            document_prompt=str(model_data["document_prompt"]),
            trust_remote_code=bool(
                model_data["trust_remote_code"]
            ),
            safe_serialization_required=bool(
                model_data["safe_serialization_required"]
            ),
            local_directory_name=str(
                model_data["local_directory_name"]
            ),
        ),
        build=BuildPolicy(
            batch_size=int(build_data["batch_size"]),
            normalize_embeddings=bool(
                build_data["normalize_embeddings"]
            ),
            storage_dtype=str(build_data["storage_dtype"]),
            device=str(build_data["device"]),
            show_progress=bool(build_data["show_progress"]),
        ),
        search=SearchPolicy(
            semantic_candidate_k=int(
                search_data["semantic_candidate_k"]
            ),
            lexical_candidate_k=int(
                search_data["lexical_candidate_k"]
            ),
            default_limit=int(search_data["default_limit"]),
            maximum_chunks_per_source=int(
                search_data["maximum_chunks_per_source"]
            ),
            snippet_characters=int(
                search_data["snippet_characters"]
            ),
            rrf_k=int(search_data["rrf_k"]),
            lexical_weight=float(search_data["lexical_weight"]),
            semantic_weight=float(
                search_data["semantic_weight"]
            ),
        ),
        benchmark=BenchmarkPolicy(
            minimum_approved_cases=int(
                benchmark_data["minimum_approved_cases"]
            ),
            candidate_sources_per_question=int(
                benchmark_data[
                    "candidate_sources_per_question"
                ]
            ),
            evaluation_k_values=tuple(
                int(value)
                for value in benchmark_data["evaluation_k_values"]
            ),
        ),
        digest=sha256_bytes(canonical_json(data)),
        source_path=source,
    )
    _validate_policy(policy)
    return policy


def _validate_policy(policy: SemanticPolicy) -> None:
    if policy.model.embedding_dimension < 8:
        raise ValueError("Embedding dimension is invalid")
    if policy.model.maximum_sequence_tokens < 32:
        raise ValueError("Maximum sequence length is invalid")
    if policy.build.storage_dtype != "float32":
        raise ValueError("Only float32 storage is supported")
    if policy.build.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if policy.search.default_limit < 1:
        raise ValueError("Search limit must be positive")
    if policy.search.rrf_k < 1:
        raise ValueError("RRF k must be positive")
    if policy.model.trust_remote_code:
        raise ValueError("Remote model code is forbidden")
    if not policy.model.safe_serialization_required:
        raise ValueError("Safe model serialization is required")


def model_root(
    vault_root: Path,
    policy: SemanticPolicy,
) -> Path:
    return (
        vault_root
        / "models"
        / "embeddings"
        / policy.model.local_directory_name
    )


def _tree_digest(
    root: Path,
    *,
    exclude_names: set[str] | None = None,
) -> str:
    exclude_names = exclude_names or set()
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in exclude_names
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _unsafe_model_files(root: Path) -> list[str]:
    unsafe_suffixes = {
        ".bin",
        ".pt",
        ".pth",
        ".pkl",
        ".pickle",
        ".joblib",
    }
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in unsafe_suffixes
    )


def _default_model_loader(
    path_or_id: str,
    *,
    revision: str | None,
    device: str | None,
    local_files_only: bool,
    policy: SemanticPolicy,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install "
            "requirements-semantic-retrieval.txt."
        ) from exc

    selected_device = None if device in {None, "", "auto"} else device
    return SentenceTransformer(
        path_or_id,
        device=selected_device,
        prompts={
            "query": policy.model.query_prompt,
            "document": policy.model.document_prompt,
        },
        trust_remote_code=False,
        revision=revision,
        local_files_only=local_files_only,
        model_kwargs={"use_safetensors": True},
    )


ModelLoader = Callable[..., Any]


def prepare_embedding_model(
    *,
    vault_root: Path,
    policy_path: Path | None = None,
    device: str = "auto",
    replace: bool = False,
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(policy_path)
    destination = model_root(vault_root, policy)
    manifest_path = destination / "alice-model-manifest.json"
    exports = vault_root / "manifests" / "exports"
    temporary_root = vault_root / "temporary"
    exports.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())

    if destination.exists() and manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        actual_digest = _tree_digest(
            destination,
            exclude_names={"alice-model-manifest.json"},
        )
        if (
            manifest.get("model_id") == policy.model.model_id
            and manifest.get("revision") == policy.model.revision
            and manifest.get("model_tree_digest") == actual_digest
        ):
            result = {
                **manifest,
                "prepared_now": False,
                "resumed_existing_model": True,
                "model_path": str(destination),
            }
            summary_path = (
                exports
                / f"embedding-model-summary-{run_id}.json"
            )
            atomic_json(summary_path, result)
            result["summary_path"] = str(summary_path)
            return result
        if not replace:
            raise FileExistsError(
                "A conflicting local embedding model exists. "
                "Investigate before using --replace."
            )
        shutil.rmtree(destination)

    loader = model_loader or _default_model_loader
    with tempfile.TemporaryDirectory(
        prefix="alice-embedding-model-",
        dir=temporary_root,
    ) as temp:
        stage = Path(temp) / policy.model.local_directory_name
        stage.mkdir()
        model = loader(
            policy.model.model_id,
            revision=policy.model.revision,
            device=device,
            local_files_only=False,
            policy=policy,
        )
        model.save_pretrained(
            str(stage),
            safe_serialization=True,
        )
        unsafe = _unsafe_model_files(stage)
        if unsafe:
            raise RuntimeError(
                "Prepared model contains unsafe serialized files: "
                + ", ".join(unsafe)
            )

        dimension = int(model.get_sentence_embedding_dimension())
        if dimension != policy.model.embedding_dimension:
            raise RuntimeError(
                f"Unexpected embedding dimension {dimension}"
            )
        max_length = int(getattr(model, "max_seq_length", 0) or 0)
        if max_length and max_length != (
            policy.model.maximum_sequence_tokens
        ):
            raise RuntimeError(
                f"Unexpected model maximum sequence length {max_length}"
            )

        tree_digest = _tree_digest(stage)
        manifest = {
            "model_manifest_schema_version": 1,
            "run_id": run_id,
            "model_id": policy.model.model_id,
            "revision": policy.model.revision,
            "license": policy.model.license,
            "language": policy.model.language,
            "embedding_dimension": dimension,
            "maximum_sequence_tokens": (
                policy.model.maximum_sequence_tokens
            ),
            "query_prompt": policy.model.query_prompt,
            "document_prompt": policy.model.document_prompt,
            "trust_remote_code": False,
            "safe_serialization": True,
            "model_tree_digest": tree_digest,
            "prepared_at": utc_now(),
            "private_data_used_during_download": False,
        }
        atomic_json(
            stage / "alice-model-manifest.json",
            manifest,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(destination)

    result = {
        **manifest,
        "prepared_now": True,
        "resumed_existing_model": False,
        "model_path": str(destination),
    }
    summary_path = (
        exports / f"embedding-model-summary-{run_id}.json"
    )
    atomic_json(summary_path, result)
    result["summary_path"] = str(summary_path)
    return result


def _load_model_manifest(
    vault_root: Path,
    policy: SemanticPolicy,
) -> tuple[Path, dict[str, Any]]:
    path = model_root(vault_root, policy)
    manifest_path = path / "alice-model-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Local embedding model is missing. Run "
            "prepare_embedding_model.py first."
        )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("model_id") != policy.model.model_id:
        raise ValueError("Local model ID does not match policy")
    if manifest.get("revision") != policy.model.revision:
        raise ValueError("Local model revision does not match policy")
    if manifest.get("embedding_dimension") != (
        policy.model.embedding_dimension
    ):
        raise ValueError("Local model dimension does not match policy")
    if _unsafe_model_files(path):
        raise ValueError("Local model contains unsafe serialized files")
    actual_digest = _tree_digest(
        path,
        exclude_names={"alice-model-manifest.json"},
    )
    if actual_digest != manifest.get("model_tree_digest"):
        raise ValueError("Local model tree digest mismatch")
    return path, manifest


def _load_local_model(
    *,
    vault_root: Path,
    policy: SemanticPolicy,
    device: str = "auto",
    model_loader: ModelLoader | None = None,
):
    path, manifest = _load_model_manifest(vault_root, policy)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    loader = model_loader or _default_model_loader
    model = loader(
        str(path),
        revision=None,
        device=device,
        local_files_only=True,
        policy=policy,
    )
    return model, manifest


def semantic_index_id(
    *,
    chunk_set_id: str,
    chunk_manifest_fingerprint: str,
    semantic_policy_digest: str,
    model_tree_digest: str,
) -> str:
    material = (
        "alice-semantic-index-v1\0"
        f"{chunk_set_id}\0"
        f"{chunk_manifest_fingerprint}\0"
        f"{semantic_policy_digest}\0"
        f"{model_tree_digest}"
    )
    return sha256_bytes(material.encode("utf-8"))[:32]


def _write_float32_matrix(
    path: Path,
    rows: Sequence[Sequence[float]],
    dimension: int,
) -> None:
    values = array("f")
    for row in rows:
        if len(row) != dimension:
            raise ValueError("Embedding row has unexpected dimension")
        values.extend(float(value) for value in row)
    if sys.byteorder != "little":
        values.byteswap()
    with path.open("wb") as handle:
        values.tofile(handle)


def _read_float32_matrix(
    path: Path,
    *,
    rows: int,
    dimension: int,
) -> array:
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, rows * dimension)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != rows * dimension:
        raise ValueError("Embedding binary has unexpected size")
    return values


def _rows_from_encoded(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = [
        [float(item) for item in row]
        for row in value
    ]
    return rows


def _norm(row: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in row))


def _encode_documents(
    model: Any,
    texts: list[str],
    policy: SemanticPolicy,
) -> list[list[float]]:
    prefixed = [
        policy.model.document_prompt + text
        for text in texts
    ]
    encoded = model.encode(
        prefixed,
        batch_size=policy.build.batch_size,
        normalize_embeddings=policy.build.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=policy.build.show_progress,
    )
    return _rows_from_encoded(encoded)


def _encode_query(
    model: Any,
    query: str,
    policy: SemanticPolicy,
) -> list[float]:
    encoded = model.encode(
        [policy.model.query_prompt + query],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    rows = _rows_from_encoded(encoded)
    if len(rows) != 1:
        raise RuntimeError("Query encoder returned unexpected shape")
    return rows[0]


def _token_length(
    model: Any,
    text: str,
    prompt: str,
) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        output = tokenizer(
            prompt + text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        ids = output.get("input_ids")
        return len(ids) if ids is not None else None
    except Exception:
        return None


def _write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL line {line_number} is not an object"
                )
            output.append(value)
    return output


def build_semantic_index(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    device: str = "auto",
    replace: bool = False,
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(policy_path)
    model_path, model_manifest = _load_model_manifest(
        vault_root,
        policy,
    )
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
    )
    chunk_manifest, records = load_chunk_catalog(chunk_root)
    records = sorted(
        records,
        key=lambda record: str(record["chunk_id"]),
    )
    index_id = semantic_index_id(
        chunk_set_id=str(chunk_manifest["chunk_set_id"]),
        chunk_manifest_fingerprint=str(
            chunk_manifest["manifest_fingerprint"]
        ),
        semantic_policy_digest=policy.digest,
        model_tree_digest=str(
            model_manifest["model_tree_digest"]
        ),
    )

    output_root = (
        vault_root
        / "derived"
        / pilot_name
        / "semantic"
        / index_id
    )
    exports = vault_root / "manifests" / "exports"
    private_root = (
        vault_root / "manifests" / "semantic" / pilot_name
    )
    temporary_root = vault_root / "temporary"
    for path in (exports, private_root, temporary_root):
        path.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    if output_root.exists():
        existing_manifest_path = (
            output_root / "semantic-manifest.json"
        )
        if existing_manifest_path.is_file():
            existing = json.loads(
                existing_manifest_path.read_text(encoding="utf-8")
            )
            if (
                existing.get("index_id") == index_id
                and existing.get("policy_digest") == policy.digest
                and existing.get("model_tree_digest")
                == model_manifest["model_tree_digest"]
                and existing.get("chunk_manifest_fingerprint")
                == chunk_manifest["manifest_fingerprint"]
            ):
                result = {
                    **existing,
                    "successful_index_build": False,
                    "resumed_existing_index": True,
                    "output_root": str(output_root),
                    "manifest_path": str(existing_manifest_path),
                }
                summary_path = (
                    exports
                    / f"semantic-index-summary-{run_id}.json"
                )
                atomic_json(summary_path, result)
                result["summary_path"] = str(summary_path)
                return result
        if not replace:
            raise FileExistsError(
                "A conflicting semantic index exists. "
                "Investigate before using --replace."
            )
        shutil.rmtree(output_root)

    model, loaded_manifest = _load_local_model(
        vault_root=vault_root,
        policy=policy,
        device=device,
        model_loader=model_loader,
    )
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension != policy.model.embedding_dimension:
        raise RuntimeError("Loaded model dimension mismatch")

    texts: list[str] = []
    chunk_map: list[dict[str, Any]] = []
    token_truncated_count = 0
    token_length_unknown_count = 0
    for record in records:
        text_path = (
            chunk_root / "text" / f"{record['chunk_id']}.txt"
        )
        text = text_path.read_text(encoding="utf-8")
        if sha256_bytes(text.encode("utf-8")) != record[
            "chunk_text_sha256"
        ]:
            raise ValueError(
                f"Chunk text hash mismatch: {record['chunk_id']}"
            )
        length = _token_length(
            model,
            text,
            policy.model.document_prompt,
        )
        if length is None:
            token_length_unknown_count += 1
        elif length > policy.model.maximum_sequence_tokens:
            token_truncated_count += 1

        texts.append(text)
        chunk_map.append(
            {
                "row_index": len(chunk_map),
                "chunk_id": str(record["chunk_id"]),
                "source_content_sha256": str(
                    record["source_content_sha256"]
                ),
                "chunk_index": int(record["chunk_index"]),
                "family": str(record["family"]),
                "source_extraction_truncated": bool(
                    record["source_extraction_truncated"]
                ),
                "chunk_text_sha256": str(
                    record["chunk_text_sha256"]
                ),
                "char_count": int(record["char_count"]),
                "provenance": list(record.get("provenance", [])),
            }
        )

    embeddings = _encode_documents(model, texts, policy)
    if len(embeddings) != len(records):
        raise RuntimeError("Encoder returned unexpected row count")
    norm_failures = 0
    for row in embeddings:
        if len(row) != dimension:
            raise RuntimeError("Encoder returned unexpected dimension")
        if not 0.98 <= _norm(row) <= 1.02:
            norm_failures += 1
    if norm_failures:
        raise RuntimeError(
            f"{norm_failures} embeddings were not normalized"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"alice-semantic-{index_id}-",
        dir=temporary_root,
    ) as temp:
        stage = Path(temp) / index_id
        stage.mkdir()
        embeddings_path = stage / "embeddings.f32"
        chunk_map_path = stage / "chunk-map.jsonl"
        _write_float32_matrix(
            embeddings_path,
            embeddings,
            dimension,
        )
        _write_jsonl(chunk_map_path, chunk_map)

        manifest = {
            "semantic_index_schema_version": (
                SEMANTIC_INDEX_SCHEMA_VERSION
            ),
            "run_id": run_id,
            "index_id": index_id,
            "pilot_name": pilot_name,
            "created_at": utc_now(),
            "policy_id": policy.policy_id,
            "policy_digest": policy.digest,
            "chunk_set_id": str(chunk_manifest["chunk_set_id"]),
            "chunk_manifest_fingerprint": str(
                chunk_manifest["manifest_fingerprint"]
            ),
            "chunk_count": len(records),
            "embedding_dimension": dimension,
            "storage_dtype": "float32",
            "normalized_embeddings": True,
            "model_id": policy.model.model_id,
            "model_revision": policy.model.revision,
            "model_tree_digest": str(
                loaded_manifest["model_tree_digest"]
            ),
            "model_path": str(model_path),
            "maximum_sequence_tokens": (
                policy.model.maximum_sequence_tokens
            ),
            "token_truncated_chunk_count": (
                token_truncated_count
            ),
            "token_length_unknown_count": (
                token_length_unknown_count
            ),
            "embeddings_sha256": sha256_file(
                embeddings_path
            ),
            "chunk_map_sha256": sha256_file(chunk_map_path),
            "source_files_modified": False,
            "private_text_uploaded": False,
        }
        manifest["manifest_fingerprint"] = sha256_bytes(
            canonical_json(
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {
                        "run_id",
                        "created_at",
                        "model_path",
                        "manifest_fingerprint",
                    }
                }
            )
        )
        atomic_json(
            stage / "semantic-manifest.json",
            manifest,
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output_root)

    private_manifest_path = (
        private_root / f"semantic-manifest-{run_id}.json"
    )
    shutil.copy2(
        output_root / "semantic-manifest.json",
        private_manifest_path,
    )
    result = {
        **manifest,
        "successful_index_build": True,
        "resumed_existing_index": False,
        "output_root": str(output_root),
        "manifest_path": str(
            output_root / "semantic-manifest.json"
        ),
        "embeddings_path": str(
            output_root / "embeddings.f32"
        ),
        "chunk_map_path": str(
            output_root / "chunk-map.jsonl"
        ),
        "private_manifest_path": str(
            private_manifest_path
        ),
    }
    summary_path = (
        exports / f"semantic-index-summary-{run_id}.json"
    )
    atomic_json(summary_path, result)
    result["summary_path"] = str(summary_path)
    return result


def _semantic_paths(
    *,
    vault_root: Path,
    pilot_name: str,
    policy: SemanticPolicy,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    _, model_manifest = _load_model_manifest(vault_root, policy)
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
    )
    chunk_manifest, _ = load_chunk_catalog(chunk_root)
    index_id = semantic_index_id(
        chunk_set_id=str(chunk_manifest["chunk_set_id"]),
        chunk_manifest_fingerprint=str(
            chunk_manifest["manifest_fingerprint"]
        ),
        semantic_policy_digest=policy.digest,
        model_tree_digest=str(
            model_manifest["model_tree_digest"]
        ),
    )
    root = (
        vault_root
        / "derived"
        / pilot_name
        / "semantic"
        / index_id
    )
    manifest_path = root / "semantic-manifest.json"
    map_path = root / "chunk-map.jsonl"
    if not manifest_path.is_file() or not map_path.is_file():
        raise FileNotFoundError(
            "Semantic index is missing. Build it first."
        )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    chunk_map = _read_jsonl(map_path)
    return root, manifest, chunk_map


def verify_semantic_index(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(policy_path)
    root, manifest, chunk_map = _semantic_paths(
        vault_root=vault_root,
        pilot_name=pilot_name,
        policy=policy,
    )
    errors: list[str] = []
    embeddings_path = root / "embeddings.f32"
    map_path = root / "chunk-map.jsonl"
    dimension = int(manifest["embedding_dimension"])
    count = int(manifest["chunk_count"])

    if manifest.get("policy_digest") != policy.digest:
        errors.append("Semantic-policy digest mismatch")
    if manifest.get("model_id") != policy.model.model_id:
        errors.append("Model ID mismatch")
    if manifest.get("model_revision") != policy.model.revision:
        errors.append("Model revision mismatch")
    if len(chunk_map) != count:
        errors.append("Chunk-map row count mismatch")
    if manifest.get("embeddings_sha256") != sha256_file(
        embeddings_path
    ):
        errors.append("Embedding binary digest mismatch")
    if manifest.get("chunk_map_sha256") != sha256_file(
        map_path
    ):
        errors.append("Chunk-map digest mismatch")
    expected_bytes = count * dimension * 4
    if embeddings_path.stat().st_size != expected_bytes:
        errors.append("Embedding binary size mismatch")

    values = _read_float32_matrix(
        embeddings_path,
        rows=count,
        dimension=dimension,
    )
    norm_failures = 0
    for row_index in range(count):
        offset = row_index * dimension
        norm = math.sqrt(
            sum(
                values[offset + column] ** 2
                for column in range(dimension)
            )
        )
        if not 0.98 <= norm <= 1.02:
            norm_failures += 1
    if norm_failures:
        errors.append(
            f"{norm_failures} embedding rows failed normalization"
        )

    chunk_ids = [
        str(record.get("chunk_id", ""))
        for record in chunk_map
    ]
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("Duplicate chunk IDs in semantic map")
    if [
        int(record.get("row_index", -1))
        for record in chunk_map
    ] != list(range(count)):
        errors.append("Semantic row indexes are not contiguous")

    return {
        "verification_schema_version": 1,
        "pilot_name": pilot_name,
        "index_id": manifest["index_id"],
        "expected_embeddings": count,
        "verified_embeddings": (
            count if not norm_failures else count - norm_failures
        ),
        "embedding_dimension": dimension,
        "token_truncated_chunk_count": int(
            manifest["token_truncated_chunk_count"]
        ),
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "error_count": len(errors),
        "errors": errors,
        "private_text_uploaded": False,
        "ready_for_semantic_search": not errors,
    }


def _matches_filters(
    record: dict[str, Any],
    filters: SearchFilters,
) -> bool:
    if filters.families and record["family"] not in (
        filters.families
    ):
        return False
    if (
        not filters.include_truncated
        and record["source_extraction_truncated"]
    ):
        return False

    provenance = list(record.get("provenance", []))
    if filters.years and not any(
        str(item.get("year_hint", "")) in filters.years
        for item in provenance
    ):
        return False
    if filters.source_buckets and not any(
        str(item.get("source_bucket", ""))
        in filters.source_buckets
        for item in provenance
    ):
        return False
    if filters.contradiction_labels and not any(
        str(item.get("known_contradiction_group", ""))
        in filters.contradiction_labels
        for item in provenance
    ):
        return False
    return True


def _dot_scores(
    values: array,
    *,
    rows: int,
    dimension: int,
    query: Sequence[float],
) -> list[float]:
    if len(query) != dimension:
        raise ValueError("Query embedding dimension mismatch")
    try:
        import numpy as np
    except ImportError:
        output = []
        for row_index in range(rows):
            offset = row_index * dimension
            output.append(
                sum(
                    values[offset + column] * query[column]
                    for column in range(dimension)
                )
            )
        return output

    matrix = np.frombuffer(
        values,
        dtype=np.float32,
    ).reshape(rows, dimension)
    vector = np.asarray(query, dtype=np.float32)
    return (matrix @ vector).astype(float).tolist()


def _snippet(text: str, length: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= length:
        return compact
    return compact[: max(0, length - 2)].rstrip() + " …"


def semantic_search(
    *,
    vault_root: Path,
    query: str,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    filters: SearchFilters | None = None,
    limit: int | None = None,
    candidate_k: int | None = None,
    device: str = "auto",
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query may not be empty")
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(policy_path)
    filters = filters or SearchFilters()
    limit = limit or policy.search.default_limit
    candidate_k = (
        candidate_k or policy.search.semantic_candidate_k
    )
    root, manifest, chunk_map = _semantic_paths(
        vault_root=vault_root,
        pilot_name=pilot_name,
        policy=policy,
    )
    model, _ = _load_local_model(
        vault_root=vault_root,
        policy=policy,
        device=device,
        model_loader=model_loader,
    )
    query_vector = _encode_query(model, query, policy)
    if not 0.98 <= _norm(query_vector) <= 1.02:
        raise RuntimeError("Query embedding was not normalized")

    count = int(manifest["chunk_count"])
    dimension = int(manifest["embedding_dimension"])
    values = _read_float32_matrix(
        root / "embeddings.f32",
        rows=count,
        dimension=dimension,
    )
    scores = _dot_scores(
        values,
        rows=count,
        dimension=dimension,
        query=query_vector,
    )
    candidates = sorted(
        (
            (score, row_index, record)
            for row_index, (score, record) in enumerate(
                zip(scores, chunk_map)
            )
            if _matches_filters(record, filters)
        ),
        key=lambda item: (-item[0], item[2]["chunk_id"]),
    )[:candidate_k]

    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
    )
    output: list[dict[str, Any]] = []
    per_source: Counter[str] = Counter()
    kept_indices: dict[str, list[int]] = {}
    for score, _, record in candidates:
        source = str(record["source_content_sha256"])
        if per_source[source] >= (
            policy.search.maximum_chunks_per_source
        ):
            continue
        indices = kept_indices.setdefault(source, [])
        chunk_index = int(record["chunk_index"])
        if any(abs(chunk_index - previous) <= 1 for previous in indices):
            continue

        text_path = (
            chunk_root / "text" / f"{record['chunk_id']}.txt"
        )
        text = text_path.read_text(encoding="utf-8")
        output.append(
            {
                "rank": len(output) + 1,
                "chunk_id": record["chunk_id"],
                "source_content_sha256": source,
                "chunk_index": chunk_index,
                "family": record["family"],
                "source_extraction_truncated": bool(
                    record["source_extraction_truncated"]
                ),
                "cosine_similarity": round(float(score), 8),
                "snippet": _snippet(
                    text,
                    policy.search.snippet_characters,
                ),
                "provenance": record["provenance"],
            }
        )
        per_source[source] += 1
        indices.append(chunk_index)
        if len(output) >= limit:
            break

    return {
        "semantic_result_schema_version": 1,
        "pilot_name": pilot_name,
        "query": query,
        "model_id": policy.model.model_id,
        "model_revision": policy.model.revision,
        "filters": {
            "families": list(filters.families),
            "years": list(filters.years),
            "source_buckets": list(filters.source_buckets),
            "contradiction_labels": list(
                filters.contradiction_labels
            ),
            "include_truncated": filters.include_truncated,
        },
        "result_count": len(output),
        "results": output,
        "private_text_uploaded": False,
    }


def hybrid_search(
    *,
    vault_root: Path,
    query: str,
    pilot_name: str = "pilot-v1",
    semantic_policy_path: Path | None = None,
    lexical_policy_path: Path | None = None,
    filters: SearchFilters | None = None,
    limit: int | None = None,
    device: str = "auto",
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(semantic_policy_path)
    filters = filters or SearchFilters()
    limit = limit or policy.search.default_limit

    try:
        lexical = search_index(
            vault_root=vault_root,
            pilot_name=pilot_name,
            policy_path=lexical_policy_path,
            query=query,
            filters=filters,
            limit=policy.search.lexical_candidate_k,
            max_chunks_per_source=1,
        )
    except ValueError:
        lexical = {"result_count": 0, "results": []}
    semantic = semantic_search(
        vault_root=vault_root,
        pilot_name=pilot_name,
        policy_path=semantic_policy_path,
        query=query,
        filters=filters,
        limit=policy.search.semantic_candidate_k,
        candidate_k=max(
            policy.search.semantic_candidate_k * 4,
            100,
        ),
        device=device,
        model_loader=model_loader,
    )

    fused: dict[str, dict[str, Any]] = {}
    for method, result, weight in (
        ("lexical", lexical, policy.search.lexical_weight),
        ("semantic", semantic, policy.search.semantic_weight),
    ):
        for item in result["results"]:
            source = str(item["source_content_sha256"])
            entry = fused.setdefault(
                source,
                {
                    "source_content_sha256": source,
                    "rrf_score": 0.0,
                    "lexical_rank": None,
                    "semantic_rank": None,
                    "lexical_result": None,
                    "semantic_result": None,
                },
            )
            rank = int(item["rank"])
            entry["rrf_score"] += weight / (
                policy.search.rrf_k + rank
            )
            entry[f"{method}_rank"] = rank
            entry[f"{method}_result"] = item

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            item["source_content_sha256"],
        ),
    )[:limit]
    output = []
    for index, entry in enumerate(ordered, start=1):
        representative = (
            entry["semantic_result"]
            or entry["lexical_result"]
        )
        output.append(
            {
                "rank": index,
                "source_content_sha256": (
                    entry["source_content_sha256"]
                ),
                "rrf_score": round(
                    float(entry["rrf_score"]),
                    10,
                ),
                "lexical_rank": entry["lexical_rank"],
                "semantic_rank": entry["semantic_rank"],
                "chunk_id": representative["chunk_id"],
                "chunk_index": representative["chunk_index"],
                "family": representative["family"],
                "source_extraction_truncated": (
                    representative[
                        "source_extraction_truncated"
                    ]
                ),
                "snippet": representative["snippet"],
                "provenance": representative["provenance"],
                "lexical_score": (
                    entry["lexical_result"].get("score")
                    if entry["lexical_result"]
                    else None
                ),
                "semantic_cosine_similarity": (
                    entry["semantic_result"].get(
                        "cosine_similarity"
                    )
                    if entry["semantic_result"]
                    else None
                ),
            }
        )

    return {
        "hybrid_result_schema_version": 1,
        "pilot_name": pilot_name,
        "query": query,
        "fusion": "reciprocal_rank_fusion",
        "rrf_k": policy.search.rrf_k,
        "lexical_candidate_count": lexical["result_count"],
        "semantic_candidate_count": semantic["result_count"],
        "result_count": len(output),
        "results": output,
        "private_text_uploaded": False,
    }

# ---------------------------------------------------------------------------
# P1.9a: token-aware semantic segmentation
#
# P1.7 chunks are optimized for provenance and lexical retrieval. They may be
# longer than a transformer's token window. The definitions below supersede
# the earlier semantic-index functions and create deterministic token-bounded
# embedding segments without changing the P1.7 chunk catalog.
# ---------------------------------------------------------------------------

import re as _semantic_re


def _semantic_segmenting_settings(
    policy: SemanticPolicy,
) -> dict[str, int | str]:
    data = json.loads(policy.source_path.read_text(encoding="utf-8"))
    value = dict(data.get("semantic_segmenting", {}))
    settings: dict[str, int | str] = {
        "strategy": str(value.get("strategy", "token_window_v1")),
        "maximum_total_tokens": int(
            value.get("maximum_total_tokens", 480)
        ),
        "overlap_tokens": int(value.get("overlap_tokens", 64)),
        "minimum_content_tokens": int(
            value.get("minimum_content_tokens", 48)
        ),
    }
    if settings["strategy"] != "token_window_v1":
        raise ValueError("Unsupported semantic segmentation strategy")
    maximum = int(settings["maximum_total_tokens"])
    overlap = int(settings["overlap_tokens"])
    minimum = int(settings["minimum_content_tokens"])
    if maximum < 64 or maximum > policy.model.maximum_sequence_tokens:
        raise ValueError("Semantic maximum token count is invalid")
    if overlap < 0 or overlap >= maximum:
        raise ValueError("Semantic token overlap is invalid")
    if minimum < 1 or minimum >= maximum:
        raise ValueError("Semantic minimum token count is invalid")
    return settings


def _token_ids_and_offsets(
    model: Any,
    text: str,
) -> tuple[list[Any], list[tuple[int, int]]]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "The embedding model tokenizer is required for token-aware "
            "semantic segmentation"
        )

    try:
        output = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        ids = list(output.get("input_ids") or [])
        raw_offsets = output.get("offset_mapping")
        if raw_offsets is not None:
            offsets = [
                (int(item[0]), int(item[1]))
                for item in raw_offsets
                if int(item[1]) > int(item[0])
            ]
            if len(offsets) == len(ids) and offsets:
                return ids, offsets
    except Exception:
        pass

    # Slow/custom tokenizers do not always expose offsets. This deterministic
    # whitespace fallback is used only for segmentation boundaries; the final
    # model-token count is still checked with the actual tokenizer.
    spans = [
        (match.start(), match.end())
        for match in _semantic_re.finditer(r"\S+", text)
    ]
    if not spans:
        return [], []
    return list(range(len(spans))), spans


def _token_overhead(
    model: Any,
    prompt: str,
) -> int:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required")
    output = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    ids = output.get("input_ids")
    if ids is None:
        raise RuntimeError("Tokenizer returned no input IDs")
    return len(ids)


def _semantic_segment_id(
    *,
    parent_chunk_id: str,
    policy_digest: str,
    segment_index: int,
    start_char: int,
    end_char: int,
    text_sha256: str,
) -> str:
    material = (
        "alice-semantic-segment-v1\0"
        f"{parent_chunk_id}\0"
        f"{policy_digest}\0"
        f"{segment_index}\0"
        f"{start_char}\0"
        f"{end_char}\0"
        f"{text_sha256}"
    )
    return sha256_bytes(material.encode("utf-8"))


def _trim_segment_span(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _segment_chunk_for_embeddings(
    *,
    model: Any,
    text: str,
    parent_chunk_id: str,
    policy: SemanticPolicy,
) -> list[dict[str, Any]]:
    settings = _semantic_segmenting_settings(policy)
    maximum_total = int(settings["maximum_total_tokens"])
    overlap = int(settings["overlap_tokens"])
    minimum = int(settings["minimum_content_tokens"])
    prompt = policy.model.document_prompt
    overhead = _token_overhead(model, prompt)
    content_budget = maximum_total - overhead
    if content_budget < minimum:
        raise RuntimeError(
            "Semantic token budget is too small after prompt/special tokens"
        )

    token_ids, offsets = _token_ids_and_offsets(model, text)
    if not offsets:
        raise ValueError("Cannot create an embedding segment from empty text")

    segments: list[dict[str, Any]] = []
    token_start = 0
    while token_start < len(offsets):
        token_end = min(len(offsets), token_start + content_budget)

        # Verify with the actual tokenizer. Token boundary arithmetic can vary
        # slightly around prefixes and special tokens, so shrink deterministically
        # until the complete prompted segment fits.
        while True:
            start_char = offsets[token_start][0]
            end_char = offsets[token_end - 1][1]
            start_char, end_char = _trim_segment_span(
                text,
                start_char,
                end_char,
            )
            segment_text = text[start_char:end_char]
            actual_tokens = _token_length(model, segment_text, prompt)
            if actual_tokens is None:
                raise RuntimeError(
                    "Tokenizer could not count semantic segment tokens"
                )
            if actual_tokens <= maximum_total:
                break
            if token_end - token_start <= minimum:
                raise RuntimeError(
                    "A semantic segment cannot be reduced below the model "
                    "token limit"
                )
            token_end -= max(1, (token_end - token_start) // 8)

        text_hash = sha256_bytes(segment_text.encode("utf-8"))
        segment_index = len(segments)
        segments.append(
            {
                "semantic_segment_id": _semantic_segment_id(
                    parent_chunk_id=parent_chunk_id,
                    policy_digest=policy.digest,
                    segment_index=segment_index,
                    start_char=start_char,
                    end_char=end_char,
                    text_sha256=text_hash,
                ),
                "segment_index": segment_index,
                "segment_start_char": start_char,
                "segment_end_char": end_char,
                "segment_char_count": len(segment_text),
                "segment_text_sha256": text_hash,
                "segment_token_count": int(actual_tokens),
                "text": segment_text,
            }
        )

        if token_end >= len(offsets):
            break
        next_start = max(token_start + 1, token_end - overlap)
        if next_start <= token_start:
            raise RuntimeError("Semantic segmentation failed to make progress")
        token_start = next_start

    return segments


def build_semantic_index(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    device: str = "auto",
    replace: bool = False,
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(policy_path)
    settings = _semantic_segmenting_settings(policy)
    model_path, model_manifest = _load_model_manifest(
        vault_root,
        policy,
    )
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
    )
    chunk_manifest, records = load_chunk_catalog(chunk_root)
    records = sorted(records, key=lambda record: str(record["chunk_id"]))
    index_id = semantic_index_id(
        chunk_set_id=str(chunk_manifest["chunk_set_id"]),
        chunk_manifest_fingerprint=str(
            chunk_manifest["manifest_fingerprint"]
        ),
        semantic_policy_digest=policy.digest,
        model_tree_digest=str(model_manifest["model_tree_digest"]),
    )

    output_root = (
        vault_root / "derived" / pilot_name / "semantic" / index_id
    )
    exports = vault_root / "manifests" / "exports"
    private_root = vault_root / "manifests" / "semantic" / pilot_name
    temporary_root = vault_root / "temporary"
    for path in (exports, private_root, temporary_root):
        path.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    if output_root.exists():
        existing_manifest_path = output_root / "semantic-manifest.json"
        if existing_manifest_path.is_file():
            existing = json.loads(
                existing_manifest_path.read_text(encoding="utf-8")
            )
            if (
                existing.get("index_id") == index_id
                and existing.get("policy_digest") == policy.digest
                and existing.get("model_tree_digest")
                == model_manifest["model_tree_digest"]
                and existing.get("chunk_manifest_fingerprint")
                == chunk_manifest["manifest_fingerprint"]
                and int(existing.get("token_truncated_segment_count", -1))
                == 0
            ):
                result = {
                    **existing,
                    "successful_index_build": False,
                    "resumed_existing_index": True,
                    "output_root": str(output_root),
                    "manifest_path": str(existing_manifest_path),
                }
                summary_path = (
                    exports / f"semantic-index-summary-{run_id}.json"
                )
                atomic_json(summary_path, result)
                result["summary_path"] = str(summary_path)
                return result
        if not replace:
            raise FileExistsError(
                "A conflicting semantic index exists. Investigate before "
                "using --replace."
            )
        shutil.rmtree(output_root)

    model, loaded_manifest = _load_local_model(
        vault_root=vault_root,
        policy=policy,
        device=device,
        model_loader=model_loader,
    )
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension != policy.model.embedding_dimension:
        raise RuntimeError("Loaded model dimension mismatch")

    segment_texts: list[str] = []
    segment_map: list[dict[str, Any]] = []
    chunks_split_for_embedding = 0
    maximum_segments_per_chunk = 0
    maximum_segment_tokens = 0

    for record in records:
        text_path = chunk_root / "text" / f"{record['chunk_id']}.txt"
        text = text_path.read_text(encoding="utf-8")
        if sha256_bytes(text.encode("utf-8")) != record[
            "chunk_text_sha256"
        ]:
            raise ValueError(
                f"Chunk text hash mismatch: {record['chunk_id']}"
            )
        segments = _segment_chunk_for_embeddings(
            model=model,
            text=text,
            parent_chunk_id=str(record["chunk_id"]),
            policy=policy,
        )
        if len(segments) > 1:
            chunks_split_for_embedding += 1
        maximum_segments_per_chunk = max(
            maximum_segments_per_chunk,
            len(segments),
        )

        for segment in segments:
            maximum_segment_tokens = max(
                maximum_segment_tokens,
                int(segment["segment_token_count"]),
            )
            segment_texts.append(str(segment.pop("text")))
            segment_map.append(
                {
                    "row_index": len(segment_map),
                    **segment,
                    "chunk_id": str(record["chunk_id"]),
                    "source_content_sha256": str(
                        record["source_content_sha256"]
                    ),
                    "chunk_index": int(record["chunk_index"]),
                    "family": str(record["family"]),
                    "source_extraction_truncated": bool(
                        record["source_extraction_truncated"]
                    ),
                    "chunk_text_sha256": str(
                        record["chunk_text_sha256"]
                    ),
                    "chunk_char_count": int(record["char_count"]),
                    "provenance": list(record.get("provenance", [])),
                }
            )

    embeddings = _encode_documents(model, segment_texts, policy)
    if len(embeddings) != len(segment_map):
        raise RuntimeError("Encoder returned unexpected row count")
    norm_failures = 0
    for row in embeddings:
        if len(row) != dimension:
            raise RuntimeError("Encoder returned unexpected dimension")
        if not 0.98 <= _norm(row) <= 1.02:
            norm_failures += 1
    if norm_failures:
        raise RuntimeError(
            f"{norm_failures} embeddings were not normalized"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"alice-semantic-{index_id}-",
        dir=temporary_root,
    ) as temp:
        stage = Path(temp) / index_id
        stage.mkdir()
        embeddings_path = stage / "embeddings.f32"
        map_path = stage / "segment-map.jsonl"
        _write_float32_matrix(
            embeddings_path,
            embeddings,
            dimension,
        )
        _write_jsonl(map_path, segment_map)

        manifest = {
            "semantic_index_schema_version": 2,
            "run_id": run_id,
            "index_id": index_id,
            "pilot_name": pilot_name,
            "created_at": utc_now(),
            "policy_id": policy.policy_id,
            "policy_digest": policy.digest,
            "chunk_set_id": str(chunk_manifest["chunk_set_id"]),
            "chunk_manifest_fingerprint": str(
                chunk_manifest["manifest_fingerprint"]
            ),
            "source_chunk_count": len(records),
            "embedding_count": len(segment_map),
            "chunk_count": len(segment_map),
            "embedding_dimension": dimension,
            "storage_dtype": "float32",
            "normalized_embeddings": True,
            "model_id": policy.model.model_id,
            "model_revision": policy.model.revision,
            "model_tree_digest": str(
                loaded_manifest["model_tree_digest"]
            ),
            "model_path": str(model_path),
            "maximum_sequence_tokens": (
                policy.model.maximum_sequence_tokens
            ),
            "semantic_segmenting_strategy": settings["strategy"],
            "semantic_maximum_total_tokens": int(
                settings["maximum_total_tokens"]
            ),
            "semantic_overlap_tokens": int(
                settings["overlap_tokens"]
            ),
            "chunks_split_for_embedding": chunks_split_for_embedding,
            "maximum_segments_per_chunk": maximum_segments_per_chunk,
            "maximum_segment_tokens": maximum_segment_tokens,
            "token_truncated_segment_count": 0,
            "embeddings_sha256": sha256_file(embeddings_path),
            "segment_map_sha256": sha256_file(map_path),
            # Compatibility alias for tools that previously expected this key.
            "chunk_map_sha256": sha256_file(map_path),
            "source_files_modified": False,
            "private_text_uploaded": False,
        }
        manifest["manifest_fingerprint"] = sha256_bytes(
            canonical_json(
                {
                    key: value
                    for key, value in manifest.items()
                    if key
                    not in {
                        "run_id",
                        "created_at",
                        "model_path",
                        "manifest_fingerprint",
                    }
                }
            )
        )
        atomic_json(stage / "semantic-manifest.json", manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output_root)

    private_manifest_path = (
        private_root / f"semantic-manifest-{run_id}.json"
    )
    shutil.copy2(
        output_root / "semantic-manifest.json",
        private_manifest_path,
    )
    result = {
        **manifest,
        "successful_index_build": True,
        "resumed_existing_index": False,
        "output_root": str(output_root),
        "manifest_path": str(output_root / "semantic-manifest.json"),
        "embeddings_path": str(output_root / "embeddings.f32"),
        "segment_map_path": str(output_root / "segment-map.jsonl"),
        "private_manifest_path": str(private_manifest_path),
    }
    summary_path = exports / f"semantic-index-summary-{run_id}.json"
    atomic_json(summary_path, result)
    result["summary_path"] = str(summary_path)
    return result


def _semantic_paths(
    *,
    vault_root: Path,
    pilot_name: str,
    policy: SemanticPolicy,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    _, model_manifest = _load_model_manifest(vault_root, policy)
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
    )
    chunk_manifest, _ = load_chunk_catalog(chunk_root)
    index_id = semantic_index_id(
        chunk_set_id=str(chunk_manifest["chunk_set_id"]),
        chunk_manifest_fingerprint=str(
            chunk_manifest["manifest_fingerprint"]
        ),
        semantic_policy_digest=policy.digest,
        model_tree_digest=str(model_manifest["model_tree_digest"]),
    )
    root = vault_root / "derived" / pilot_name / "semantic" / index_id
    manifest_path = root / "semantic-manifest.json"
    map_path = root / "segment-map.jsonl"
    if not manifest_path.is_file() or not map_path.is_file():
        raise FileNotFoundError(
            "Token-aware semantic index is missing. Build it first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segment_map = _read_jsonl(map_path)
    return root, manifest, segment_map


def verify_semantic_index(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(policy_path)
    root, manifest, segment_map = _semantic_paths(
        vault_root=vault_root,
        pilot_name=pilot_name,
        policy=policy,
    )
    errors: list[str] = []
    embeddings_path = root / "embeddings.f32"
    map_path = root / "segment-map.jsonl"
    dimension = int(manifest["embedding_dimension"])
    count = int(manifest["embedding_count"])
    maximum_total = int(
        _semantic_segmenting_settings(policy)["maximum_total_tokens"]
    )

    if manifest.get("semantic_index_schema_version") != 2:
        errors.append("Semantic index is not token-aware schema v2")
    if manifest.get("policy_digest") != policy.digest:
        errors.append("Semantic-policy digest mismatch")
    if manifest.get("model_id") != policy.model.model_id:
        errors.append("Model ID mismatch")
    if manifest.get("model_revision") != policy.model.revision:
        errors.append("Model revision mismatch")
    if len(segment_map) != count:
        errors.append("Segment-map row count mismatch")
    if manifest.get("embeddings_sha256") != sha256_file(
        embeddings_path
    ):
        errors.append("Embedding binary digest mismatch")
    if manifest.get("segment_map_sha256") != sha256_file(map_path):
        errors.append("Segment-map digest mismatch")
    expected_bytes = count * dimension * 4
    if embeddings_path.stat().st_size != expected_bytes:
        errors.append("Embedding binary size mismatch")

    values = _read_float32_matrix(
        embeddings_path,
        rows=count,
        dimension=dimension,
    )
    norm_failures = 0
    for row_index in range(count):
        offset = row_index * dimension
        norm = math.sqrt(
            sum(
                values[offset + column] ** 2
                for column in range(dimension)
            )
        )
        if not 0.98 <= norm <= 1.02:
            norm_failures += 1
    if norm_failures:
        errors.append(
            f"{norm_failures} embedding rows failed normalization"
        )

    segment_ids = [
        str(record.get("semantic_segment_id", ""))
        for record in segment_map
    ]
    if len(segment_ids) != len(set(segment_ids)):
        errors.append("Duplicate semantic segment IDs")
    if [
        int(record.get("row_index", -1))
        for record in segment_map
    ] != list(range(count)):
        errors.append("Semantic row indexes are not contiguous")

    over_limit = sum(
        int(record.get("segment_token_count", maximum_total + 1))
        > maximum_total
        for record in segment_map
    )
    if over_limit:
        errors.append(
            f"{over_limit} semantic segments exceed the token limit"
        )
    if int(manifest.get("token_truncated_segment_count", -1)) != 0:
        errors.append("Manifest reports truncated semantic segments")

    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
    )
    slice_failures = 0
    for record in segment_map:
        path = chunk_root / "text" / f"{record['chunk_id']}.txt"
        text = path.read_text(encoding="utf-8")
        segment_text = text[
            int(record["segment_start_char"]):
            int(record["segment_end_char"])
        ]
        if sha256_bytes(segment_text.encode("utf-8")) != record[
            "segment_text_sha256"
        ]:
            slice_failures += 1
    if slice_failures:
        errors.append(
            f"{slice_failures} semantic segment source slices failed"
        )

    return {
        "verification_schema_version": 2,
        "pilot_name": pilot_name,
        "index_id": manifest["index_id"],
        "source_chunk_count": int(manifest["source_chunk_count"]),
        "expected_embeddings": count,
        "verified_embeddings": (
            count if not norm_failures else count - norm_failures
        ),
        "embedding_dimension": dimension,
        "chunks_split_for_embedding": int(
            manifest["chunks_split_for_embedding"]
        ),
        "maximum_segments_per_chunk": int(
            manifest["maximum_segments_per_chunk"]
        ),
        "maximum_segment_tokens": int(
            manifest["maximum_segment_tokens"]
        ),
        "token_truncated_segment_count": int(
            manifest["token_truncated_segment_count"]
        ),
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "error_count": len(errors),
        "errors": errors,
        "private_text_uploaded": False,
        "ready_for_semantic_search": not errors,
    }


def semantic_search(
    *,
    vault_root: Path,
    query: str,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    filters: SearchFilters | None = None,
    limit: int | None = None,
    candidate_k: int | None = None,
    device: str = "auto",
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query may not be empty")
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_semantic_policy(policy_path)
    filters = filters or SearchFilters()
    limit = limit or policy.search.default_limit
    candidate_k = candidate_k or policy.search.semantic_candidate_k
    root, manifest, segment_map = _semantic_paths(
        vault_root=vault_root,
        pilot_name=pilot_name,
        policy=policy,
    )
    model, _ = _load_local_model(
        vault_root=vault_root,
        policy=policy,
        device=device,
        model_loader=model_loader,
    )
    query_vector = _encode_query(model, query, policy)
    if not 0.98 <= _norm(query_vector) <= 1.02:
        raise RuntimeError("Query embedding was not normalized")

    count = int(manifest["embedding_count"])
    dimension = int(manifest["embedding_dimension"])
    values = _read_float32_matrix(
        root / "embeddings.f32",
        rows=count,
        dimension=dimension,
    )
    scores = _dot_scores(
        values,
        rows=count,
        dimension=dimension,
        query=query_vector,
    )
    candidates = sorted(
        (
            (score, row_index, record)
            for row_index, (score, record) in enumerate(
                zip(scores, segment_map)
            )
            if _matches_filters(record, filters)
        ),
        key=lambda item: (
            -item[0],
            item[2]["semantic_segment_id"],
        ),
    )[:candidate_k]

    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
    )
    output: list[dict[str, Any]] = []
    per_source: Counter[str] = Counter()
    kept_chunks: dict[str, set[str]] = {}
    for score, _, record in candidates:
        source = str(record["source_content_sha256"])
        if per_source[source] >= policy.search.maximum_chunks_per_source:
            continue
        parent_chunks = kept_chunks.setdefault(source, set())
        parent_chunk_id = str(record["chunk_id"])
        if parent_chunk_id in parent_chunks:
            continue

        text_path = chunk_root / "text" / f"{parent_chunk_id}.txt"
        parent_text = text_path.read_text(encoding="utf-8")
        segment_text = parent_text[
            int(record["segment_start_char"]):
            int(record["segment_end_char"])
        ]
        output.append(
            {
                "rank": len(output) + 1,
                "semantic_segment_id": record[
                    "semantic_segment_id"
                ],
                "chunk_id": parent_chunk_id,
                "source_content_sha256": source,
                "chunk_index": int(record["chunk_index"]),
                "segment_index": int(record["segment_index"]),
                "segment_start_char": int(
                    record["segment_start_char"]
                ),
                "segment_end_char": int(record["segment_end_char"]),
                "segment_token_count": int(
                    record["segment_token_count"]
                ),
                "family": record["family"],
                "source_extraction_truncated": bool(
                    record["source_extraction_truncated"]
                ),
                "cosine_similarity": round(float(score), 8),
                "snippet": _snippet(
                    segment_text,
                    policy.search.snippet_characters,
                ),
                "provenance": record["provenance"],
            }
        )
        per_source[source] += 1
        parent_chunks.add(parent_chunk_id)
        if len(output) >= limit:
            break

    return {
        "semantic_result_schema_version": 2,
        "pilot_name": pilot_name,
        "query": query,
        "model_id": policy.model.model_id,
        "model_revision": policy.model.revision,
        "filters": {
            "families": list(filters.families),
            "years": list(filters.years),
            "source_buckets": list(filters.source_buckets),
            "contradiction_labels": list(filters.contradiction_labels),
            "include_truncated": filters.include_truncated,
        },
        "result_count": len(output),
        "results": output,
        "private_text_uploaded": False,
    }
