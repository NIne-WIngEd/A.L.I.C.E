"""Read-only bridge from Phase 2 memory to frozen Phase 1 evidence catalogs."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class Phase1BridgeError(RuntimeError):
    """Base error for the read-only Phase 1 evidence bridge."""


class Phase1CatalogNotFoundError(Phase1BridgeError):
    """Raised when a requested Phase 1 catalog does not exist."""


class Phase1CatalogSchemaError(Phase1BridgeError):
    """Raised when a Phase 1 catalog does not match the expected contract."""


class Phase1CatalogIntegrityError(Phase1BridgeError):
    """Raised when SQLite reports integrity problems in a Phase 1 catalog."""


class Phase1EvidenceNotFoundError(Phase1BridgeError):
    """Raised when requested Phase 1 evidence cannot be found."""


class Phase1EvidenceMismatchError(Phase1BridgeError):
    """Raised when expected evidence identity does not match the catalog."""


@dataclass(frozen=True)
class Phase1ProvenancePath:
    """Private-safe provenance identifiers for one Phase 1 chunk path."""

    file_id: str
    role: str | None
    family: str | None
    source_bucket: str | None
    year_hint: str | None
    duplicate_control_group: str | None
    known_contradiction_group: str | None


@dataclass(frozen=True)
class Phase1ChunkEvidence:
    """Verified Phase 1 chunk identity without source plaintext."""

    pilot_name: str
    chunk_set_id: str
    chunk_id: str
    source_content_sha256: str
    source_text_sha256: str
    normalized_source_text_sha256: str
    chunk_text_sha256: str
    chunk_index: int
    start_char: int
    end_char: int
    char_count: int
    family: str | None
    parser_id: str | None
    source_extraction_truncated: bool
    provenance_paths: tuple[Phase1ProvenancePath, ...]

    @property
    def source_ref(self) -> str:
        """Return the canonical Phase 1 chunk reference."""
        return (
            f"phase1:pilot:{self.pilot_name}:"
            f"chunk-set:{self.chunk_set_id}:chunk:{self.chunk_id}"
        )


_REQUIRED_COLUMNS = {
    "chunk_sets": {
        "chunk_set_id",
        "pilot_name",
        "policy_id",
        "policy_digest",
        "pilot_manifest_sha256",
        "extraction_registry_digest",
        "chunk_count",
        "source_count",
        "created_at",
    },
    "chunks": {
        "chunk_id",
        "chunk_set_id",
        "source_content_sha256",
        "source_text_sha256",
        "normalized_source_text_sha256",
        "chunk_index",
        "start_char",
        "end_char",
        "char_count",
        "chunk_text_sha256",
        "family",
        "parser_id",
        "source_extraction_truncated",
        "provenance_path_count",
        "text_relative_path",
    },
    "chunk_provenance": {
        "chunk_id",
        "file_id",
        "original_relative_path",
        "filename",
        "role",
        "family",
        "source_bucket",
        "year_hint",
        "duplicate_control_group",
        "known_contradiction_group",
    },
}


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64:
        raise Phase1CatalogSchemaError(
            f"{field_name} must be a 64-character SHA-256 digest."
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise Phase1CatalogSchemaError(
            f"{field_name} must contain hexadecimal SHA-256 characters."
        ) from exc


def _readonly_sqlite_uri(path: Path) -> str:
    return f"{path.resolve(strict=True).as_uri()}?mode=ro"


@contextmanager
def open_phase1_catalog(
    catalog_path: Path,
) -> Iterator[sqlite3.Connection]:
    """Open a Phase 1 SQLite catalog in SQLite-enforced read-only mode."""
    try:
        resolved = catalog_path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise Phase1CatalogNotFoundError(
            f"Phase 1 catalog does not exist: {catalog_path}"
        ) from exc

    if not resolved.is_file():
        raise Phase1CatalogNotFoundError(
            f"Phase 1 catalog is not a file: {resolved}"
        )

    connection = sqlite3.connect(
        _readonly_sqlite_uri(resolved),
        uri=True,
    )
    connection.row_factory = sqlite3.Row

    try:
        verify_phase1_catalog_schema(connection)
        verify_phase1_catalog_integrity(connection)
        yield connection
    finally:
        connection.close()


def verify_phase1_catalog_schema(
    connection: sqlite3.Connection,
) -> None:
    """Validate the frozen Phase 1 chunk-catalog table contract."""
    table_rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    available_tables = {str(row[0]) for row in table_rows}

    for table_name, required_columns in _REQUIRED_COLUMNS.items():
        if table_name not in available_tables:
            raise Phase1CatalogSchemaError(
                f"Phase 1 catalog is missing required table {table_name!r}."
            )

        column_rows = connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
        available_columns = {str(row[1]) for row in column_rows}
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise Phase1CatalogSchemaError(
                "Phase 1 catalog table "
                f"{table_name!r} is missing columns: "
                f"{sorted(missing_columns)}"
            )


def verify_phase1_catalog_integrity(
    connection: sqlite3.Connection,
) -> None:
    """Fail closed unless SQLite reports a clean quick check."""
    row = connection.execute("PRAGMA quick_check").fetchone()
    result = None if row is None else str(row[0])

    if result != "ok":
        raise Phase1CatalogIntegrityError(
            f"Phase 1 catalog quick check failed: {result!r}"
        )


def load_phase1_chunk_evidence(
    catalog_path: Path,
    *,
    chunk_id: str,
    expected_source_content_sha256: str | None = None,
    expected_source_text_sha256: str | None = None,
    expected_chunk_text_sha256: str | None = None,
    expected_file_id: str | None = None,
    verify_chunk_text_file: bool = True,
) -> Phase1ChunkEvidence:
    """Load and validate one Phase 1 chunk without returning source plaintext."""
    with open_phase1_catalog(catalog_path) as connection:
        row = connection.execute(
            """
            SELECT
                c.chunk_id,
                c.chunk_set_id,
                c.source_content_sha256,
                c.source_text_sha256,
                c.normalized_source_text_sha256,
                c.chunk_index,
                c.start_char,
                c.end_char,
                c.char_count,
                c.chunk_text_sha256,
                c.family,
                c.parser_id,
                c.source_extraction_truncated,
                c.provenance_path_count,
                c.text_relative_path,
                s.pilot_name
            FROM chunks AS c
            JOIN chunk_sets AS s
              ON s.chunk_set_id = c.chunk_set_id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()

        if row is None:
            raise Phase1EvidenceNotFoundError(
                f"Phase 1 chunk not found: {chunk_id}"
            )

        source_content_sha256 = str(row["source_content_sha256"])
        source_text_sha256 = str(row["source_text_sha256"])
        normalized_source_text_sha256 = str(
            row["normalized_source_text_sha256"]
        )
        chunk_text_sha256 = str(row["chunk_text_sha256"])

        for field_name, value in (
            ("source_content_sha256", source_content_sha256),
            ("source_text_sha256", source_text_sha256),
            (
                "normalized_source_text_sha256",
                normalized_source_text_sha256,
            ),
            ("chunk_text_sha256", chunk_text_sha256),
        ):
            _validate_sha256(
                value,
                field_name=field_name,
            )

        expected_pairs = (
            (
                "source_content_sha256",
                expected_source_content_sha256,
                source_content_sha256,
            ),
            (
                "source_text_sha256",
                expected_source_text_sha256,
                source_text_sha256,
            ),
            (
                "chunk_text_sha256",
                expected_chunk_text_sha256,
                chunk_text_sha256,
            ),
        )
        for field_name, expected, actual in expected_pairs:
            if expected is not None and expected != actual:
                raise Phase1EvidenceMismatchError(
                    f"Phase 1 {field_name} mismatch for chunk {chunk_id}: "
                    f"expected {expected}, found {actual}."
                )

        provenance_rows = connection.execute(
            """
            SELECT
                file_id,
                role,
                family,
                source_bucket,
                year_hint,
                duplicate_control_group,
                known_contradiction_group
            FROM chunk_provenance
            WHERE chunk_id = ?
            ORDER BY file_id
            """,
            (chunk_id,),
        ).fetchall()

        expected_provenance_count = int(row["provenance_path_count"])
        if len(provenance_rows) != expected_provenance_count:
            raise Phase1CatalogIntegrityError(
                "Phase 1 provenance count mismatch for chunk "
                f"{chunk_id}: expected {expected_provenance_count}, "
                f"found {len(provenance_rows)}."
            )

        provenance_paths = tuple(
            Phase1ProvenancePath(
                file_id=str(provenance["file_id"]),
                role=provenance["role"],
                family=provenance["family"],
                source_bucket=provenance["source_bucket"],
                year_hint=provenance["year_hint"],
                duplicate_control_group=provenance[
                    "duplicate_control_group"
                ],
                known_contradiction_group=provenance[
                    "known_contradiction_group"
                ],
            )
            for provenance in provenance_rows
        )

        if expected_file_id is not None and expected_file_id not in {
            item.file_id for item in provenance_paths
        }:
            raise Phase1EvidenceMismatchError(
                f"Expected file_id {expected_file_id!r} is not provenance "
                f"for Phase 1 chunk {chunk_id}."
            )

        if verify_chunk_text_file:
            text_relative_path = str(row["text_relative_path"])
            chunk_text_path = (
                catalog_path.expanduser().resolve(strict=True).parent
                / text_relative_path
            ).resolve(strict=True)

            expected_root = (
                catalog_path.expanduser().resolve(strict=True).parent
            )
            if (
                chunk_text_path != expected_root
                and expected_root not in chunk_text_path.parents
            ):
                raise Phase1CatalogIntegrityError(
                    "Phase 1 chunk text path escapes its chunk-set root."
                )

            actual_chunk_text_sha256 = hashlib.sha256(
                chunk_text_path.read_bytes()
            ).hexdigest()
            if actual_chunk_text_sha256 != chunk_text_sha256:
                raise Phase1CatalogIntegrityError(
                    f"Phase 1 chunk text hash mismatch: {chunk_id}"
                )

        return Phase1ChunkEvidence(
            pilot_name=str(row["pilot_name"]),
            chunk_set_id=str(row["chunk_set_id"]),
            chunk_id=str(row["chunk_id"]),
            source_content_sha256=source_content_sha256,
            source_text_sha256=source_text_sha256,
            normalized_source_text_sha256=normalized_source_text_sha256,
            chunk_text_sha256=chunk_text_sha256,
            chunk_index=int(row["chunk_index"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
            char_count=int(row["char_count"]),
            family=row["family"],
            parser_id=row["parser_id"],
            source_extraction_truncated=bool(
                row["source_extraction_truncated"]
            ),
            provenance_paths=provenance_paths,
        )
