from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence


ANALYSIS_SCHEMA_VERSION = 1
HIGH_RISK_EXTENSIONS = {
    ".exe", ".dll", ".msi", ".com", ".scr", ".bat", ".cmd", ".ps1", ".vbs",
}
SERIALIZED_CODE_EXTENSIONS = {".pth", ".pt", ".pkl", ".pickle", ".joblib"}
ARCHIVE_EXTENSIONS = {".zip"}
MAILBOX_EXTENSIONS = {".mbox"}
OPAQUE_EXTENSIONS = {".dat", ".bin"}
LOW_TRUST_SIGNATURE_EXTENSIONS = {".xxx"}
TEXT_LIKE_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".html", ".htm", ".xml", ".css",
    ".js", ".py", ".cpp", ".vcf", ".ics", ".srt", ".atom", ".webmanifest",
}
MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp3", ".wav", ".ogg", ".mp4",
}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx"}
EXTENSION_ALIASES = [
    {".jpg", ".jpeg", ".jfif"},
    {".html", ".htm"},
    {".json", ".ipynb", ".webmanifest"},
    {".xml", ".atom"},
    {".tif", ".tiff"},
    {".yaml", ".yml"},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_extension(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def extensions_equivalent(left: str, right: str) -> bool:
    left = normalize_extension(left)
    right = normalize_extension(right)
    if left == right:
        return True
    return any(left in aliases and right in aliases for aliases in EXTENSION_ALIASES)


@dataclass(frozen=True)
class SignatureResult:
    extension: str = ""
    mime_type: str = ""
    description: str = ""
    confidence: float | None = None
    status: str = "unknown"
    error: str | None = None


@dataclass(frozen=True)
class ArchiveResult:
    is_zip: bool = False
    detected_extension: str = ""
    member_count: int = 0
    file_member_count: int = 0
    compressed_bytes: int = 0
    uncompressed_bytes: int = 0
    compression_ratio: float = 0.0
    encrypted_members: int = 0
    unsafe_path_members: int = 0
    oversized_members: int = 0
    corrupt: bool = False
    error: str | None = None


class SignatureDetector(Protocol):
    def detect(self, path: Path) -> SignatureResult:
        ...


def detect_iso_bmff(path: Path) -> SignatureResult | None:
    """Recognize ISO Base Media files from the leading ftyp box."""
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
    except OSError as exc:
        return SignatureResult(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12].decode("ascii", errors="replace").strip()
        description = "ISO Base Media File"
        if brand:
            description += f" ({brand})"
        return SignatureResult(
            extension=".mp4",
            mime_type="video/mp4",
            description=description,
            confidence=1.0,
            status="identified",
        )
    return None


class PureMagicDetector:
    """Header-oriented signature detection with puremagic deep scan disabled."""

    def __init__(self) -> None:
        os.environ["PUREMAGIC_DEEPSCAN"] = "0"
        try:
            import puremagic
        except ImportError as exc:
            raise RuntimeError(
                "puremagic is not installed. Run: "
                "py -m pip install -r requirements-phase1.txt"
            ) from exc
        self._puremagic = puremagic

    def detect(self, path: Path) -> SignatureResult:
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            return SignatureResult(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        if size_bytes == 0:
            return SignatureResult(
                description="Empty file",
                confidence=1.0,
                status="empty",
            )

        iso_bmff = detect_iso_bmff(path)
        if iso_bmff is not None:
            return iso_bmff

        try:
            matches = self._puremagic.magic_file(str(path))
        except Exception as exc:
            message = str(exc)
            lowered = message.casefold()
            if "no match" in lowered or "could not identify" in lowered:
                return SignatureResult(status="unknown")
            return SignatureResult(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        if not matches:
            return SignatureResult(status="unknown")

        first = matches[0]

        if hasattr(first, "extension"):
            extension = getattr(first, "extension", "") or ""
            mime_type = getattr(first, "mime_type", "") or ""
            description = getattr(first, "name", "") or ""
            confidence = getattr(first, "confidence", None)
        else:
            extension = first[0] if len(first) > 0 else ""
            mime_type = first[1] if len(first) > 1 else ""
            description = first[2] if len(first) > 2 else ""
            confidence = first[3] if len(first) > 3 else None

        normalized_extension = normalize_extension(extension)
        if normalized_extension in LOW_TRUST_SIGNATURE_EXTENSIONS:
            return SignatureResult(
                mime_type=mime_type,
                description=description,
                confidence=float(confidence) if confidence is not None else None,
                status="ambiguous",
            )

        return SignatureResult(
            extension=normalized_extension,
            mime_type=mime_type,
            description=description,
            confidence=float(confidence) if confidence is not None else None,
            status="identified",
        )


def _unsafe_archive_name(name: str) -> bool:
    if "\x00" in name:
        return True
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    if re.match(r"^[A-Za-z]:", normalized):
        return True
    parts = PurePosixPath(normalized).parts
    return ".." in parts


def inspect_zip(path: Path) -> ArchiveResult:
    if not zipfile.is_zipfile(path):
        return ArchiveResult(is_zip=False)

    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            names = {item.filename.replace("\\", "/") for item in members}
            detected_extension = ".zip"

            if "word/document.xml" in names:
                detected_extension = ".docx"
            elif "xl/workbook.xml" in names:
                detected_extension = ".xlsx"
            elif "ppt/presentation.xml" in names:
                detected_extension = ".pptx"

            file_members = [item for item in members if not item.is_dir()]
            compressed = sum(item.compress_size for item in file_members)
            uncompressed = sum(item.file_size for item in file_members)
            ratio = uncompressed / max(compressed, 1)
            encrypted = sum(bool(item.flag_bits & 0x1) for item in file_members)
            unsafe = sum(_unsafe_archive_name(item.filename) for item in members)
            oversized = sum(item.file_size > 10 * 1024**3 for item in file_members)

            return ArchiveResult(
                is_zip=True,
                detected_extension=detected_extension,
                member_count=len(members),
                file_member_count=len(file_members),
                compressed_bytes=compressed,
                uncompressed_bytes=uncompressed,
                compression_ratio=round(ratio, 3),
                encrypted_members=encrypted,
                unsafe_path_members=unsafe,
                oversized_members=oversized,
            )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        return ArchiveResult(
            is_zip=True,
            corrupt=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def classify_match(
    claimed_extension: str,
    detected_extension: str,
    signature_status: str,
) -> str:
    claimed = normalize_extension(claimed_extension)
    detected = normalize_extension(detected_extension)

    if signature_status == "error":
        return "signature_error"
    if signature_status == "empty":
        return "empty"
    if signature_status == "ambiguous":
        return "ambiguous"
    if not detected:
        return "unknown"
    if not claimed:
        return "missing_extension"
    if claimed in OPAQUE_EXTENSIONS:
        return "opaque_identified"
    if claimed in SERIALIZED_CODE_EXTENSIONS and detected == ".zip":
        return "serialized_container"
    if extensions_equivalent(claimed, detected):
        return "match"
    return "mismatch"


def risk_and_recommendation(
    *,
    claimed_extension: str,
    detected_extension: str,
    match_status: str,
    archive: ArchiveResult,
    size_bytes: int,
) -> tuple[list[str], str]:
    claimed = normalize_extension(claimed_extension)
    detected = normalize_extension(detected_extension)
    flags: list[str] = []

    if claimed in HIGH_RISK_EXTENSIONS or detected in HIGH_RISK_EXTENSIONS:
        flags.append("executable_or_script")
    if claimed in SERIALIZED_CODE_EXTENSIONS:
        flags.append("serialized_code_or_model")
    if claimed in MAILBOX_EXTENSIONS:
        flags.append("mailbox_container")
    # OOXML documents are intentionally ZIP-packaged. A structurally
    # recognized DOCX/XLSX/PPTX is a document container, not a generic
    # archive requiring specialized review. A literal .zip, or a ZIP whose
    # structure cannot be classified as OOXML, remains an archive container.
    if claimed in ARCHIVE_EXTENSIONS or (archive.is_zip and detected == ".zip"):
        flags.append("archive_container")
    if claimed in OPAQUE_EXTENSIONS:
        flags.append("opaque_file")
    if not claimed:
        flags.append("extensionless")
    if match_status == "mismatch":
        flags.append("extension_signature_mismatch")
    if match_status == "opaque_identified":
        flags.append("opaque_content_identified")
    if match_status == "serialized_container":
        flags.append("serialized_archive_container")
    if match_status == "signature_error":
        flags.append("signature_error")
    if match_status == "ambiguous":
        flags.append("ambiguous_signature")
    if match_status == "empty":
        flags.append("empty_file")
    if archive.unsafe_path_members:
        flags.append("archive_unsafe_paths")
    if archive.encrypted_members:
        flags.append("archive_encrypted_members")
    if archive.corrupt:
        flags.append("archive_corrupt")
    if archive.compression_ratio > 200:
        flags.append("archive_extreme_compression")
    if archive.member_count > 100_000:
        flags.append("archive_excessive_members")
    if archive.oversized_members:
        flags.append("archive_oversized_member")

    if (
        "executable_or_script" in flags
        or "archive_unsafe_paths" in flags
        or "archive_corrupt" in flags
    ):
        recommendation = "quarantine_recommended"
    elif any(
        item in flags
        for item in (
            "serialized_code_or_model",
            "mailbox_container",
            "archive_container",
            "opaque_file",
            "archive_encrypted_members",
            "archive_extreme_compression",
            "archive_excessive_members",
            "archive_oversized_member",
        )
    ):
        recommendation = "specialized_review"
    elif match_status in {
        "mismatch",
        "signature_error",
        "missing_extension",
        "ambiguous",
    }:
        recommendation = "manual_review"
    elif match_status == "empty":
        recommendation = "inventory_only"
    elif claimed in DOCUMENT_EXTENSIONS and size_bytes <= 25 * 1024**2:
        recommendation = "pilot_candidate"
    elif claimed in TEXT_LIKE_EXTENSIONS and size_bytes <= 10 * 1024**2:
        recommendation = "pilot_candidate"
    elif claimed in MEDIA_EXTENSIONS:
        recommendation = "metadata_only"
    else:
        recommendation = "inventory_only"

    return sorted(set(flags)), recommendation


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            analysis_run_id TEXT PRIMARY KEY,
            inventory_run_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            unique_content_count INTEGER NOT NULL DEFAULT 0,
            analyzed_file_count INTEGER NOT NULL DEFAULT 0,
            mismatch_count INTEGER NOT NULL DEFAULT 0,
            unknown_count INTEGER NOT NULL DEFAULT 0,
            quarantine_count INTEGER NOT NULL DEFAULT 0,
            specialized_review_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS content_analysis (
            analysis_run_id TEXT NOT NULL REFERENCES analysis_runs(analysis_run_id)
                ON DELETE CASCADE,
            content_key TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            representative_file_id TEXT NOT NULL,
            detected_extension TEXT NOT NULL,
            detected_mime_type TEXT NOT NULL,
            description TEXT NOT NULL,
            confidence REAL,
            signature_status TEXT NOT NULL,
            risk_flags_json TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            archive_json TEXT NOT NULL,
            error TEXT,
            PRIMARY KEY (analysis_run_id, content_key)
        );

        CREATE TABLE IF NOT EXISTS file_analysis (
            analysis_run_id TEXT NOT NULL REFERENCES analysis_runs(analysis_run_id)
                ON DELETE CASCADE,
            file_id TEXT NOT NULL,
            content_key TEXT NOT NULL,
            claimed_extension TEXT NOT NULL,
            detected_extension TEXT NOT NULL,
            match_status TEXT NOT NULL,
            risk_flags_json TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            PRIMARY KEY (analysis_run_id, file_id)
        );

        CREATE INDEX IF NOT EXISTS idx_analysis_inventory
            ON analysis_runs(inventory_run_id);
        CREATE INDEX IF NOT EXISTS idx_file_analysis_recommendation
            ON file_analysis(recommendation);
        CREATE INDEX IF NOT EXISTS idx_file_analysis_match
            ON file_analysis(match_status);
        """
    )
    connection.commit()


def _latest_sha256_run(connection: sqlite3.Connection) -> sqlite3.Row:
    run = connection.execute(
        """
        SELECT *
        FROM inventory_runs
        WHERE scan_mode='sha256' AND status='complete'
        ORDER BY completed_at DESC
        LIMIT 1
        """
    ).fetchone()
    if run is None:
        raise RuntimeError("No completed SHA-256 inventory run was found.")
    return run


def analyze_inventory(
    *,
    vault_root: Path,
    detector: SignatureDetector | None = None,
    progress_every: int = 250,
) -> dict[str, object]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    database = vault_root / "manifests" / "inventory.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"Inventory database not found: {database}")

    detector = detector or PureMagicDetector()
    connection = _connect(database)
    _initialize_schema(connection)
    inventory_run = _latest_sha256_run(connection)
    inventory_run_id = inventory_run["run_id"]

    rows = connection.execute(
        """
        SELECT file_id, original_path, relative_path, extension,
               size_bytes, sha256, duplicate_of
        FROM files
        WHERE run_id=? AND sha256 IS NOT NULL
        ORDER BY sha256, size_bytes, duplicate_of IS NOT NULL, relative_path
        """,
        (inventory_run_id,),
    ).fetchall()

    groups: dict[tuple[str, int], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[(row["sha256"], row["size_bytes"])].append(row)

    analysis_run_id = str(uuid.uuid4())
    started = utc_now()
    connection.execute(
        """
        INSERT INTO analysis_runs(
            analysis_run_id, inventory_run_id, started_at, status
        ) VALUES (?, ?, ?, 'running')
        """,
        (analysis_run_id, inventory_run_id, started),
    )
    connection.commit()

    mismatch_count = 0
    unknown_count = 0
    quarantine_count = 0
    specialized_count = 0
    error_count = 0
    recommendation_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    detected_extension_counts: Counter[str] = Counter()

    try:
        for index, ((digest, size_bytes), members) in enumerate(
            groups.items(), start=1
        ):
            representative = members[0]
            path = Path(representative["original_path"])
            content_key = f"{digest}:{size_bytes}"
            signature = detector.detect(path)
            archive = inspect_zip(path)

            detected_extension = (
                archive.detected_extension
                if archive.is_zip and archive.detected_extension
                else signature.extension
            )
            signature_status = signature.status
            if archive.is_zip and not archive.corrupt:
                signature_status = "identified"

            claimed_extensions = sorted(
                {normalize_extension(item["extension"]) for item in members}
            )
            per_file_results: list[tuple[sqlite3.Row, str, list[str], str]] = []

            content_flags: set[str] = set()
            content_recommendations: list[str] = []

            for member in members:
                match_status = classify_match(
                    member["extension"],
                    detected_extension,
                    signature_status,
                )
                flags, recommendation = risk_and_recommendation(
                    claimed_extension=member["extension"],
                    detected_extension=detected_extension,
                    match_status=match_status,
                    archive=archive,
                    size_bytes=size_bytes,
                )
                per_file_results.append(
                    (member, match_status, flags, recommendation)
                )
                content_flags.update(flags)
                content_recommendations.append(recommendation)

                match_counts[match_status] += 1
                recommendation_counts[recommendation] += 1
                for flag in flags:
                    risk_counts[flag] += 1
                if match_status == "mismatch":
                    mismatch_count += 1
                elif match_status == "unknown":
                    unknown_count += 1
                if recommendation == "quarantine_recommended":
                    quarantine_count += 1
                elif recommendation == "specialized_review":
                    specialized_count += 1

            recommendation_priority = {
                "quarantine_recommended": 5,
                "specialized_review": 4,
                "manual_review": 3,
                "inventory_only": 2,
                "metadata_only": 1,
                "pilot_candidate": 0,
            }
            content_recommendation = max(
                content_recommendations,
                key=lambda item: recommendation_priority[item],
            )

            error = signature.error or archive.error
            if error:
                error_count += 1

            detected_extension_counts[detected_extension or "[unknown]"] += 1

            connection.execute(
                """
                INSERT INTO content_analysis(
                    analysis_run_id, content_key, sha256, size_bytes,
                    representative_file_id, detected_extension,
                    detected_mime_type, description, confidence,
                    signature_status, risk_flags_json, recommendation,
                    archive_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_run_id,
                    content_key,
                    digest,
                    size_bytes,
                    representative["file_id"],
                    detected_extension,
                    signature.mime_type,
                    signature.description,
                    signature.confidence,
                    signature_status,
                    json.dumps(sorted(content_flags)),
                    content_recommendation,
                    json.dumps(asdict(archive)),
                    error,
                ),
            )

            for member, match_status, flags, recommendation in per_file_results:
                connection.execute(
                    """
                    INSERT INTO file_analysis(
                        analysis_run_id, file_id, content_key,
                        claimed_extension, detected_extension, match_status,
                        risk_flags_json, recommendation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_run_id,
                        member["file_id"],
                        content_key,
                        normalize_extension(member["extension"]),
                        detected_extension,
                        match_status,
                        json.dumps(flags),
                        recommendation,
                    ),
                )

            if index % 100 == 0:
                connection.commit()
            if progress_every and index % progress_every == 0:
                print(
                    f"Analyzed {index:,}/{len(groups):,} unique contents"
                )

        completed = utc_now()
        connection.execute(
            """
            UPDATE analysis_runs
            SET completed_at=?, status='complete',
                unique_content_count=?, analyzed_file_count=?,
                mismatch_count=?, unknown_count=?, quarantine_count=?,
                specialized_review_count=?, error_count=?
            WHERE analysis_run_id=?
            """,
            (
                completed,
                len(groups),
                len(rows),
                mismatch_count,
                unknown_count,
                quarantine_count,
                specialized_count,
                error_count,
                analysis_run_id,
            ),
        )
        connection.commit()

    except Exception as exc:
        connection.execute(
            """
            UPDATE analysis_runs
            SET completed_at=?, status='failed', notes=?
            WHERE analysis_run_id=?
            """,
            (
                utc_now(),
                f"{type(exc).__name__}: {exc}",
                analysis_run_id,
            ),
        )
        connection.commit()
        connection.close()
        raise

    duplicate_group_row = connection.execute(
        """
        SELECT
            COUNT(DISTINCT duplicate_of) AS duplicate_groups,
            COUNT(*) AS duplicate_files,
            COALESCE(SUM(size_bytes), 0) AS reclaimable_bytes
        FROM files
        WHERE run_id=? AND duplicate_of IS NOT NULL
        """,
        (inventory_run_id,),
    ).fetchone()

    summary = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_run_id": analysis_run_id,
        "inventory_run_id": inventory_run_id,
        "started_at": started,
        "completed_at": completed,
        "analyzed_files": len(rows),
        "unique_contents": len(groups),
        "duplicate_files": duplicate_group_row["duplicate_files"],
        "duplicate_groups": duplicate_group_row["duplicate_groups"],
        "potential_reclaimable_bytes": duplicate_group_row["reclaimable_bytes"],
        "potential_reclaimable_gib": round(
            duplicate_group_row["reclaimable_bytes"] / (1024**3), 3
        ),
        "mismatch_count": mismatch_count,
        "unknown_count": unknown_count,
        "quarantine_recommended_count": quarantine_count,
        "specialized_review_count": specialized_count,
        "error_count": error_count,
        "match_status_counts": dict(match_counts.most_common()),
        "recommendation_counts": dict(recommendation_counts.most_common()),
        "risk_flag_counts": dict(risk_counts.most_common()),
        "detected_extension_counts": dict(
            detected_extension_counts.most_common()
        ),
        "database_path": str(database),
    }

    exports = vault_root / "manifests" / "exports"
    exports.mkdir(parents=True, exist_ok=True)

    summary_path = exports / f"analysis-summary-{analysis_run_id}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    details_path = exports / f"analysis-files-{analysis_run_id}.csv"
    detail_rows = connection.execute(
        """
        SELECT
            f.relative_path,
            f.filename,
            f.size_bytes,
            f.sha256,
            f.duplicate_of,
            a.claimed_extension,
            a.detected_extension,
            a.match_status,
            a.risk_flags_json,
            a.recommendation
        FROM file_analysis a
        JOIN files f ON f.file_id=a.file_id
        WHERE a.analysis_run_id=?
        ORDER BY f.relative_path
        """,
        (analysis_run_id,),
    )
    fields = [
        "relative_path",
        "filename",
        "size_bytes",
        "sha256",
        "duplicate_of",
        "claimed_extension",
        "detected_extension",
        "match_status",
        "risk_flags_json",
        "recommendation",
    ]
    with details_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(dict(row))

    review_path = exports / f"analysis-review-{analysis_run_id}.csv"
    review_rows = connection.execute(
        """
        SELECT
            f.relative_path,
            f.filename,
            f.size_bytes,
            a.claimed_extension,
            a.detected_extension,
            a.match_status,
            a.risk_flags_json,
            a.recommendation
        FROM file_analysis a
        JOIN files f ON f.file_id=a.file_id
        WHERE a.analysis_run_id=?
          AND a.recommendation IN (
              'quarantine_recommended',
              'specialized_review',
              'manual_review'
          )
        ORDER BY
            CASE a.recommendation
                WHEN 'quarantine_recommended' THEN 1
                WHEN 'specialized_review' THEN 2
                ELSE 3
            END,
            f.relative_path
        """,
        (analysis_run_id,),
    )
    review_fields = [
        "relative_path",
        "filename",
        "size_bytes",
        "claimed_extension",
        "detected_extension",
        "match_status",
        "risk_flags_json",
        "recommendation",
    ]
    with review_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(dict(row))

    connection.close()
    summary["summary_path"] = str(summary_path)
    summary["details_path"] = str(details_path)
    summary["review_path"] = str(review_path)
    return summary
