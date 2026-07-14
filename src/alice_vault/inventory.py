from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import sqlite3
import stat
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

VALID_MODES = {"metadata", "sha256"}
VALID_CLASSES = {"PUBLIC", "INTERNAL", "PRIVATE", "HIGHLY_SENSITIVE", "SECRETS"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contains(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_layout(vault: Path) -> None:
    for rel in [
        "incoming", "raw", "quarantine", "derived/extracted_text",
        "derived/metadata", "derived/thumbnails", "derived/chunks",
        "manifests/exports", "logs", "temporary", "backups",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS inventory_runs(
          run_id TEXT PRIMARY KEY,
          source_root TEXT NOT NULL,
          vault_root TEXT NOT NULL,
          scan_mode TEXT NOT NULL,
          default_classification TEXT NOT NULL,
          started_at TEXT NOT NULL,
          completed_at TEXT,
          status TEXT NOT NULL,
          file_count INTEGER DEFAULT 0,
          total_bytes INTEGER DEFAULT 0,
          hashed_bytes INTEGER DEFAULT 0,
          duplicate_count INTEGER DEFAULT 0,
          error_count INTEGER DEFAULT 0,
          skipped_link_count INTEGER DEFAULT 0,
          notes TEXT
        );
        CREATE TABLE IF NOT EXISTS files(
          file_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL REFERENCES inventory_runs(run_id) ON DELETE CASCADE,
          original_path TEXT NOT NULL,
          relative_path TEXT NOT NULL,
          filename TEXT NOT NULL,
          extension TEXT NOT NULL,
          size_bytes INTEGER,
          created_time_ns INTEGER,
          modified_time_ns INTEGER,
          sha256 TEXT,
          preliminary_mime_type TEXT,
          scan_status TEXT NOT NULL,
          duplicate_of TEXT REFERENCES files(file_id),
          classification TEXT NOT NULL,
          error TEXT,
          discovered_at TEXT NOT NULL,
          UNIQUE(run_id, relative_path)
        );
        CREATE INDEX IF NOT EXISTS idx_files_run ON files(run_id);
        CREATE INDEX IF NOT EXISTS idx_files_sha ON files(sha256);
        CREATE INDEX IF NOT EXISTS idx_files_dup ON files(duplicate_of);
        """
    )
    con.commit()
    return con


def _reparse(entry: os.DirEntry[str]) -> bool:
    try:
        attrs = entry.stat(follow_symlinks=False).st_file_attributes
        return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (AttributeError, OSError):
        return False


def discover(source: Path, excluded: Sequence[str]) -> tuple[list[tuple[Path, str]], int]:
    excluded_names = {x.casefold() for x in excluded}
    found: list[tuple[Path, str]] = []
    skipped = 0

    def walk(folder: Path) -> None:
        nonlocal skipped
        try:
            with os.scandir(folder) as it:
                entries = sorted(it, key=lambda e: e.name.casefold())
        except OSError:
            return
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink() or _reparse(entry):
                    skipped += 1
                elif entry.is_dir(follow_symlinks=False):
                    if entry.name.casefold() not in excluded_names:
                        walk(path)
                elif entry.is_file(follow_symlinks=False):
                    found.append((path, path.relative_to(source).as_posix()))
            except OSError:
                continue

    walk(source)
    return found, skipped


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(
    source: Path,
    vault: Path,
    mode: str = "metadata",
    classification: str = "HIGHLY_SENSITIVE",
    excluded: Sequence[str] = (),
) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}")
    if classification not in VALID_CLASSES:
        raise ValueError(f"Invalid classification: {classification}")

    source = source.expanduser().resolve(strict=True)
    vault = vault.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source is not a directory: {source}")
    if contains(vault, source) or contains(source, vault):
        raise ValueError("Source and vault must not contain one another")

    ensure_layout(vault)
    database = vault / "manifests" / "inventory.sqlite3"
    con = connect_db(database)
    run_id = str(uuid.uuid4())
    started = utc_now()
    con.execute(
        "INSERT INTO inventory_runs(run_id,source_root,vault_root,scan_mode,default_classification,started_at,status) VALUES(?,?,?,?,?,?,?)",
        (run_id, str(source), str(vault), mode, classification, started, "running"),
    )
    con.commit()

    count = total = hashed = duplicates = errors = 0
    extension_counts: Counter[str] = Counter()
    seen: dict[tuple[str, int], str] = {}

    try:
        files, skipped = discover(source, excluded)
        for index, (path, relative) in enumerate(files, start=1):
            file_id = str(uuid.uuid4())
            extension = path.suffix.lower()
            mime = mimetypes.guess_type(path.name, strict=False)[0]
            size = created = modified = None
            digest = duplicate_of = error = None
            status = "metadata_recorded"

            try:
                before = path.stat()
                size = before.st_size
                created = before.st_ctime_ns
                modified = before.st_mtime_ns
                if mode == "sha256":
                    digest = file_sha256(path)
                    after = path.stat()
                    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
                        digest = None
                        status = "changed_during_scan"
                        error = "File changed while hashing"
                        errors += 1
                    else:
                        status = "hashed"
                        hashed += size
                        key = (digest, size)
                        duplicate_of = seen.get(key)
                        if duplicate_of:
                            status = "exact_duplicate"
                            duplicates += 1
                        else:
                            seen[key] = file_id
                count += 1
                total += size
                extension_counts[extension or "[no extension]"] += 1
            except (OSError, PermissionError) as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                errors += 1

            con.execute(
                """
                INSERT INTO files(
                  file_id,run_id,original_path,relative_path,filename,extension,
                  size_bytes,created_time_ns,modified_time_ns,sha256,
                  preliminary_mime_type,scan_status,duplicate_of,classification,
                  error,discovered_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    file_id, run_id, str(path), relative, path.name, extension,
                    size, created, modified, digest, mime, status, duplicate_of,
                    classification, error, utc_now(),
                ),
            )
            if index % 100 == 0:
                con.commit()
            if index % 250 == 0:
                print(f"Scanned {index:,}/{len(files):,} files ({total/(1024**3):.2f} GiB)")

        completed = utc_now()
        con.execute(
            """
            UPDATE inventory_runs SET completed_at=?,status='complete',file_count=?,
              total_bytes=?,hashed_bytes=?,duplicate_count=?,error_count=?,
              skipped_link_count=? WHERE run_id=?
            """,
            (completed, count, total, hashed, duplicates, errors, skipped, run_id),
        )
        con.commit()
    except Exception as exc:
        con.execute(
            "UPDATE inventory_runs SET completed_at=?,status='failed',notes=? WHERE run_id=?",
            (utc_now(), f"{type(exc).__name__}: {exc}", run_id),
        )
        con.commit()
        con.close()
        raise

    summary = {
        "run_id": run_id,
        "source_root": str(source),
        "vault_root": str(vault),
        "mode": mode,
        "classification": classification,
        "started_at": started,
        "completed_at": completed,
        "file_count": count,
        "total_bytes": total,
        "total_gib": round(total / (1024**3), 3),
        "hashed_bytes": hashed,
        "hashed_gib": round(hashed / (1024**3), 3),
        "exact_duplicate_count": duplicates,
        "error_count": errors,
        "skipped_link_count": skipped,
        "extension_counts": dict(extension_counts.most_common()),
        "database_path": str(database),
    }

    exports = vault / "manifests" / "exports"
    summary_path = exports / f"inventory-summary-{run_id}.json"
    csv_path = exports / f"inventory-files-{run_id}.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = con.execute(
        """
        SELECT relative_path,filename,extension,size_bytes,created_time_ns,
          modified_time_ns,sha256,preliminary_mime_type,scan_status,
          duplicate_of,classification,error
        FROM files WHERE run_id=? ORDER BY relative_path
        """,
        (run_id,),
    )
    fields = [
        "relative_path", "filename", "extension", "size_bytes",
        "created_time_ns", "modified_time_ns", "sha256",
        "preliminary_mime_type", "scan_status", "duplicate_of",
        "classification", "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    con.close()
    summary["summary_path"] = str(summary_path)
    summary["csv_path"] = str(csv_path)
    return summary
