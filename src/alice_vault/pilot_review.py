from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REVIEW_SCHEMA_VERSION = 1
ALLOWED_DECISIONS = {"pending", "approve", "reject"}
BOOLEAN_VALUES = {"", "yes", "no"}
REQUIRED_REVIEW_COLUMNS = {
    "item_index",
    "file_id",
    "content_key",
    "relative_path",
    "filename",
    "size_bytes",
    "sha256",
    "role",
    "family",
    "source_bucket",
    "year_hint",
    "duplicate_control_group",
    "selection_reason",
    "decision",
    "review_notes",
    "known_contradiction_group",
    "contains_identity_document",
    "contains_credentials_or_secrets",
}
CORE_FAMILIES = {"json", "html", "csv", "pdf", "docx"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _latest_proposal_id(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        """
        SELECT proposal_id
        FROM pilot_proposal_runs
        WHERE status='complete'
        ORDER BY completed_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("No completed pilot proposal was found.")
    return str(row["proposal_id"])


def _proposal_path(vault_root: Path, proposal_id: str) -> Path:
    return (
        vault_root
        / "manifests"
        / "exports"
        / f"pilot-proposal-{proposal_id}.csv"
    )


def _review_path(vault_root: Path, proposal_id: str) -> Path:
    return (
        vault_root
        / "manifests"
        / "exports"
        / f"pilot-review-{proposal_id}.csv"
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_review(
    *,
    vault_root: Path,
    proposal_id: str | None = None,
) -> dict[str, object]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    database = vault_root / "manifests" / "inventory.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"Inventory database not found: {database}")

    connection = _connect(database)
    proposal_id = proposal_id or _latest_proposal_id(connection)
    connection.close()

    proposal_path = _proposal_path(vault_root, proposal_id)
    if not proposal_path.is_file():
        raise FileNotFoundError(f"Proposal CSV not found: {proposal_path}")

    fieldnames, rows = _read_csv(proposal_path)
    missing = REQUIRED_REVIEW_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(
            "Proposal CSV is missing review columns: "
            + ", ".join(sorted(missing))
        )

    review_path = _review_path(vault_root, proposal_id)
    created = False
    if not review_path.exists():
        _write_csv(review_path, fieldnames, rows)
        created = True

    guide_path = (
        vault_root
        / "manifests"
        / "exports"
        / f"pilot-review-guide-{proposal_id}.txt"
    )
    guide_path.write_text(
        (
            "A.L.I.C.E. PILOT HUMAN REVIEW\n"
            "============================\n\n"
            "Edit only these columns:\n"
            "  decision: approve | reject | pending\n"
            "  review_notes: private notes explaining your decision\n"
            "  known_contradiction_group: a shared label for 2+ files "
            "that contain conflicting or superseded information\n"
            "  contains_identity_document: yes | no\n"
            "  contains_credentials_or_secrets: yes | no\n\n"
            "Rules:\n"
            "- Inspect the actual content, not only the filename.\n"
            "- Reject identity documents and files containing credentials.\n"
            "- Keep both members of a duplicate-control pair approved or "
            "reject both.\n"
            "- Label at least two contradiction groups, with at least two "
            "approved files in each group.\n"
            "- Do not change file_id, hash, path, role, family, or other "
            "generated columns.\n"
            "- Keep this CSV inside the private vault.\n"
        ),
        encoding="utf-8",
    )

    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "proposal_id": proposal_id,
        "review_path": str(review_path),
        "guide_path": str(guide_path),
        "created": created,
        "row_count": len(rows),
    }


@dataclass(frozen=True)
class ReviewValidation:
    proposal_id: str
    total_rows: int
    approved_count: int
    rejected_count: int
    pending_count: int
    approved_bytes: int
    approved_families: dict[str, int]
    approved_source_buckets: int
    approved_known_years: int
    duplicate_groups_approved: int
    contradiction_groups_approved: int
    blocking_errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "total_rows": self.total_rows,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "pending_count": self.pending_count,
            "approved_bytes": self.approved_bytes,
            "approved_mib": round(self.approved_bytes / 1024**2, 3),
            "approved_gib": round(self.approved_bytes / 1024**3, 3),
            "approved_families": self.approved_families,
            "approved_source_bucket_count": self.approved_source_buckets,
            "approved_known_year_count": self.approved_known_years,
            "duplicate_groups_approved": self.duplicate_groups_approved,
            "contradiction_groups_approved": (
                self.contradiction_groups_approved
            ),
            "blocking_errors": self.blocking_errors,
            "warnings": self.warnings,
            "ready_to_finalize": not self.blocking_errors,
        }


def validate_review(
    *,
    vault_root: Path,
    proposal_id: str | None = None,
    minimum_approved: int = 100,
    minimum_contradiction_groups: int = 2,
) -> ReviewValidation:
    vault_root = vault_root.expanduser().resolve(strict=True)
    database = vault_root / "manifests" / "inventory.sqlite3"
    connection = _connect(database)
    proposal_id = proposal_id or _latest_proposal_id(connection)
    connection.close()

    review_path = _review_path(vault_root, proposal_id)
    if not review_path.is_file():
        raise FileNotFoundError(
            f"Review CSV not found. Prepare it first: {review_path}"
        )

    fieldnames, rows = _read_csv(review_path)
    missing = REQUIRED_REVIEW_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(
            "Review CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    errors: list[str] = []
    warnings: list[str] = []
    decisions = Counter()
    family_counts: Counter[str] = Counter()
    source_buckets: set[str] = set()
    years: set[str] = set()
    duplicate_decisions: dict[str, list[str]] = defaultdict(list)
    contradiction_members: dict[str, list[int]] = defaultdict(list)
    approved_bytes = 0
    approved_file_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        decision = (row.get("decision") or "").strip().lower()
        identity = (
            row.get("contains_identity_document") or ""
        ).strip().lower()
        credentials = (
            row.get("contains_credentials_or_secrets") or ""
        ).strip().lower()

        if decision not in ALLOWED_DECISIONS:
            errors.append(
                f"Row {row_number}: invalid decision {decision!r}"
            )
            continue
        if identity not in BOOLEAN_VALUES:
            errors.append(
                f"Row {row_number}: identity flag must be yes, no, or blank"
            )
        if credentials not in BOOLEAN_VALUES:
            errors.append(
                f"Row {row_number}: credentials flag must be yes, no, or blank"
            )

        decisions[decision] += 1
        duplicate_group = (
            row.get("duplicate_control_group") or ""
        ).strip()
        if duplicate_group:
            duplicate_decisions[duplicate_group].append(decision)

        if decision != "approve":
            continue

        file_id = (row.get("file_id") or "").strip()
        if file_id in approved_file_ids:
            errors.append(f"Row {row_number}: duplicate approved file_id")
        approved_file_ids.add(file_id)

        if identity != "no":
            errors.append(
                f"Row {row_number}: approved item must explicitly mark "
                "contains_identity_document=no"
            )
        if credentials != "no":
            errors.append(
                f"Row {row_number}: approved item must explicitly mark "
                "contains_credentials_or_secrets=no"
            )

        try:
            approved_bytes += int(row["size_bytes"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"Row {row_number}: invalid size_bytes")

        family_counts[(row.get("family") or "").strip()] += 1
        source_buckets.add((row.get("source_bucket") or "").strip())
        year = (row.get("year_hint") or "").strip()
        if year and year != "[unknown]":
            years.add(year)

        contradiction = (
            row.get("known_contradiction_group") or ""
        ).strip()
        if contradiction:
            contradiction_members[contradiction].append(row_number)

    pending_count = decisions["pending"]
    approved_count = decisions["approve"]
    rejected_count = decisions["reject"]

    if pending_count:
        errors.append(f"{pending_count} review rows are still pending")
    if approved_count < minimum_approved:
        errors.append(
            f"Only {approved_count} items approved; minimum is "
            f"{minimum_approved}"
        )
    if approved_bytes > 2 * 1024**3:
        errors.append("Approved pilot exceeds the 2 GiB limit")

    missing_core = CORE_FAMILIES.difference(
        family for family, count in family_counts.items() if count > 0
    )
    if missing_core:
        errors.append(
            "Approved pilot is missing core families: "
            + ", ".join(sorted(missing_core))
        )
    if len(family_counts) < 8:
        errors.append(
            f"Approved pilot covers only {len(family_counts)} families; "
            "minimum is 8"
        )
    if len(source_buckets) < 5:
        errors.append(
            f"Approved pilot covers only {len(source_buckets)} source "
            "buckets; minimum is 5"
        )
    if len(years) < 5:
        warnings.append(
            f"Approved pilot covers only {len(years)} known years"
        )

    complete_duplicate_groups = 0
    for group, group_decisions in sorted(duplicate_decisions.items()):
        approved_in_group = group_decisions.count("approve")
        rejected_in_group = group_decisions.count("reject")
        pending_in_group = group_decisions.count("pending")
        if pending_in_group:
            errors.append(
                f"Duplicate-control group {group} is still pending"
            )
        elif approved_in_group == len(group_decisions) == 2:
            complete_duplicate_groups += 1
        elif rejected_in_group == len(group_decisions):
            continue
        else:
            errors.append(
                f"Duplicate-control group {group} must approve both "
                "members or reject both"
            )

    valid_contradiction_groups = {
        group: members
        for group, members in contradiction_members.items()
        if len(members) >= 2
    }
    invalid_contradiction_groups = {
        group: members
        for group, members in contradiction_members.items()
        if len(members) < 2
    }
    for group in sorted(invalid_contradiction_groups):
        errors.append(
            f"Contradiction group {group!r} has fewer than 2 approved items"
        )
    if len(valid_contradiction_groups) < minimum_contradiction_groups:
        errors.append(
            f"Only {len(valid_contradiction_groups)} valid contradiction "
            f"groups; minimum is {minimum_contradiction_groups}"
        )

    return ReviewValidation(
        proposal_id=proposal_id,
        total_rows=len(rows),
        approved_count=approved_count,
        rejected_count=rejected_count,
        pending_count=pending_count,
        approved_bytes=approved_bytes,
        approved_families=dict(family_counts),
        approved_source_buckets=len(source_buckets),
        approved_known_years=len(years),
        duplicate_groups_approved=complete_duplicate_groups,
        contradiction_groups_approved=len(valid_contradiction_groups),
        blocking_errors=errors,
        warnings=warnings,
    )


def _safe_object_extension(row: dict[str, str]) -> str:
    extension = Path(row.get("filename") or "").suffix.lower()
    if not extension:
        extension = (row.get("family") or "").strip().lower()
        extension = f".{extension}" if extension else ".bin"
    if len(extension) > 12 or any(
        character not in ".abcdefghijklmnopqrstuvwxyz0123456789"
        for character in extension
    ):
        return ".bin"
    return extension


def finalize_pilot(
    *,
    vault_root: Path,
    proposal_id: str | None = None,
    pilot_name: str = "pilot-v1",
    minimum_approved: int = 100,
    minimum_contradiction_groups: int = 2,
) -> dict[str, object]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    validation = validate_review(
        vault_root=vault_root,
        proposal_id=proposal_id,
        minimum_approved=minimum_approved,
        minimum_contradiction_groups=minimum_contradiction_groups,
    )
    if validation.blocking_errors:
        raise ValueError(
            "Review cannot be finalized:\n- "
            + "\n- ".join(validation.blocking_errors)
        )

    proposal_id = validation.proposal_id
    review_path = _review_path(vault_root, proposal_id)
    fieldnames, rows = _read_csv(review_path)
    approved = [
        row
        for row in rows
        if (row.get("decision") or "").strip().lower() == "approve"
    ]

    database = vault_root / "manifests" / "inventory.sqlite3"
    connection = _connect(database)

    target_root = vault_root / "raw" / pilot_name
    if target_root.exists():
        connection.close()
        raise FileExistsError(
            f"Pilot snapshot already exists and will not be overwritten: "
            f"{target_root}"
        )

    temporary_root = (
        vault_root / "temporary" / f"{pilot_name}-{uuid.uuid4().hex}"
    )
    objects_root = temporary_root / "objects"
    objects_root.mkdir(parents=True, exist_ok=False)

    copied_by_hash: dict[tuple[str, int], str] = {}
    manifest_items: list[dict[str, object]] = []
    copied_unique_bytes = 0

    try:
        for row in approved:
            file_id = row["file_id"]
            source_row = connection.execute(
                """
                SELECT original_path, relative_path, size_bytes, sha256
                FROM files
                WHERE file_id=?
                """,
                (file_id,),
            ).fetchone()
            if source_row is None:
                raise RuntimeError(f"Inventory record not found: {file_id}")

            source = Path(source_row["original_path"])
            expected_size = int(source_row["size_bytes"])
            expected_hash = str(source_row["sha256"])
            if not source.is_file():
                raise FileNotFoundError(f"Source file missing: {source}")

            stat_before = source.stat()
            if stat_before.st_size != expected_size:
                raise RuntimeError(
                    f"Source size changed since inventory: "
                    f"{source_row['relative_path']}"
                )
            actual_hash = sha256_file(source)
            stat_after = source.stat()
            if (
                actual_hash != expected_hash
                or stat_after.st_size != stat_before.st_size
                or stat_after.st_mtime_ns != stat_before.st_mtime_ns
            ):
                raise RuntimeError(
                    "Source content changed or failed verification: "
                    f"{source_row['relative_path']}"
                )

            content_key = (expected_hash, expected_size)
            object_relative = copied_by_hash.get(content_key)
            if object_relative is None:
                extension = _safe_object_extension(row)
                object_name = f"{expected_hash}{extension}"
                destination = objects_root / object_name
                shutil.copy2(source, destination)
                if sha256_file(destination) != expected_hash:
                    raise RuntimeError(
                        f"Copied object failed verification: {object_name}"
                    )
                os.chmod(destination, stat.S_IREAD)
                object_relative = f"objects/{object_name}"
                copied_by_hash[content_key] = object_relative
                copied_unique_bytes += expected_size

            manifest_items.append(
                {
                    "item_index": int(row["item_index"]),
                    "file_id": file_id,
                    "original_relative_path": row["relative_path"],
                    "filename": row["filename"],
                    "role": row["role"],
                    "family": row["family"],
                    "source_bucket": row["source_bucket"],
                    "year_hint": row["year_hint"],
                    "duplicate_control_group": (
                        row["duplicate_control_group"]
                    ),
                    "known_contradiction_group": (
                        row["known_contradiction_group"]
                    ),
                    "review_notes": row["review_notes"],
                    "size_bytes": expected_size,
                    "sha256": expected_hash,
                    "object_path": object_relative,
                }
            )

        snapshot = {
            "pilot_snapshot_schema_version": 1,
            "pilot_name": pilot_name,
            "proposal_id": proposal_id,
            "created_at": utc_now(),
            "approved_item_count": len(manifest_items),
            "unique_content_count": len(copied_by_hash),
            "logical_bytes": validation.approved_bytes,
            "physical_unique_bytes": copied_unique_bytes,
            "duplicate_bytes_avoided": (
                validation.approved_bytes - copied_unique_bytes
            ),
            "family_counts": validation.approved_families,
            "source_bucket_count": validation.approved_source_buckets,
            "known_year_count": validation.approved_known_years,
            "duplicate_groups": validation.duplicate_groups_approved,
            "contradiction_groups": (
                validation.contradiction_groups_approved
            ),
            "default_classification": "HIGHLY_SENSITIVE",
            "source_files_modified": False,
            "items": manifest_items,
        }

        (temporary_root / "pilot-manifest.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        item_fields = list(manifest_items[0].keys())
        _write_csv(
            temporary_root / "pilot-items.csv",
            item_fields,
            manifest_items,
        )
        (temporary_root / "README.txt").write_text(
            (
                "A.L.I.C.E. approved parser pilot snapshot.\n"
                "Objects are content-addressed, verified copies.\n"
                "Do not edit files in this directory.\n"
                "Original source files were not modified.\n"
            ),
            encoding="utf-8",
        )

        target_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root.replace(target_root)

        private_manifest_root = (
            vault_root / "manifests" / "pilots" / pilot_name
        )
        private_manifest_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(
            target_root / "pilot-manifest.json",
            private_manifest_root / "pilot-manifest.json",
        )
        shutil.copy2(
            target_root / "pilot-items.csv",
            private_manifest_root / "pilot-items.csv",
        )
        shutil.copy2(
            review_path,
            private_manifest_root / review_path.name,
        )

    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
        connection.close()
        raise

    connection.close()

    result = {
        key: value
        for key, value in snapshot.items()
        if key != "items"
    }
    result["snapshot_path"] = str(target_root)
    result["manifest_path"] = str(
        target_root / "pilot-manifest.json"
    )
    return result
