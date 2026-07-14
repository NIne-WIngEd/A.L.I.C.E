from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.pilot_review import (
    finalize_pilot,
    prepare_review,
    validate_review,
)


FIELDS = [
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
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PilotReviewTests(unittest.TestCase):
    def make_vault(self, base: Path) -> tuple[Path, str, list[dict[str, str]]]:
        vault = base / "vault"
        exports = vault / "manifests" / "exports"
        exports.mkdir(parents=True)
        (vault / "raw").mkdir()
        (vault / "temporary").mkdir()

        database = vault / "manifests" / "inventory.sqlite3"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE pilot_proposal_runs(
                proposal_id TEXT PRIMARY KEY,
                status TEXT,
                completed_at TEXT
            );
            CREATE TABLE files(
                file_id TEXT PRIMARY KEY,
                original_path TEXT,
                relative_path TEXT,
                size_bytes INTEGER,
                sha256 TEXT
            );
            """
        )
        proposal_id = "proposal-1"
        connection.execute(
            """
            INSERT INTO pilot_proposal_runs
            VALUES (?, 'complete', '2026-07-14T00:00:00+00:00')
            """,
            (proposal_id,),
        )

        rows: list[dict[str, str]] = []
        families = [
            "json",
            "html",
            "csv",
            "pdf",
            "docx",
            "text",
            "calendar",
            "contacts",
        ]
        for index in range(1, 101):
            family = families[(index - 1) % len(families)]
            suffix = {
                "json": ".json",
                "html": ".html",
                "csv": ".csv",
                "pdf": ".pdf",
                "docx": ".docx",
                "text": ".txt",
                "calendar": ".ics",
                "contacts": ".vcf",
            }[family]
            content = f"content-{index}".encode()
            if index == 100:
                content = b"content-99"
            file_path = base / f"source-{index}{suffix}"
            file_path.write_bytes(content)
            file_id = f"file-{index}"
            sha = digest(content)
            connection.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
                (
                    file_id,
                    str(file_path),
                    file_path.name,
                    len(content),
                    sha,
                ),
            )
            duplicate_group = "dup-1" if index in {99, 100} else ""
            contradiction = (
                "education-current"
                if index in {1, 2}
                else "project-status"
                if index in {3, 4}
                else ""
            )
            rows.append(
                {
                    "item_index": str(index),
                    "file_id": file_id,
                    "content_key": f"{sha}:{len(content)}",
                    "relative_path": file_path.name,
                    "filename": file_path.name,
                    "size_bytes": str(len(content)),
                    "sha256": sha,
                    "role": (
                        "duplicate_control"
                        if duplicate_group
                        else "primary"
                    ),
                    "family": family,
                    "source_bucket": f"bucket-{index % 5}",
                    "year_hint": str(2020 + index % 5),
                    "duplicate_control_group": duplicate_group,
                    "selection_reason": "test",
                    "decision": "approve",
                    "review_notes": "",
                    "known_contradiction_group": contradiction,
                    "contains_identity_document": "no",
                    "contains_credentials_or_secrets": "no",
                }
            )
        connection.commit()
        connection.close()

        proposal_path = exports / f"pilot-proposal-{proposal_id}.csv"
        with proposal_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        return vault, proposal_id, rows

    def test_prepare_and_validate_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault, proposal_id, _ = self.make_vault(Path(temp))
            prepared = prepare_review(
                vault_root=vault,
                proposal_id=proposal_id,
            )
            self.assertTrue(prepared["created"])
            validation = validate_review(
                vault_root=vault,
                proposal_id=proposal_id,
            )
            self.assertFalse(validation.blocking_errors)
            self.assertEqual(validation.approved_count, 100)
            self.assertEqual(validation.duplicate_groups_approved, 1)
            self.assertEqual(validation.contradiction_groups_approved, 2)

    def test_approved_secret_blocks_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault, proposal_id, _ = self.make_vault(Path(temp))
            prepared = prepare_review(
                vault_root=vault,
                proposal_id=proposal_id,
            )
            review = Path(prepared["review_path"])
            with review.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["contains_credentials_or_secrets"] = "yes"
            with review.open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            validation = validate_review(
                vault_root=vault,
                proposal_id=proposal_id,
            )
            self.assertTrue(
                any(
                    "contains_credentials_or_secrets=no" in error
                    for error in validation.blocking_errors
                )
            )

    def test_finalize_copies_unique_content_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault, proposal_id, _ = self.make_vault(Path(temp))
            prepare_review(
                vault_root=vault,
                proposal_id=proposal_id,
            )
            result = finalize_pilot(
                vault_root=vault,
                proposal_id=proposal_id,
            )
            self.assertEqual(result["approved_item_count"], 100)
            self.assertEqual(result["unique_content_count"], 99)
            snapshot = Path(result["snapshot_path"])
            self.assertEqual(
                len(list((snapshot / "objects").iterdir())),
                99,
            )
            self.assertTrue(
                (snapshot / "pilot-manifest.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
