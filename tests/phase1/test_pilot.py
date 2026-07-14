from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.pilot import propose_pilot, scaled_quotas


class PilotProposalTests(unittest.TestCase):
    def _database(self, vault: Path) -> sqlite3.Connection:
        database = vault / "manifests" / "inventory.sqlite3"
        database.parent.mkdir(parents=True)
        con = sqlite3.connect(database)
        con.executescript(
            """
            CREATE TABLE inventory_runs(
                run_id TEXT PRIMARY KEY,
                scan_mode TEXT,
                status TEXT,
                completed_at TEXT
            );
            CREATE TABLE files(
                file_id TEXT PRIMARY KEY,
                run_id TEXT,
                relative_path TEXT,
                filename TEXT,
                size_bytes INTEGER,
                sha256 TEXT,
                duplicate_of TEXT
            );
            CREATE TABLE analysis_runs(
                analysis_run_id TEXT PRIMARY KEY,
                inventory_run_id TEXT,
                completed_at TEXT,
                status TEXT
            );
            CREATE TABLE file_analysis(
                analysis_run_id TEXT,
                file_id TEXT,
                content_key TEXT,
                claimed_extension TEXT,
                detected_extension TEXT,
                match_status TEXT,
                risk_flags_json TEXT,
                recommendation TEXT
            );
            """
        )
        con.execute(
            "INSERT INTO inventory_runs VALUES('inv','sha256','complete','2026-01-01')"
        )
        con.execute(
            "INSERT INTO analysis_runs VALUES('analysis','inv','2026-01-02','complete')"
        )
        return con

    def _add(
        self,
        con: sqlite3.Connection,
        index: int,
        extension: str,
        *,
        duplicate_of: str | None = None,
        sha: str | None = None,
        path_prefix: str = "source",
        recommendation: str = "pilot_candidate",
        risks: str = "[]",
        filename: str | None = None,
    ) -> str:
        file_id = f"f{index}"
        digest = sha or f"sha{index}"
        name = filename or f"file-{index}{extension}"
        path = f"{path_prefix}/202{index % 6}/{name}"
        con.execute(
            "INSERT INTO files VALUES(?,?,?,?,?,?,?)",
            (file_id, "inv", path, name, 100 + index, digest, duplicate_of),
        )
        con.execute(
            "INSERT INTO file_analysis VALUES(?,?,?,?,?,?,?,?)",
            (
                "analysis", file_id, f"{digest}:{100 + index}",
                extension, extension, "match", risks, recommendation,
            ),
        )
        return file_id

    def test_scaled_quotas_sum_to_target(self) -> None:
        self.assertEqual(sum(scaled_quotas(110).values()), 110)
        self.assertEqual(sum(scaled_quotas(73).values()), 73)

    def test_proposal_is_balanced_private_and_review_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            con = self._database(vault)
            extensions = [
                ".json", ".html", ".csv", ".txt", ".pdf", ".docx",
                ".xlsx", ".pptx", ".ics", ".vcf", ".srt", ".xml",
            ]
            index = 1
            for extension in extensions:
                for _ in range(15):
                    self._add(
                        con, index, extension,
                        path_prefix=f"bucket-{index % 8}",
                    )
                    index += 1

            # Five exact duplicate groups, each with two different paths.
            for group in range(5):
                first_index = index
                first = self._add(
                    con, index, ".json", sha=f"dup-{group}",
                    path_prefix=f"dup-a-{group}",
                )
                index += 1
                # Match content_key by matching both sha and size.
                second_id = f"f{index}"
                size = 100 + first_index
                name = f"copy-{group}.json"
                con.execute(
                    "INSERT INTO files VALUES(?,?,?,?,?,?,?)",
                    (
                        second_id, "inv", f"dup-b-{group}/{name}", name,
                        size, f"dup-{group}", first,
                    ),
                )
                con.execute(
                    "INSERT INTO file_analysis VALUES(?,?,?,?,?,?,?,?)",
                    (
                        "analysis", second_id, f"dup-{group}:{size}",
                        ".json", ".json", "match", "[]",
                        "pilot_candidate",
                    ),
                )
                # Correct first content_key to the shared size.
                con.execute(
                    "UPDATE file_analysis SET content_key=? WHERE file_id=?",
                    (f"dup-{group}:{size}", first),
                )
                index += 1

            # Must be excluded by sensitive-path screening.
            self._add(
                con, index, ".pdf", filename="passport-copy.pdf",
                path_prefix="private",
            )
            con.commit()
            con.close()

            summary = propose_pilot(
                vault_root=vault,
                target_total=60,
                duplicate_groups=5,
                selection_seed="test-seed",
            )

            self.assertEqual(summary["selected_count"], 60)
            self.assertEqual(summary["primary_selected"], 50)
            self.assertEqual(summary["duplicate_control_files_selected"], 10)
            self.assertGreaterEqual(summary["sensitive_path_excluded_count"], 1)
            self.assertLess(summary["selected_gib"], 2)
            self.assertEqual(summary["pending_human_review_count"], 60)
            self.assertEqual(summary["warnings"], [])

            proposal = Path(summary["proposal_path"])
            with proposal.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 60)
            self.assertTrue(all(row["decision"] == "pending" for row in rows))
            self.assertFalse(any("passport" in row["relative_path"] for row in rows))
            control_groups = {
                row["duplicate_control_group"]
                for row in rows
                if row["role"] == "duplicate_control"
            }
            self.assertEqual(len(control_groups), 5)


if __name__ == "__main__":
    unittest.main()
