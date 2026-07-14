from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.inventory import inventory


class InventoryTests(unittest.TestCase):
    def test_hash_inventory_duplicate_and_source_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            vault = base / "vault"
            (source / "nested").mkdir(parents=True)
            (source / "a.txt").write_bytes(b"same")
            (source / "nested" / "b.txt").write_bytes(b"same")
            (source / "c.bin").write_bytes(b"different")
            before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}

            result = inventory(source, vault, "sha256")

            self.assertEqual(result["file_count"], 3)
            self.assertEqual(result["exact_duplicate_count"], 1)
            self.assertEqual(result["error_count"], 0)
            after = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

            con = sqlite3.connect(vault / "manifests" / "inventory.sqlite3")
            statuses = [row[0] for row in con.execute("SELECT scan_status FROM files")]
            con.close()
            self.assertEqual(statuses.count("exact_duplicate"), 1)

    def test_metadata_mode_does_not_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            source.mkdir()
            (source / "a.txt").write_text("hello", encoding="utf-8")
            result = inventory(source, base / "vault", "metadata")
            self.assertEqual(result["hashed_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
