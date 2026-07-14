from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from alice_vault.analysis import (
    SignatureResult,
    analyze_inventory,
    inspect_zip,
)
from alice_vault.inventory import inventory


class FakeDetector:
    def detect(self, path: Path) -> SignatureResult:
        mapping = {
            "photo.jpg": SignatureResult(
                extension=".jpg",
                mime_type="image/jpeg",
                description="JPEG",
                confidence=0.9,
                status="identified",
            ),
            "wrong.jpg": SignatureResult(
                extension=".pdf",
                mime_type="application/pdf",
                description="PDF",
                confidence=0.9,
                status="identified",
            ),
            "document.docx": SignatureResult(
                extension=".zip",
                mime_type="application/zip",
                description="ZIP",
                confidence=0.8,
                status="identified",
            ),
            "unsafe.zip": SignatureResult(
                extension=".zip",
                mime_type="application/zip",
                description="ZIP",
                confidence=0.8,
                status="identified",
            ),
            "unknown.dat": SignatureResult(status="unknown"),
        }
        return mapping[path.name]


class AnalysisTests(unittest.TestCase):
    def test_zip_inspection_does_not_extract_and_finds_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_path = base / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../../escape.txt", "do not extract")

            result = inspect_zip(archive_path)

            self.assertTrue(result.is_zip)
            self.assertEqual(result.unsafe_path_members, 1)
            self.assertFalse((base.parent / "escape.txt").exists())

    def test_analysis_flags_mismatch_archive_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            vault = base / "vault"
            source.mkdir()

            (source / "photo.jpg").write_bytes(b"jpeg-like")
            (source / "wrong.jpg").write_bytes(b"pdf-like")
            (source / "unknown.dat").write_bytes(b"opaque")

            with zipfile.ZipFile(source / "document.docx", "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
                archive.writestr("word/document.xml", "<document/>")

            with zipfile.ZipFile(source / "unsafe.zip", "w") as archive:
                archive.writestr("../../escape.txt", "do not extract")

            before = {
                path.relative_to(source).as_posix(): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
            }

            inventory(
                source=source,
                vault=vault,
                mode="sha256",
                classification="HIGHLY_SENSITIVE",
            )
            summary = analyze_inventory(
                vault_root=vault,
                detector=FakeDetector(),
                progress_every=0,
            )

            after = {
                path.relative_to(source).as_posix(): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
            }

            self.assertEqual(before, after)
            self.assertEqual(summary["analyzed_files"], 5)
            self.assertEqual(summary["unique_contents"], 5)
            self.assertEqual(summary["mismatch_count"], 1)
            self.assertEqual(summary["quarantine_recommended_count"], 1)
            self.assertGreaterEqual(summary["specialized_review_count"], 2)
            self.assertFalse((base / "escape.txt").exists())

            connection = sqlite3.connect(
                vault / "manifests" / "inventory.sqlite3"
            )
            row = connection.execute(
                """
                SELECT a.detected_extension, a.match_status
                FROM file_analysis a
                JOIN files f ON f.file_id=a.file_id
                WHERE f.filename='document.docx'
                """
            ).fetchone()
            connection.close()

            self.assertEqual(row, (".docx", "match"))


if __name__ == "__main__":
    unittest.main()
