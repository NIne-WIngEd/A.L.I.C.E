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
    ArchiveResult,
    PureMagicDetector,
    SignatureResult,
    analyze_inventory,
    classify_match,
    detect_iso_bmff,
    extensions_equivalent,
    inspect_zip,
    risk_and_recommendation,
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
    def test_format_family_and_opaque_calibration(self) -> None:
        self.assertTrue(extensions_equivalent(".jpg", ".jfif"))
        self.assertTrue(extensions_equivalent(".ipynb", ".json"))
        self.assertTrue(extensions_equivalent(".atom", ".xml"))
        self.assertEqual(
            classify_match(".dat", ".png", "identified"),
            "opaque_identified",
        )
        self.assertEqual(
            classify_match(".pth", ".zip", "identified"),
            "serialized_container",
        )
        self.assertEqual(
            classify_match(".txt", "", "empty"),
            "empty",
        )

    def test_iso_bmff_header_overrides_ambiguous_magic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video.mp4"
            path.write_bytes(
                b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
            )
            result = detect_iso_bmff(path)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.extension, ".mp4")
            self.assertEqual(result.status, "identified")

    def test_empty_file_is_not_a_signature_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty.txt"
            path.write_bytes(b"")
            result = object.__new__(PureMagicDetector).detect(path)
            self.assertEqual(result.status, "empty")
            self.assertIsNone(result.error)


    def test_ooxml_package_is_pilot_candidate_not_generic_archive(self) -> None:
        flags, recommendation = risk_and_recommendation(
            claimed_extension=".docx",
            detected_extension=".docx",
            match_status="match",
            archive=ArchiveResult(
                is_zip=True,
                detected_extension=".docx",
                member_count=2,
                file_member_count=2,
            ),
            size_bytes=1024,
        )
        self.assertNotIn("archive_container", flags)
        self.assertEqual(recommendation, "pilot_candidate")

        zip_flags, zip_recommendation = risk_and_recommendation(
            claimed_extension=".zip",
            detected_extension=".zip",
            match_status="match",
            archive=ArchiveResult(is_zip=True, detected_extension=".zip"),
            size_bytes=1024,
        )
        self.assertIn("archive_container", zip_flags)
        self.assertEqual(zip_recommendation, "specialized_review")

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
            self.assertGreaterEqual(summary["specialized_review_count"], 1)
            self.assertFalse((base / "escape.txt").exists())

            connection = sqlite3.connect(
                vault / "manifests" / "inventory.sqlite3"
            )
            row = connection.execute(
                """
                SELECT a.detected_extension, a.match_status, a.recommendation
                FROM file_analysis a
                JOIN files f ON f.file_id=a.file_id
                WHERE f.filename='document.docx'
                """
            ).fetchone()
            connection.close()

            self.assertEqual(row, (".docx", "match", "pilot_candidate"))


if __name__ == "__main__":
    unittest.main()
