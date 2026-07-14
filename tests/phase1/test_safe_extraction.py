from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.safe_extraction import (
    extract_pilot,
    verify_pilot_extraction,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SafeExtractionTests(unittest.TestCase):
    def make_pilot(self, base: Path) -> Path:
        vault = base / "vault"
        snapshot = vault / "raw" / "pilot-v1"
        objects = snapshot / "objects"
        objects.mkdir(parents=True)
        (vault / "derived").mkdir()
        (vault / "temporary").mkdir()
        (vault / "manifests" / "exports").mkdir(parents=True)

        sources = [
            ("json", ".json", b'{"project": "AFM", "status": "complete"}'),
            ("text", ".txt", b"Personal research notes"),
            (
                "contacts",
                ".vcf",
                b"BEGIN:VCARD\r\nFN:Example Person\r\n"
                b"PHOTO;ENCODING=b:AAAA\r\nEND:VCARD\r\n",
            ),
        ]

        items = []
        for index, (family, suffix, content) in enumerate(sources, start=1):
            source_hash = hashlib.sha256(content).hexdigest()
            path = objects / f"{source_hash}{suffix}"
            path.write_bytes(content)
            items.append(
                {
                    "item_index": index,
                    "file_id": f"file-{index}",
                    "original_relative_path": f"source-{index}{suffix}",
                    "filename": f"source-{index}{suffix}",
                    "role": "primary",
                    "family": family,
                    "source_bucket": "test",
                    "year_hint": "2026",
                    "duplicate_control_group": "",
                    "known_contradiction_group": "",
                    "review_notes": "",
                    "size_bytes": len(content),
                    "sha256": source_hash,
                    "object_path": f"objects/{path.name}",
                }
            )

        duplicate = dict(items[0])
        duplicate["item_index"] = 4
        duplicate["file_id"] = "file-4"
        duplicate["original_relative_path"] = "duplicate.json"
        duplicate["filename"] = "duplicate.json"
        duplicate["role"] = "duplicate_control"
        duplicate["duplicate_control_group"] = "dup-1"
        items[0]["role"] = "duplicate_control"
        items[0]["duplicate_control_group"] = "dup-1"
        items.append(duplicate)

        manifest = {
            "pilot_snapshot_schema_version": 1,
            "pilot_name": "pilot-v1",
            "approved_item_count": len(items),
            "unique_content_count": 3,
            "items": items,
        }
        (snapshot / "pilot-manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return vault

    def test_extracts_unique_objects_once_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_pilot(Path(temp))
            summary = extract_pilot(
                vault_root=vault,
                registry_path=ROOT / "policies" / "parser_registry.json",
                fail_on_error=True,
            )
            self.assertEqual(summary["approved_item_records"], 4)
            self.assertEqual(summary["unique_content_objects"], 3)
            self.assertEqual(summary["successful_extractions"], 3)
            self.assertEqual(summary["failed_extractions"], 0)

            verification = verify_pilot_extraction(
                vault_root=vault,
                registry_path=ROOT / "policies" / "parser_registry.json",
            )
            self.assertTrue(verification["ready_for_chunking"])
            self.assertEqual(verification["verified_unique_contents"], 3)

            text_files = list(
                (vault / "derived" / "pilot-v1" / "extracted" / "text")
                .glob("*.txt")
            )
            self.assertEqual(len(text_files), 3)

            resumed = extract_pilot(
                vault_root=vault,
                registry_path=ROOT / "policies" / "parser_registry.json",
            )
            self.assertEqual(resumed["resumed_extractions"], 3)

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_pilot(Path(temp))
            manifest_path = vault / "raw" / "pilot-v1" / "pilot-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            object_path = (
                vault
                / "raw"
                / "pilot-v1"
                / manifest["items"][0]["object_path"]
            )
            object_path.write_bytes(b"tampered")
            summary = extract_pilot(
                vault_root=vault,
                registry_path=ROOT / "policies" / "parser_registry.json",
            )
            self.assertGreater(summary["failed_extractions"], 0)


if __name__ == "__main__":
    unittest.main()
