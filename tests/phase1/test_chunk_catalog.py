from __future__ import annotations
import hashlib, json, sqlite3, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.chunk_catalog import build_pilot_chunks, verify_pilot_chunks

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class ChunkCatalogTests(unittest.TestCase):
    def make_vault(self, base: Path) -> Path:
        vault = base / "vault"
        snapshot = vault / "raw" / "pilot-v1"
        objects = snapshot / "objects"
        extracted = vault / "derived" / "pilot-v1" / "extracted"
        for path in (objects, extracted / "text", extracted / "metadata", vault / "temporary", vault / "manifests" / "exports"):
            path.mkdir(parents=True, exist_ok=True)
        items = []
        contents = [
            ("json", ".json", ("Project research progress. " * 300).strip(), False),
            ("text", ".txt", ("Personal education timeline. " * 240).strip(), True),
        ]
        for index, (family, suffix, text, truncated) in enumerate(contents, 1):
            raw = text.encode()
            source_sha = hashlib.sha256(raw).hexdigest()
            (objects / f"{source_sha}{suffix}").write_bytes(raw)
            text_path = extracted / "text" / f"{source_sha}.txt"
            text_path.write_text(text, encoding="utf-8")
            metadata = {
                "status": "success", "run_id": "extract-run", "source_sha256": source_sha,
                "text_sha256": digest(text_path), "parser_id": f"{family}-test-v1",
                "registry_digest": "r" * 64, "truncated": truncated,
                "warnings": ["output_character_limit_reached"] if truncated else [],
            }
            (extracted / "metadata" / f"{source_sha}.json").write_text(json.dumps(metadata), encoding="utf-8")
            items.append({
                "item_index": index, "file_id": f"file-{index}",
                "original_relative_path": f"source-{index}{suffix}", "filename": f"source-{index}{suffix}",
                "role": "primary", "family": family, "source_bucket": f"bucket-{index}",
                "year_hint": str(2024 + index), "duplicate_control_group": "",
                "known_contradiction_group": "project-status" if index == 1 else "",
                "size_bytes": len(raw), "sha256": source_sha,
                "object_path": f"objects/{source_sha}{suffix}",
            })
        duplicate = dict(items[0])
        duplicate.update({"item_index": 3, "file_id": "file-3", "original_relative_path": "duplicate.json", "filename": "duplicate.json", "role": "duplicate_control", "duplicate_control_group": "dup-1"})
        items[0]["role"] = "duplicate_control"
        items[0]["duplicate_control_group"] = "dup-1"
        items.append(duplicate)
        (snapshot / "pilot-manifest.json").write_text(json.dumps({"pilot_snapshot_schema_version": 1, "pilot_name": "pilot-v1", "approved_item_count": 3, "unique_content_count": 2, "items": items}, indent=2), encoding="utf-8")
        return vault

    def test_build_verify_resume_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_vault(Path(temp))
            policy = ROOT / "policies" / "chunking_policy.json"
            first = build_pilot_chunks(vault, policy_path=policy)
            self.assertTrue(first["successful_chunk_build"])
            self.assertEqual(first["source_count"], 2)
            self.assertEqual(first["duplicate_provenance_paths"], 1)
            self.assertEqual(first["truncated_source_count"], 1)
            verification = verify_pilot_chunks(vault, policy_path=policy)
            self.assertTrue(verification["ready_for_indexing"])
            self.assertTrue(verification["deterministic_rebuild_match"])
            second = build_pilot_chunks(vault, policy_path=policy)
            self.assertTrue(second["resumed_existing_chunk_set"])
            self.assertEqual(second["manifest_fingerprint"], first["manifest_fingerprint"])
            con = sqlite3.connect(Path(first["output_root"]) / "chunks.sqlite3")
            provenance_count = con.execute("SELECT COUNT(*) FROM chunk_provenance").fetchone()[0]
            con.close()
            self.assertGreater(provenance_count, first["chunk_count"])

    def test_tampered_chunk_fails_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_vault(Path(temp))
            policy = ROOT / "policies" / "chunking_policy.json"
            summary = build_pilot_chunks(vault, policy_path=policy)
            chunk_file = next((Path(summary["output_root"]) / "text").glob("*.txt"))
            chunk_file.chmod(0o644)
            chunk_file.write_text("tampered", encoding="utf-8")
            verification = verify_pilot_chunks(vault, policy_path=policy)
            self.assertFalse(verification["ready_for_indexing"])
            self.assertGreater(verification["error_count"], 0)

if __name__ == "__main__":
    unittest.main()
