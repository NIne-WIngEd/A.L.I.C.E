"""The paid MFM run must load exactly the CPU-staged pinned model bytes."""

from __future__ import annotations

from hashlib import sha1, sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.mfm import stage_gemma4_model as stage


class StagedGemmaModelTests(unittest.TestCase):
    def _upstream(self, root: Path):
        snapshot = root / "snapshot"
        snapshot.mkdir()
        siblings = {}
        metadata = {}
        for name in sorted(stage._ESSENTIAL_FILES | {".gitattributes", "README.md"}):
            content = f"pinned:{name}\n".encode()
            (snapshot / name).write_bytes(content)
            blob = sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            # Simulate Xet-backed content with no LFS object exposed by
            # model_info() or list_repo_tree(), requiring resolve HEAD ETag.
            siblings[name] = types.SimpleNamespace(
                rfilename=name, path=name, size=len(content), lfs=None, blob_id=blob)
            metadata[name] = types.SimpleNamespace(
                etag=sha256(content).hexdigest()
                if name in {"model.safetensors", "tokenizer.json"} else blob,
                commit_hash=stage.GEMMA_4_12B_REVISION, size=len(content))

        class FakeApi:
            def model_info(self, model, *, revision, files_metadata):
                self.assert_pinned(model, revision, files_metadata)
                return types.SimpleNamespace(
                    sha=stage.GEMMA_4_12B_REVISION, siblings=list(siblings.values()))

            def list_repo_tree(self, model, *, revision, recursive, expand):
                self.assert_pinned(model, revision, recursive and expand)
                return iter(siblings.values())

            @staticmethod
            def assert_pinned(model, revision, flag):
                assert model == stage.GEMMA_4_12B_MODEL
                assert revision == stage.GEMMA_4_12B_REVISION
                assert flag

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        fake_hub.get_hf_file_metadata = lambda url: metadata[url]
        fake_hub.hf_hub_url = lambda *, repo_id, filename, revision: filename
        fake_hub.snapshot_download = lambda **kw: self.fail(
            "an existing local snapshot must not be downloaded again")
        return snapshot, metadata, fake_hub

    def test_xet_fallback_seals_and_verifies_full_and_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, metadata, fake_hub = self._upstream(root)
            receipt_path = root / "model-receipt.json"
            with patch.object(stage, "PINNED_WEIGHT_SHA256",
                              metadata["model.safetensors"].etag), \
                 patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                sealed = stage.stage_model(snapshot, receipt_path)
                self.assertEqual(sealed["revision"], stage.GEMMA_4_12B_REVISION)
                self.assertEqual(sealed["files"][-2]["path"], "tokenizer.json")
                self.assertEqual(sealed["snapshot_path"], str(snapshot))
                self.assertEqual(stage.verify_staged_model(
                    receipt_path, stage.GEMMA_4_12B_MODEL,
                    stage.GEMMA_4_12B_REVISION), (snapshot, sealed))
                self.assertEqual(stage.verify_staged_model(
                    receipt_path, stage.GEMMA_4_12B_MODEL,
                    stage.GEMMA_4_12B_REVISION, rehash=False), (snapshot, sealed))
                relocated = root / "relocated"
                shutil.copytree(snapshot, relocated)
                self.assertEqual(stage.verify_staged_model(
                    receipt_path, stage.GEMMA_4_12B_MODEL,
                    stage.GEMMA_4_12B_REVISION, snapshot_override=relocated),
                    (relocated, sealed))
                # A Windows staging receipt can be copied to a Linux CPU node.
                # Its original absolute path is not a Linux absolute path.
                windows_receipt = dict(sealed, snapshot_path=r"C:\mfm\gemma4-12b-it")
                unsigned = {key: value for key, value in windows_receipt.items()
                            if key != "receipt_sha256"}
                windows_receipt["receipt_sha256"] = sha256(
                    stage._canonical(unsigned)).hexdigest()
                receipt_path.write_text(json.dumps(windows_receipt))
                self.assertEqual(stage.verify_staged_model(
                    receipt_path, stage.GEMMA_4_12B_MODEL,
                    stage.GEMMA_4_12B_REVISION, snapshot_override=relocated),
                    (relocated, windows_receipt))
                with self.assertRaisesRegex(stage.StagedModelError, "not absolute"):
                    stage.verify_staged_model(
                        receipt_path, stage.GEMMA_4_12B_MODEL,
                        stage.GEMMA_4_12B_REVISION)
                with self.assertRaisesRegex(stage.StagedModelError, "already exists"):
                    stage.stage_model(snapshot, receipt_path)

    def test_tampering_and_extra_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, metadata, fake_hub = self._upstream(root)
            receipt_path = root / "model-receipt.json"
            with patch.object(stage, "PINNED_WEIGHT_SHA256",
                              metadata["model.safetensors"].etag), \
                 patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                stage.stage_model(snapshot, receipt_path)
                item = snapshot / "tokenizer_config.json"
                item.write_bytes(b"X" * item.stat().st_size)
                # Other ranks use fast structural checks only; rank 0 must
                # rehash all bytes before they proceed to GPU work.
                stage.verify_staged_model(
                    receipt_path, stage.GEMMA_4_12B_MODEL,
                    stage.GEMMA_4_12B_REVISION, rehash=False)
                with self.assertRaisesRegex(stage.StagedModelError, "file changed"):
                    stage.verify_staged_model(
                        receipt_path, stage.GEMMA_4_12B_MODEL,
                        stage.GEMMA_4_12B_REVISION)
                item.write_bytes(b"longer")
                with self.assertRaisesRegex(stage.StagedModelError, "size changed"):
                    stage.verify_staged_model(
                        receipt_path, stage.GEMMA_4_12B_MODEL,
                        stage.GEMMA_4_12B_REVISION, rehash=False)
                item.write_bytes(b"pinned:tokenizer_config.json\n")
                (snapshot / "unexpected.py").write_text("bad")
                with self.assertRaisesRegex(stage.StagedModelError, "added, removed"):
                    stage.verify_staged_model(
                        receipt_path, stage.GEMMA_4_12B_MODEL,
                        stage.GEMMA_4_12B_REVISION, rehash=False)

    def test_self_digest_and_weight_anchor_are_distinct_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, metadata, fake_hub = self._upstream(root)
            receipt_path = root / "model-receipt.json"
            with patch.object(stage, "PINNED_WEIGHT_SHA256",
                              metadata["model.safetensors"].etag), \
                 patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                stage.stage_model(snapshot, receipt_path)
                record = json.loads(receipt_path.read_text())
                record["revision"] = "f" * 40
                receipt_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(stage.StagedModelError, "digest mismatch"):
                    stage.verify_staged_model(
                        receipt_path, stage.GEMMA_4_12B_MODEL,
                        stage.GEMMA_4_12B_REVISION)
                record["revision"] = stage.GEMMA_4_12B_REVISION
                for item in record["files"]:
                    if item["path"] == "model.safetensors":
                        item["sha256"] = "f" * 64
                        item["upstream"]["digest"] = "f" * 64
                unsigned = {key: value for key, value in record.items()
                            if key != "receipt_sha256"}
                record["receipt_sha256"] = sha256(stage._canonical(unsigned)).hexdigest()
                receipt_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(stage.StagedModelError, "pinned anchor"):
                    stage.verify_staged_model(
                        receipt_path, stage.GEMMA_4_12B_MODEL,
                        stage.GEMMA_4_12B_REVISION, rehash=False)


if __name__ == "__main__":
    unittest.main()
