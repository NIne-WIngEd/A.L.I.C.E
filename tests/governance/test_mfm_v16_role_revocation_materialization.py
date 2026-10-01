"""One exact revoked-source candidate; no issued rights."""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.mfm import materialize_v16_role_revocation_train as materializer


class RoleRevocationMaterializationTests(unittest.TestCase):
    def test_source_and_target_exact_without_original_revoked_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "owner-private"
            result = materializer.materialize(output)
            receipt = json.loads(Path(result["receipt_path"]).read_bytes())
            candidate = json.loads((output / receipt["candidate"]["path"]).read_bytes())
            self.assertEqual(receipt["original_corpus_sha256"],
                             materializer.ORIGINAL_CORPUS_SHA256)
            self.assertEqual(candidate["case_id"], materializer.CASE_ID)
            self.assertEqual(candidate["authorization_id"],
                             "owner-directed-mfm-v16-synthetic")
            self.assertEqual(receipt["generator_family"],
                             "codex-fictional-role-chronicle-20261001")
            self.assertEqual({row["proposal"]["kind"]
                              for row in candidate["target"]["proposals"]},
                             {"host_observation", "revocation_request", "deletion_request"})
            self.assertEqual(len(receipt["sources"]), len(candidate["sources"]))
            for source, draft, opened in zip(receipt["sources"], receipt["rights_drafts"],
                                             candidate["sources"], strict=True):
                text = (output / source["path"]).read_bytes()
                self.assertEqual(text, b64decode(opened["content_b64"]))
                self.assertNotIn(b"I checked the harbor tide chart before each", text)
                rights = json.loads((output / draft["path"]).read_bytes())
                self.assertEqual(rights["source_id"], source["source_id"])
                self.assertEqual(rights["source_sha256"], sha256(text).hexdigest())
                self.assertIsNone(rights["issuer_id"])
                self.assertIsNone(rights["model_distribution"])
                self.assertTrue(rights["draft_unissued"])
            provenance = json.loads((output / receipt["converter_provenance"]["path"]).read_bytes())
            self.assertEqual(provenance["case_sha256"], receipt["candidate"]["sha256"])
            self.assertEqual(provenance["generator_output_sha256"],
                             receipt["generator_output"]["sha256"])
            self.assertFalse(receipt["source_rights_created"])
            self.assertFalse(receipt["qualified_for_product"])

    def test_frozen_generator_pin_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private"
            with patch.object(materializer, "ORIGINAL_CORPUS_SHA256", "0" * 64):
                with self.assertRaisesRegex(ValueError, "differs from pin"):
                    materializer.materialize(output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
