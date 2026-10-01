"""Fictional-source rights issue is explicit, narrow and not an owner signature."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import canonical_json_bytes
from scripts.mfm.issue_v16_synthetic_teacher_corpus import (
    ATTESTATION_METHOD, AUTHORIZATION, ISSUER, OWNER_AUTHORIZATION_DATE,
    _issued_rights, issue_and_assemble,
)


def write(root: Path, name: str, data: bytes) -> dict:
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
    return {"path": name, "sha256": sha256(data).hexdigest()}


class SyntheticRightsIssuanceTests(unittest.TestCase):
    def _fixture(self, root: Path, *, development=False):
        source = {"source_id": "fictional-unit-source",
                  **write(root, "source.txt", b"I made up this test source."),
                  "parent_source_ids": []}
        if development:
            draft = {"schema": "mfm-source-rights-unissued-draft-v1",
                     "status": "unissued-not-admissible"}
        else:
            draft = {"schema": "mfm-source-rights-v1",
                     "draft_unissued": True}
        draft.update({"source_id": source["source_id"],
                      "source_sha256": source["sha256"],
                      "host_family": "fictional-unit-host",
                      "issuer_id": None, "authority_ref": None,
                      "formation_training": None, "formation_evaluation": None,
                      "model_distribution": None, "revoked": None})
        receipt = write(root, "draft.json", canonical_json_bytes(draft) + b"\n")
        return source, receipt

    def test_train_and_development_are_differently_scoped(self):
        for development in (False, True):
            with self.subTest(development=development), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, draft = self._fixture(root, development=development)
                original = (root / draft["path"]).read_bytes()
                issued, binding = _issued_rights(
                    root, source, draft, host="fictional-unit-host",
                    split="development" if development else "train",
                    issued_at_utc="2026-10-01T22:00:00Z")
                self.assertEqual((root / draft["path"]).read_bytes(), original)
                self.assertEqual(binding["source_id"], source["source_id"])
                self.assertEqual(issued["issuer_id"], ISSUER)
                self.assertEqual(issued["authority_ref"], AUTHORIZATION)
                self.assertEqual(issued["attestation_method"], ATTESTATION_METHOD)
                self.assertEqual(issued["owner_authorization_date"], OWNER_AUTHORIZATION_DATE)
                self.assertEqual(issued["issued_at_utc"], "2026-10-01T22:00:00Z")
                self.assertEqual(issued["source_origin"],
                                 "project-authored-fictional-synthetic")
                self.assertTrue(issued["owner_statement_is_simulated"])
                self.assertFalse(issued["owner_signature_claimed"])
                self.assertFalse(issued["independent_rights_review_claimed"])
                self.assertEqual(issued["formation_training"], not development)
                self.assertEqual(issued["model_distribution"], not development)
                self.assertEqual(issued["formation_evaluation"], development)

    def test_tamper_and_premature_rights_assertion_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, draft = self._fixture(root)
            draft["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "exact bytes"):
                _issued_rights(root, source, draft, host="fictional-unit-host",
                               split="train", issued_at_utc="2026-10-02T01:00:00Z")
            draft["sha256"] = sha256((root / draft["path"]).read_bytes()).hexdigest()
            doc = json.loads((root / draft["path"]).read_bytes())
            doc["formation_training"] = True
            draft = write(root, "draft.json", canonical_json_bytes(doc) + b"\n")
            with self.assertRaisesRegex(ValueError, "asserted permission"):
                _issued_rights(root, source, draft, host="fictional-unit-host",
                               split="train", issued_at_utc="2026-10-02T01:00:00Z")

    def test_failed_issuance_does_not_expose_partial_rights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "final"

            def fail_after_write(*, output_dir, **_kwargs):
                (output_dir / "issued-rights").mkdir()
                (output_dir / "issued-rights" / "partial.json").write_text("partial")
                raise ValueError("invalid source byte")

            with patch("scripts.mfm.issue_v16_synthetic_teacher_corpus._issue_staged",
                       side_effect=fail_after_write):
                with self.assertRaisesRegex(ValueError, "invalid source byte"):
                    issue_and_assemble(
                        seed_bundle=root, seed_receipt_sha256="0" * 64,
                        role_bundle=root, role_receipt_sha256="0" * 64,
                        development_bundle=root, development_index_sha256="0" * 64,
                        output_dir=output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
