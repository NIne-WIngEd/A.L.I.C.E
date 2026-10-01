"""Exact original v1.6 train seed custody, without invented rights."""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import canonical_json_bytes
from cognitive_kernel.formation_dataset_admission import (
    AdmittedCase, CorpusAdmission, PayloadRef,
)
from cognitive_kernel.formation_learning_v16 import admitted_rows_v16
from scripts.mfm import materialize_v16_authoring_seed_train as materializer


class SeedTrainMaterializationTests(unittest.TestCase):
    def test_exact_frozen_train_rows_and_target_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "owner-private"
            result = materializer.materialize(output)
            receipt = json.loads(Path(result["receipt_path"]).read_bytes())
            self.assertEqual(result["train_cases"], 15)
            self.assertEqual(result["receipt_sha256"], sha256(
                (output / "receipt.json").read_bytes()).hexdigest())
            self.assertEqual(receipt["frozen_corpus_sha256"], materializer.CORPUS_SHA256)
            self.assertEqual(receipt["generator_family"],
                             "assistant-authored-v16-seed-20260930")
            self.assertEqual(len(receipt["cases"]), 15)
            self.assertFalse(any(path.name.startswith("rights-issued") for path in output.rglob("*")))
            for item in receipt["cases"]:
                candidate_bytes = (output / item["candidate"]["path"]).read_bytes()
                candidate = json.loads(candidate_bytes)
                self.assertEqual(candidate["split"], "train")
                self.assertEqual(candidate["status"]["qualification"],
                                 "diagnostic-only-unadmitted")
                self.assertEqual(candidate["authorization_id"], materializer.AUTHORIZATION)
                self.assertEqual(item["parent_case_ids"],
                                 candidate["lineage"]["parent_case_ids"])
                original_candidate = (output / item["original_candidate"]["path"]).read_bytes()
                original = json.loads(original_candidate)
                self.assertEqual(item["original_candidate"]["sha256"],
                                 sha256(original_candidate).hexdigest())
                if item["case_id"] == "v16-seed-03":
                    self.assertNotEqual(item["candidate"]["sha256"],
                                        item["original_candidate"]["sha256"])
                    self.assertEqual(item["source_ref_remapping"], {
                        "v16-pair-temperament-evidence-flip-later":
                        "v16-seed-03-temperament-later"})
                    self.assertEqual(original["sources"][1]["content_b64"],
                                     candidate["sources"][1]["content_b64"])
                    self.assertEqual(original["target"]["adjudications"],
                                     candidate["target"]["adjudications"])
                else:
                    self.assertEqual(item["candidate"]["sha256"],
                                     item["original_candidate"]["sha256"])
                self.assertEqual(len(item["sources"]), len(candidate["sources"]))
                self.assertEqual(len(item["rights_drafts"]), len(candidate["sources"]))
                for source, opened, right in zip(item["sources"], candidate["sources"],
                                                item["rights_drafts"], strict=True):
                    self.assertEqual(source["source_id"], opened["ref_id"])
                    self.assertEqual((output / source["path"]).read_bytes(),
                                     b64decode(opened["content_b64"]))
                    draft = json.loads((output / right["path"]).read_bytes())
                    self.assertTrue(draft["draft_unissued"])
                    self.assertIsNone(draft["issuer_id"])
                    self.assertIsNone(draft["authority_ref"])
                    self.assertIsNone(draft["formation_training"])
                    self.assertIsNone(draft["model_distribution"])
                    self.assertIsNone(draft["revoked"])
                    self.assertEqual(draft["source_id"], source["source_id"])
                    self.assertEqual(draft["source_sha256"], source["sha256"])
                target = json.loads((output / item["generator_output"]["path"]).read_bytes())
                self.assertEqual(target, {
                    "schema": "mfm-formation-target-v1.6", "context": candidate["context"],
                    "source_ids": [source["ref_id"] for source in candidate["sources"]],
                    **{field: candidate["target"][field] for field in (
                        "proposals", "dispositions", "adjudications")}})
                self.assertEqual((output / item["generator_output"]["path"]).read_bytes(),
                                 canonical_json_bytes(target) + b"\n")
                provenance = json.loads((output / item["converter_provenance"]["path"]).read_bytes())
                self.assertEqual(provenance["case_sha256"], item["candidate"]["sha256"])
                self.assertEqual(provenance["generator_output_sha256"],
                                 item["generator_output"]["sha256"])
                self.assertEqual(provenance["rendered_prompt_sha256"],
                                 item["generator_input"]["sha256"])
                self.assertEqual(provenance["converter_sha256"],
                                 receipt["target_converter"]["sha256"])
            self.assertNotIn("v16-seed-13", [row["case_id"] for row in receipt["cases"]])
            self.assertFalse(receipt["source_rights_created"])
            self.assertTrue(receipt["pending_rights_drafts_created"])
            self.assertFalse(receipt["qualified_for_product"])

    def test_shared_exact_source_path_is_audited_once_but_read_for_both_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "owner-private"
            materializer.materialize(output)
            receipt = json.loads((output / "receipt.json").read_bytes())
            rows = [item for item in receipt["cases"] if item["case_id"] in {
                "v16-seed-19", "v16-seed-20"}]
            self.assertEqual(rows[0]["sources"][0]["path"], rows[1]["sources"][0]["path"])
            cases = []
            for row in rows:
                rights_refs = []
                for index, source in enumerate(row["sources"]):
                    raw = canonical_json_bytes({
                        "schema": "mfm-source-rights-v1", "issuer_id": "unit-only",
                        "authority_ref": "fictional-not-issued",
                        "host_family": row["host_family"],
                        "source_id": source["source_id"],
                        "source_sha256": source["sha256"],
                        "formation_training": True, "formation_evaluation": True,
                        "model_distribution": True, "revoked": False}) + b"\n"
                    path = output / "unit-only" / row["case_id"] / f"rights-{index}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                    rights_refs.append(PayloadRef(str(path.relative_to(output)),
                                                  sha256(raw).hexdigest()))
                cases.append(AdmittedCase(
                    row["case_id"], "train",
                    tuple(PayloadRef(s["path"], s["sha256"]) for s in row["sources"]),
                    PayloadRef(row["generator_output"]["path"],
                               row["generator_output"]["sha256"]),
                    tuple(rights_refs)))
            admission = CorpusAdmission("unit-only", "f" * 64, tuple(cases), (), (), output)
            self.assertEqual(len(tuple(admitted_rows_v16(admission, split="train"))), 2)

    def test_wrong_frozen_pin_or_existing_directory_fails_before_emitting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private"
            with patch.object(materializer, "CORPUS_SHA256", "0" * 64):
                with self.assertRaisesRegex(ValueError, "differs from pinned"):
                    materializer.materialize(output)
            self.assertFalse(output.exists())
            materializer.materialize(output)
            with self.assertRaisesRegex(ValueError, "new private output directory"):
                materializer.materialize(output)


if __name__ == "__main__":
    unittest.main()
