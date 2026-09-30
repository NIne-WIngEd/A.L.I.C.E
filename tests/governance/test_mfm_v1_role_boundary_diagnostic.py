from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from urllib import request

from scripts.mfm.evaluate_base_behavior import DiagnosticError, _input_bytes, load_diagnostic
from cognitive_kernel.formation_learning import OUTPUT_SCHEMA
from scripts.mfm.qualify_v1_role_boundary import (
    SOURCE_SHA256, assess, verify_local_lineage,
)


class RoleBoundaryDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_diagnostic()
        cls.prompt_digest = sha256(_input_bytes(cls.cases)).hexdigest()
        cls.receipt_digest = "a" * 64

    @staticmethod
    def lineage():
        return {"run_manifest_sha256": "1" * 64,
                "prepared_base_parent_sha256": SOURCE_SHA256,
                "prepared_base_sha256": SOURCE_SHA256,
                "prepared_base_receipt_sha256": "e" * 64,
                "max_target_tokens": 8192,
                "trained": {"weight_sha256": "b" * 64,
                            "receipt_sha256": "c" * 64},
                "seeded-untrained": {"weight_sha256": "f" * 64,
                                     "receipt_sha256": "d" * 64}}

    def rows(self, *, reference: bool = True) -> dict[str, dict[str, dict]]:
        roles = ("trained", "seeded-untrained", "untouched") if reference else (
            "trained", "seeded-untrained")
        result = {role: {} for role in roles}
        lineage = self.lineage()
        for case_id, (gold, _, _) in self.cases.items():
            answer = json.dumps({"schema": OUTPUT_SCHEMA,
                                 "proposals": [p.record() for p in gold.expected],
                                 "dispositions": [d.record() for d in gold.expected_dispositions]})
            modalities = sorted({ref.modality for ref in gold.context.evidence})
            for role in roles:
                row = {"case_id": case_id, "context_digest": gold.context.content_digest(),
                       "prompt_set_sha256": self.prompt_digest,
                       "source_repository": "google/gemma-4-12B",
                       "source_revision": "023679ed352de9bb66cc873c9009ce3482585c08",
                       "status": "generated", "output_text": answer,
                       "processed_modalities": modalities}
                if role == "untouched":
                    row.update(generation={"do_sample": False, "num_beams": 1,
                                           "max_new_tokens": 1024, "seed": 20260930,
                                           "attention_implementation": "eager", "dtype": "bfloat16"},
                               model_artifact_digest=SOURCE_SHA256,
                               model_receipt=self.receipt_digest)
                    if case_id == "blank-image-no-biography":
                        row.update(status="unexercised", output_text=None,
                                   processed_modalities=["text"])
                else:
                    row.update(base_source_sha256=SOURCE_SHA256,
                               prepared_base_parent_sha256=SOURCE_SHA256,
                               prepared_base_sha256=SOURCE_SHA256,
                               prepared_base_receipt_sha256="e" * 64,
                               formation_component_sha256=lineage[role]["weight_sha256"],
                               model_artifact_digest=lineage[role]["weight_sha256"],
                               model_receipt=lineage[role]["receipt_sha256"],
                               control_kind=role, raw_output_text=answer,
                               generated_token_ids=[123, 456], eos_observed=True,
                               validation_status="grounded_proposal_only",
                               validation_error=None,
                               generation={"do_sample": False, "max_new_tokens": 8192,
                                           "decoder": "specialist-greedy-bos-eos-v1"})
                result[role][case_id] = row
        return result

    def test_real_control_shape_and_reference_coverage_are_separate(self):
        rows = self.rows()
        seed = rows["seeded-untrained"]["neutral-owner-event"]
        seed.update(output_text="As an AI language model, I remember Ada.",
                    raw_output_text="As an AI language model, I remember Ada.",
                    validation_status="invalid", validation_error="invalid JSON")
        with patch.object(socket, "create_connection", side_effect=AssertionError("network")), \
                patch.object(request, "urlopen", side_effect=AssertionError("network")):
            unverified = assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
            report = assess(self.cases, rows, source_receipt_sha256=self.receipt_digest,
                            verified_lineage=self.lineage())
        self.assertFalse(unverified["role_boundary_diagnostic_pass"])
        self.assertFalse(unverified["artifact_binding_verified"])
        self.assertIn("neutral-owner-event", report["formation_contribution_cases"])
        self.assertEqual(report["coverage_gaps"], [])
        self.assertIn("blank-image-no-biography:publisher_reference_unexercised",
                      report["reference_coverage_gaps"])
        image = next(row for row in report["cases"]
                     if row["case_id"] == "blank-image-no-biography")
        self.assertEqual(image["trained"]["status"], "generated")
        self.assertEqual(image["seeded-untrained"]["status"], "generated")
        self.assertTrue(report["role_boundary_diagnostic_pass"])
        self.assertFalse(report["qualification_claim"])
        self.assertIn("inference_execution_authenticity", report["unmeasured"])

    def test_without_reference_and_without_effect(self):
        rows = self.rows(reference=False)
        report = assess(self.cases, rows, source_receipt_sha256=self.receipt_digest,
                        verified_lineage=self.lineage())
        self.assertEqual(report["formation_contribution_cases"], [])
        self.assertEqual(report["reference_coverage_gaps"], [])
        self.assertFalse(report["role_boundary_diagnostic_pass"])

    def test_training_failure_and_wrong_schema_are_visible(self):
        rows = self.rows()
        trained = rows["trained"]["forwarded-third-party"]
        body = json.loads(trained["output_text"])
        body["proposals"].append({**body["proposals"][0],
            "proposal_id": "invented-host-preference", "kind": "preference",
            "domain": "host", "subject_ref": "fictional-ada",
            "value_text": "Ada wants to move to Denver.", "value_ref": "invented-value"})
        trained["output_text"] = trained["raw_output_text"] = json.dumps(body)
        failed = assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        self.assertTrue(any("unsupported_personal_assertion" in item
                            for item in failed["trained_failures"]))
        trained = rows["trained"]["neutral-owner-event"]
        trained["output_text"] = trained["raw_output_text"] = (
            trained["output_text"].replace(OUTPUT_SCHEMA, "unknown-schema"))
        trained.update(validation_status="invalid", validation_error="schema differs")
        failed = assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        self.assertTrue(any("schema differs" in failure for failure in
                            failed["trained_failures"]))

    def test_rejects_fictitious_disabled_control_or_missing_runner_fields(self):
        rows = self.rows()
        rows["ablated"] = rows.pop("seeded-untrained")
        with self.assertRaisesRegex(DiagnosticError, "need trained and seeded"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        rows = self.rows()
        row = rows["seeded-untrained"]["neutral-owner-event"]
        row["control_kind"] = "ablated"
        with self.assertRaisesRegex(DiagnosticError, "control kind"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        rows = self.rows()
        del rows["trained"]["neutral-owner-event"]["generated_token_ids"]
        with self.assertRaisesRegex(DiagnosticError, "generation record"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)

    def test_rejects_mismatched_base_receipt_component_and_decoding(self):
        rows = self.rows()
        rows["seeded-untrained"]["neutral-owner-event"]["prepared_base_receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "prepared_base_receipt_sha256 changed"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        rows = self.rows()
        rows["trained"]["neutral-owner-event"]["model_receipt"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "model_receipt changed"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        rows = self.rows()
        for row in rows["seeded-untrained"].values():
            row["formation_component_sha256"] = row["model_artifact_digest"] = "b" * 64
        with self.assertRaisesRegex(DiagnosticError, "distinct artifacts"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        rows = self.rows()
        for row in rows["seeded-untrained"].values():
            row["generation"]["max_new_tokens"] = 1024
        with self.assertRaisesRegex(DiagnosticError, "decoding differs between specialist"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        rows = self.rows()
        # The untouched publisher is allowed a different cap and decoder.
        report = assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        self.assertFalse(report["role_boundary_diagnostic_pass"])

    def test_rejects_different_prompt_and_verified_artifact_bytes(self):
        rows = self.rows()
        rows["trained"]["neutral-owner-event"]["prompt_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "prompt digest"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)
        rows = self.rows()
        lineage = self.lineage()
        lineage["trained"]["weight_sha256"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "rehashed local artifacts"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest,
                   verified_lineage=lineage)
        rows = self.rows()
        rows["trained"]["neutral-owner-event"]["processed_modalities"] = ["text"]
        if "image" not in {r.modality for r in self.cases["neutral-owner-event"][0].context.evidence}:
            # Mismatch the image case, where the specialist must exercise the image.
            rows["trained"]["blank-image-no-biography"]["processed_modalities"] = ["text"]
        with self.assertRaisesRegex(DiagnosticError, "modality coverage"):
            assess(self.cases, rows, source_receipt_sha256=self.receipt_digest)

    def test_local_artifact_binding_rehashes_distinct_files_and_shared_run(self):
        from scripts.mfm import run_v1_formation_specialist as inference
        from scripts.mfm import train_v1_formation_specialist as training

        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            trained_file = root / "trained.bin"
            seed_file = root / "seed.bin"
            trained_file.write_bytes(b"trained")
            seed_file.write_bytes(b"seed")
            prepared = {"receipt_sha256": "e" * 64}
            preflight = {"max_target_tokens": 8192}
            component = {"run_manifest_sha256": "1" * 64,
                         "prepared_base_sha256": SOURCE_SHA256,
                         "prepared_base_parent_sha256": SOURCE_SHA256,
                         "record_sha256": "c" * 64}
            seed = {"run_manifest_sha256": "1" * 64, "record_sha256": "d" * 64}

            def verified(*_args, control):
                return prepared, preflight, component, (
                    trained_file if control == "trained" else seed_file)

            with patch.object(inference, "verify_artifacts", side_effect=verified), \
                    patch.object(training, "_read_sealed", return_value=seed):
                bound = verify_local_lineage(root, root, root / "base.json", root / "pre.json")
                self.assertEqual(bound["trained"]["weight_sha256"], sha256(b"trained").hexdigest())
                self.assertNotEqual(bound["trained"]["weight_sha256"],
                                    bound["seeded-untrained"]["weight_sha256"])
                seed["run_manifest_sha256"] = "0" * 64
                with self.assertRaisesRegex(DiagnosticError, "share a sealed run manifest"):
                    verify_local_lineage(root, root, root / "base.json", root / "pre.json")


if __name__ == "__main__":
    unittest.main()
