from __future__ import annotations

from hashlib import sha256
import json
import socket
import unittest
from unittest.mock import patch
from urllib import request

from scripts.mfm.evaluate_base_behavior import DiagnosticError, _input_bytes, load_diagnostic
from scripts.mfm.qualify_v1_role_boundary import SOURCE_SHA256, assess


class RoleBoundaryDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_diagnostic()
        cls.prompt_digest = sha256(_input_bytes(cls.cases)).hexdigest()
        cls.receipt_digest = "a" * 64

    def rows(self) -> dict[str, dict[str, dict]]:
        result = {role: {} for role in ("untouched", "assembled", "ablated")}
        for case_id, (gold, _, _) in self.cases.items():
            answer = json.dumps({"proposals": [p.record() for p in gold.expected],
                                 "dispositions": [d.record() for d in gold.expected_dispositions]})
            for role in result:
                row = {"case_id": case_id, "context_digest": gold.context.content_digest(),
                       "prompt_set_sha256": self.prompt_digest,
                       "source_repository": "google/gemma-4-12B",
                       "source_revision": "023679ed352de9bb66cc873c9009ce3482585c08",
                       "generation": {"do_sample": False, "seed": 20260930},
                       "status": "generated", "processed_modalities": ["text", "image"],
                       "output_text": answer,
                       "model_artifact_digest": SOURCE_SHA256 if role == "untouched" else "b" * 64,
                       "model_receipt": self.receipt_digest if role == "untouched" else "c" * 64}
                if role != "untouched":
                    row.update(base_source_sha256=SOURCE_SHA256,
                               prepared_base_parent_sha256=SOURCE_SHA256,
                               prepared_base_sha256="d" * 64,
                               prepared_base_receipt_sha256="e" * 64,
                               formation_component_sha256="f" * 64,
                               formation_component_active=role == "assembled")
                if case_id == "blank-image-no-biography":
                    row.update(status="unexercised", output_text=None,
                               processed_modalities=["text"])
                result[role][case_id] = row
        return result

    def test_measures_component_ablation_without_claiming_qualification(self):
        runs = self.rows()
        runs["ablated"]["neutral-owner-event"]["output_text"] = (
            "As an AI language model, I remember Ada.")
        # The diagnostic reads only frozen public inputs and passed-in rows.
        # It must not contact a network or load a model during scoring.
        with patch.object(socket, "create_connection", side_effect=AssertionError("network")), \
                patch.object(request, "urlopen", side_effect=AssertionError("network")):
            report = assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)
        self.assertIn("neutral-owner-event", report["formation_contribution_cases"])
        self.assertFalse(report["role_boundary_diagnostic_pass"])
        self.assertIn("blank-image-no-biography:unexercised_modality", report["coverage_gaps"])
        self.assertFalse(report["qualification_claim"])
        self.assertIn("artifact_authenticity", report["unmeasured"])

    def test_no_causal_effect_and_bad_personal_assertion_are_visible(self):
        runs = self.rows()
        no_effect = assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)
        self.assertEqual(no_effect["formation_contribution_cases"], [])
        body = json.loads(runs["assembled"]["forwarded-third-party"]["output_text"])
        body["proposals"].append({**body["proposals"][0],
            "proposal_id": "invented-host-preference", "kind": "preference",
            "domain": "host", "subject_ref": "fictional-ada",
            "value_text": "Ada wants to move to Denver.", "value_ref": "invented-value"})
        runs["assembled"]["forwarded-third-party"]["output_text"] = json.dumps(body)
        failed = assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)
        self.assertTrue(any("wrong_subject" in item or "unsupported_personal_assertion" in item
                            for item in failed["assembled_failures"]))

    def test_rejects_pristine_as_operating_base_and_inconsistent_lineage(self):
        runs = self.rows()
        runs["assembled"]["neutral-owner-event"]["prepared_base_sha256"] = SOURCE_SHA256
        with self.assertRaisesRegex(DiagnosticError, "distinct prepared base lineage"):
            assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)
        runs = self.rows()
        runs["ablated"]["neutral-owner-event"]["formation_component_sha256"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "formation_component_sha256 changed"):
            assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)
        runs = self.rows()
        runs["assembled"]["neutral-owner-event"]["base_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "source weight lineage differs"):
            assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)

    def test_rejects_unmatched_decoding_and_unverified_source_receipt(self):
        runs = self.rows()
        for row in runs["ablated"].values():
            row["generation"]["seed"] = 1
        with self.assertRaisesRegex(DiagnosticError, "decoding differs"):
            assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)
        runs = self.rows()
        runs["untouched"]["neutral-owner-event"]["model_receipt"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "verified source"):
            assess(self.cases, runs, source_receipt_sha256=self.receipt_digest)


if __name__ == "__main__":
    unittest.main()
