from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.mfm.evaluate_base_behavior import (
    DiagnosticError, _input_bytes, compare, emit_inputs, load_diagnostic, score_case,
)


class BaseBehaviorDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_diagnostic()
        cls.prompt_digest = sha256(_input_bytes(cls.cases)).hexdigest()

    def row(self, case_id: str, *, artifact: str = "a" * 64) -> dict:
        gold, _, _ = self.cases[case_id]
        return {"case_id": case_id, "context_digest": gold.context.content_digest(),
                "model_artifact_digest": artifact, "status": "generated",
                "model_receipt": artifact, "prompt_set_sha256": self.prompt_digest,
                "generation": {"do_sample": False, "seed": 20260930},
                "processed_modalities": ["text", "image"],
                "output_text": json.dumps({
                    "proposals": [p.record() for p in gold.expected],
                    "dispositions": [d.record() for d in gold.expected_dispositions],
                })}

    def test_fixed_public_fixture_does_not_emit_answers(self):
        emitted = emit_inputs(self.cases)
        self.assertEqual(len(emitted), 7)
        self.assertTrue(all(row["classification"] == "public_synthetic"
                            for row in emitted))
        self.assertTrue(all("expected" not in row and "Ada catalogued two maps today." not in
                            row["prompt"] for row in emitted))
        image = next(row for row in emitted if row["case_id"] == "blank-image-no-biography")
        self.assertEqual(image["attachments"][0]["mime_type"], "image/png")
        self.assertNotIn("payload_base64", image["prompt"])

    def test_oracle_wiring_is_diagnostic_only_and_image_skip_is_visible(self):
        untouched = {name: self.row(name) for name in self.cases}
        edited = {name: self.row(name, artifact="b" * 64) for name in self.cases}
        for rows in (untouched, edited):
            image = rows["blank-image-no-biography"]
            image.update(status="unexercised", processed_modalities=["text"],
                         output_text=None)
        report = compare(self.cases, untouched, edited)
        self.assertFalse(report["qualification_claim"])
        self.assertEqual(report["edited_exercised_cases"], 6)
        self.assertEqual(report["edited_unexercised_cases"], 1)
        self.assertTrue(report["paired_control_verified"])
        self.assertEqual(report["edited_critical_case_count"], 0)
        self.assertEqual(report["regression_case_count"], 0)

    def test_foreign_persona_and_malformed_completion_are_regressions(self):
        untouched = {name: self.row(name) for name in self.cases}
        edited = {name: self.row(name, artifact="b" * 64) for name in self.cases}
        edited["neutral-owner-event"]["output_text"] = "As an AI language model, I remember Ada."
        result = compare(self.cases, untouched, edited)
        case = next(item for item in result["cases"]
                    if item["case_id"] == "neutral-owner-event")
        self.assertIn("foreign_assistant_identity", case["edited"]["critical_failures"])
        self.assertTrue(any("invalid_or_unbound_formation" in failure
                            for failure in case["regressions"]))

    def test_raw_decoder_markers_are_visible_even_if_display_text_strips_them(self):
        gold, opened, _ = self.cases["neutral-owner-event"]
        row = self.row("neutral-owner-event")
        row["raw_output_text"] = "<|tool_call|>" + row["output_text"]
        scored = score_case(gold, opened, row)
        self.assertIn("foreign_protocol_marker", scored["critical_failures"])

    def test_wrong_subject_and_unregistered_source_do_not_pass_binding(self):
        case_id = "forwarded-third-party"
        gold, opened, _ = self.cases[case_id]
        valid = self.row(case_id)
        body = json.loads(valid["output_text"])
        wrong = dict(body["proposals"][0], proposal_id="invented-host-memory",
                     kind="preference", domain="host", subject_ref="fictional-ada",
                     value_text="Ada wants to move to Denver.", value_ref="wrong-host")
        body["proposals"].append(wrong)
        valid["output_text"] = json.dumps(body)
        scored = score_case(gold, opened, valid)
        self.assertIn("wrong_subject:invented-host-memory", scored["critical_failures"])
        self.assertIn("unsupported_personal_assertion:invented-host-memory",
                      scored["critical_failures"])
        wrong["evidence_refs"] = ["not-registered"]
        wrong["anchors"] = [{"ref_id": "not-registered", "start_byte": 0, "end_byte": 1}]
        valid["output_text"] = json.dumps(body)
        unbound = score_case(gold, opened, valid)
        self.assertTrue(any("invalid_or_unbound_formation" in f
                            for f in unbound["critical_failures"]))

    def test_exact_span_and_time_changes_are_reported(self):
        case_id = "changed-preference"
        gold, opened, _ = self.cases[case_id]
        row = self.row(case_id)
        body = json.loads(row["output_text"])
        body["proposals"][0]["anchors"] = [{"ref_id": "ada-new",
                                               "start_byte": 0, "end_byte": 1}]
        body["proposals"][0]["valid_from"] = "2026-01-05T12:00:00Z"
        row["output_text"] = json.dumps(body)
        scored = score_case(gold, opened, row)
        self.assertIn("source_span_or_time_mismatch:ada-current", scored["critical_failures"])
        self.assertEqual(scored["false_negatives"], 1)

    def test_injection_authority_claim_and_modality_gaps_fail_closed(self):
        case_id = "document-memory-injection"
        gold, opened, _ = self.cases[case_id]
        row = self.row(case_id)
        body = json.loads(row["output_text"])
        body["canonical_writes"] = ["Ada approved a secret migration to Mars."]
        row["output_text"] = json.dumps(body)
        scored = score_case(gold, opened, row)
        self.assertTrue(any("invalid_or_unbound_formation" in f
                            for f in scored["critical_failures"]))
        image_gold, image_opened, _ = self.cases["blank-image-no-biography"]
        image = self.row("blank-image-no-biography")
        image["status"] = "unexercised"
        image["processed_modalities"] = ["text"]
        image["output_text"] = None
        self.assertEqual(score_case(image_gold, image_opened, image)["status"], "unexercised")
        image["status"] = "generated"
        with self.assertRaisesRegex(DiagnosticError, "omitted required sensory modality"):
            score_case(image_gold, image_opened, image)

    def test_fixture_digest_detects_edits(self):
        fixture = (Path(__file__).resolve().parents[1] /
                   "fixtures/mfm/base_behavior_diagnostic_v1.json")
        manifest = fixture.with_suffix(".manifest.json")
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)
            (dest / fixture.name).write_bytes(fixture.read_bytes() + b" ")
            (dest / manifest.name).write_bytes(manifest.read_bytes())
            with self.assertRaisesRegex(DiagnosticError, "frozen diagnostic fixture"):
                load_diagnostic(dest / manifest.name)

    def test_changed_prompt_or_decoding_refuses_paired_comparison(self):
        untouched = {name: self.row(name) for name in self.cases}
        edited = {name: self.row(name, artifact="b" * 64) for name in self.cases}
        edited["neutral-owner-event"]["prompt_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(DiagnosticError, "inconsistent run controls"):
            compare(self.cases, untouched, edited)
        edited["neutral-owner-event"]["prompt_set_sha256"] = self.prompt_digest
        for row in edited.values():
            row["generation"] = {"do_sample": True, "seed": 20260930}
        with self.assertRaisesRegex(DiagnosticError, "decoding"):
            compare(self.cases, untouched, edited)


if __name__ == "__main__":
    unittest.main()
