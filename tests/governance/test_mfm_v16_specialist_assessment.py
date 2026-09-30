"""Fake outputs exercise the assessment boundary, never model performance."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_dataset_admission import AdmittedCase, CorpusAdmission, PayloadRef
from cognitive_kernel.formation_contracts import FormationDisposition
from cognitive_kernel.formation_semantics_v16 import (
    FULL_ROLE_ADJUDICATION_DIMENSIONS, FormationGoldCaseV16,
)
from scripts.mfm import qualify_v16_formation_specialist as qualifier
from scripts.mfm import run_v16_formation_specialist as runner
from tests.governance.test_formation_semantics_v16 import SOURCE, fixture
from tests.governance.test_formation_learning_v16 import case_record


def gold_case():
    context, bundle = fixture()
    return FormationGoldCaseV16(
        "case-1", context, bundle.proposals, bundle.base.dispositions, (),
        FULL_ROLE_ADJUDICATION_DIMENSIONS, ("reviewer-a", "reviewer-b"),
        (("source-1", SOURCE),))


def lineage():
    return {"weights": {"trained": "a" * 64, "seeded-untrained": "b" * 64},
            "receipts": {"trained": "c" * 64, "seeded-untrained": "d" * 64},
            "prepared_base_sha256": "e" * 64,
            "prepared_base_parent_sha256": runner.training.shared.SOURCE_WEIGHT_SHA256,
            "prepared_base_receipt_sha256": "f" * 64,
            "source_repository": "google/gemma-4-12B",
            "source_revision": "023679ed352de9bb66cc873c9009ce3482585c08",
            "max_target_tokens": 8192,
            "probe_only": True, "optimizer_steps": 1}


def input_row(case):
    from base64 import b64encode
    return {"case_id": case.case_id, "context": case.context.record(),
            "context_digest": case.context.content_digest(),
            "opened_sources": [{"ref_id": ref, "payload_base64": b64encode(raw).decode()}
                               for ref, raw in case.opened_sources]}


def output_row(case, role, input_sha, *, invalid=False, altered=False):
    output = fixture()[1].output_record()
    if altered:
        output["proposals"][0]["proposal"]["value_text"] = "An unsupported host memory."
        output["proposals"][0]["proposal"]["subject_ref"] = "other-person"
    raw = "not-json" if invalid else json.dumps(output)
    return {"case_id": case.case_id, "status": "probe_generated",
            "context_digest": case.context.content_digest(),
            "processed_modalities": ["text"],
            "model_artifact_digest": lineage()["weights"][role],
            "model_receipt": lineage()["receipts"][role],
            "formation_component_sha256": lineage()["weights"][role],
            "control_kind": role, "probe_only": True,
            "qualified_for_product": False,
            "source_repository": lineage()["source_repository"],
            "source_revision": lineage()["source_revision"],
            "base_source_sha256": runner.training.shared.SOURCE_WEIGHT_SHA256,
            "prepared_base_sha256": lineage()["prepared_base_sha256"],
            "prepared_base_parent_sha256": lineage()["prepared_base_parent_sha256"],
            "prepared_base_receipt_sha256": lineage()["prepared_base_receipt_sha256"],
            "prompt_set_sha256": input_sha,
            "generation": {"decoder": "v1.6-specialist-greedy-bos-eos",
                           "do_sample": False, "max_new_tokens": 8192},
            "inference_runner_sha256": sha256(Path(runner.__file__).read_bytes()).hexdigest(),
            "output_text": raw, "raw_output_text": raw,
            "generated_token_ids": [1, 2], "eos_observed": not invalid,
            "validation_status": "invalid" if invalid else "grounded_probe_proposal_only",
            "validation_error": "no EOS inside declared v1.6 target budget" if invalid else None}


class MFMV16SpecialistAssessmentTests(unittest.TestCase):
    def test_matched_control_evaluates_exact_v16_gold_but_cannot_qualify(self):
        case = gold_case()
        case.validate()
        input_sha = "0" * 64
        rows = {"trained": [output_row(case, "trained", input_sha)],
                "seeded-untrained": [output_row(case, "seeded-untrained",
                                                input_sha, invalid=True)]}
        report = qualifier.assess_pair({case.case_id: case}, input_sha, rows, lineage())
        self.assertEqual(report["trained_improvement_cases"], [case.case_id])
        self.assertEqual(report["trained_failure_cases"], [])
        self.assertTrue(report["full_fit_artifact_not_supported"])
        self.assertFalse(report["qualification_claim"])
        self.assertFalse(report["inference_execution_authenticity_verified"])

    def test_grounded_false_subject_and_false_memory_are_critical(self):
        case = gold_case()
        input_sha = "0" * 64
        rows = {"trained": [output_row(case, "trained", input_sha, altered=True)],
                "seeded-untrained": [output_row(case, "seeded-untrained",
                                                input_sha, invalid=True)]}
        report = qualifier.assess_pair({case.case_id: case}, input_sha, rows, lineage())
        score = report["cases"][0]["trained"]
        self.assertIn("false_memory", score["critical_failures"])
        self.assertIn("wrong_subject", score["critical_failures"])
        self.assertEqual(score["false_positives"], 1)
        self.assertEqual(report["trained_failure_cases"], [case.case_id])

    def test_missing_abstention_and_mismatched_sensitivity_are_reported(self):
        case = gold_case()
        input_sha = "0" * 64
        hint = output_row(case, "trained", input_sha)
        altered = json.loads(hint["output_text"])
        altered["proposals"][0]["sensitivity_hint"] = None
        hint["output_text"] = hint["raw_output_text"] = json.dumps(altered)
        pair = {"trained": [hint], "seeded-untrained": [output_row(
            case, "seeded-untrained", input_sha, invalid=True)]}
        report = qualifier.assess_pair({case.case_id: case}, input_sha, pair, lineage())
        self.assertIn("sensitivity_mismatch",
                      report["cases"][0]["trained"]["critical_failures"])

        abstain_case = replace(
            case, expected=(), expected_dispositions=(FormationDisposition(
                "synthetic-scope", "abstain", ("source-1",), ()),))
        abstain_case.validate()
        pair = {"trained": [output_row(abstain_case, "trained", input_sha)],
                "seeded-untrained": [output_row(
                    abstain_case, "seeded-untrained", input_sha, invalid=True)]}
        report = qualifier.assess_pair({case.case_id: abstain_case}, input_sha,
                                       pair, lineage())
        self.assertIn("missing_abstention",
                      report["cases"][0]["trained"]["critical_failures"])

    def test_rejects_tampered_run_receipt_or_unmatched_decoding(self):
        case = gold_case()
        input_sha = "0" * 64
        rows = {"trained": [output_row(case, "trained", input_sha)],
                "seeded-untrained": [output_row(case, "seeded-untrained",
                                                input_sha, invalid=True)]}
        rows["trained"][0]["model_receipt"] = "f" * 64
        with self.assertRaisesRegex(CognitiveKernelContractError, "receipt differs"):
            qualifier.assess_pair({case.case_id: case}, input_sha, rows, lineage())
        rows["trained"][0]["model_receipt"] = "c" * 64
        rows["trained"][0]["generation"]["max_new_tokens"] = 64
        with self.assertRaisesRegex(CognitiveKernelContractError, "decoder settings"):
            qualifier.assess_pair({case.case_id: case}, input_sha, rows, lineage())

    def test_rejects_untrusted_or_changed_input_and_duplicate_json_keys(self):
        case = gold_case()
        row = input_row(case)
        qualifier._assert_input([row], {case.case_id: case})
        row["opened_sources"][0]["payload_base64"] = "dGFtcGVyZWQ="
        with self.assertRaisesRegex(CognitiveKernelContractError, "source digest"):
            qualifier._assert_input([row], {case.case_id: case})
        with self.assertRaisesRegex(CognitiveKernelContractError, "duplicate JSON key"):
            qualifier._rows(b'{"case_id":"a","case_id":"b"}\n', "run")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_bytes(b'{"case_id":"a"}\n')
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "external digest"):
                qualifier._bound_bytes(path, "0" * 64, "run")

    def test_final_guard_runs_before_any_final_payload_is_opened(self):
        args = type("Args", (), {"split": "final", "final_custodian": False})()
        with patch.object(qualifier, "admit_formation_corpus") as admission:
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "separate custodian mode"):
                qualifier.run(args)
            admission.assert_not_called()

    def test_final_custodian_parser_rechecks_exact_target_and_complete_labels(self):
        """Hand-constructed fixture exercises FINAL parser, not signed admission."""
        row = case_record(split="development")
        target = {**row["target"], "context": row["context"],
                  "source_ids": ["source-1"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path, target_path = root / "source.bin", root / "target.json"
            source_path.write_bytes(SOURCE)
            target_path.write_bytes(json.dumps(target).encode())
            case = AdmittedCase(
                "synthetic-codec-case", "final",
                (PayloadRef(source_path.name, sha256(SOURCE).hexdigest()),),
                PayloadRef(target_path.name, sha256(target_path.read_bytes()).hexdigest()), ())
            admission = CorpusAdmission("unit-only", "1" * 64, (), (), (case,), root)
            manifest = {"cases": [{"case_id": case.case_id, "host_family": "owner-one",
                                    "sources": [{"source_id": "source-1"}],
                                    "reviews": [{"reviewer_id": "fixture-reviewer-a"},
                                                {"reviewer_id": "fixture-reviewer-b"}]}]}
            parsed = qualifier._gold_cases(admission, manifest, "final")
            self.assertEqual(parsed[case.case_id].opened_sources, (("source-1", SOURCE),))
            self.assertEqual(len(parsed[case.case_id].expected), 1)
            target["adjudications"]["episode"] = "unknown"
            target_path.write_bytes(json.dumps(target).encode())
            changed = replace(case, target_payload=PayloadRef(
                target_path.name, sha256(target_path.read_bytes()).hexdigest()))
            admission = replace(admission, final_metadata=(changed,))
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "unknown or inconsistent"):
                qualifier._gold_cases(admission, manifest, "final")

    def test_invalid_row_cannot_hide_a_grounded_eos_output(self):
        case = gold_case()
        row = output_row(case, "trained", "0" * 64)
        row["validation_status"] = "invalid"
        row["validation_error"] = "fabricated rejection"
        with self.assertRaisesRegex(CognitiveKernelContractError, "grounded EOS"):
            qualifier._check_output_row(row, case, "trained", "0" * 64, lineage())

    def test_one_step_probe_is_the_only_supported_artifact(self):
        case = gold_case()
        local_lineage = lineage()
        local_lineage["optimizer_steps"] = 2
        with self.assertRaisesRegex(CognitiveKernelContractError, "one-step"):
            qualifier.assess_pair({case.case_id: case}, "0" * 64,
                                  {"trained": [output_row(case, "trained", "0" * 64)],
                                   "seeded-untrained": [output_row(
                                       case, "seeded-untrained", "0" * 64)]},
                                  local_lineage)


if __name__ == "__main__":
    unittest.main()
