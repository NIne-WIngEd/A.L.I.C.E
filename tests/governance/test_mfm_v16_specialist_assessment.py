"""Fake outputs exercise the assessment boundary, never model performance."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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
            "run_manifest_sha256": "1" * 64,
            "preflight_sha256": "2" * 64,
            "training_input_sha256": "3" * 64,
            "trust_roster_sha256": None,
            "signed_review_receipt_sha256": None,
            "full_fit": False, "probe_only": True, "optimizer_steps": 1}


def input_row(case):
    from base64 import b64encode
    return {"case_id": case.case_id, "context": case.context.record(),
            "context_digest": case.context.content_digest(),
            "opened_sources": [{"ref_id": ref, "payload_base64": b64encode(raw).decode()}
                               for ref, raw in case.opened_sources]}


def output_row(case, role, input_sha, *, invalid=False, altered=False,
               local_lineage=None):
    local_lineage = local_lineage or lineage()
    output = fixture()[1].output_record()
    if altered:
        output["proposals"][0]["proposal"]["value_text"] = "An unsupported host memory."
        output["proposals"][0]["proposal"]["subject_ref"] = "other-person"
    raw = "not-json" if invalid else json.dumps(output)
    full_fit = local_lineage["full_fit"]
    return {"case_id": case.case_id,
            "status": "fit_generated" if full_fit else "probe_generated",
            "context_digest": case.context.content_digest(),
            "processed_modalities": ["text"],
            "model_artifact_digest": local_lineage["weights"][role],
            "model_receipt": local_lineage["receipts"][role],
            "formation_component_sha256": local_lineage["weights"][role],
            "control_kind": role, "full_fit": full_fit,
            "probe_only": local_lineage["probe_only"],
            "optimizer_steps": local_lineage["optimizer_steps"],
            "qualified_for_product": False,
            "run_manifest_sha256": local_lineage["run_manifest_sha256"],
            "preflight_sha256": local_lineage["preflight_sha256"],
            "training_input_sha256": local_lineage["training_input_sha256"],
            "trust_roster_sha256": local_lineage["trust_roster_sha256"],
            "signed_review_receipt_sha256": local_lineage["signed_review_receipt_sha256"],
            "source_repository": local_lineage["source_repository"],
            "source_revision": local_lineage["source_revision"],
            "base_source_sha256": runner.training.shared.SOURCE_WEIGHT_SHA256,
            "prepared_base_sha256": local_lineage["prepared_base_sha256"],
            "prepared_base_parent_sha256": local_lineage["prepared_base_parent_sha256"],
            "prepared_base_receipt_sha256": local_lineage["prepared_base_receipt_sha256"],
            "prompt_set_sha256": input_sha,
            "generation": {"decoder": "v1.6-specialist-greedy-bos-eos",
                           "do_sample": False, "max_new_tokens": 8192},
            "inference_runner_sha256": sha256(Path(runner.__file__).read_bytes()).hexdigest(),
            "output_text": raw, "raw_output_text": raw,
            "generated_token_ids": [1, 2], "eos_observed": not invalid,
            "validation_status": "invalid" if invalid else (
                "grounded_fit_proposal_only" if full_fit else
                "grounded_probe_proposal_only"),
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

    def test_private_isolation_precedes_corpus_admission(self):
        args = SimpleNamespace(split="development", final_custodian=False)
        with patch.object(qualifier.training.shared,
                          "_require_private_network_isolation",
                          side_effect=CognitiveKernelContractError("no egress")), \
                patch.object(qualifier, "admit_formation_corpus") as admission:
            with self.assertRaisesRegex(CognitiveKernelContractError, "no egress"):
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

    def test_signed_full_fit_pair_scored_without_qualification_claim(self):
        case = gold_case()
        fit = {**lineage(), "full_fit": True, "probe_only": False,
               "optimizer_steps": 17, "trust_roster_sha256": "4" * 64,
               "signed_review_receipt_sha256": "5" * 64}
        pair = {role: [output_row(case, role, "0" * 64, local_lineage=fit,
                                  invalid=role == "seeded-untrained")]
                for role in qualifier.CONTROL_ROLES}
        report = qualifier.assess_pair({case.case_id: case}, "0" * 64, pair, fit)
        self.assertEqual(report["assessment_mode"], "signed_full_fit")
        self.assertFalse(report["full_fit_artifact_not_supported"])
        self.assertEqual(report["trained_improvement_cases"], [case.case_id])
        self.assertFalse(report["qualification_claim"])
        self.assertFalse(report["inference_execution_authenticity_verified"])
        pair["trained"][0]["signed_review_receipt_sha256"] = "6" * 64
        with self.assertRaisesRegex(CognitiveKernelContractError, "receipt differs"):
            qualifier.assess_pair({case.case_id: case}, "0" * 64, pair, fit)

    def test_full_fit_artifact_requires_exact_external_manifest_roster_and_review(self):
        shared = runner.training.shared
        a, b, c, d, e, f = (digit * 64 for digit in "abcdef")
        prepared = {"receipt_sha256": a,
                    "files": [{"path": "model.safetensors", "sha256": b}]}
        cfg = SimpleNamespace(heads=2, layers=1, max_target_tokens=8)
        common = {"full_fit": True, "trust_roster_sha256": c,
                  "signed_review_receipt_sha256": d}
        preflight = {**common, "record_sha256": e,
                     "objective": runner.training.OBJECTIVE_VERSION_V16,
                     "output_schema": "mfm-formation-output-v1.6",
                     "context_schema": "mfm-formation-context-v1.6",
                     "prepared_base_receipt_sha256": a,
                     "prepared_base_parent_weight_sha256": shared.SOURCE_WEIGHT_SHA256,
                     "prepared_base_weight_sha256": b,
                     "prepared_base_kind": "licensed-verified-mfm-role-clone",
                     "foundation_commit": shared.FOUNDATION_COMMIT,
                     "foundation_verifier_sha256": shared.FOUNDATION_VERIFIER_SHA256,
                     "foundation_inventory_sha256": shared.FOUNDATION_INVENTORY_SHA256,
                     "source_template_sha256": sha256(shared.SOURCE_TEMPLATE.encode()).hexdigest(),
                     "source_instruction_sha256": sha256(
                         runner.training.SOURCE_INSTRUCTION.encode()).hexdigest(),
                     "source_builder_sha256": a, "codec_sha256": a,
                     "semantics_sha256": a, "decoder_implementation_sha256": a,
                     "trainer_sha256": a, "max_source_tokens": 64,
                     "max_target_tokens": 8, "specialist_heads": 2,
                     "specialist_layers": 1, "corpus_sha256": f,
                     "corpus_status": "admitted-signed-review-final-sealed-unqualified",
                     "transformers_version": "fixture"}
        run = {**common, "record_sha256": "1" * 64,
               "objective": runner.training.OBJECTIVE_VERSION_V16,
               "preflight_sha256": e, "prepared_base_receipt_sha256": a,
               "corpus_sha256": f, "specialist_config": {},
               "device_placement": {"base": "cuda:0", "specialist": "cuda:0",
                                    "strategy": "single-gpu"},
               "transformers_version": "fixture", "probe_only": False,
               "qualified_for_product": False}
        component = {**common, "record_sha256": "2" * 64,
                     "objective": runner.training.OBJECTIVE_VERSION_V16,
                     "run_manifest_sha256": "1" * 64,
                     "formation_component_sha256": "3" * 64,
                     "prepared_base_sha256": b,
                     "prepared_base_parent_sha256": shared.SOURCE_WEIGHT_SHA256,
                     "prepared_base_receipt_sha256": a,
                     "prepared_base_kind": "licensed-verified-mfm-role-clone",
                     "device_placement": run["device_placement"],
                     "specialist_config": {}, "seed_control_sha256": "4" * 64,
                     "probe_only": False, "optimizer_steps": 1,
                     "qualified_for_product": False}
        receipts = {"preflight.json": preflight, "run.json": run,
                    "formation-component.json": component}
        def digest(path):
            if path.name == "formation-specialist.safetensors":
                return "3" * 64
            if path.name == "seed-control.safetensors":
                return "4" * 64
            return a
        with patch.object(shared, "_prepared_base", return_value=prepared), \
                patch.object(shared, "_read_sealed",
                             side_effect=lambda path, schema: receipts[path.name]), \
                patch.object(shared, "_prepared_kind",
                             return_value="licensed-verified-mfm-role-clone"), \
                patch.object(shared, "_digest", side_effect=digest), \
                patch.object(runner, "_config", return_value=cfg), \
                patch.object(runner.training, "_codec_sha256", return_value=a), \
                patch.object(runner.training, "_semantics_sha256", return_value=a), \
                patch.object(runner.training, "_decoder_sha256", return_value=a), \
                patch.object(runner.training, "_verify_seed_control",
                             return_value={"specialist_sha256": "4" * 64}):
            for role in qualifier.CONTROL_ROLES:
                runner.verify_artifacts(Path("/fixture"), Path("/base"), Path("/base.json"),
                                        Path("/preflight.json"), control=role,
                                        expected_manifest_sha256=f,
                                        expected_roster_sha256=c,
                                        expected_review_sha256=d)
            with self.assertRaisesRegex(CognitiveKernelContractError, "externally pinned"):
                runner.verify_artifacts(Path("/fixture"), Path("/base"), Path("/base.json"),
                                        Path("/preflight.json"), control="trained")
            with self.assertRaisesRegex(CognitiveKernelContractError, "signed full-fit"):
                runner.verify_artifacts(Path("/fixture"), Path("/base"), Path("/base.json"),
                                        Path("/preflight.json"), control="trained",
                                        expected_manifest_sha256=f,
                                        expected_roster_sha256="0" * 64,
                                        expected_review_sha256=d)
            component["signed_review_receipt_sha256"] = "0" * 64
            with self.assertRaisesRegex(CognitiveKernelContractError, "signed full-fit"):
                runner.verify_artifacts(Path("/fixture"), Path("/base"), Path("/base.json"),
                                        Path("/preflight.json"), control="trained",
                                        expected_manifest_sha256=f,
                                        expected_roster_sha256=c,
                                        expected_review_sha256=d)

    def test_signed_binding_requires_isolation_before_admission(self):
        with patch.object(runner.training.shared, "_require_private_network_isolation",
                          side_effect=CognitiveKernelContractError("loopback-only")), \
                patch.object(runner, "admit_formation_corpus") as admission:
            with self.assertRaisesRegex(CognitiveKernelContractError, "loopback-only"):
                runner.verify_signed_binding(Path("/private/manifest.json"), "a" * 64,
                                             Path("/private/roster.json"), "b" * 64)
            admission.assert_not_called()

    def test_final_checks_fit_lineage_before_opening_gold_payload(self):
        fields = {name: Path(f"/fixture/{name}") for name in (
            "manifest", "roster", "input_jsonl", "trained", "seeded_untrained",
            "component_dir", "prepared_base_dir", "prepared_base_receipt",
            "preflight_receipt", "output")}
        args = SimpleNamespace(**fields, split="final", final_custodian=True,
                               manifest_sha256="a" * 64, roster_sha256="b" * 64)
        admission = SimpleNamespace(manifest_sha256="a" * 64)
        review = {"corpus_manifest_sha256": "a" * 64,
                  "externally_pinned_roster_sha256": "b" * 64,
                  "rights_and_review_signatures_verified": True,
                  "final_payloads_opened": False}
        with patch.object(qualifier, "admit_formation_corpus", return_value=admission), \
                patch.object(qualifier.training.shared, "_require_private_network_isolation"), \
                patch.object(qualifier, "verify_adjudicated_corpus_v16",
                             return_value=review), \
                patch.object(qualifier.training.shared, "_read_sealed",
                             return_value={"full_fit": True}), \
                patch.object(qualifier, "verify_local_controls",
                             side_effect=CognitiveKernelContractError("signed full-fit lineage")), \
                patch.object(qualifier, "_gold_cases") as gold:
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "signed full-fit lineage"):
                qualifier.run(args)
            gold.assert_not_called()


if __name__ == "__main__":
    unittest.main()
