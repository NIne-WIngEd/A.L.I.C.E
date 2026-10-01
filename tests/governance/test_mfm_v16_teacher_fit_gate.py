"""CPU-only owner-teacher training admission; no capability or rights claim."""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes
from cognitive_kernel.formation_dataset_admission import (
    TEACHER_SCHEMA, admit_formation_corpus,
)
from cognitive_kernel.formation_learning_v16 import FULL_ROLE_DIMENSIONS, admitted_rows_v16
from scripts.mfm import run_v16_formation_specialist as runner
from scripts.mfm import train_v16_formation_specialist as trainer
from scripts.mfm.build_v16_authoring_seed import render


def _write(path: Path, value: dict) -> str:
    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def _args(**updates):
    values = dict(mode="train", full_fit=False, teacher_fit=True, probe_only=False,
                  admitted_manifest=None, public_synthetic_curriculum=None,
                  teacher_training_manifest=Path("/owner/teacher.json"),
                  input_sha256="a" * 64, owner_authorization_ref="owner-teacher-v16",
                  trust_roster=None, trust_roster_sha256=None, resume_checkpoint=None,
                  max_source_tokens=128, max_target_tokens=128,
                  specialist_width=8, specialist_layers=1, specialist_heads=2,
                  logit_chunk_tokens=2, epochs=1, gradient_accumulation=1,
                  save_every_steps=1, learning_rate=1e-4, max_cross_attention_pairs=128,
                  seed=7)
    values.update(updates)
    return SimpleNamespace(**values)


class TeacherFitGateTests(unittest.TestCase):
    def _runner_receipts(self):
        from hashlib import sha256 as digest
        config = SimpleNamespace(heads=2, layers=1, max_target_tokens=128)
        placement = {"base": "cuda:0", "specialist": "cuda:0", "strategy": "single-gpu"}
        prepared = {"receipt_sha256": "b" * 64,
                    "files": [{"path": "model.safetensors", "sha256": "e" * 64}]}
        preflight = {
            "record_sha256": "c" * 64, "objective": trainer.OBJECTIVE_VERSION_V16,
            "output_schema": "mfm-formation-output-v1.6",
            "context_schema": "mfm-formation-context-v1.6",
            "prepared_base_receipt_sha256": "b" * 64,
            "prepared_base_parent_weight_sha256": trainer.shared.SOURCE_WEIGHT_SHA256,
            "prepared_base_weight_sha256": "e" * 64,
            "prepared_base_kind": "licensed-clone",
            "foundation_commit": trainer.shared.FOUNDATION_COMMIT,
            "foundation_verifier_sha256": trainer.shared.FOUNDATION_VERIFIER_SHA256,
            "foundation_inventory_sha256": trainer.shared.FOUNDATION_INVENTORY_SHA256,
            "source_template_sha256": digest(trainer.shared.SOURCE_TEMPLATE.encode()).hexdigest(),
            "source_instruction_sha256": digest(trainer.SOURCE_INSTRUCTION.encode()).hexdigest(),
            "source_builder_sha256": "f" * 64,
            "codec_sha256": "f" * 64, "semantics_sha256": "f" * 64,
            "decoder_implementation_sha256": "f" * 64, "trainer_sha256": "f" * 64,
            "corpus_sha256": "a" * 64, "corpus_status":
            "owner-authorized-teacher-fit-diagnostic-development-unqualified",
            "owner_authorization_ref": "unit-owner-teacher",
            "max_source_tokens": 256, "max_target_tokens": 128,
            "specialist_heads": 2, "specialist_layers": 1,
            "transformers_version": "unit", "full_fit": False,
            "teacher_fit": True, "trust_roster_sha256": None,
            "signed_review_receipt_sha256": None,
        }
        run = {
            "record_sha256": "d" * 64, "objective": trainer.OBJECTIVE_VERSION_V16,
            "preflight_sha256": "c" * 64, "prepared_base_receipt_sha256": "b" * 64,
            "corpus_sha256": "a" * 64, "specialist_config": {"width": 8},
            "transformers_version": "unit", "device_placement": placement,
            "full_fit": False, "teacher_fit": True, "probe_only": False,
            "qualified_for_product": False,
            "trust_roster_sha256": None, "signed_review_receipt_sha256": None,
        }
        component = {
            "objective": trainer.OBJECTIVE_VERSION_V16,
            "run_manifest_sha256": "d" * 64,
            "prepared_base_receipt_sha256": "b" * 64,
            "prepared_base_sha256": "e" * 64,
            "prepared_base_parent_sha256": trainer.shared.SOURCE_WEIGHT_SHA256,
            "prepared_base_kind": "licensed-clone", "specialist_config": {"width": 8},
            "seed_control_sha256": "1" * 64,
            "formation_component_sha256": "f" * 64,
            "device_placement": placement, "full_fit": False,
            "teacher_fit": True, "probe_only": False, "optimizer_steps": 9,
            "qualified_for_product": False,
            "trust_roster_sha256": None, "signed_review_receipt_sha256": None,
        }
        return config, prepared, preflight, run, component

    def test_runner_requires_separate_teacher_pin_and_never_accepts_probe_lineage(self):
        config, prepared, preflight, run, component = self._runner_receipts()
        records = {"preflight.json": preflight, "run.json": run,
                   "formation-component.json": component}
        def read(path, schema):
            return records[path.name]
        with patch.object(runner.training.shared, "_prepared_base", return_value=prepared), \
                patch.object(runner.training.shared, "_read_sealed", side_effect=read), \
                patch.object(runner.training.shared, "_prepared_kind",
                             return_value="licensed-clone"), \
                patch.object(runner.training.shared, "_digest", return_value="f" * 64), \
                patch.object(runner.training, "_codec_sha256", return_value="f" * 64), \
                patch.object(runner.training, "_semantics_sha256", return_value="f" * 64), \
                patch.object(runner.training, "_decoder_sha256", return_value="f" * 64), \
                patch.object(runner.training, "_verify_seed_control",
                             return_value={"specialist_sha256": "1" * 64}), \
                patch.object(runner, "_config", return_value=config):
            kwargs = dict(component_dir=Path("/unit"), prepared_base_dir=Path("/unit"),
                          prepared_base_receipt=Path("/unit/clone.json"),
                          preflight_receipt=Path("/unit/preflight.json"), control="trained")
            with self.assertRaisesRegex(CognitiveKernelContractError, "teacher-fit lineage"):
                runner.verify_artifacts(**kwargs)
            selected = runner.verify_artifacts(
                **kwargs, expected_teacher_manifest_sha256="a" * 64)[3]
            self.assertEqual(selected.name, "formation-specialist.safetensors")
            preflight["corpus_status"] = "admitted-signed-review-final-sealed-unqualified"
            with self.assertRaisesRegex(CognitiveKernelContractError, "teacher-fit lineage"):
                runner.verify_artifacts(
                    **kwargs, expected_teacher_manifest_sha256="a" * 64)

    def _fixture(self, root: Path):
        """Fictional unit bytes only; these are not real rights attestations."""
        selected = [row for row in map(json.loads, render().splitlines())
                    if row["case_id"] in {"v16-seed-01", "v16-seed-13"}]
        cases = []
        converter_path = root / "unit-converter.py"
        converter_path.write_bytes(b"fictional test converter v1\n")
        converter_sha = sha256(converter_path.read_bytes()).hexdigest()
        for row in selected:
            split = row["split"]
            host = row["context"]["base_context"]["scope"]["host_instance_id"]
            sources = []
            for index, source in enumerate(row["sources"]):
                raw = b64decode(source["content_b64"])
                path = f"{split}/source-{index}.bin"
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(raw)
                source_sha = sha256(raw).hexdigest()
                rights_path = f"{split}/rights-{index}.json"
                rights_sha = _write(root / rights_path, {
                    "schema": "mfm-source-rights-v1", "issuer_id": "test-owner",
                    "source_id": source["ref_id"], "source_sha256": source_sha,
                    "host_family": host, "authority_ref": "fictional-unit-only",
                    "revoked": False, "formation_training": True,
                    "model_distribution": True, "formation_evaluation": True,
                })
                sources.append({"source_id": source["ref_id"], "path": path,
                                "sha256": source_sha, "parent_source_ids": [],
                                "rights_path": rights_path, "rights_sha256": rights_sha})
            target_path = f"{split}/target.json"
            target_sha = _write(root / target_path, {
                "schema": row["target"]["schema"], "context": row["context"],
                "source_ids": [source["ref_id"] for source in row["sources"]],
                **{name: row["target"][name] for name in (
                    "proposals", "dispositions", "adjudications")},
            })
            prompt_path = f"{split}/prompt.txt"
            response_path = f"{split}/response.txt"
            (root / prompt_path).write_bytes(f"fictional prompt {split}".encode())
            (root / response_path).write_bytes(f"fictional response {split}".encode())
            cases.append({"case_id": row["case_id"], "split": split,
                          "author_id": "test-teacher", "host_family": host,
                          "source_family": f"unit-{split}",
                          "generator_family": f"unit-{split}-generator",
                          "scenario_family": f"unit-{split}-scenario",
                          "duplicate_group": f"unit-{split}-duplicate",
                          "parent_case_ids": [], "authorization_id": "unit-owner-teacher",
                          "target_origin": "owner-authorized-service-teacher",
                          "target_provenance": {
                              "producer_id": "test-teacher", "producer_version": "unit-v1",
                              "input": {"path": prompt_path, "sha256": sha256(
                                  (root / prompt_path).read_bytes()).hexdigest()},
                              "output": {"path": response_path, "sha256": sha256(
                                  (root / response_path).read_bytes()).hexdigest()},
                              "conversion": {"path": converter_path.name,
                                             "sha256": converter_sha}},
                          "sources": sources,
                          "target": {"path": target_path, "sha256": target_sha},
                          "reviews": []})
        manifest = {"schema": TEACHER_SCHEMA, "corpus_id": "fictional-unit-corpus",
                    "status": "owner-authorized-teacher-training-only-unqualified",
                    "authorization_id": "unit-owner-teacher", "case_count": len(cases),
                    "cases": cases}
        path = root / "manifest.json"
        return path, manifest, _write(path, manifest)

    def test_teacher_admission_binds_v16_target_source_rights_and_split_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, digest = self._fixture(root)
            admission = admit_formation_corpus(
                path, expected_sha256=digest, teacher_training=True,
                owner_authorization_ref="unit-owner-teacher")
            self.assertEqual((len(admission.train), len(admission.development),
                              len(admission.final_metadata)), (1, 1, 0))
            self.assertEqual(tuple(map(len, (tuple(admitted_rows_v16(admission, split="train")),
                                             tuple(admitted_rows_v16(admission, split="development"))))),
                             (1, 1))
            gradient_paths = tuple(ref.path for case in admission.train for ref in (
                *case.source_payloads, case.target_payload))
            audit = admission.audit_handoff(gradient_paths=gradient_paths,
                                            development_paths=())
            self.assertEqual({item["path"] for item in audit["gradient_inputs"]},
                             set(gradient_paths))
            self.assertNotIn("train/prompt.txt", gradient_paths)
            self.assertNotIn("train/response.txt", gradient_paths)
            self.assertNotIn("unit-converter.py", gradient_paths)
            manifest["cases"][1]["generator_family"] = manifest["cases"][0]["generator_family"]
            digest = _write(path, manifest)
            with self.assertRaisesRegex(CognitiveKernelContractError, "lineage leaks"):
                admit_formation_corpus(path, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="unit-owner-teacher")

    def test_teacher_rights_are_rechecked_and_revocation_blocks_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, _ = self._fixture(root)
            rights_ref = manifest["cases"][0]["sources"][0]
            rights_path = root / rights_ref["rights_path"]
            rights = json.loads(rights_path.read_bytes())
            rights["revoked"] = True
            rights_ref["rights_sha256"] = _write(rights_path, rights)
            digest = _write(path, manifest)
            with self.assertRaisesRegex(CognitiveKernelContractError, "revoked"):
                admit_formation_corpus(path, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="unit-owner-teacher")

    def test_teacher_prompt_and_conversion_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, digest = self._fixture(root)
            prompt = root / manifest["cases"][0]["target_provenance"]["input"]["path"]
            prompt.write_bytes(b"changed prompt")
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "target provenance input bytes differ"):
                admit_formation_corpus(path, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="unit-owner-teacher")
            manifest["cases"][0]["target_provenance"]["input"]["sha256"] = sha256(
                prompt.read_bytes()).hexdigest()
            digest = _write(path, manifest)
            (root / "unit-converter.py").write_bytes(b"changed converter")
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "target provenance conversion bytes differ"):
                admit_formation_corpus(path, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="unit-owner-teacher")

    def test_service_teacher_provenance_rejects_all_path_aliases(self):
        for changed, donor in (
                ("input", "output"), ("input", "conversion"),
                ("input", "target"), ("output", "conversion"),
                ("output", "source"), ("conversion", "source"),
                ("conversion", "target")):
            with self.subTest(changed=changed, donor=donor), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, manifest, _ = self._fixture(root)
                case = manifest["cases"][0]
                refs = case["target_provenance"]
                replacement = (case["target"] if donor == "target" else
                               case["sources"][0] if donor == "source" else refs[donor])
                refs[changed] = {"path": replacement["path"],
                                 "sha256": replacement["sha256"]}
                digest = _write(path, manifest)
                with self.assertRaisesRegex(CognitiveKernelContractError,
                                            "provenance paths must be distinct"):
                    admit_formation_corpus(path, expected_sha256=digest,
                                           teacher_training=True,
                                           owner_authorization_ref="unit-owner-teacher")

    def test_symlink_alias_is_rejected_by_canonical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, _ = self._fixture(root)
            case = manifest["cases"][0]
            alias = root / "train/source-alias.txt"
            alias.symlink_to(root / case["sources"][0]["path"])
            case["target_provenance"]["input"] = {
                "path": "train/source-alias.txt",
                "sha256": case["sources"][0]["sha256"]}
            digest = _write(path, manifest)
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "provenance paths must be distinct"):
                admit_formation_corpus(path, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="unit-owner-teacher")

    def test_deterministic_output_can_be_exact_target_but_other_aliases_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, _ = self._fixture(root)
            case = manifest["cases"][0]
            case["target_origin"] = "licensed-deterministic-generator"
            case["target_provenance"]["output"] = dict(case["target"])
            digest = _write(path, manifest)
            admitted = admit_formation_corpus(path, expected_sha256=digest,
                                              teacher_training=True,
                                              owner_authorization_ref="unit-owner-teacher")
            self.assertEqual(len(admitted.train), 1)
            case["target_provenance"]["conversion"] = dict(case["target"])
            digest = _write(path, manifest)
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "input and conversion paths must be distinct"):
                admit_formation_corpus(path, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="unit-owner-teacher")

    def test_current_21_case_jsonl_is_not_a_teacher_training_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v16-diagnostic.jsonl"
            raw = render()
            path.write_bytes(raw)
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "not one JSON object"):
                admit_formation_corpus(path, expected_sha256=sha256(raw).hexdigest(),
                                       teacher_training=True,
                                       owner_authorization_ref="owner-directed-mfm-v16-synthetic")

    def test_duplicate_manifest_key_fails_even_when_new_hash_is_supplied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            raw = b'{"schema":"mfm-v16-owner-teacher-corpus-v1","schema":"mfm-v16-owner-teacher-corpus-v1"}'
            path.write_bytes(raw)
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "duplicate JSON key"):
                admit_formation_corpus(path, expected_sha256=sha256(raw).hexdigest(),
                                       teacher_training=True,
                                       owner_authorization_ref="unit-owner-teacher")

    def test_teacher_policy_requires_explicit_owner_origin_exact_count_and_rights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"case_id": "case-a", "split": "train", "author_id": "teacher",
                   "host_family": "host-a", "source_family": "source-a",
                   "generator_family": "generator-a", "scenario_family": "scenario-a",
                   "duplicate_group": "duplicate-a", "parent_case_ids": [],
                   "authorization_id": "owner-teacher-v16",
                   "target_origin": "owner-authorized-service-teacher",
                   "sources": [{"source_id": "source-a", "path": "train/source.txt",
                                "sha256": "0" * 64, "parent_source_ids": [],
                                "rights_path": "train/rights.json", "rights_sha256": "0" * 64}],
                   "target": {"path": "train/target.json", "sha256": "0" * 64},
                   "reviews": []}
            manifest = {"schema": TEACHER_SCHEMA, "corpus_id": "teacher-example",
                        "status": "owner-authorized-teacher-training-only-unqualified",
                        "authorization_id": "owner-teacher-v16", "case_count": 3,
                        "cases": [row, {**row, "case_id": "case-b", "split": "development"}]}
            file = root / "manifest.json"
            digest = _write(file, manifest)
            with self.assertRaisesRegex(CognitiveKernelContractError, "count differs"):
                admit_formation_corpus(file, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="owner-teacher-v16")
            manifest["case_count"] = 1
            manifest["cases"] = [row]
            digest = _write(file, manifest)
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "source cannot be read"):
                admit_formation_corpus(file, expected_sha256=digest,
                                       teacher_training=True,
                                       owner_authorization_ref="owner-teacher-v16")
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "unsupported formation corpus schema"):
                admit_formation_corpus(file, expected_sha256=digest)

    def test_teacher_flags_fail_before_private_open_if_mixed_or_missing(self):
        for updates in (
                {"full_fit": True}, {"teacher_fit": False},
                {"teacher_training_manifest": None},
                {"admitted_manifest": Path("/other")},
                {"public_synthetic_curriculum": Path("/diagnostic.jsonl")},
                {"owner_authorization_ref": None},
                {"trust_roster": Path("/fake-roster")},
        ):
            with self.subTest(updates=updates), \
                    patch.object(trainer, "arguments", return_value=_args(**updates)), \
                    patch.object(trainer, "_examples",
                                 side_effect=AssertionError("private bytes opened")), \
                    self.assertRaises(CognitiveKernelContractError):
                trainer.main()

    def test_teacher_examples_require_corpus_policy_and_complete_dimensions(self):
        args = _args()
        complete = SimpleNamespace(case_id="train-a", adjudications=tuple(
            (name, "present") for name in sorted(FULL_ROLE_DIMENSIONS)))
        dev = SimpleNamespace(case_id="dev-a", adjudications=complete.adjudications)
        with patch.object(trainer, "admit_formation_corpus",
                          return_value=object()) as admit, \
                patch.object(trainer, "admitted_rows_v16",
                             side_effect=(iter((complete,)), iter((dev,)))) as rows, \
                patch.object(trainer, "_require_critical_construct_coverage"), \
                patch.object(trainer, "supervised_output_record_v16"), \
                patch.object(trainer, "model_input_sha256_v16",
                             side_effect=("c" * 64, "d" * 64)):
            train, development, status = trainer._examples(args)
        self.assertEqual((train, development), ((complete,), (dev,)))
        self.assertEqual(status,
                         "owner-authorized-teacher-fit-diagnostic-development-unqualified")
        self.assertEqual(admit.call_args.kwargs,
                         {"expected_sha256": "a" * 64,
                          "teacher_training": True,
                          "owner_authorization_ref": "owner-teacher-v16"})
        self.assertEqual([call.kwargs["split"] for call in rows.call_args_list],
                         ["train", "development"])
        self.assertFalse(hasattr(args, "signed_review_receipt_sha256"))

    def test_teacher_run_digest_is_separate_from_signed_fit(self):
        args = _args()
        prepared = {"receipt_sha256": "b" * 64}
        preflight = {"record_sha256": "c" * 64, "transformers_version": "test"}
        config = SimpleNamespace(record=lambda: {"width": 8})
        with patch.object(trainer.shared, "_prepared_kind", return_value="licensed-clone"):
            teacher = trainer._run_manifest_v16(args, prepared, preflight, config, "test")
            args.teacher_fit = False
            probe = trainer._run_manifest_v16(args, prepared, preflight, config, "test")
        self.assertTrue(teacher["teacher_fit"])
        self.assertFalse(teacher["full_fit"])
        self.assertIsNone(teacher["trust_roster_sha256"])
        self.assertIsNone(teacher["signed_review_receipt_sha256"])
        self.assertNotEqual(trainer.shared._record_hash(teacher),
                            trainer.shared._record_hash(probe))

    def test_teacher_binding_rejects_diagnostic_jsonl_without_rights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.jsonl"
            raw = render()
            path.write_bytes(raw)
            with patch.object(runner.training.shared, "_require_private_network_isolation"):
                with self.assertRaisesRegex(CognitiveKernelContractError,
                                            "not one JSON object"):
                    runner.verify_teacher_binding(path, sha256(raw).hexdigest(),
                                                  "owner-directed-mfm-v16-synthetic")


if __name__ == "__main__":
    unittest.main()
