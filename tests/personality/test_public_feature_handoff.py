"""Tiny PUBLIC mechanics stand-ins; no Gemma/FewRel competence or source proof.

Only producer source verification and runtime are injected. Real Torch BF16
banks, own readout optimization, receipt/token hashes and portable import run.
Every fixture closure is PUBLIC_MECHANICS_ONLY; production import rejects it.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from src.alice_personality.gemma_n0 import public_feature_handoff as handoff
from src.alice_personality.gemma_n0 import public_semantic_experiment as producer
from tests.personality.test_gemma_n0_semantic_experiment import _Provider, _Tokenizer


class PublicFeatureHandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        data, snapshot = self.root / "public-fixture-data", self.root / "public-fixture-publisher"
        data.mkdir()
        snapshot.mkdir()
        rows = []
        for split, family in (("train", "P1"), ("train", "P2"), ("dev", "P3")):
            rid = f"fewrel:{split}:" + sha256(family.encode()).hexdigest()[:20]
            rows.append({"id": rid, "split": split, "target_relation_key": family,
                "candidate_relation_keys": ["P1", "P2", "P3"],
                "target_candidate_index": int(family[-1]) - 1,
                "sentence": f"Ada public {split} relation-{family[-1]} knows Bea.",
                "head": {"text": "Ada", "type": "Qhumanhead"}, "tail": {"text": "Bea", "type": "Qhumantail"}})
        files = {"rows": data / "train_dev_rows.jsonl", "bank": data / "train_dev_bank.json",
            "manifest": data / "manifest.json", "audit": data / "audit.json"}
        files["rows"].write_bytes(b"".join(handoff._canonical(row) + b"\n" for row in rows))
        files["bank"].write_bytes(handoff._canonical({"relations": {
            "P1": {"semantic_text": "One entity knows another."},
            "P2": {"semantic_text": "One entity assists another."},
            "P3": {"semantic_text": "One entity opposes another."}}}) + b"\n")
        for name in ("manifest", "audit"):
            files[name].write_bytes(handoff._canonical({"scope": "tiny_public_fixture_only"}) + b"\n")
        self.source_file, self.prepared_file, self.clone_file = (self.root / name for name in
            ("source.json", "prepared.json", "clone.json"))
        self.source = handoff._seal({"schema": "alice-personality-gemma-n0-public-fewrel-admission-v1",
            "state": "ELIGIBLE_PUBLIC_SOURCE_UNQUALIFIED", "public_train_dev_declared": True,
            "target_keys_indices_and_ids_are_model_input": False, "private_identity_data": False,
            "n0_approved": False, "statistics": {"normalized_sentence_only_overlap": 0},
            "provenance": {"scope": "tiny_human_written_public_fixtures_only"},
            "files": [{"kind": name, "path": str(path), "size": path.stat().st_size,
                "sha256": handoff._hash(path)} for name, path in files.items()]})
        clone = handoff._seal({"schema": "alice-gemma4-v1-clone-v1", "repository": handoff.MODEL,
            "revision": handoff.REVISION, "role": "personality", "files": [],
            "snapshot_path": str(snapshot), "qualification": "unqualified",
            "meaning": "fixture metadata only; no publisher bytes"})
        self.prepared = handoff._seal({"schema": "alice-personality-gemma-n0-preparation-v2",
            "repository": handoff.MODEL, "revision": handoff.REVISION, "role": "personality",
            "state": "PREPARED_UNQUALIFIED", "behavior_qualification": None,
            "snapshot_path": str(snapshot), "clone_receipt_path": str(self.clone_file),
            "clone_receipt_sha256": clone["receipt_sha256"], "files": [],
            "runtime": {"backend": "transformers", "dtype": "bfloat16",
                "torch_version": str(torch.__version__), "transformers_version": "fixture-only"},
            "model_geometry": {"hidden_state_count": 3, "hidden_size": 8,
                "num_hidden_layers": 2, "max_position_embeddings": 128}})
        for path, value in ((self.source_file, self.source), (self.prepared_file, self.prepared), (self.clone_file, clone)):
            handoff._write(path, value)
        self.tokenizer, self.provider = _Tokenizer(), _Provider()
        self.provider_factory = Mock(return_value=self.provider)
        for name, value in (("_verify_source", lambda path: deepcopy(self.source)),
            ("_verify_preparation", lambda path: deepcopy(self.prepared)),
            ("_runtime", lambda prepared: (torch, self.tokenizer, self.provider_factory))):
            mock = patch.object(producer, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.recipe = {"train_rows_per_family": 1, "dev_rows_per_family": 1,
            "style_rows_per_dev_family": 1, "width": 4, "steps": 2, "seed": 37,
            "learning_rate": 0.001, "weight_decay": 0.01}
        self.plan_file, self.cache = self.root / "plan.json", self.root / "cache"
        producer.create_plan(source_admission=self.source_file, preparation_receipt=self.prepared_file,
            output_plan=self.plan_file, recipe=self.recipe)
        with redirect_stdout(io.StringIO()):
            self.experiment = producer.run_experiment(plan_path=self.plan_file,
                output_directory=self.root / "run", cache_directory=self.cache)
        self.experiment_file = self.root / "run/experiment.json"
        self.experiment_sha = handoff._hash(self.experiment_file)
        self.provider_factory.reset_mock()
        self.provider_factory.side_effect = AssertionError("handoff cannot load or forward a publisher model")

    def export(self, name="package"):
        self.package = self.root / name
        self.pins = handoff.export_public_features(experiment_path=self.experiment_file,
            expected_experiment_sha256=self.experiment_sha, plan_path=self.plan_file,
            cache_directory=self.cache, output_directory=self.package, mechanics_only=True)
        return self.pins

    def admit(self, allow_mechanics=True):
        return handoff.import_public_features(self.package,
            expected_manifest_sha256=self.pins["manifest_sha256"],
            expected_closure_sha256=self.pins["closure_sha256"], allow_mechanics_only=allow_mechanics)

    def repin(self):
        """Adversarial stand-in external pins: validation still must reject bad semantics."""
        closure = handoff._read(self.package / "closure.json")
        closure.pop("receipt_sha256")
        # Preserve every enclosing integrity link to isolate semantic validation.
        # These invented pins are test mechanics, never trusted producer custody.
        exp_path = self.package / "original/experiment.json"
        experiment = handoff._read(exp_path)
        experiment.pop("receipt_sha256")
        for record in experiment["cache"]:
            for kind, suffix in (("metadata", ".json"), ("tensor", ".pt")):
                record[kind + "_sha256"] = handoff._hash(self.package / f"features/{record['key']}{suffix}")
        experiment["binding"]["cache_tensor_bindings"] = [
            {"key": row["key"], "tensor_sha256": row["tensor_sha256"]}
            for row in sorted(experiment["cache"], key=lambda row: row["key"])]
        exp_path.write_bytes(handoff._canonical(handoff._seal(experiment)) + b"\n")
        closure["original_file_sha256"]["experiment"] = handoff._hash(exp_path)
        closure["completed_experiment_file_sha256"] = handoff._hash(exp_path)
        for row in closure["inventory"]:
            path = self.package / row["path"]
            row["size"], row["sha256"] = path.stat().st_size, handoff._hash(path)
        (self.package / "closure.json").write_bytes(handoff._canonical(handoff._seal(closure)) + b"\n")
        manifest = handoff._read(self.package / "manifest.json")
        manifest.pop("receipt_sha256")
        manifest["files"] = [*closure["inventory"], {"path": "closure.json",
            "size": (self.package / "closure.json").stat().st_size,
            "sha256": handoff._hash(self.package / "closure.json")}]
        manifest["files"].sort(key=lambda row: row["path"])
        manifest["closure_file_sha256"] = handoff._hash(self.package / "closure.json")
        (self.package / "manifest.json").write_bytes(handoff._canonical(handoff._seal(manifest)) + b"\n")
        self.pins = {"manifest_sha256": handoff._hash(self.package / "manifest.json"),
            "closure_sha256": handoff._hash(self.package / "closure.json")}

    def test_reproducible_closed_mechanics_export_and_source_free_import(self):
        first = self.export()
        originals = {name: (self.package / f"original/{name}.json").read_bytes()
            for name in ("experiment", "plan", "preparation", "source_admission", "feature_budget", "clone")}
        self.assertEqual(originals["plan"], self.plan_file.read_bytes())
        first_files = {path.relative_to(self.package).as_posix(): handoff._hash(path)
            for path in self.package.rglob("*") if path.is_file()}
        second = self.export("another-package")
        self.assertEqual(first, second)
        self.assertEqual(first_files, {path.relative_to(self.package).as_posix(): handoff._hash(path)
            for path in self.package.rglob("*") if path.is_file()})
        with patch.object(producer, "_load_plan", side_effect=AssertionError("consumer source read")), \
            patch.object(producer, "_runtime", side_effect=AssertionError("consumer model/processor load")):
            result = self.admit()
        self.provider_factory.assert_not_called()
        self.assertEqual(result.geometry["hidden_state_count"], 3)
        self.assertEqual(len(result.examples), 3)
        self.assertEqual(len(next(row for row in result.examples if row["split"] == "dev")["styles"]), 2)
        self.assertTrue(result.binding["mechanics_only"])
        self.assertFalse(result.binding["publisher_checkpoint_loaded"])
        self.assertFalse(result.binding["consumer_retokenized"])
        self.assertFalse(result.binding["n0_approved"])
        self.assertEqual(result.plan["recipe"], self.recipe)
        for row in result.examples:
            self.assertEqual(row["source"].layers.dtype, torch.bfloat16)
            self.assertFalse(row["source"].layers.requires_grad)
        for name, raw in originals.items():
            self.assertEqual(raw, (self.package / f"original/{name}.json").read_bytes())
        output = self.root / "import.json"
        handoff.write_import_receipt(result, output)
        self.assertEqual(handoff._read(output), result.binding)
        with self.assertRaises(handoff.HandoffError):
            handoff.write_import_receipt(result, output)
        with self.assertRaisesRegex(handoff.HandoffError, "outside source"):
            handoff.write_import_receipt(result, self.package / "import.json")

    def test_fixture_scope_external_pins_and_create_only(self):
        self.export()
        with self.assertRaisesRegex(handoff.HandoffError, "mechanics-only"):
            self.admit(False)
        with self.assertRaisesRegex(handoff.HandoffError, "wrong pinned Gemma geometry"):
            handoff.export_public_features(experiment_path=self.experiment_file,
                expected_experiment_sha256=self.experiment_sha, plan_path=self.plan_file,
                cache_directory=self.cache, output_directory=self.root / "production-claim")
        with self.assertRaisesRegex(handoff.HandoffError, "external pin"):
            handoff.import_public_features(self.package, expected_manifest_sha256="0" * 64,
                expected_closure_sha256=self.pins["closure_sha256"], allow_mechanics_only=True)
        with self.assertRaisesRegex(handoff.HandoffError, "fresh absolute"):
            self.export()
        with self.assertRaisesRegex(handoff.HandoffError, "external pin"):
            handoff.export_public_features(experiment_path=self.experiment_file,
                expected_experiment_sha256="0" * 64, plan_path=self.plan_file,
                cache_directory=self.cache, output_directory=self.root / "bad-pin", mechanics_only=True)

    def test_missing_bank_extra_member_and_directory_rejected(self):
        self.export()
        file = next((self.package / "features").glob("*.pt"))
        saved = file.read_bytes()
        file.unlink()
        with self.assertRaises(handoff.HandoffError):
            self.admit()
        file.write_bytes(saved)
        (self.package / "unauthorized.json").write_text("{}")
        with self.assertRaisesRegex(handoff.HandoffError, "extra or missing"):
            self.admit()
        (self.package / "unauthorized.json").unlink()
        (self.package / "extra-empty-directory").mkdir()
        with self.assertRaisesRegex(handoff.HandoffError, "unsupported directories"):
            self.admit()

    def test_resealed_flag_and_token_evidence_mutation_rejected(self):
        self.export()
        closure_path = self.package / "closure.json"
        raw = closure_path.read_bytes()
        closure = handoff._read(closure_path)
        closure.pop("receipt_sha256")
        closure["n0_approved"] = 0
        closure_path.write_bytes(handoff._canonical(handoff._seal(closure)) + b"\n")
        self.repin()
        with self.assertRaisesRegex(handoff.HandoffError, "unqualified boundary"):
            self.admit()
        closure_path.write_bytes(raw)
        tensor_path = next((self.package / "features").glob("*.pt"))
        payload = torch.load(tensor_path, weights_only=True)
        payload["input_ids"][0] += 1
        torch.save(payload, tensor_path)
        metadata_path = tensor_path.with_suffix(".json")
        metadata = handoff._read(metadata_path)
        metadata.pop("receipt_sha256")
        metadata["tensor_sha256"] = handoff._hash(tensor_path)
        metadata_path.write_bytes(handoff._canonical(handoff._seal(metadata)) + b"\n")
        self.repin()
        with self.assertRaisesRegex(handoff.HandoffError, "token IDs/masks/roles"):
            self.admit()

    def test_nonfinite_and_partial_states_rejected_despite_resealed_inventory(self):
        self.export()
        tensor_path = next((self.package / "features").glob("*.pt"))
        original = torch.load(tensor_path, weights_only=True)
        for kind in ("nonfinite", "partial", "invalid_mask"):
            with self.subTest(kind=kind):
                payload = deepcopy(original)
                if kind == "nonfinite":
                    payload["layers"][0, 0, 0] = float("nan")
                elif kind == "partial":
                    payload["layers"] = payload["layers"][:-1]
                else:
                    payload["attention_mask"][:] = False
                torch.save(payload, tensor_path)
                metadata_path = tensor_path.with_suffix(".json")
                metadata = handoff._read(metadata_path)
                metadata.pop("receipt_sha256")
                metadata["tensor_sha256"] = handoff._hash(tensor_path)
                metadata_path.write_bytes(handoff._canonical(handoff._seal(metadata)) + b"\n")
                self.repin()
                with self.assertRaises(ValueError):
                    self.admit()

    def test_source_change_during_closure_preserves_partial_without_closed_receipt(self):
        original = producer._load_plan
        count = 0
        def changed(path):
            nonlocal count
            count += 1
            result = original(path)
            if count == 2:
                self.source_file.write_bytes(self.source_file.read_bytes() + b" ")
            return result
        with patch.object(producer, "_load_plan", side_effect=changed), \
            self.assertRaisesRegex(handoff.HandoffError, "changed before closure"):
            self.export()
        self.assertTrue(self.package.is_dir())
        self.assertFalse((self.package / "closure.json").exists())
        self.assertFalse((self.package / "manifest.json").exists())
        self.provider_factory.assert_not_called()

    def test_inventory_path_alias_extra_and_duplicate_json_fail_before_tensor_load(self):
        self.export()
        manifest_path = self.package / "manifest.json"
        original = manifest_path.read_bytes()
        for bad in ("../outside.pt", "features/../outside.pt", "/outside.pt", "features\\bad.pt"):
            with self.subTest(member=bad):
                manifest = json.loads(original)
                manifest.pop("receipt_sha256")
                manifest["files"][0]["path"] = bad
                manifest_path.write_bytes(handoff._canonical(handoff._seal(manifest)) + b"\n")
                self.pins["manifest_sha256"] = handoff._hash(manifest_path)
                with patch.object(torch, "load", side_effect=AssertionError("unsafe import reached tensors")), \
                    self.assertRaises(handoff.HandoffError):
                    self.admit()
        manifest_path.write_bytes(original[:-2] + b',"schema":"duplicate"}\n')
        self.pins["manifest_sha256"] = handoff._hash(manifest_path)
        with self.assertRaisesRegex(handoff.HandoffError, "duplicate JSON key"):
            self.admit()


if __name__ == "__main__":
    unittest.main()
