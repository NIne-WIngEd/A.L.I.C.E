"""Offline custody/admission tests using tiny publisher and code fixtures.

PINNED_FILES is patched to eight tiny public files. These tests exercise actual
foundation clone verification, not Gemma weights, model inference, library
compatibility, inherited-personality suppression, or behavior qualification.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.alice_foundation import gemma4_v1 as foundation
from src.alice_personality.gemma_n0 import preparation


RUNTIME = {"backend": "transformers", "dtype": "bfloat16",
           "torch_version": "2.7.1+fixture", "transformers_version": "5.0.0.fixture"}


class GemmaN0PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        config = {"architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "model_type": "gemma4_unified"}
        contents = {
            ".gitattributes": b"public fixture lfs notice",
            "README.md": b"public fixture model notice",
            "config.json": json.dumps(config).encode(),
            "generation_config.json": b"{}",
            "model.safetensors": b"public custody fixture, not real model tensors",
            "processor_config.json": b"{}",
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b"{}",
        }
        for name, payload in contents.items():
            (self.source / name).write_bytes(payload)
        pins = {name: (len(data), sha256(data).hexdigest()) for name, data in contents.items()}
        pin = patch.dict(foundation.PINNED_FILES, pins, clear=True)
        pin.start()
        self.addCleanup(pin.stop)
        self.code = self.root / "code"
        self.code.mkdir()
        self.code_paths = {}
        for name in ("preparation.py", "backbone.py"):
            path = self.code / name
            path.write_text(f"# public {name} fixture; no inference\n", encoding="utf-8")
            self.code_paths[name] = path
        implementation = patch.dict(preparation.IMPLEMENTATION_PATHS, self.code_paths, clear=True)
        implementation.start()
        self.addCleanup(implementation.stop)
        self.clone = self.root / "personality"
        self.clone_receipt = self.root / "clone.json"
        self.prepared_receipt = self.root / "prepared.json"
        self.clone_record = foundation.clone_verified_source(
            self.source, self.clone, self.clone_receipt, role="personality")

    def _prepare(self, **kwargs):
        return preparation.prepare(self.clone_receipt, self.prepared_receipt,
                                   runtime=dict(RUNTIME), **kwargs)

    def _reseal(self, path, update):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        update(receipt)
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = sha256(preparation._canonical(receipt)).hexdigest()
        path.write_bytes(preparation._canonical(receipt) + b"\n")

    def test_role_clone_preparation_roundtrip_is_unqualified_and_unchanged(self):
        before = {path.name: path.read_bytes() for path in self.clone.iterdir()}
        receipt = self._prepare(snapshot=self.clone)
        verified = preparation.verify_prepared(
            self.prepared_receipt, snapshot=self.clone, expected_runtime=RUNTIME)
        self.assertEqual(receipt, verified)
        self.assertEqual(receipt["state"], "PREPARED_UNQUALIFIED")
        self.assertIsNone(receipt["behavior_qualification"])
        self.assertEqual(receipt["clone_receipt_sha256"], self.clone_record["receipt_sha256"])
        self.assertEqual(receipt["files"], self.clone_record["files"])
        self.assertEqual(len(receipt["implementation"]), 2)
        self.assertEqual(receipt["interfaces"], {"input": None, "output": None})
        self.assertEqual(receipt["feature_scope"]["personality_authority"], ["N1", "N2", "N3", "EIPM"])
        self.assertFalse(receipt["feature_scope"]["personality_judgments"])
        self.assertFalse(receipt["feature_scope"]["text_generation"])
        self.assertFalse(receipt["feature_scope"]["upstream_tensor_mutation"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.clone.iterdir()})
        self.assertEqual(self.prepared_receipt.read_bytes(), preparation._canonical(receipt) + b"\n")

    def test_wrong_role_clone_is_rejected(self):
        self._reseal(self.clone_receipt, lambda receipt: receipt.update(role="mfm"))
        with self.assertRaisesRegex(preparation.PreparationError, "wrong role"):
            self._prepare()
        self.assertFalse(self.prepared_receipt.exists())

    def test_derivative_or_source_receipt_cannot_enter_personality_v1(self):
        for schema in (foundation.DERIVATIVE_SCHEMA, foundation.SOURCE_SCHEMA):
            with self.subTest(schema=schema):
                self._reseal(self.clone_receipt, lambda receipt: receipt.update(schema=schema))
                with self.assertRaisesRegex(preparation.PreparationError, "unexpected receipt schema"):
                    self._prepare()
        self.assertFalse(self.prepared_receipt.exists())

    def test_missing_publisher_file_prevents_preparation(self):
        (self.clone / "model.safetensors").unlink()
        with self.assertRaisesRegex(preparation.PreparationError, "file set changed"):
            self._prepare()
        self.assertFalse(self.prepared_receipt.exists())

    def test_loading_rehashes_snapshot_instead_of_trusting_old_receipts(self):
        self._prepare()
        weights = self.clone / "model.safetensors"
        data = weights.read_bytes()
        weights.write_bytes(b"X" + data[1:])
        with self.assertRaisesRegex(preparation.PreparationError, "changed since receipt"):
            preparation.verify_prepared(self.prepared_receipt)

    def test_loading_rejects_unexpected_snapshot_file(self):
        self._prepare()
        (self.clone / "unexpected.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(preparation.PreparationError, "file set changed"):
            preparation.verify_prepared(self.prepared_receipt)

    def test_loading_rejects_changed_implementation(self):
        self._prepare()
        self.code_paths["backbone.py"].write_text("# changed execution code\n", encoding="utf-8")
        with self.assertRaisesRegex(preparation.PreparationError, "implementation changed"):
            preparation.verify_prepared(self.prepared_receipt)

    def test_current_module_implementation_hashes_are_bound_without_importing_model_code(self):
        real_paths = {
            "preparation.py": Path(preparation.__file__),
            "backbone.py": Path(preparation.__file__).with_name("backbone.py"),
        }
        with patch.dict(preparation.IMPLEMENTATION_PATHS, real_paths, clear=True):
            receipt = self._prepare()
            self.assertEqual(receipt["implementation"], [
                {"path": name, "sha256": sha256(path.read_bytes()).hexdigest()}
                for name, path in sorted(real_paths.items())])
            self.assertEqual(preparation.verify_prepared(self.prepared_receipt), receipt)

    def test_both_implementation_files_must_exist_before_preparation(self):
        self.code_paths["backbone.py"].unlink()
        with self.assertRaisesRegex(preparation.PreparationError, "implementation backbone.py"):
            self._prepare()
        self.assertFalse(self.prepared_receipt.exists())

    def test_runtime_is_explicit_and_complete(self):
        invalid = [None, {}, {**RUNTIME, "torch_version": ""},
                   {**RUNTIME, "transformers_version": " 5.0.0"},
                   {**RUNTIME, "backend": "remote_api"},
                   {**RUNTIME, "dtype": ["bfloat16"]},
                   {**RUNTIME, "trust_remote_code": True}]
        for value in invalid:
            with self.subTest(runtime=value), self.assertRaises(preparation.PreparationError):
                preparation.prepare(self.clone_receipt, self.prepared_receipt, runtime=value)
        self.assertFalse(self.prepared_receipt.exists())

    def test_actual_runtime_version_or_dtype_mismatch_fails_closed(self):
        self._prepare()
        for field, changed in (("torch_version", "2.7.2"), ("transformers_version", "5.0.1"),
                               ("dtype", "float32")):
            with self.subTest(field=field):
                with self.assertRaisesRegex(preparation.PreparationError, "runtime does not match"):
                    preparation.verify_prepared(self.prepared_receipt,
                                                expected_runtime={**RUNTIME, field: changed})

    def test_preparation_receipt_cannot_be_overwritten(self):
        self._prepare()
        original = self.prepared_receipt.read_bytes()
        with self.assertRaisesRegex(preparation.PreparationError, "already exists"):
            self._prepare()
        self.assertEqual(original, self.prepared_receipt.read_bytes())

    def test_preparation_receipt_cannot_be_written_into_snapshot(self):
        forbidden = self.clone / "prepared.json"
        with self.assertRaisesRegex(preparation.PreparationError, "outside the checkpoint"):
            preparation.prepare(self.clone_receipt, forbidden, runtime=RUNTIME)
        self.assertFalse(forbidden.exists())
        self.assertEqual(foundation.verify_clone(self.clone_receipt)["files"], self.clone_record["files"])

    def test_unsealed_edit_is_rejected(self):
        self._prepare()
        data = json.loads(self.prepared_receipt.read_text(encoding="utf-8"))
        data["state"] = "APPROVED"
        self.prepared_receipt.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(preparation.PreparationError, "digest mismatch"):
            preparation.verify_prepared(self.prepared_receipt)

    def test_resealed_behavior_or_role_claim_is_still_rejected(self):
        changes = [{"state": "APPROVED"}, {"behavior_qualification": "PASS"},
                   {"role": "mfm"},
                   {"feature_scope": {**preparation.FEATURE_SCOPE, "personality_judgments": True}},
                   {"feature_scope": {**preparation.FEATURE_SCOPE, "personality_judgments": 0}}]
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                receipt_path = self.root / f"prepared-{index}.json"
                preparation.prepare(self.clone_receipt, receipt_path, runtime=RUNTIME)
                self._reseal(receipt_path, lambda receipt: receipt.update(change))
                with self.assertRaises(preparation.PreparationError):
                    preparation.verify_prepared(receipt_path)

    def test_resealed_model_or_revision_substitution_is_rejected(self):
        for field, changed in (("repository", "google/gemma-4-12B-it"),
                               ("revision", "unreviewed-new-head")):
            with self.subTest(field=field):
                receipt_path = self.root / f"prepared-{field}.json"
                preparation.prepare(self.clone_receipt, receipt_path, runtime=RUNTIME)
                self._reseal(receipt_path, lambda receipt: receipt.update({field: changed}))
                with self.assertRaisesRegex(preparation.PreparationError, "exact publisher pin"):
                    preparation.verify_prepared(receipt_path)

    def test_clone_binding_reseal_does_not_admit_changed_clone_role(self):
        self._prepare()
        self._reseal(self.clone_receipt, lambda receipt: receipt.update(role="mfm"))
        with self.assertRaisesRegex(preparation.PreparationError, "wrong role"):
            preparation.verify_prepared(self.prepared_receipt)

    def test_requested_snapshot_must_match_receipted_personality_clone(self):
        with self.assertRaisesRegex(preparation.PreparationError, "path differs"):
            self._prepare(snapshot=self.source)

    def test_existing_resolved_interface_contract_is_hashed_and_reverified(self):
        contract = self.root / "public-feature-interface.json"
        contract.write_text('{"purpose":"public feature interface fixture"}', encoding="utf-8")
        receipt = self._prepare(output_contract={"contract_id": "existing-public-interface.v1",
                                                 "path": contract})
        binding = receipt["interfaces"]["output"]
        self.assertEqual(binding["sha256"], sha256(contract.read_bytes()).hexdigest())
        self.assertEqual(preparation.verify_prepared(self.prepared_receipt), receipt)
        contract.write_text('{"purpose":"changed fixture"}', encoding="utf-8")
        with self.assertRaisesRegex(preparation.PreparationError, "interface contract changed"):
            preparation.verify_prepared(self.prepared_receipt)

    def test_cli_requires_runtime_file_and_verifies_actual_runtime(self):
        runtime_path = self.root / "runtime.json"
        runtime_path.write_text(json.dumps(RUNTIME), encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            result = preparation.main(["prepare", str(self.clone_receipt),
                                       str(self.prepared_receipt), "--runtime-json", str(runtime_path)])
            self.assertEqual(result, 0)
            result = preparation.main(["verify-prepared", str(self.prepared_receipt),
                                       "--runtime-json", str(runtime_path)])
            self.assertEqual(result, 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(all(row["state"] == "PREPARED_UNQUALIFIED" for row in rows))
        runtime_path.write_text(json.dumps({**RUNTIME, "torch_version": "mismatch"}), encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(preparation.main(["verify-prepared", str(self.prepared_receipt),
                                               "--runtime-json", str(runtime_path)]), 2)

    def test_cli_rejects_half_resolved_contract_without_writing_artifact(self):
        runtime_path = self.root / "runtime.json"
        runtime_path.write_text(json.dumps(RUNTIME), encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            result = preparation.main(["prepare", str(self.clone_receipt), str(self.prepared_receipt),
                                       "--runtime-json", str(runtime_path),
                                       "--input-contract-id", "existing-contract.v1"])
        self.assertEqual(result, 2)
        self.assertFalse(self.prepared_receipt.exists())


if __name__ == "__main__":
    unittest.main()
