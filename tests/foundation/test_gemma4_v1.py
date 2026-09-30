"""The source-to-role chain is exercised with tiny local safetensors bytes."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from src.alice_foundation import gemma4_v1 as foundation
from src.alice_foundation import gemma4_inventory as inspector


def _bf16(*values: float) -> bytes:
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16)
                    for v in values)


def _weights(*, modified: bool = False) -> bytes:
    payloads = {
        "model.language_model.embed_tokens.weight": ([3, 2], _bf16(1, 2, 3, 4, 5, 6)),
        "model.language_model.norm.weight": ([2], _bf16(1, 1)),
        "model.language_model.layers.0.input_layernorm.weight":
            ([2], _bf16(1, 2 if modified else 1)),
    }
    header = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    for name, (shape, values) in payloads.items():
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [len(data), len(data) + len(values)]}
        data.extend(values)
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + data


class FoundationCustodyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        config = {"architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "model_type": "gemma4_unified", "dtype": "bfloat16",
                  "text_config": {"hidden_size": 2, "vocab_size": 3,
                                  "num_hidden_layers": 1,
                                  "layer_types": ["sliding_attention"]}}
        contents = {"config.json": json.dumps(config).encode(),
                    "model.safetensors": _weights(),
                    "README.md": b"licensed model card", ".gitattributes": b"lfs marker"}
        for name, data in contents.items():
            (self.source / name).write_bytes(data)
        pins = {name: (len(data), sha256(data).hexdigest()) for name, data in contents.items()}
        pin = patch.dict(foundation.PINNED_FILES, pins, clear=True)
        pin.start()
        self.addCleanup(pin.stop)

    def _clone(self):
        self.source_receipt = self.root / "source.json"
        self.clone = self.root / "clone"
        self.clone_receipt = self.root / "clone.json"
        source = foundation.seal_source(self.source, self.source_receipt)
        clone = foundation.clone_verified_source(self.source, self.clone,
                                                 self.clone_receipt, role="mfm")
        self.assertEqual(clone["parent_source_receipt_sha256"], source["receipt_sha256"])
        self.assertEqual((self.source / "model.safetensors").read_bytes(),
                         (self.clone / "model.safetensors").read_bytes())

    def _operation(self):
        path = self.root / "operation.json"
        path.write_text(json.dumps({
            "schema": foundation.OPERATION_SCHEMA,
            "parent_receipt_sha256": foundation.read_receipt(self.clone_receipt)["receipt_sha256"],
            "transform_id": "mfm_formation_v1", "implementation_ref": "git:test-edit@abcdef",
            "parameters": {
                "tensor_name": "model.language_model.layers.0.input_layernorm.weight",
                "expected_tensor_sha256": sha256(_bf16(1, 1)).hexdigest(),
                "replacement_sha256": sha256(_bf16(1, 2)).hexdigest(),
                "replacement_size": len(_bf16(1, 2)),
            },
            "intent": "Exercise tracked byte changes without behavioral claims",
            "evaluation_receipt": None,
        }))
        return path

    def _stage(self):
        operation = self._operation()
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(_bf16(1, 2))
        candidate = self.root / "candidate"
        stage_receipt = self.root / "stage.json"
        foundation.stage_tensor_replacement(
            self.clone_receipt, replacement, candidate, operation, stage_receipt)
        return candidate, operation, stage_receipt

    def test_clone_then_changed_derivative_materializes_with_unqualified_receipt(self):
        self._clone()
        candidate, operation, stage_receipt = self._stage()
        destination = self.root / "derived"
        receipt_path = self.root / "derived.json"
        derivative = foundation.materialize_transform(
            self.clone_receipt, candidate, destination, operation, receipt_path,
            stage_receipt_path=stage_receipt)
        self.assertEqual(derivative["qualification"], "unqualified")
        self.assertEqual(derivative["changed_weight_files"], ["model.safetensors"])
        self.assertEqual(derivative["parent_receipt_sha256"],
                         foundation.read_receipt(self.clone_receipt)["receipt_sha256"])
        self.assertEqual((destination / "model.safetensors").read_bytes(),
                         (candidate / "model.safetensors").read_bytes())
        self.assertNotEqual((destination / "model.safetensors").read_bytes(),
                            (self.clone / "model.safetensors").read_bytes())
        self.assertEqual(foundation.read_receipt(receipt_path), derivative)
        self.assertEqual(foundation.verify_derivative(
            receipt_path, snapshot=destination, expected_role="mfm"), derivative)
        with self.assertRaisesRegex(foundation.FoundationError, "wrong role"):
            foundation.verify_derivative(receipt_path, expected_role="personality")
        (destination / "model.safetensors").write_bytes(_weights())
        with self.assertRaisesRegex(foundation.FoundationError, "changed since receipt"):
            foundation.verify_derivative(receipt_path, expected_role="mfm")
        self.assertFalse((destination / "qualified.json").exists())

    def test_pristine_copy_cannot_be_relabelled_derivative(self):
        self._clone()
        candidate, operation, stage_receipt = self._stage()
        (candidate / "model.safetensors").write_bytes(_weights())
        with self.assertRaisesRegex(foundation.FoundationError, "candidate weight digest"):
            foundation.materialize_transform(self.clone_receipt, candidate,
                                             self.root / "derived", operation,
                                             self.root / "derived.json",
                                             stage_receipt_path=stage_receipt)
        self.assertFalse((self.root / "derived").exists())

    def test_undeclared_second_tensor_edit_fails_even_with_rehashed_stage_receipt(self):
        self._clone()
        candidate, operation, stage_receipt = self._stage()
        weights_path = candidate / "model.safetensors"
        payload = bytearray(weights_path.read_bytes())
        header, data_start, _, _ = inspector._read_header(weights_path)
        other = header["model.language_model.embed_tokens.weight"]
        payload[data_start + other["data_offsets"][0]] ^= 1
        weights_path.write_bytes(payload)
        # Simulate a rewritten, internally consistent stage receipt. A file
        # digest alone cannot prove which tensor changed.
        staged = json.loads(stage_receipt.read_text())
        staged["candidate_weight_sha256"] = sha256(payload).hexdigest()
        staged.pop("receipt_sha256")
        staged["receipt_sha256"] = sha256(foundation._canonical(staged)).hexdigest()
        stage_receipt.write_text(json.dumps(staged))
        with self.assertRaisesRegex(foundation.FoundationError, "undeclared bytes"):
            foundation.materialize_transform(
                self.clone_receipt, candidate, self.root / "derived", operation,
                self.root / "derived.json", stage_receipt_path=stage_receipt)
        self.assertFalse((self.root / "derived").exists())

    def test_stage_receipt_rejects_later_operation_change(self):
        self._clone()
        candidate, operation, stage_receipt = self._stage()
        amended = json.loads(operation.read_text())
        amended["intent"] = "A different stated intervention"
        operation.write_text(json.dumps(amended))
        with self.assertRaisesRegex(foundation.FoundationError, "stage receipt does not bind"):
            foundation.materialize_transform(
                self.clone_receipt, candidate, self.root / "derived", operation,
                self.root / "derived.json", stage_receipt_path=stage_receipt)
        self.assertFalse((self.root / "derived").exists())

    def test_named_tensor_patch_preserves_all_other_bytes_and_seals_derivative(self):
        self._clone()
        tensor = "model.language_model.layers.0.input_layernorm.weight"
        old = _bf16(1, 1)
        new = _bf16(1, 2)
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(new)
        operation = self._operation()
        candidate = self.root / "candidate"
        stage_receipt = self.root / "stage.json"
        staged = foundation.stage_tensor_replacement(
            self.clone_receipt, replacement, candidate, operation, stage_receipt)
        self.assertEqual(staged["qualification"], "unqualified")
        self.assertEqual(foundation.read_receipt(stage_receipt), staged)
        original = (self.clone / "model.safetensors").read_bytes()
        changed = (candidate / "model.safetensors").read_bytes()
        self.assertEqual(len(original), len(changed))
        header, data_start, _, _ = inspector._read_header(self.clone / "model.safetensors")
        start = data_start + header[tensor]["data_offsets"][0]
        end = data_start + header[tensor]["data_offsets"][1]
        self.assertEqual(original[:start], changed[:start])
        self.assertEqual(original[start:end], old)
        self.assertEqual(changed[start:end], new)
        self.assertEqual(original[end:], changed[end:])
        for name in ("README.md", ".gitattributes", "config.json"):
            self.assertEqual((candidate / name).read_bytes(), (self.clone / name).read_bytes())
        derivative = foundation.materialize_transform(
            self.clone_receipt, candidate, self.root / "derived", operation,
            self.root / "derived.json", stage_receipt_path=stage_receipt)
        self.assertEqual(derivative["qualification"], "unqualified")
        self.assertEqual(derivative["stage_receipt_sha256"], staged["receipt_sha256"])
        self.assertEqual(foundation.verify_derivative(self.root / "derived.json",
                                                      expected_role="mfm"), derivative)

    def test_tensor_patch_refuses_wrong_before_or_after_digest(self):
        self._clone()
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(_bf16(1, 2))
        operation = self._operation()
        content = json.loads(operation.read_text())
        content["parameters"] = {
            "tensor_name": "model.language_model.layers.0.input_layernorm.weight",
            "expected_tensor_sha256": "0" * 64,
            "replacement_sha256": sha256(replacement.read_bytes()).hexdigest(),
            "replacement_size": 4,
        }
        operation.write_text(json.dumps(content))
        with self.assertRaisesRegex(foundation.FoundationError, "changed during patch"):
            foundation.stage_tensor_replacement(self.clone_receipt, replacement,
                                                self.root / "candidate", operation,
                                                self.root / "stage.json")
        self.assertFalse((self.root / "candidate").exists())
        content["parameters"]["expected_tensor_sha256"] = sha256(_bf16(1, 1)).hexdigest()
        content["parameters"]["replacement_sha256"] = "f" * 64
        operation.write_text(json.dumps(content))
        with self.assertRaisesRegex(foundation.FoundationError, "hash or size mismatch"):
            foundation.stage_tensor_replacement(self.clone_receipt, replacement,
                                                self.root / "candidate", operation,
                                                self.root / "stage.json")

    def test_modified_parent_and_mismatched_operation_fail_closed(self):
        self._clone()
        candidate, operation, stage_receipt = self._stage()
        (self.clone / "model.safetensors").write_bytes(_weights(modified=True))
        with self.assertRaisesRegex(foundation.FoundationError, "parent snapshot changed"):
            foundation.materialize_transform(self.clone_receipt, candidate,
                                             self.root / "derived", operation,
                                             self.root / "derived.json",
                                             stage_receipt_path=stage_receipt)
        (self.clone / "model.safetensors").write_bytes(_weights())
        payload = json.loads(operation.read_text())
        payload["parent_receipt_sha256"] = "0" * 64
        operation.write_text(json.dumps(payload))
        with self.assertRaisesRegex(foundation.FoundationError, "unrelated"):
            foundation.materialize_transform(self.clone_receipt, candidate,
                                             self.root / "derived", operation,
                                             self.root / "derived.json",
                                             stage_receipt_path=stage_receipt)

    def test_license_notice_and_structure_are_required(self):
        self._clone()
        candidate, operation, stage_receipt = self._stage()
        (candidate / "README.md").unlink()
        with self.assertRaisesRegex(foundation.FoundationError, "nonweight files or file set"):
            foundation.materialize_transform(self.clone_receipt, candidate,
                                             self.root / "derived", operation,
                                             self.root / "derived.json",
                                             stage_receipt_path=stage_receipt)
        shutil.copy2(self.source / "README.md", candidate / "README.md")
        (candidate / "model.safetensors").write_bytes(b"not a tensor file")
        with self.assertRaisesRegex(foundation.FoundationError, "derivative structure"):
            foundation.materialize_transform(self.clone_receipt, candidate,
                                             self.root / "derived", operation,
                                             self.root / "derived.json",
                                             stage_receipt_path=stage_receipt)

    def test_unexpected_source_file_and_symlink_fail_admission(self):
        (self.source / "extra.txt").write_text("unexpected")
        with self.assertRaisesRegex(foundation.FoundationError, "source file set mismatch"):
            foundation.verify_source(self.source)
        (self.source / "extra.txt").unlink()
        (self.source / "link").symlink_to(self.source / "README.md")
        with self.assertRaisesRegex(foundation.FoundationError, "nonregular entry"):
            foundation.verify_source(self.source)


if __name__ == "__main__":
    unittest.main()
