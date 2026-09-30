"""Structural checkpoint inventory works without downloading or executing a model."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct
import tempfile
import unittest

from src.alice_foundation import gemma4_inventory as inspector


def _bf16(*values: float) -> bytes:
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16)
                    for v in values)


class CheckpointInventoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.config = {
            "model_type": "gemma4_unified", "dtype": "bfloat16",
            "text_config": {"hidden_size": 2, "vocab_size": 3,
                            "num_hidden_layers": 2,
                            "layer_types": ["sliding_attention", "full_attention"]},
            "audio_config": {"hidden_size": 4},
            "vision_config": {"mm_embed_dim": 5},
        }
        self.payloads = {
            "model.language_model.embed_tokens.weight": ([3, 2], _bf16(1, 2, 3, 4, 5, 6)),
            "model.language_model.norm.weight": ([2], _bf16(1, 1)),
            "model.language_model.layers.0.input_layernorm.weight": ([2], _bf16(1, 1)),
            "model.language_model.layers.0.self_attn.q_proj.weight": ([2, 2], _bf16(1, 2, 3, 4)),
            "model.language_model.layers.1.input_layernorm.weight": ([2], _bf16(1, 1)),
            "model.language_model.layers.1.self_attn.q_proj.weight": ([2, 2], _bf16(2, 4, 6, 8)),
            "model.embed_audio.embedding_projection.weight": ([2, 4], _bf16(*range(8))),
            "model.embed_vision.embedding_projection.weight": ([2, 5], _bf16(*range(10))),
        }
        self._write_checkpoint()

    def _write_checkpoint(self, *, header_override=None, trailing=b""):
        header = {"__metadata__": {"format": "pt"}}
        data = bytearray()
        for name, (shape, payload) in self.payloads.items():
            header[name] = {"dtype": "BF16", "shape": shape,
                            "data_offsets": [len(data), len(data) + len(payload)]}
            data.extend(payload)
        if header_override:
            header_override(header)
        encoded = json.dumps(header).encode("utf-8")
        (self.snapshot / "model.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + data + trailing)
        (self.snapshot / "config.json").write_text(json.dumps(self.config))

    def test_full_hash_and_config_cross_checks(self):
        digest = sha256((self.snapshot / "model.safetensors").read_bytes()).hexdigest()
        result = inspector.inspect_checkpoint(
            self.snapshot, revision="a" * 40, expected_sha256=digest,
            sample_bf16_tensors=3, sample_elements=6)
        self.assertEqual(result["summary"]["tensor_count"], 8)
        self.assertEqual(result["summary"]["dtypes"], {"BF16": 8})
        self.assertEqual(result["provenance"]["full_weight_sha256"], digest)
        self.assertEqual(result["provenance"]["declared_revision"], "a" * 40)
        self.assertEqual(result["groups"]["layers"]["model.language_model.layers.1"][
            "configured_attention_type"], "full_attention")
        self.assertEqual(result["groups"]["name_modalities"]["vision_named"]["tensors"], 1)
        self.assertEqual(result["groups"]["module_templates"][
            "model.language_model.layers.*.self_attn.q_proj"]["tensors"], 2)
        self.assertEqual(len(result["bf16_samples"]), 3)
        self.assertEqual(result["manifest_sha256"], sha256(inspector._canonical({
            k: v for k, v in result.items() if k != "manifest_sha256"})).hexdigest())
        self.assertIn("do not identify", result["interpretation_limit"])

    def test_expected_digest_must_match_and_requires_hash(self):
        with self.assertRaisesRegex(inspector.InventoryError, "differs"):
            inspector.inspect_checkpoint(self.snapshot, expected_sha256="f" * 64)
        with self.assertRaisesRegex(inspector.InventoryError, "requires"):
            inspector.inspect_checkpoint(self.snapshot, expected_sha256="f" * 64,
                                         hash_weights=False)
        result = inspector.inspect_checkpoint(self.snapshot, hash_weights=False)
        self.assertIsNone(result["provenance"]["full_weight_sha256"])
        self.assertFalse(result["provenance"]["full_weight_hash_performed"])

    def test_gap_overlap_shape_and_trailing_bytes_fail_closed(self):
        def gap(header):
            header["model.language_model.norm.weight"]["data_offsets"][0] += 1
        self._write_checkpoint(header_override=gap)
        with self.assertRaisesRegex(inspector.InventoryError, "offset or shape"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)

        def overlap(header):
            header["model.language_model.norm.weight"]["data_offsets"] = [10, 14]
        self._write_checkpoint(header_override=overlap)
        with self.assertRaisesRegex(inspector.InventoryError, "overlap or gap"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)

        def shape(header):
            header["model.language_model.norm.weight"]["shape"] = [3]
        self._write_checkpoint(header_override=shape)
        with self.assertRaisesRegex(inspector.InventoryError, "offset or shape"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)

        self._write_checkpoint(trailing=b"extra")
        with self.assertRaisesRegex(inspector.InventoryError, "trailing"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)

    def test_missing_layer_and_wrong_config_dimensions_fail(self):
        self.config["text_config"]["num_hidden_layers"] = 3
        self.config["text_config"]["layer_types"].append("full_attention")
        self._write_checkpoint()
        with self.assertRaisesRegex(inspector.InventoryError, "layer indices"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)
        self.config["text_config"]["num_hidden_layers"] = 2
        self.config["text_config"]["layer_types"].pop()
        self.config["audio_config"]["hidden_size"] = 99
        self._write_checkpoint()
        with self.assertRaisesRegex(inspector.InventoryError, "audio projection"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)

    def test_duplicate_json_keys_and_bad_header_length_fail(self):
        path = self.snapshot / "model.safetensors"
        payload = path.read_bytes()
        duplicate_header = b'{"x":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]},"x":{}}'
        path.write_bytes(struct.pack("<Q", len(duplicate_header)) + duplicate_header)
        with self.assertRaisesRegex(inspector.InventoryError, "duplicate JSON key"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)
        path.write_bytes(struct.pack("<Q", inspector.MAX_HEADER_BYTES + 1) + payload[8:])
        with self.assertRaisesRegex(inspector.InventoryError, "header length"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)

    def test_bad_dtype_and_null_metadata_fail(self):
        def dtype(header):
            header["model.language_model.norm.weight"]["dtype"] = []
        self._write_checkpoint(header_override=dtype)
        with self.assertRaisesRegex(inspector.InventoryError, "unsupported dtype"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)
        self._write_checkpoint(header_override=lambda header: header.update({"__metadata__": None}))
        with self.assertRaisesRegex(inspector.InventoryError, "metadata"):
            inspector.inspect_checkpoint(self.snapshot, hash_weights=False)

    def test_cli_writes_only_outside_snapshot_and_never_overwrites(self):
        output = self.root / "inventory.json"
        self.assertEqual(inspector.main(["--snapshot", str(self.snapshot),
                                         "--output", str(output), "--sample-bf16-tensors", "2"]), 0)
        self.assertEqual(json.loads(output.read_text())["summary"]["tensor_count"], 8)
        with self.assertRaises(SystemExit):
            inspector.main(["--snapshot", str(self.snapshot), "--output", str(output)])
        with self.assertRaises(SystemExit):
            inspector.main(["--snapshot", str(self.snapshot),
                            "--output", str(self.snapshot / "inventory.json")])


if __name__ == "__main__":
    unittest.main()
