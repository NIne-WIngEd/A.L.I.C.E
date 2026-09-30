"""Cost-bearing training must reject stale input and unsafe restart roots on CPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
from contextlib import nullcontext
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from scripts.mfm.multimodal_paid_run import (
    PREFLIGHT_SCHEMA, bind_run, probe_indices, read_sealed, require_complete_weight_export,
    require_preflight, require_zero3, source_fingerprint, write_sealed,
)
from scripts.mfm import train_formation_model, train_multimodal_formation


class PaidRunBindingsTests(unittest.TestCase):
    def test_pretrained_routes_require_explicit_nonproduct_research_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = ["train", "--curriculum", str(root / "train.jsonl"),
                      "--input-sha256", "a" * 64,
                      "--output-dir", str(root / "run"),
                      "--max-sequence-tokens", "32768"]
            with patch.object(sys, "argv", common), \
                 patch.object(train_multimodal_formation, "_dataset") as dataset:
                with self.assertRaisesRegex(CognitiveKernelContractError,
                                            "Gemma-derived MFM route is superseded"):
                    train_multimodal_formation.main()
                dataset.assert_not_called()
            with patch.object(sys, "argv", common):
                with self.assertRaisesRegex(CognitiveKernelContractError,
                                            "pretrained MFM route is superseded"):
                    train_formation_model.main()
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "cannot be loaded as the personal core"):
                train_formation_model.load_hf_candidate(
                    root, inference_run_id="research", max_input_tokens=1,
                    max_new_tokens=1)
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "cannot be loaded as the personal core"):
                train_multimodal_formation.load_multimodal_candidate(
                    root, inference_run_id="research", max_input_tokens=1,
                    max_new_tokens=1)

    def test_processor_receipt_binds_source_and_exact_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            base = {"schema": PREFLIGHT_SCHEMA, "input_sha256": "a" * 64,
                    "train_cases": 90, "source_fingerprint": source_fingerprint(),
                    "longest_case_index": 89,
                    "probe_indices": probe_indices([(n, n) for n in range(90)])}
            write_sealed(path, base)
            self.assertEqual(require_preflight(path, {"input_sha256": "a" * 64})[
                "longest_case_index"], 89)
            with self.assertRaisesRegex(CognitiveKernelContractError, "differs: input_sha256"):
                require_preflight(path, {"input_sha256": "b" * 64})
            changed = json.loads(path.read_text())
            changed["longest_case_index"] = 0
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(CognitiveKernelContractError, "digest differs"):
                require_preflight(path, {"input_sha256": "a" * 64})

    def test_zero3_config_requires_sharding_bf16_and_complete_export(self):
        config = (Path(__file__).resolve().parents[2] / "scripts/mfm" /
                  "deepspeed_zero3_a100_4gpu.json")
        self.assertEqual(len(require_zero3(config)), 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            changed = json.loads(config.read_text())
            changed["zero_optimization"]["stage"] = 2
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(CognitiveKernelContractError, "stage 3"):
                require_zero3(path)
            changed["zero_optimization"]["stage"] = 3
            changed["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(CognitiveKernelContractError, "complete 16-bit"):
                require_zero3(path)

    def test_resume_cannot_rebind_or_restart_a_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            record = {"schema": "test-run", "input_sha256": "a" * 64}
            bind_run(root, record, None)
            with self.assertRaisesRegex(CognitiveKernelContractError, "occupied"):
                bind_run(root, record, None)
            checkpoint = root / "checkpoints/checkpoint-50"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text("{}")
            bind_run(root, record, checkpoint)
            with self.assertRaisesRegex(CognitiveKernelContractError, "manifest differs"):
                bind_run(root, {**record, "input_sha256": "b" * 64}, checkpoint)
            other = Path(directory) / "other/checkpoint-50"
            other.mkdir(parents=True)
            (other / "trainer_state.json").write_text("{}")
            with self.assertRaisesRegex(CognitiveKernelContractError, "exact run"):
                bind_run(root, record, other)
            self.assertEqual(read_sealed(root / "run-manifest.json")["input_sha256"], "a" * 64)

    def test_probe_selection_is_unique_and_includes_longest(self):
        lengths = [(n, n) for n in range(1000)]
        chosen = probe_indices(lengths)
        self.assertEqual(len(chosen), 64)
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertIn(999, chosen)

    def test_export_rejects_partial_or_invalid_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(CognitiveKernelContractError, "lacks complete"):
                require_complete_weight_export(root)
            (root / "model.safetensors").write_bytes(b"not weights")
            with self.assertRaisesRegex(CognitiveKernelContractError, "smaller"):
                require_complete_weight_export(root)
            tensor = b"\x00\x00\x00\x00"
            header = json.dumps({"model.embed_tokens.weight": {
                "dtype": "BF16", "shape": [2], "data_offsets": [0, len(tensor)]}}).encode()
            (root / "model.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + tensor)
            summary = require_complete_weight_export(root, min_bytes=4, min_parameters=2)
            self.assertEqual(summary["parameters"], 2)
            scalar_header = json.dumps({"model.scalar": {
                "dtype": "BF16", "shape": [], "data_offsets": [0, 2]}}).encode()
            (root / "model.safetensors").write_bytes(
                len(scalar_header).to_bytes(8, "little") + scalar_header + tensor[:2])
            self.assertEqual(require_complete_weight_export(
                root, min_bytes=2, min_parameters=1)["parameters"], 1)
            (root / "model.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + tensor[:2])
            with self.assertRaisesRegex(CognitiveKernelContractError, "invalid tensor metadata"):
                require_complete_weight_export(root, min_bytes=1, min_parameters=2)

    def test_cpu_processor_preflight_can_use_staged_snapshot_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "processor.json"
            snapshot = root / "snapshot"
            calls = []

            class FakeProcessor:
                audio_seq_length = 750
                video_processor = object()

            class FakeLabels:
                def __eq__(self, value):
                    return types.SimpleNamespace(sum=lambda: types.SimpleNamespace(item=lambda: 10))

                def numel(self):
                    return 20

            fake_batch = {"input_ids": types.SimpleNamespace(shape=(1, 20)),
                          "labels": FakeLabels()}
            transformers = types.ModuleType("transformers")
            transformers.__version__ = "5.17.0"
            transformers.AutoProcessor = types.SimpleNamespace(
                from_pretrained=lambda *args, **kw: (calls.append((args, kw)) or FakeProcessor()))
            transformers.AutoModelForMultimodalLM = object()
            transformers.Trainer = object()
            transformers.TrainingArguments = object()
            argv = ["train", "--curriculum", str(root / "train.jsonl"),
                    "--input-sha256", "a" * 64, "--owner-authorization-ref", "owner-test",
                    "--output-dir", str(root / "unused"), "--preflight-receipt", str(receipt),
                    "--staged-model-receipt", str(root / "staged.json"),
                    "--max-sequence-tokens", "32768", "--preflight-only",
                    "--research-derivative-only"]
            with (patch.object(sys, "argv", argv),
                  patch.dict(sys.modules, {"torch": types.ModuleType("torch"),
                                           "transformers": transformers}),
                  patch("builtins.print"),
                  patch.object(train_multimodal_formation, "_dataset", return_value=(
                      (types.SimpleNamespace(context=types.SimpleNamespace(evidence=())),), (),
                      "owner-authorized-training-only-unqualified")),
                  patch.object(train_multimodal_formation, "formation_media_messages",
                               side_effect=lambda _: nullcontext([])),
                  patch.object(train_multimodal_formation, "supervised_multimodal_batch",
                               return_value=fake_batch),
                  patch.object(train_multimodal_formation, "verify_staged_model",
                               return_value=(snapshot, {"receipt_sha256": "b" * 64}))):
                train_multimodal_formation.main()
            self.assertEqual(calls[0][0], (snapshot,))
            self.assertTrue(calls[0][1]["local_files_only"])
            self.assertEqual(read_sealed(receipt)["processor_snapshot_sha256"], "b" * 64)

    def test_paid_probe_shards_before_load_and_exits_after_one_step(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "probe"
            receipt = root / "processor.json"
            config = (Path(__file__).resolve().parents[2] / "scripts/mfm" /
                      "deepspeed_zero3_a100_4gpu.json")
            write_sealed(receipt, {
                "schema": PREFLIGHT_SCHEMA, "objective": "mfm-grounded-multimodal-disposition-objective-v1",
                "weight_lineage": "third-party-pretrained-derivative-research-only",
                "qualified_for_product": False,
                "model": "google/gemma-4-12B-it",
                "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                "input_sha256": "a" * 64, "owner_authorization_ref": "owner-test",
                "corpus_status": "owner-authorized-training-only-unqualified",
                "train_cases": 64, "development_cases": 0,
                "max_sequence_tokens": 32768, "max_new_tokens": 8192,
                "transformers_version": "5.17.0", "source_fingerprint": source_fingerprint(),
                "longest_case_index": 63, "longest_processed_tokens": 400,
                "probe_indices": list(range(64)),
            })

            class FakeProcessor:
                audio_seq_length = 750
                video_processor = object()

                def save_pretrained(self, target):
                    (Path(target) / "processor.json").write_text("{}")

            class FakeModel:
                def __init__(self):
                    self.config = types.SimpleNamespace(
                        text_config=types.SimpleNamespace(max_position_embeddings=32768),
                        use_cache=True)

            class FakeTrainingArguments:
                def __init__(self, **kwargs):
                    events.append("args")
                    self.output_dir = kwargs["output_dir"]
                    self.optim = "adamw_torch_fused"
                    self.max_steps = kwargs["max_steps"]
                    self.save_steps = kwargs["save_steps"]

            class FakeTrainer:
                def __init__(self, *, args, **kwargs):
                    self.args = args
                    self.output = output

                def train(self, *, resume_from_checkpoint):
                    events.append(("train", resume_from_checkpoint))
                    checkpoint = Path(self.args.output_dir) / "checkpoint-1"
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "trainer_state.json").write_text("{}")
                    return types.SimpleNamespace(global_step=1)

                def save_model(self, final):
                    events.append("save")
                    target = Path(final)
                    target.mkdir()
                    (target / "model.safetensors").write_bytes(b"fake-sharded-export")
                    (target / "config.json").write_text("{}")
                    manifest = read_sealed(self.output / "run-manifest.json")
                    for rank in range(1, 4):
                        write_sealed(self.output / f"gpu-rank-{rank}.json", {
                            "rank": rank, "world_size": 4, "gpu": "A100-SXM4-80GB",
                            "run_manifest_sha256": manifest["record_sha256"],
                            "optimizer_steps": 1})

            torch = types.ModuleType("torch")
            torch.__version__ = "2.9-fake"
            torch.bfloat16 = "bfloat16"
            torch.cuda = types.SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 4,
                get_device_properties=lambda _: types.SimpleNamespace(
                    name="A100-SXM4-80GB", total_memory=80 * 1024 ** 3),
                set_device=lambda _: None, is_bf16_supported=lambda: True,
                reset_peak_memory_stats=lambda _: None, synchronize=lambda _: None,
                max_memory_reserved=lambda _: 60 * 1024 ** 3,
                get_device_name=lambda _: "A100-SXM4-80GB")
            torch.distributed = types.SimpleNamespace(is_initialized=lambda: False)
            torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
            transformers = types.ModuleType("transformers")
            transformers.__version__ = "5.17.0"
            transformers.AutoProcessor = types.SimpleNamespace(
                from_pretrained=lambda *a, **kw: FakeProcessor())

            def load_model(*a, **kwargs):
                events.append("model")
                self.assertTrue(kwargs["local_files_only"])
                return FakeModel()

            transformers.AutoModelForMultimodalLM = types.SimpleNamespace(
                from_pretrained=load_model)
            transformers.TrainingArguments = FakeTrainingArguments
            transformers.Trainer = FakeTrainer
            argv = ["train", "--curriculum-manifest", str(root / "mixture.json"),
                    "--input-sha256", "a" * 64, "--owner-authorization-ref", "owner-test",
                    "--preflight-receipt", str(receipt), "--output-dir", str(output),
                    "--staged-model-receipt", str(root / "staged.json"),
                    "--deepspeed-config", str(config), "--max-sequence-tokens", "32768",
                    "--probe-only", "--research-derivative-only"]
            env = {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"}
            with (patch.object(sys, "argv", argv), patch.dict(os.environ, env),
                  patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
                  patch("builtins.print"),
                  patch.object(train_multimodal_formation, "verify_staged_model",
                               return_value=(root / "snapshot", {"receipt_sha256": "b" * 64})),
                  patch.object(train_multimodal_formation.shutil, "disk_usage",
                               return_value=types.SimpleNamespace(free=1024 ** 4)),
                  patch.object(train_multimodal_formation, "require_complete_weight_export",
                               return_value={"weight_bytes": 24_000_000_000}),
                  patch.object(train_multimodal_formation, "_dataset", return_value=(
                      tuple(object() for _ in range(64)), (),
                      "owner-authorized-training-only-unqualified"))):
                train_multimodal_formation.main()
            self.assertEqual(events[:2], ["args", "model"])
            self.assertEqual(events[2:], [("train", None), "save"])
            self.assertTrue((output / "checkpoints/checkpoint-1/trainer_state.json").exists())
            artifact = json.loads((output / "artifact-receipt.json").read_text())
            self.assertFalse(artifact["provenance"]["qualified_for_product"])
            self.assertEqual(artifact["provenance"]["run_mode"], "hardware-qualification")


if __name__ == "__main__":
    unittest.main()
