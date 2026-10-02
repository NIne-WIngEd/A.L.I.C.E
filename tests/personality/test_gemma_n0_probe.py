"""CPU fixtures test public-input and diagnostic boundaries, not model behavior."""

from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.eipm.gemma_n0 import probe_prepared as probe


class _PublicTokenizer:
    def __init__(self):
        self.calls = []
        self.alter_padded_source = False

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("a public source diagnostic must never apply a chat template")

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        if kwargs["truncation"] is not False:
            raise AssertionError("truncation cannot be enabled")
        rows = [[1, *range(3, 3 + len(text.split())), 2] for text in texts]
        if self.alter_padded_source and kwargs["padding"]:
            rows[0] = rows[0][:-1]
        width = max(map(len, rows))
        ids = [row + [0] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class _FeatureAdapter:
    max_source_tokens = 512
    hidden_size = 3

    def __init__(self):
        self.calls = 0
        self.nonfinite = False
        self.misaligned = False
        self.on_forward = None

    def extract_features(self, *, input_ids, attention_mask):
        self.calls += 1
        if self.on_forward is not None:
            self.on_forward()
        states = torch.ones((*input_ids.shape, self.hidden_size), dtype=torch.float32)
        if self.nonfinite:
            states[0, 0, 0] = float("nan")
        returned_mask = attention_mask.bool()
        if self.misaligned:
            returned_mask = returned_mask.clone()
            returned_mask[0, 0] = False
        return SimpleNamespace(hidden_states=states, attention_mask=returned_mask)


class PublicProbeBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.prepared_path = self.root / "prepared.json"
        self.prepared_path.write_text('{"test_fixture":"not_a_real_model_receipt"}', encoding="utf-8")
        self.output = self.root / "diagnostic.json"
        self.prepared = {
            "receipt_sha256": "a" * 64,
            "snapshot_path": str(self.snapshot),
            "repository": "fixture-only/no-model-loaded",
            "revision": "b" * 40,
            "files": [{"path": "fixture-only", "size": 1, "sha256": "c" * 64}],
            "runtime": {"backend": "transformers", "dtype": "float32",
                        "torch_version": str(torch.__version__),
                        "transformers_version": "fixture-only-no-transformers-loaded"},
        }
        self.tokenizer = _PublicTokenizer()
        self.adapter = _FeatureAdapter()
        self.processor_factory = SimpleNamespace(from_pretrained=Mock(
            return_value=SimpleNamespace(tokenizer=self.tokenizer)))
        self.backbone_loader = Mock(return_value=self.adapter)
        runtime = probe._Runtime(torch, self.processor_factory, self.backbone_loader,
                                 self.prepared["runtime"]["transformers_version"])
        for target, replacement in (("_verified_preparation", Mock(return_value=self.prepared)),
                                    ("_load_runtime", Mock(return_value=runtime))):
            patcher = patch.object(probe, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_fixture(self, **kwargs):
        return probe.run_probe(preparation_receipt=self.prepared_path,
                               output_receipt=self.output, **kwargs)

    def test_forward_metadata_is_source_bound_and_always_unqualified(self):
        receipt = self.run_fixture(batch_size=3)
        self.assertEqual(receipt["qualification"], "unqualified")
        self.assertIsNone(receipt["behavioral_scores"])
        self.assertFalse(receipt["hidden_states_retained"])
        self.assertEqual(receipt["public_plan_sha256"], sha256(probe.PLAN_PATH.read_bytes()).hexdigest())
        self.assertEqual(receipt["prepared_receipt_file_sha256"],
                         sha256(self.prepared_path.read_bytes()).hexdigest())
        self.assertEqual(receipt["runtime"], self.prepared["runtime"])
        self.assertEqual(receipt["source_files"], self.prepared["files"])
        self.assertEqual(sum(len(batch["probe_ids"]) for batch in receipt["batches"]),
                         receipt["example_count"])
        self.assertTrue(all(batch["all_features_finite"] for batch in receipt["batches"]))
        self.assertIsNone(receipt["memory"]["cpu_process_peak_bytes"])
        self.assertTrue(receipt["code"]["current_files"])
        self.assertTrue(all(len(row["sha256"]) == 64 for row in receipt["code"]["current_files"]))
        self.processor_factory.from_pretrained.assert_called_once_with(
            str(self.snapshot), local_files_only=True, trust_remote_code=False)
        self.backbone_loader.assert_called_once_with(self.prepared_path, device="cpu", dtype="float32")
        self.assertTrue(all(call[1]["truncation"] is False for call in self.tokenizer.calls))
        stored = json.loads(self.output.read_bytes())
        supplied_digest = stored.pop("receipt_sha256")
        self.assertEqual(supplied_digest, sha256(probe._canonical(stored)).hexdigest())
        self.assertNotIn("hidden_states", stored)

    def test_custom_private_text_label_cannot_enter_public_plan(self):
        altered = json.loads(probe.PLAN_PATH.read_bytes())
        altered["examples"][0]["text"] = "This is not the published fictional fixture."
        private_claim = self.root / "public-labelled.json"
        private_claim.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(probe.ProbeError, "immutable public fixture"):
            self.run_fixture(public_plan=private_claim)
        probe._load_runtime.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_full_source_over_budget_is_refused_before_any_forward(self):
        with self.assertRaisesRegex(probe.ProbeError, "truncation is forbidden"):
            self.run_fixture(maximum_source_tokens=5)
        self.assertEqual(self.adapter.calls, 0)
        self.assertFalse(self.output.exists())
        self.assertTrue(all(call[1]["truncation"] is False for call in self.tokenizer.calls))

    def test_padding_cannot_silently_drop_source_suffix(self):
        self.tokenizer.alter_padded_source = True
        with self.assertRaisesRegex(probe.ProbeError, "altered or truncated"):
            self.run_fixture(batch_size=2)
        self.assertEqual(self.adapter.calls, 0)
        self.assertFalse(self.output.exists())

    def test_existing_receipt_is_not_overwritten_or_reexecuted(self):
        self.output.write_bytes(b"earlier diagnostic receipt\n")
        with self.assertRaisesRegex(probe.ProbeError, "append-only"):
            self.run_fixture()
        probe._load_runtime.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b"earlier diagnostic receipt\n")

    def test_receipt_cannot_be_written_inside_snapshot(self):
        self.output = self.snapshot / "diagnostic.json"
        with self.assertRaisesRegex(probe.ProbeError, "outside the prepared model snapshot"):
            self.run_fixture()
        probe._load_runtime.assert_not_called()

    def test_runtime_mismatch_refuses_processor_and_model_loading(self):
        self.prepared["runtime"]["torch_version"] = "incompatible-version"
        with self.assertRaisesRegex(probe.ProbeError, "installed runtime differs"):
            self.run_fixture()
        self.processor_factory.from_pretrained.assert_not_called()
        self.backbone_loader.assert_not_called()

    def test_missing_processor_tokenizer_cannot_fall_back_to_chat_or_download(self):
        self.processor_factory.from_pretrained.return_value = SimpleNamespace()
        with self.assertRaisesRegex(probe.ProbeError, "no local raw-text tokenizer"):
            self.run_fixture()
        self.backbone_loader.assert_not_called()

    def test_nonfinite_features_cannot_receive_a_diagnostic_receipt(self):
        self.adapter.nonfinite = True
        with self.assertRaisesRegex(probe.ProbeError, "finite detached source alignment"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_returned_mask_must_match_complete_source_positions(self):
        self.adapter.misaligned = True
        with self.assertRaisesRegex(probe.ProbeError, "source alignment"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_changed_preparation_during_forward_cannot_publish_stale_receipt(self):
        self.adapter.on_forward = lambda: self.prepared_path.write_bytes(b"changed during forward")
        with self.assertRaisesRegex(probe.ProbeError, "changed during the diagnostic"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_changed_code_cannot_publish_receipt_with_earlier_source_binding(self):
        with patch.object(probe, "_code_binding", side_effect=[{"current": 1}, {"current": 2}]):
            with self.assertRaisesRegex(probe.ProbeError, "code changed"):
                self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_cli_reports_diagnostic_status_without_behavioral_pass(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = probe.main([str(self.prepared_path), str(self.output), "--batch-size", "4"])
        self.assertEqual(status, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["qualification"], "unqualified")
        self.assertNotIn("pass", summary)


if __name__ == "__main__":
    unittest.main()
