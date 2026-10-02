"""Lossless representation, authority, gradient and real restart checks on CPU."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record
from tests.governance.test_formation_learning_v16 import case_record
from scripts.mfm import formation_feature_bank as bank
from scripts.mfm import export_v16_frozen_features as exporter
from scripts.mfm import train_v16_cached_specialist as cached

try:
    import torch
    from safetensors.torch import load_file
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch and safetensors required")
class FrozenFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.example = learning_example_v16_from_record(case_record())
        self.states = torch.randn(1, 5, 12).bfloat16()
        self.encoded = {"input_ids": torch.tensor([[2, 3, 4, 5, 0]]),
                        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])}
        self.preflight = bank.shared._seal({
            "schema": "fixture", "corpus_sha256": "a" * 64,
            "prepared_base_receipt_sha256": "b" * 64,
            "prepared_base_weight_sha256": "c" * 64,
            "transformers_version": "fixture-no-publisher-model"})
        binding = bank.feature_binding(self.preflight, {"receipt_sha256": "b" * 64}, torch.__version__)
        self.export = bank.shared._write_new(self.root / "export.json",
                                            {"schema": bank.EXPORT_SCHEMA, **binding})

    def write(self, example=None):
        return bank.write_case(self.root, example or self.example, self.encoded,
                               self.states, self.export["record_sha256"])

    def test_lossless_live_cached_loss_and_every_gradient_match(self):
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
        self.write()
        _, restored = bank.read_case(self.root, self.example, self.export["record_sha256"], width=12)
        for key, value in {"states": self.states, **self.encoded}.items():
            self.assertTrue(torch.equal(value, restored[key]))
        model = FormationSpecialist(SpecialistConfig(12, 31, 16, 2, 4, 8, 0, 2, 1, 2))
        target = torch.tensor([[2, 9, 8]])
        labels = torch.tensor([[9, 8, 1]])
        def gradients(states, mask):
            model.zero_grad(set_to_none=True)
            _, loss = model(base_states=states, source_mask=mask, input_ids=target, labels=labels)
            loss.backward()
            return loss.detach().clone(), {k: p.grad.clone() for k, p in model.named_parameters()}
        live_loss, live_gradients = gradients(self.states, self.encoded["attention_mask"])
        cached_loss, cached_gradients = gradients(restored["states"], restored["attention_mask"])
        self.assertTrue(torch.equal(live_loss, cached_loss))
        self.assertGreater(sum(g.abs().sum().item() for g in live_gradients.values()), 0)
        for name in live_gradients:
            self.assertTrue(torch.equal(live_gradients[name], cached_gradients[name]), name)

    def test_case_identity_split_and_export_are_bound(self):
        self.write()
        for example in (replace(self.example, case_id="other-case"),
                        replace(self.example, split="development")):
            with self.subTest(example=example.case_id, split=example.split), self.assertRaises(CognitiveKernelContractError):
                bank.read_case(self.root, example, self.export["record_sha256"])
        with self.assertRaises(CognitiveKernelContractError):
            bank.read_case(self.root, self.example, "d" * 64)
        with self.assertRaisesRegex(CognitiveKernelContractError, "FINAL"):
            bank.case_key(replace(self.example, split="final"))

    def test_corrupt_payload_and_wrong_hidden_width_fail(self):
        self.write()
        with self.assertRaises(CognitiveKernelContractError):
            bank.read_case(self.root, self.example, self.export["record_sha256"], width=13)
        path = self.root / "cases" / bank.case_key(self.example) / "features.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(CognitiveKernelContractError, "bytes differ"):
            bank.read_case(self.root, self.example, self.export["record_sha256"])

    def test_nonfinite_wrong_dtype_and_invalid_mask_fail(self):
        for mode in ("nan", "fp32", "empty-mask", "fractional-mask"):
            states = self.states.clone()
            tensors = {"states": states, **self.encoded}
            if mode == "nan":
                states[0, 0, 0] = float("nan")
            elif mode == "fp32":
                tensors["states"] = states.float()
            elif mode == "empty-mask":
                tensors["attention_mask"] = torch.zeros_like(self.encoded["attention_mask"])
            else:
                tensors["attention_mask"] = torch.full_like(self.encoded["attention_mask"], 2)
            with self.subTest(mode=mode), self.assertRaises(CognitiveKernelContractError):
                bank._validate_tensors(tensors)

    def completed_fixture(self):
        self.write()
        processor = self.root / "processor"
        processor.mkdir()
        (processor / "fixture.json").write_bytes(b"processor-fixture")
        rows = [{"path": "fixture.json", "size": 17,
                 "sha256": bank.shared._digest(processor / "fixture.json")}]
        receipt = bank.finalize(self.root, self.export, [self.example], rows)
        return receipt, rows

    def test_complete_bank_requires_external_pin_and_has_no_extra_payloads(self):
        receipt, rows = self.completed_fixture()
        with patch.object(bank, "training_prepared_files", return_value=rows):
            result, _ = bank.verify_bank(self.root, expected_sha256=receipt["record_sha256"],
                                         examples=[self.example], preflight=self.preflight)
            self.assertTrue(result["complete"])
            with self.assertRaises(CognitiveKernelContractError):
                bank.verify_bank(self.root, expected_sha256="d" * 64,
                                 examples=[self.example], preflight=self.preflight)
            (self.root / "unexpected-target.json").write_text("not-allowed")
            with self.assertRaisesRegex(CognitiveKernelContractError, "unlisted"):
                bank.verify_bank(self.root, expected_sha256=receipt["record_sha256"],
                                 examples=[self.example], preflight=self.preflight)

    def test_incomplete_bank_and_changed_model_preflight_fail(self):
        self.write()
        with self.assertRaises(FileNotFoundError):
            bank.verify_bank(self.root, expected_sha256="a" * 64,
                             examples=[self.example], preflight=self.preflight)
        (self.root / "processor").mkdir()
        receipt = bank.finalize(self.root, self.export, [self.example], [])
        changed = dict(self.preflight, prepared_base_weight_sha256="d" * 64)
        with self.assertRaises(CognitiveKernelContractError):
            bank.verify_bank(self.root, expected_sha256=receipt["record_sha256"],
                             examples=[self.example], preflight=changed)

    def test_processor_cannot_be_replaced_by_self_issued_bank(self):
        receipt, rows = self.completed_fixture()
        with patch.object(bank, "training_prepared_files", return_value=[]), self.assertRaisesRegex(
                CognitiveKernelContractError, "processor file set"):
            bank.verify_bank(self.root, expected_sha256=receipt["record_sha256"],
                             examples=[self.example], preflight=self.preflight)

    def test_atomic_case_resume_does_not_rewrite_and_incomplete_stage_blocks_finalization(self):
        first = self.write()
        self.assertEqual(self.write(), first)
        (self.root / "cases" / ".staging-interrupted").mkdir()
        with self.assertRaisesRegex(CognitiveKernelContractError, "incomplete"):
            bank.finalize(self.root, self.export, [self.example], [])

    def test_paths_cannot_escape_custody(self):
        for value in ("../outside", "/absolute", "cases/./x", "cases//x", "C:/outside", "cases\\x"):
            with self.subTest(value=value), self.assertRaises(CognitiveKernelContractError):
                bank._within(self.root, value)

    def test_development_and_final_cannot_enter_optimizer(self):
        for split in ("development", "final"):
            with self.subTest(split=split), self.assertRaisesRegex(CognitiveKernelContractError, "only admitted train"):
                cached.optimizer_step(None, None, [replace(self.example, split=split)],
                                      root=None, export_digest=None, tokenizer=None, device=None)

    def test_private_entrypoints_stop_before_reading_inputs(self):
        for route, action in ((exporter, exporter.export), (cached, cached.run)):
            with patch.object(route.training.shared, "_require_private_network_isolation",
                              side_effect=CognitiveKernelContractError("blocked-network")), \
                    patch.object(route.training, "_examples") as opener:
                with self.assertRaisesRegex(CognitiveKernelContractError, "blocked-network"):
                    action(SimpleNamespace())
                opener.assert_not_called()

    def test_actual_multistep_restart_restores_weights_and_adam_moments(self):
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
        self.write()
        config = SpecialistConfig(12, 31, 16, 2, 4, 8, 0, 2, 1, 2)
        model = FormationSpecialist(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        args = SimpleNamespace(output_dir=self.root)
        # Only tokenization is replaced by a short fixture. Serialization,
        # gradients, AdamW state, checkpoint disk reload and continuation run.
        with patch.object(cached.training, "_target_ids", return_value=([2, 9, 8], [9, 8, 1])):
            result = cached.probe(model, optimizer, [self.example], args=args,
                run_digest="d" * 64, step_kwargs=dict(root=self.root,
                export_digest=self.export["record_sha256"], tokenizer=None, device=torch.device("cpu")))
        self.assertTrue(result["checkpoint_reload_executed"])
        self.assertEqual(result["optimizer_steps"], 3)
        self.assertEqual(result["model_max_abs_difference"], 0)
        self.assertEqual(result["optimizer_max_abs_difference"], 0)
        self.assertTrue(optimizer.state)
        saved = load_file(str(self.root / "probe-specialist.safetensors"))
        self.assertTrue(all(torch.equal(value, saved[name]) for name, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
