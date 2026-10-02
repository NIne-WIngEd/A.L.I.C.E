"""Generated control and cache artifact boundaries; fixtures are not model scores."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record
from tests.governance.test_formation_learning_v16 import case_record
from scripts.mfm import run_v16_cached_diagnostic as diagnostic
from scripts.mfm import train_v16_cached_specialist as cached

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch required")
class CachedDiagnosticTests(unittest.TestCase):
    def test_generated_output_retains_invalid_no_eos_and_duplicate_keys(self):
        example = learning_example_v16_from_record(case_record())
        body = json.dumps(example.target.output_record())
        self.assertEqual(diagnostic.validate_generated(body, True, example, "a" * 64,
            "diagnostic-fixture")["validation_status"], "grounded_teacher_proposal_only")
        for raw, ended in ((body, False), ("bad-json", True), ("{\"schema\":1,\"schema\":2}", True)):
            result = diagnostic.validate_generated(raw, ended, example, "a" * 64, "diagnostic-fixture")
            self.assertEqual(result["raw_output_text"], raw)
            self.assertEqual(result["validation_status"], "invalid")
            self.assertTrue(result["validation_error"])

    def test_fp32_reference_decode_under_outer_autocast_and_missing_eos(self):
        class Tokens(torch.nn.Module):
            def __init__(self, sequence):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(start_token_id=2, end_token_id=1, max_target_tokens=8)
                self.sequence = sequence
            def next_token_logits(self, *, base_states, source_mask, input_ids):
                assert not torch.is_autocast_enabled("cpu")
                assert input_ids[0, 0].item() == 2
                values = torch.zeros(1, 9)
                values[0, self.sequence[input_ids.shape[1] - 1]] = 1
                return values
        kwargs = dict(states=torch.randn(1, 3, 4).bfloat16(), mask=torch.ones(1, 3, dtype=torch.long))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(diagnostic.greedy_fp32(Tokens([3, 4, 1]), max_new_tokens=3, **kwargs), ([3, 4], True))
        self.assertEqual(diagnostic.greedy_fp32(Tokens([3, 4]), max_new_tokens=2, **kwargs), ([3, 4], False))
        with self.assertRaisesRegex(CognitiveKernelContractError, "FP32"):
            diagnostic.greedy_fp32(Tokens([1]).bfloat16(), max_new_tokens=1, **kwargs)

    def artifact(self, root):
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
        shared = cached.training.shared
        model = FormationSpecialist(SpecialistConfig(12, 31, 16, 2, 4, 8, 0, 2, 1, 2))
        preflight = {"record_sha256": "b" * 64, "specialist_heads": 4,
                     "specialist_layers": 2, "max_target_tokens": 8}
        run = shared._write_new(root / "run.json", {
            "schema": cached.RUN_SCHEMA, "objective": cached.training.OBJECTIVE_VERSION_V16,
            "corpus_sha256": "c" * 64, "feature_bank_sha256": "a" * 64,
            "preflight_sha256": "b" * 64, "specialist_dtype": "float32",
            "trainer_sha256": shared._digest(Path(cached.__file__)),
            "specialist_config": model.config.record(), "teacher_fit": True,
            "probe_only": False, "full_fit": False, "qualified_for_product": False})
        seed = cached.training._seed_control(root, model, run["record_sha256"], model.config)
        with torch.no_grad():
            model.output_bias.add_(0.01)
        path = root / "formation-specialist.safetensors"
        save_file({name: value.contiguous() for name, value in model.state_dict().items()}, str(path))
        component = shared._write_new(root / "component.json", {
            "schema": cached.ARTIFACT_SCHEMA, "run_sha256": run["record_sha256"],
            "specialist_config": model.config.record(), "feature_bank_sha256": "a" * 64,
            "teacher_fit": True, "probe_only": False, "full_fit": False,
            "qualified_for_product": False, "result": {"all_train_epochs_completed": True,
                "optimizer_steps": 1, "trained_weights_sha256": shared._digest(path)}})
        return component, preflight, seed

    def test_artifact_chain_external_pin_weight_tamper_and_seed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component, preflight, seed = self.artifact(root)
            kwargs = dict(expected_sha256=component["record_sha256"], bank_digest="a" * 64,
                          preflight=preflight, corpus_digest="c" * 64)
            _, _, _, paths = diagnostic.verify_component(root, **kwargs)
            self.assertNotEqual(paths["trained"][1], paths["seeded-untrained"][1])
            with self.assertRaises(CognitiveKernelContractError):
                diagnostic.verify_component(root, **{**kwargs, "expected_sha256": "d" * 64})
            path = root / "formation-specialist.safetensors"
            path.write_bytes((root / "seed-control.safetensors").read_bytes())
            with self.assertRaises(CognitiveKernelContractError):
                diagnostic.verify_component(root, **kwargs)

    def test_incomplete_or_probe_component_cannot_be_a_completed_fit_diagnostic(self):
        for mode in ("probe", "incomplete"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                component, preflight, _ = self.artifact(root)
                body = {k: v for k, v in component.items() if k != "record_sha256"}
                if mode == "probe":
                    body["probe_only"] = True
                else:
                    body["result"]["all_train_epochs_completed"] = False
                (root / "component.json").unlink()
                altered = cached.training.shared._write_new(root / "component.json", body)
                with self.assertRaises(CognitiveKernelContractError):
                    diagnostic.verify_component(root, expected_sha256=altered["record_sha256"],
                        bank_digest="a" * 64, preflight=preflight, corpus_digest="c" * 64)

    def test_isolation_precedes_admission(self):
        with patch.object(cached.training.shared, "_require_private_network_isolation",
                          side_effect=CognitiveKernelContractError("blocked")), \
                patch.object(cached.training, "_examples") as opener:
            with self.assertRaisesRegex(CognitiveKernelContractError, "blocked"):
                diagnostic.run(SimpleNamespace())
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
