"""CPU-only placement and detached source bridge for the v1.6 GPU route."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_sha256
from scripts.mfm import train_v16_formation_specialist as trainer
from scripts.mfm import run_v16_formation_specialist as runner


class TensorStandIn:
    def __init__(self, shape, trace, *, device="cuda:0", grad=True):
        self.shape = shape
        self.device = device
        self.grad = grad
        self.trace = trace

    def detach(self):
        self.trace.append("detach")
        return TensorStandIn(self.shape, self.trace, device=self.device, grad=False)

    def to(self, device):
        self.trace.append(("to", str(device), self.grad))
        return TensorStandIn(self.shape, self.trace, device=str(device), grad=self.grad)


class V16RoleSplitTests(unittest.TestCase):
    def test_explicit_split_and_single_gpu_allocation(self):
        default = trainer._device_plan(SimpleNamespace(), cuda_devices=1)
        self.assertEqual(default, {"base": "cuda:0", "specialist": "cuda:0",
                                   "strategy": "single-gpu"})
        split = SimpleNamespace(base_device="cuda:0", specialist_device="cuda:1")
        self.assertEqual(trainer._device_plan(split, cuda_devices=2), {
            "base": "cuda:0", "specialist": "cuda:1",
            "strategy": "frozen-base-role-split"})
        with self.assertRaisesRegex(CognitiveKernelContractError, "unavailable"):
            trainer._device_plan(split, cuda_devices=1)
        for invalid in ("cuda", "cuda:01", "cpu", "disk", "cuda:-1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    CognitiveKernelContractError, "explicit CUDA ordinal"):
                trainer._device_plan(SimpleNamespace(base_device=invalid,
                                                      specialist_device="cuda:0"))

    def test_training_run_hash_changes_with_role_placement(self):
        args = SimpleNamespace(full_fit=False, trust_roster_sha256=None,
                               signed_review_receipt_sha256=None,
                               input_sha256="a" * 64, epochs=2, learning_rate=1e-4,
                               gradient_accumulation=16, seed=73129,
                               max_cross_attention_pairs=1000, probe_only=True,
                               base_device="cuda:0", specialist_device="cuda:0")
        prepared = {"receipt_sha256": "b" * 64, "role": "mfm"}
        preflight = {"record_sha256": "c" * 64,
                     "transformers_version": "test-version"}
        config = SimpleNamespace(record=lambda: {"width": 8})
        from unittest.mock import patch
        with patch.object(trainer.shared, "_prepared_kind", return_value="role-clone"):
            one = trainer._run_manifest_v16(args, prepared, preflight, config, "cpu-test")
            args.specialist_device = "cuda:1"
            two = trainer._run_manifest_v16(args, prepared, preflight, config, "cpu-test")
        self.assertNotEqual(canonical_sha256(one), canonical_sha256(two))
        self.assertEqual(two["device_placement"]["specialist"], "cuda:1")

    def test_bridge_detaches_base_states_and_routes_mask(self):
        trace = []
        states = TensorStandIn((1, 7, 4), trace)
        mask = TensorStandIn((1, 7), trace, grad=False)
        moved_states, moved_mask = trainer._specialist_sources(
            states, mask, "cuda:1")
        self.assertEqual(trace, ["detach", ("to", "cuda:1", False),
                                 ("to", "cuda:1", False)])
        self.assertFalse(moved_states.grad)
        self.assertEqual((moved_states.device, moved_mask.device), ("cuda:1", "cuda:1"))
        with self.assertRaisesRegex(CognitiveKernelContractError, "align"):
            trainer._specialist_sources(states, TensorStandIn((1, 6), []), "cuda:1")

    def test_runner_rejects_half_specified_split_before_private_source(self):
        args = SimpleNamespace(base_device="cuda:0", specialist_device=None)
        with self.assertRaisesRegex(CognitiveKernelContractError, "both explicit"):
            runner.run(args)

    def test_probe_hardware_reports_both_devices_without_estimating_fit(self):
        class Cuda:
            def max_memory_allocated(self, name):
                return {"cuda:0": 31, "cuda:1": 41}[name]

            def max_memory_reserved(self, name):
                return {"cuda:0": 32, "cuda:1": 42}[name]

            def get_device_properties(self, name):
                return SimpleNamespace(total_memory=48)

        receipt = trainer._probe_hardware(SimpleNamespace(cuda=Cuda()),
                                          ("cuda:0", "cuda:1"),
                                          model_load_seconds=3.0,
                                          optimizer_step_seconds=5.0)
        self.assertEqual(receipt["optimizer_step_seconds"], 5.0)
        self.assertEqual(set(receipt["devices"]), {"cuda:0", "cuda:1"})
        self.assertEqual(receipt["devices"]["cuda:1"], {
            "peak_allocated_bytes": 41, "peak_reserved_bytes": 42,
            "total_bytes": 48})


if __name__ == "__main__":
    unittest.main()
