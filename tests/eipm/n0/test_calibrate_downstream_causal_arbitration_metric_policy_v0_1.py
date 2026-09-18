import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/eipm/n0/calibrate_downstream_causal_arbitration_metric_policy_v0_1.py"
)
SPEC = importlib.util.spec_from_file_location("calibrate_arb_policy_v01", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class CalibrateDownstreamArbitrationMetricPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prepare = MOD.load_sibling(
            "calibrate_test_prepare",
            "prepare_downstream_causal_arbitration_v0_1.py",
        )
        self.base_identity = {
            "source_revision": "a" * 40,
            "stack_sha": "b" * 64,
            "stack_fingerprint": "c" * 64,
            "graph_sha": "d" * 64,
            "latent_sha": "e" * 64,
            "eval_sha": "f" * 64,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def result_payload(self, offset: float = 0.0):
        metrics = {}
        counterfactual = {}
        for index, path in enumerate(self.prepare.EXPECTED_METRIC_DIRECTIONS):
            value = 0.25 + index * 0.01 + offset
            root, leaf = path.split(".", 1)
            if root == "metrics":
                metrics[leaf] = value
            else:
                counterfactual[leaf] = value
        return {
            "protocol_version": MOD.EXPECTED_EVALUATOR_PROTOCOL,
            "status": MOD.EXPECTED_STATUS,
            "source_revision": self.base_identity["source_revision"],
            "stack_manifest": {
                "sha256": self.base_identity["stack_sha"],
                "stack_binding_fingerprint": self.base_identity["stack_fingerprint"],
            },
            "graph_condition": {"sha256": self.base_identity["graph_sha"]},
            "common_external_inputs": {
                "latent_checkpoint": {"sha256": self.base_identity["latent_sha"]},
                "evaluation_set": {"sha256": self.base_identity["eval_sha"]},
            },
            "metrics": metrics,
            "counterfactual_metrics": counterfactual,
            "invariants": {
                "diagnostic_only": True,
                "training_enabled": False,
                "gradient_performed": False,
                "parents_mutated": False,
                "graph_checkpoint_is_only_arm_variant": True,
                "proxy_graph_transform_used": False,
            },
        }

    def write_result(self, name: str, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_calibration_freezes_policy_from_canonical_repeats(self):
        a = self.write_result("a.json", self.result_payload(0.0))
        b = self.write_result("b.json", self.result_payload(2e-6))
        policy = self.root / "policy.json"
        receipt = self.root / "receipt.json"

        result = MOD.derive_policy([a, b], policy, receipt)

        self.assertEqual(result["status"], "CALIBRATION_COMPLETE")
        self.assertFalse(result["candidate_arm_result_observed"])

        raw = json.loads(policy.read_text(encoding="utf-8"))
        self.assertTrue(raw["selected_before_arm_results"])
        self.assertFalse(raw["results_observed"])
        self.assertEqual(
            len(raw["metrics"]),
            len(self.prepare.EXPECTED_METRIC_DIRECTIONS),
        )
        self.assertTrue(all(item["rtol"] == 0.0 for item in raw["metrics"]))
        expected = max(
            MOD.ABSOLUTE_FLOOR,
            MOD.REPEATABILITY_MULTIPLIER * 2e-6,
        )
        self.assertTrue(
            all(abs(item["atol"] - expected) < 1e-12 for item in raw["metrics"])
        )

    def test_exact_repeats_use_frozen_absolute_floor(self):
        a = self.write_result("a.json", self.result_payload())
        b = self.write_result("b.json", self.result_payload())
        policy = self.root / "policy.json"
        receipt = self.root / "receipt.json"

        MOD.derive_policy([a, b], policy, receipt)

        raw = json.loads(policy.read_text(encoding="utf-8"))
        self.assertTrue(
            all(item["atol"] == MOD.ABSOLUTE_FLOOR for item in raw["metrics"])
        )

    def test_mismatched_frozen_identity_fails(self):
        first = self.result_payload()
        second = self.result_payload()
        second["graph_condition"]["sha256"] = "0" * 64
        a = self.write_result("a.json", first)
        b = self.write_result("b.json", second)

        with self.assertRaises(MOD.CalibrationError):
            MOD.derive_policy(
                [a, b],
                self.root / "policy.json",
                self.root / "receipt.json",
            )

    def test_requires_two_repeats(self):
        a = self.write_result("a.json", self.result_payload())
        with self.assertRaises(MOD.CalibrationError):
            MOD.derive_policy(
                [a],
                self.root / "policy.json",
                self.root / "receipt.json",
            )

    def test_refuses_to_overwrite_outputs(self):
        a = self.write_result("a.json", self.result_payload())
        b = self.write_result("b.json", self.result_payload())
        policy = self.root / "policy.json"
        receipt = self.root / "receipt.json"
        policy.write_text("{}", encoding="utf-8")

        with self.assertRaises(MOD.CalibrationError):
            MOD.derive_policy([a, b], policy, receipt)


if __name__ == "__main__":
    unittest.main()
