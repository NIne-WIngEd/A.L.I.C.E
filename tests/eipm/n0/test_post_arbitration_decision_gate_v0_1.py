import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

DRIVER = Path(__file__).parents[3] / "scripts/eipm/n0/post_arbitration_decision_gate_v0_1.py"
SPEC = importlib.util.spec_from_file_location("post_arb_gate_v01", DRIVER)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

class PostArbitrationDecisionGateV01Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.revision = "a" * 40
        self.manifest_sha = "b" * 64
        self.fingerprint = "c" * 64

    def tearDown(self):
        self.tmp.cleanup()

    def receipt(self, classification="IMPROVEMENT", states=None):
        states = states or [classification]
        return {
            "protocol_version": MOD.ARBITRATION_PROTOCOL_VERSION,
            "experiment_id": "unit-arbitration",
            "source_revision": self.revision,
            "manifest_sha256": self.manifest_sha,
            "common_fingerprint": self.fingerprint,
            "status": "COMPLETE",
            "classification": classification,
            "comparisons": [{"path": f"metric.{i}", "result": state} for i, state in enumerate(states)],
            "arm_common_fingerprints": {"canonical": self.fingerprint, "candidate": self.fingerprint},
            "ratifies_repair": False,
            "scale_authorized": False,
            "promotion_authorized": False,
            "n0_complete": False,
        }

    def write_config(self, receipt):
        receipt_path = self.root / "arbitration.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        config = {
            "protocol_version": MOD.PROTOCOL_VERSION,
            "decision_id": "unit-gate",
            "arbitration_receipt": {
                "path": str(receipt_path),
                "sha256": MOD.sha256_file(receipt_path),
            },
            "expected_source_revision": self.revision,
            "expected_arbitration_manifest_sha256": self.manifest_sha,
            "expected_common_fingerprint": self.fingerprint,
            "invariants": {
                "ratifies_repair": False,
                "training_enabled": False,
                "promotion_authorized": False,
                "scale_authorized": False,
                "n0_ready": False,
            },
        }
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def run_case(self, receipt):
        return MOD.run_gate(self.write_config(receipt), self.root / "decision.json")

    def test_improvement_authorizes_exactly_one_final_challenge(self):
        result = self.run_case(self.receipt("IMPROVEMENT", ["IMPROVEMENT", "EQUIVALENT"]))
        self.assertEqual(result["decision"], "AUTHORIZE_ONE_FINAL_FROZEN_CHALLENGE")
        self.assertTrue(result["final_frozen_challenge_authorized"])
        self.assertEqual(result["final_frozen_challenge_limit"], 1)
        self.assertTrue(result["canonical_remains_current_baseline"])
        self.assertFalse(result["candidate_graph_promoted"])
        self.assertFalse(result["ratifies_repair"])
        self.assertFalse(result["n0_ready"])

    def test_harm_retains_canonical(self):
        result = self.run_case(self.receipt("HARM", ["IMPROVEMENT", "HARM"]))
        self.assertEqual(result["decision"], "RETAIN_CANONICAL_DOWNSTREAM_HARM")
        self.assertFalse(result["final_frozen_challenge_authorized"])

    def test_equivalence_retains_canonical(self):
        result = self.run_case(self.receipt("EQUIVALENT", ["EQUIVALENT", "EQUIVALENT"]))
        self.assertEqual(result["decision"], "RETAIN_CANONICAL_DOWNSTREAM_EQUIVALENCE")
        self.assertFalse(result["final_frozen_challenge_authorized"])

    def test_inconsistent_classification_fails_closed(self):
        receipt = self.receipt("IMPROVEMENT", ["IMPROVEMENT", "HARM"])
        with self.assertRaises(MOD.GateError):
            self.run_case(receipt)

    def test_preflight_receipt_cannot_decide(self):
        receipt = self.receipt()
        receipt["status"] = "PREFLIGHT_OK"
        with self.assertRaises(MOD.GateError):
            self.run_case(receipt)

    def test_receipt_hash_mismatch_fails(self):
        config_path = self.write_config(self.receipt())
        receipt_path = self.root / "arbitration.json"
        receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(MOD.GateError):
            MOD.run_gate(config_path, self.root / "decision.json")

    def test_wrong_common_fingerprint_fails(self):
        receipt = self.receipt()
        receipt["common_fingerprint"] = "d" * 64
        receipt["arm_common_fingerprints"] = {"canonical": "d" * 64, "candidate": "d" * 64}
        with self.assertRaises(MOD.GateError):
            self.run_case(receipt)

    def test_positive_authority_never_promotes_or_completes_n0(self):
        result = self.run_case(self.receipt("IMPROVEMENT"))
        for key in ("candidate_graph_promoted", "ratifies_repair", "training_authorized",
                    "promotion_authorized", "scale_authorized", "n0_ready", "n0_complete"):
            self.assertFalse(result[key])

    def test_refuses_output_overwrite(self):
        config_path = self.write_config(self.receipt())
        output = self.root / "decision.json"
        output.write_text("{}", encoding="utf-8")
        with self.assertRaises(MOD.GateError):
            MOD.run_gate(config_path, output)

if __name__ == "__main__":
    unittest.main()
