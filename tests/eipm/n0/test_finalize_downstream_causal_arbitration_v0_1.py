import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

FINALIZER = (
    Path(__file__).parents[3]
    / "scripts/eipm/n0/finalize_downstream_causal_arbitration_v0_1.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_arb_v01", FINALIZER)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FinalizeDownstreamCausalArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.revision = "a" * 40
        self.arb = MOD.load_sibling(
            "finalizer_test_arb", "downstream_causal_arbitration_v0_2.py"
        )

        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import argparse, json\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--graph', required=True)\n"
            "p.add_argument('--latent-checkpoint', required=True)\n"
            "p.add_argument('--config', required=True)\n"
            "p.add_argument('--evaluation-set', required=True)\n"
            "p.add_argument('--output', required=True)\n"
            "a=p.parse_args()\n"
            "score=2.0 if open(a.graph,'rb').read()==b'candidate' else 1.0\n"
            "with open(a.output,'w',encoding='utf-8') as h: json.dump({'score':score},h)\n",
            encoding="utf-8",
        )
        for name in ("latent.bin", "config.json", "evaluation.jsonl"):
            (self.root / name).write_text(name, encoding="utf-8")
        (self.root / "canonical.bin").write_bytes(b"canonical")
        (self.root / "candidate.bin").write_bytes(b"candidate")

        def spec(name):
            path = self.root / name
            return {"path": str(path), "sha256": self.arb.sha256_file(path)}

        self.manifest = self.root / "manifest.json"
        manifest = {
            "protocol_version": self.arb.PROTOCOL_VERSION,
            "experiment_id": "finalizer-unit-test",
            "source_revision": self.revision,
            "invariants": {
                "diagnostic_only": True,
                "training_enabled": False,
                "ratifies_repair": False,
                "scale_authorized": False,
                "promotion_authorized": False,
            },
            "common_inputs": {
                "evaluator": spec("evaluator.py"),
                "latent_checkpoint": spec("latent.bin"),
                "evaluation_config": spec("config.json"),
                "evaluation_set": spec("evaluation.jsonl"),
            },
            "arms": {
                "canonical": {
                    "label": "canonical_graph",
                    "graph": spec("canonical.bin"),
                },
                "candidate": {
                    "label": "relation_endpoint_repair_step_00000200",
                    "graph": spec("candidate.bin"),
                },
            },
            "evaluator_argv": [
                sys.executable,
                "{input:evaluator}",
                "--graph",
                "{graph}",
                "--latent-checkpoint",
                "{input:latent_checkpoint}",
                "--config",
                "{input:evaluation_config}",
                "--evaluation-set",
                "{input:evaluation_set}",
                "--output",
                "{output}",
            ],
            "metrics": [
                {
                    "path": "score",
                    "direction": "higher",
                    "atol": 0.0,
                    "rtol": 0.0,
                }
            ],
        }
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.preflight = self.root / "preflight.json"
        self.result = self.root / "result.json"
        self.arb.run_protocol(self.manifest, self.preflight, preflight_only=True)
        self.arb.run_protocol(self.manifest, self.result, preflight_only=False)

        self.gate_config = self.root / "gate-config.json"
        self.gate_receipt = self.root / "gate-receipt.json"
        self.final_receipt = self.root / "finalization-receipt.json"

    def tearDown(self):
        self.tmp.cleanup()

    def finalize(self):
        return MOD.finalize(
            preflight_path=self.preflight,
            result_path=self.result,
            gate_config_output=self.gate_config,
            gate_receipt_output=self.gate_receipt,
            finalization_receipt_output=self.final_receipt,
            enforce_source_freeze=False,
        )

    def test_complete_improvement_materializes_gate_and_authorizes_one_challenge(self):
        receipt = self.finalize()
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["classification"], "IMPROVEMENT")
        self.assertEqual(
            receipt["decision"], "AUTHORIZE_ONE_FINAL_FROZEN_CHALLENGE"
        )
        self.assertTrue(receipt["final_frozen_challenge_authorized"])
        self.assertEqual(receipt["final_frozen_challenge_limit"], 1)

        gate = json.loads(self.gate_receipt.read_text(encoding="utf-8"))
        self.assertFalse(gate["candidate_graph_promoted"])
        self.assertFalse(gate["ratifies_repair"])
        self.assertFalse(gate["training_authorized"])
        self.assertFalse(gate["promotion_authorized"])
        self.assertFalse(gate["scale_authorized"])
        self.assertFalse(gate["n0_ready"])
        self.assertFalse(gate["n0_complete"])

    def test_tampered_raw_arm_output_is_refused(self):
        result = json.loads(self.result.read_text(encoding="utf-8"))
        candidate_raw = Path(result["raw_outputs"]["candidate"]["path"])
        candidate_raw.write_text('{"score":999.0}\n', encoding="utf-8")
        with self.assertRaises(MOD.FinalizationError):
            self.finalize()

    def test_tampered_comparison_is_refused_even_with_complete_status(self):
        result = json.loads(self.result.read_text(encoding="utf-8"))
        result["comparisons"][0]["candidate"] = 999.0
        self.result.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaises(MOD.FinalizationError):
            self.finalize()

    def test_preflight_frozen_field_drift_is_refused(self):
        result = json.loads(self.result.read_text(encoding="utf-8"))
        result["common_fingerprint"] = "f" * 64
        result["arm_common_fingerprints"] = {
            "canonical": "f" * 64,
            "candidate": "f" * 64,
        }
        self.result.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaises(MOD.FinalizationError):
            self.finalize()

    def test_result_classification_is_recomputed_from_raw_evidence(self):
        result = json.loads(self.result.read_text(encoding="utf-8"))
        result["classification"] = "EQUIVALENT"
        self.result.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaises(MOD.FinalizationError):
            self.finalize()

    def test_refuses_to_overwrite_any_finalization_output(self):
        self.gate_config.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(MOD.FinalizationError):
            self.finalize()


if __name__ == "__main__":
    unittest.main()
