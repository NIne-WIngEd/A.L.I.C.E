import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PREP = Path(__file__).parents[3] / "scripts/eipm/n0/prepare_downstream_causal_arbitration_v0_1.py"
SPEC = importlib.util.spec_from_file_location("prepare_arb_v01", PREP)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class PrepareDownstreamArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.revision = "a" * 40
        self.full_eval = MOD.load_sibling(
            "prep_test_full_eval",
            "downstream_full_stack_graph_evaluator_v0_1.py",
        )

        self.paths = {}
        for role in [
            "latent_config",
            "semantic_config",
            "structured_config",
            "evidence_adapter",
            "fusion_config",
            "fusion_ratification",
        ]:
            path = self.root / f"{role}.bin"
            path.write_text(role, encoding="utf-8")
            self.paths[role] = path

        for role in [
            "semantic_checkpoint",
            "tokenizer_dir",
            "structured_checkpoint",
            "fusion_checkpoint",
        ]:
            path = self.root / role
            path.mkdir()
            (path / "artifact.bin").write_text(role, encoding="utf-8")
            self.paths[role] = path

        self.latent = self.root / "latent.safetensors"
        self.latent.write_bytes(b"latent")
        self.evaluation = self.root / "eval.jsonl"
        self.evaluation.write_text('{"id":"x"}\n', encoding="utf-8")
        self.canonical = self.root / "canonical.safetensors"
        self.canonical.write_bytes(b"canonical")
        self.candidate = self.root / "candidate.safetensors"
        self.candidate.write_bytes(b"candidate")

        self.policy = self.root / "policy.json"
        self.policy.write_text(
            json.dumps(
                {
                    "protocol_version": MOD.METRIC_POLICY_VERSION,
                    "selected_before_arm_results": True,
                    "results_observed": False,
                    "metrics": [
                        {
                            "path": path,
                            "direction": direction,
                            "atol": 1e-6,
                            "rtol": 1e-5,
                        }
                        for path, direction in MOD.EXPECTED_METRIC_DIRECTIONS.items()
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self):
        return MOD.prepare(
            experiment_id="unit-test",
            source_revision=self.revision,
            metric_policy_path=self.policy,
            evaluation_set=self.evaluation,
            latent_checkpoint=self.latent,
            latent_config=self.paths["latent_config"],
            semantic_config=self.paths["semantic_config"],
            semantic_checkpoint=self.paths["semantic_checkpoint"],
            tokenizer_dir=self.paths["tokenizer_dir"],
            structured_config=self.paths["structured_config"],
            structured_checkpoint=self.paths["structured_checkpoint"],
            evidence_adapter=self.paths["evidence_adapter"],
            fusion_checkpoint=self.paths["fusion_checkpoint"],
            fusion_config=self.paths["fusion_config"],
            fusion_ratification=self.paths["fusion_ratification"],
            canonical_graph=self.canonical,
            candidate_graph=self.candidate,
            stack_manifest_output=self.root / "stack.json",
            arbitration_manifest_output=self.root / "manifest.json",
            preflight_output=self.root / "preflight.json",
            evaluator_path=Path(__file__).parents[3] / "scripts/eipm/n0/downstream_full_stack_graph_evaluator_v0_1.py",
        )

    def test_prepare_writes_inner_outer_and_preflight(self):
        result = self.prepare()
        self.assertEqual(result["status"], "PREPARED_AND_PREFLIGHTED")
        receipt = json.loads((self.root / "preflight.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "PREFLIGHT_OK")
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {item["path"] for item in manifest["metrics"]},
            set(MOD.EXPECTED_METRIC_DIRECTIONS),
        )
        self.assertEqual(manifest["source_revision"], self.revision)

    def test_policy_missing_metric_fails(self):
        raw = json.loads(self.policy.read_text(encoding="utf-8"))
        raw["metrics"].pop()
        self.policy.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(MOD.PreparationError):
            self.prepare()

    def test_policy_wrong_direction_fails(self):
        raw = json.loads(self.policy.read_text(encoding="utf-8"))
        raw["metrics"][0]["direction"] = (
            "lower" if raw["metrics"][0]["direction"] == "higher" else "higher"
        )
        self.policy.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(MOD.PreparationError):
            self.prepare()

    def test_identical_graph_bytes_fail(self):
        self.candidate.write_bytes(self.canonical.read_bytes())
        with self.assertRaises(MOD.PreparationError):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
