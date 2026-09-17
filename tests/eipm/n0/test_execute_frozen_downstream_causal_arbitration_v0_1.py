import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

EXEC = Path(__file__).parents[3] / "scripts/eipm/n0/execute_frozen_downstream_causal_arbitration_v0_1.py"
SPEC = importlib.util.spec_from_file_location("execute_arb_v01", EXEC)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class ExecuteFrozenArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.revision = "a" * 40
        self.arb = MOD.load_sibling(
            "exec_test_arb", "downstream_causal_arbitration_v0_2.py"
        )
        self.full_eval = MOD.load_sibling(
            "exec_test_full_eval",
            "downstream_full_stack_graph_evaluator_v0_1.py",
        )

        self.evaluator = self.root / "eval.py"
        self.evaluator.write_text("print('unused')\n", encoding="utf-8")
        for name in ["latent.bin", "config.json", "evaluation.jsonl"]:
            (self.root / name).write_text(name, encoding="utf-8")
        (self.root / "canonical.bin").write_bytes(b"canonical")
        (self.root / "candidate.bin").write_bytes(b"candidate")

        self.policy = self.root / "policy.json"
        self.policy.write_text("{}\n", encoding="utf-8")

        def spec(name):
            p = self.root / name
            return {"path": str(p), "sha256": self.full_eval.sha256_file(p)}

        self.manifest = self.root / "manifest.json"
        payload = {
            "protocol_version": self.arb.PROTOCOL_VERSION,
            "experiment_id": "unit-test",
            "source_revision": self.revision,
            "invariants": {
                "diagnostic_only": True,
                "training_enabled": False,
                "ratifies_repair": False,
                "scale_authorized": False,
                "promotion_authorized": False,
            },
            "metric_policy": {
                "path": str(self.policy),
                "sha256": self.full_eval.sha256_file(self.policy),
            },
            "common_inputs": {
                "evaluator": spec("eval.py"),
                "latent_checkpoint": spec("latent.bin"),
                "evaluation_config": spec("config.json"),
                "evaluation_set": spec("evaluation.jsonl"),
            },
            "arms": {
                "canonical": {"label": "canonical", "graph": spec("canonical.bin")},
                "candidate": {"label": "candidate", "graph": spec("candidate.bin")},
            },
            "evaluator_argv": [
                "python",
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
            "metrics": [{"path": "score", "direction": "higher", "atol": 0.0, "rtol": 0.0}],
        }
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        self.preflight = self.root / "preflight.json"
        self.arb.run_protocol(self.manifest, self.preflight, preflight_only=True)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, revision=None):
        return MOD.verify_preflight_freeze(
            manifest_path=self.manifest,
            preflight_receipt_path=self.preflight,
            observed_source_revision=revision or self.revision,
            arb=self.arb,
            full_eval=self.full_eval,
        )

    def test_frozen_preflight_verifies(self):
        frozen = self.verify()
        self.assertEqual(frozen["source_revision"], self.revision)

    def test_manifest_change_after_preflight_fails(self):
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        raw["experiment_id"] = "changed"
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(MOD.ExecutionFreezeError):
            self.verify()

    def test_source_revision_change_after_preflight_fails(self):
        with self.assertRaises(MOD.ExecutionFreezeError):
            self.verify("b" * 40)

    def test_metric_policy_change_after_preflight_fails(self):
        self.policy.write_text('{"changed":true}\n', encoding="utf-8")
        with self.assertRaises(MOD.ExecutionFreezeError):
            self.verify()

    def test_repository_source_freeze_accepts_clean_tracked_tree(self):
        clean = SimpleNamespace(returncode=0, stdout=self.revision + "\n")
        diff = SimpleNamespace(returncode=0, stdout="")
        with mock.patch.object(MOD.subprocess, "run", side_effect=[clean, diff]):
            result = MOD.verify_repository_source_freeze(
                repo_root=self.root,
                expected_revision=self.revision,
            )
        self.assertTrue(result["tracked_worktree_clean"])
        self.assertEqual(result["source_revision"], self.revision)

    def test_repository_source_freeze_rejects_tracked_drift(self):
        clean = SimpleNamespace(returncode=0, stdout=self.revision + "\n")
        dirty = SimpleNamespace(returncode=1, stdout="")
        with mock.patch.object(MOD.subprocess, "run", side_effect=[clean, dirty]):
            with self.assertRaises(MOD.ExecutionFreezeError):
                MOD.verify_repository_source_freeze(
                    repo_root=self.root,
                    expected_revision=self.revision,
                )

    def test_repository_source_freeze_rejects_head_drift(self):
        changed = SimpleNamespace(returncode=0, stdout=("b" * 40) + "\n")
        with mock.patch.object(MOD.subprocess, "run", return_value=changed):
            with self.assertRaises(MOD.ExecutionFreezeError):
                MOD.verify_repository_source_freeze(
                    repo_root=self.root,
                    expected_revision=self.revision,
                )


if __name__ == "__main__":
    unittest.main()
