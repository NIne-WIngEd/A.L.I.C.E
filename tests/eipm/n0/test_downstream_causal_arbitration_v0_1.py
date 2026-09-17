import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


DRIVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "eipm"
    / "n0"
    / "downstream_causal_arbitration_v0_1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "downstream_causal_arbitration_v0_1",
    DRIVER_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArbitrationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.evaluator = self.root / "dummy_eval.py"
        self.evaluator.write_text(
            """
import argparse, json
p=argparse.ArgumentParser()
p.add_argument('--graph', required=True)
p.add_argument('--latent-checkpoint', required=True)
p.add_argument('--config', required=True)
p.add_argument('--evaluation-set', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
with open(a.graph, 'r', encoding='utf-8') as f:
    graph=json.load(f)
with open(a.output, 'w', encoding='utf-8') as f:
    json.dump({'metrics': {'support': graph['support'], 'loss': graph['loss']}}, f)
""".lstrip(),
            encoding="utf-8",
        )
        self.latent = self.root / "latent.bin"
        self.latent.write_bytes(b"frozen-latent-checkpoint")
        self.config = self.root / "eval_config.json"
        self.config.write_text('{"mode":"eval"}\n', encoding="utf-8")
        self.eval_set = self.root / "eval.jsonl"
        self.eval_set.write_text('{"id":1}\n', encoding="utf-8")
        self.canonical = self.root / "canonical.json"
        self.candidate = self.root / "candidate.json"
        self.manifest_path = self.root / "manifest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_graphs(
        self,
        canonical_support=0.8,
        candidate_support=0.9,
        canonical_loss=0.4,
        candidate_loss=0.3,
    ):
        self.canonical.write_text(
            json.dumps(
                {"support": canonical_support, "loss": canonical_loss}
            ),
            encoding="utf-8",
        )
        self.candidate.write_text(
            json.dumps(
                {"support": candidate_support, "loss": candidate_loss}
            ),
            encoding="utf-8",
        )

    def _manifest(self):
        return {
            "protocol_version": MODULE.PROTOCOL_VERSION,
            "experiment_id": "unit-test",
            "source_revision": "deadbeef",
            "invariants": {
                "diagnostic_only": True,
                "training_enabled": False,
                "ratifies_repair": False,
                "scale_authorized": False,
                "promotion_authorized": False,
            },
            "common_inputs": {
                "evaluator_script": {
                    "path": str(self.evaluator),
                    "sha256": _sha(self.evaluator),
                },
                "latent_checkpoint": {
                    "path": str(self.latent),
                    "sha256": _sha(self.latent),
                },
                "evaluator_config": {
                    "path": str(self.config),
                    "sha256": _sha(self.config),
                },
                "evaluation_set": {
                    "path": str(self.eval_set),
                    "sha256": _sha(self.eval_set),
                },
            },
            "arms": {
                "canonical": {
                    "label": "canonical",
                    "graph": {
                        "path": str(self.canonical),
                        "sha256": _sha(self.canonical),
                    },
                },
                "candidate": {
                    "label": "candidate",
                    "graph": {
                        "path": str(self.candidate),
                        "sha256": _sha(self.candidate),
                    },
                },
            },
            "evaluator_argv": [
                sys.executable,
                "{evaluator_script}",
                "--graph",
                "{graph_path}",
                "--latent-checkpoint",
                "{latent_checkpoint}",
                "--config",
                "{evaluator_config}",
                "--evaluation-set",
                "{evaluation_set}",
                "--output",
                "{output_path}",
            ],
            "metrics": [
                {
                    "path": "metrics.support",
                    "direction": "higher",
                    "atol": 1e-12,
                    "rtol": 0.0,
                },
                {
                    "path": "metrics.loss",
                    "direction": "lower",
                    "atol": 1e-12,
                    "rtol": 0.0,
                },
            ],
        }

    def _run(self, manifest, name="result.json", preflight=False):
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        output = self.root / name
        result = MODULE.run_protocol(
            self.manifest_path,
            output,
            preflight,
        )
        self.assertEqual(
            result,
            json.loads(output.read_text(encoding="utf-8")),
        )
        return result

    def test_improvement(self):
        self._write_graphs(candidate_support=0.9, candidate_loss=0.3)
        result = self._run(self._manifest())
        self.assertEqual(result["classification"], "IMPROVEMENT")
        self.assertFalse(result["ratifies_repair"])
        self.assertFalse(result["scale_authorized"])
        self.assertFalse(result["promotion_authorized"])

    def test_harm_if_any_metric_harms(self):
        self._write_graphs(candidate_support=0.95, candidate_loss=0.6)
        result = self._run(self._manifest())
        self.assertEqual(result["classification"], "HARM")

    def test_equivalent_inside_tolerance(self):
        self._write_graphs(
            candidate_support=0.8000001,
            candidate_loss=0.3999999,
        )
        manifest = self._manifest()
        for metric in manifest["metrics"]:
            metric["atol"] = 1e-3
        result = self._run(manifest)
        self.assertEqual(result["classification"], "EQUIVALENT")

    def test_hash_mismatch_fails_closed(self):
        self._write_graphs()
        manifest = self._manifest()
        manifest["arms"]["candidate"]["graph"]["sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_manifest(manifest, self.manifest_path)

    def test_invariant_weakening_fails_closed(self):
        self._write_graphs()
        manifest = self._manifest()
        manifest["invariants"]["ratifies_repair"] = True
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_manifest(manifest, self.manifest_path)

    def test_training_argument_is_rejected(self):
        self._write_graphs()
        manifest = self._manifest()
        manifest["evaluator_argv"].append("--train")
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_manifest(manifest, self.manifest_path)

    def test_preflight_does_not_execute_evaluator(self):
        self._write_graphs()
        result = self._run(
            self._manifest(),
            name="preflight.json",
            preflight=True,
        )
        self.assertEqual(result["status"], "PREFLIGHT_OK")
        self.assertIsNone(result["classification"])
        self.assertFalse(
            (self.root / ".preflight.canonical.raw.json").exists()
        )
        self.assertFalse(
            (self.root / ".preflight.candidate.raw.json").exists()
        )

    def test_two_arms_share_command_template_and_common_inputs(self):
        self._write_graphs()
        result = self._run(self._manifest())
        a = result["commands"]["canonical"]
        b = result["commands"]["candidate"]
        self.assertEqual(len(a), len(b))
        differing = [
            index
            for index, (left, right) in enumerate(zip(a, b))
            if left != right
        ]
        self.assertEqual(len(differing), 2)
        self.assertEqual(a[differing[0]], str(self.canonical.resolve()))
        self.assertEqual(b[differing[0]], str(self.candidate.resolve()))
        self.assertIn("canonical.raw", a[differing[1]])
        self.assertIn("candidate.raw", b[differing[1]])


if __name__ == "__main__":
    unittest.main()
