"""Tiny public Torch mechanics fixtures; no actual Gemma/FewRel qualification.

The fixture publisher/verifiers are injected and closures remain explicitly
PUBLIC_MECHANICS_ONLY. Real cached BF16 banks, readout exports and exact metric
reproduction exercise the consumer without source or publisher loading.
"""
from copy import deepcopy
from hashlib import sha256
import unittest
from unittest.mock import patch

import torch

from scripts.eipm.gemma_n0 import diagnose_public_features as cli
from src.alice_personality.gemma_n0 import public_feature_diagnostics as diagnostic
from src.alice_personality.gemma_n0 import public_feature_handoff as handoff
from src.alice_personality.gemma_n0 import public_semantic_experiment as producer
from tests.personality import test_public_feature_handoff as fixtures


class PublicFeatureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PublicFeatureHandoffTests("test_fixture_scope_external_pins_and_create_only")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.export()
        self.root = self.fixture.root
        self.export = self.root / "run/semantic-readout.pt"
        self.export_sha = handoff._hash(self.export)

    def run_diagnostic(self, name="diagnostic", **kwargs):
        return diagnostic.diagnose_public_features(self.fixture.package,
            expected_manifest_sha256=self.fixture.pins["manifest_sha256"],
            expected_closure_sha256=self.fixture.pins["closure_sha256"],
            semantic_export_path=self.export, expected_export_sha256=self.export_sha,
            output_directory=self.root / name, allow_mechanics_only=True, **kwargs)

    def test_actual_fixture_export_reloads_reproduces_and_stays_source_free(self):
        original_package = {path.relative_to(self.fixture.package).as_posix(): handoff._hash(path)
                            for path in self.fixture.package.rglob("*") if path.is_file()}
        original_export = self.export.read_bytes()
        rng = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        with patch.object(producer, "_load_plan", side_effect=AssertionError("original source reopening forbidden")), \
                patch.object(producer, "_runtime", side_effect=AssertionError("publisher runtime loading forbidden")):
            receipt = self.run_diagnostic()
        self.fixture.provider_factory.assert_not_called()
        self.assertEqual(receipt["status"], "PUBLIC_MECHANICS_ONLY")
        self.assertEqual(receipt["original_metrics_reproduced_exact"]["learned"], self.fixture.experiment["learned"])
        self.assertEqual(receipt["original_metrics_reproduced_exact"]["fixed_semantic_control"],
                         self.fixture.experiment["baselines"]["fixed_semantic_control"])
        self.assertTrue(receipt["original_semantic_export_reloaded"])
        self.assertTrue(receipt["original_features_reverified"])
        for name in ("training_performed", "publisher_checkpoint_loaded", "publisher_forward_performed",
                     "upstream_gradient", "upstream_tensor_mutation", "private_identity_data",
                     "final_payload_opened", "n0_approved", "personality_qualified", "gemma_neutrality_established"):
            self.assertIs(receipt[name], False)
        self.assertEqual(len(receipt["per_example"]), 3)
        self.assertEqual(receipt["learned_layer_mixture"]["state_count"], 3)
        self.assertAlmostEqual(sum(receipt["learned_layer_mixture"]["softmax_weights"]), 1.0, places=6)
        self.assertTrue(receipt["learned_layer_mixture"]["global_query_independent"])
        for row in receipt["per_example"]:
            self.assertEqual(len(row["candidate_features"]), row["candidate_count"])
            self.assertEqual(len(row["source_feature"]["tensor_sha256"]), 64)
            self.assertEqual(set(row["results"]), {"original", "constant_train_source", "within_split_cyclic_source"})
            for condition in row["results"].values():
                self.assertEqual(set(condition), {"learned", "fixed_semantic_control"})
                self.assertEqual(len(condition["learned"]["logits"]), row["candidate_count"])
        self.assertFalse(receipt["controls"]["counterfactual_truth"])
        self.assertFalse(receipt["controls"]["entity_role_reversal"])
        self.assertFalse(receipt["controls"]["within_split_cyclic_source"]["eligible_splits"]["dev"])
        self.assertEqual(receipt["aggregates"]["within_split_cyclic_source"]["learned"]["dev"]["source_dependence"]["actually_replaced_rows"], 0)
        dev = receipt["aggregates"]["original"]["learned"]["dev"]
        self.assertEqual(dev["strata"]["uniform_random_expected_accuracy"], 1 / 3)
        self.assertEqual(set(dev["strata"]["target_description_seen_positive_in_train"]), {"False"})
        plan = handoff._read(self.root / "diagnostic/diagnostic-plan.json")
        self.assertEqual(receipt["binding"]["diagnostic_plan_content_sha256"], plan["receipt_sha256"])
        self.assertEqual(handoff._read(self.root / "diagnostic/diagnostics.json"), receipt)
        self.assertEqual(original_package, {path.relative_to(self.fixture.package).as_posix(): handoff._hash(path)
                                           for path in self.fixture.package.rglob("*") if path.is_file()})
        self.assertEqual(self.export.read_bytes(), original_export)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(torch.get_num_threads(), threads)

    def test_fixture_production_boundary_and_wrong_export_rejected(self):
        with self.assertRaisesRegex(handoff.HandoffError, "mechanics-only"):
            diagnostic.diagnose_public_features(self.fixture.package,
                expected_manifest_sha256=self.fixture.pins["manifest_sha256"],
                expected_closure_sha256=self.fixture.pins["closure_sha256"],
                semantic_export_path=self.export, expected_export_sha256=self.export_sha,
                output_directory=self.root / "false-production")
        self.export.write_bytes(self.export.read_bytes() + b"PUBLIC mutation")
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "export external pin"):
            self.run_diagnostic()
        self.export_sha = handoff._hash(self.export)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "original completed experiment"):
            self.run_diagnostic()

    def test_runtime_and_create_only_output_fail_before_scoring(self):
        with patch.object(torch, "__version__", "different-public-runtime"), \
                self.assertRaisesRegex(diagnostic.DiagnosticError, "original Torch runtime"):
            self.run_diagnostic()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "fresh and outside"):
            self.run_diagnostic(name="package")
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "fresh and outside"):
            self.run_diagnostic(name="package/nested-diagnostic")
        self.assertFalse((self.fixture.package / "nested-diagnostic").exists())
        self.assertFalse((self.root / "diagnostic").exists())

    def test_reproduction_failure_preserves_predeclared_plan_without_success(self):
        evaluate = producer._evaluate
        def wrong(*args, **kwargs):
            metrics = evaluate(*args, **kwargs)
            metrics["mean_cross_entropy"] += 0.01
            return metrics
        with patch.object(producer, "_evaluate", side_effect=wrong), \
                self.assertRaisesRegex(diagnostic.DiagnosticError, "did not reproduce exactly"):
            self.run_diagnostic()
        self.assertTrue((self.root / "diagnostic/diagnostic-plan.json").is_file())
        self.assertFalse((self.root / "diagnostic/diagnostics.json").exists())

    def test_in_memory_feature_mutation_refuses_success(self):
        captured = {}
        banks = diagnostic._banks
        dependence = diagnostic._dependence
        def capture(imported):
            captured.update(banks(imported))
            return captured
        def mutate(*args, **kwargs):
            result = dependence(*args, **kwargs)
            next(iter(captured.values())).layers[0, 0, 0] += 1
            return result
        with patch.object(diagnostic, "_banks", side_effect=capture), \
                patch.object(diagnostic, "_dependence", side_effect=mutate), \
                self.assertRaisesRegex(diagnostic.DiagnosticError, "in-memory frozen features changed"):
            self.run_diagnostic()
        self.assertFalse((self.root / "diagnostic/diagnostics.json").exists())

    def test_package_mutation_during_controls_fails_fresh_admission(self):
        dependence = diagnostic._dependence
        path = next((self.fixture.package / "features").glob("*.json"))
        def mutate(*args, **kwargs):
            result = dependence(*args, **kwargs)
            path.write_bytes(path.read_bytes() + b" ")
            return result
        with patch.object(diagnostic, "_dependence", side_effect=mutate), \
                self.assertRaisesRegex(handoff.HandoffError, "member size/hash differs"):
            self.run_diagnostic()
        self.assertFalse((self.root / "diagnostic/diagnostics.json").exists())

    def test_diagnostic_code_recheck_refuses_success(self):
        code = diagnostic._own_code()
        wrong = {**code, "public_feature_diagnostics.py": "0" * 64}
        with patch.object(diagnostic, "_own_code", side_effect=[code, wrong]), \
                self.assertRaisesRegex(diagnostic.DiagnosticError, "diagnostic code, export or declared plan changed"):
            self.run_diagnostic()
        self.assertFalse((self.root / "diagnostic/diagnostics.json").exists())

    def test_controls_use_metadata_hash_not_targets_and_rotate_within_split(self):
        rows = []
        for split in ("train", "dev"):
            for index in range(3):
                rows.append({"split": split, "id_metadata": f"PUBLIC-{split}-{index}",
                    "row_sha256": sha256(f"PUBLIC:{split}:{index}".encode()).hexdigest(),
                    "source": {"text": f"PUBLIC {split} source {index}", "spans": {}},
                    "target_index_metadata": index})
        actual = diagnostic._controls(rows)
        altered = deepcopy(rows)
        for row in altered:
            row["target_index_metadata"] = 100 - row["target_index_metadata"]
        self.assertEqual(actual, diagnostic._controls(list(reversed(altered))))
        replacements = actual["within_split_cyclic_source"]["source_by_example_id"]
        lookup = {row["id_metadata"]: row for row in rows}
        for rid, replacement in replacements.items():
            self.assertNotEqual(rid, replacement)
            self.assertEqual(lookup[rid]["split"], lookup[replacement]["split"])
        self.assertEqual(len(set(replacements.values())), len(rows))
        self.assertTrue(all(actual["within_split_cyclic_source"]["eligible_splits"].values()))

    def test_cli_passes_only_public_production_arguments(self):
        summary = {"schema": diagnostic.SCHEMA, "state": "PUBLIC_FEATURE_DIAGNOSTICS_UNQUALIFIED",
                   "status": "CLOSED_PUBLIC_DIAGNOSTICS_COMPLETE", "receipt_sha256": "0" * 64,
                   "n0_approved": False, "personality_qualified": False}
        with patch.object(cli, "diagnose_public_features", return_value=summary) as call, \
                patch("builtins.print"):
            self.assertEqual(cli.main(["--package-directory", str(self.fixture.package),
                "--manifest-sha256", self.fixture.pins["manifest_sha256"],
                "--closure-sha256", self.fixture.pins["closure_sha256"],
                "--semantic-export", str(self.export), "--export-sha256", self.export_sha,
                "--output-directory", str(self.root / "cli-diagnostic"), "--cpu-threads", "4"]), 0)
        self.assertNotIn("allow_mechanics_only", call.call_args.kwargs)
        self.assertEqual(call.call_args.kwargs["expected_export_sha256"], self.export_sha)


if __name__ == "__main__":
    unittest.main()
