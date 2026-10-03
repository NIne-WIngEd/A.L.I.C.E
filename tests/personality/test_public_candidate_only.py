"""Synthetic PUBLIC mechanics only; no Gemma, FewRel or personality acceptance.

Real Torch fitting/export/checkpoint/banks exercise the new consumer against a
closed fixture produced by the injected stand-in publisher. Production rejects
this closure. Nothing loads an actual model, private source or FINAL payload.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

import torch

from scripts.eipm.gemma_n0 import fit_public_candidate_only as cli
from src.alice_personality.gemma_n0 import public_candidate_only as candidate
from src.alice_personality.gemma_n0 import public_feature_handoff as handoff
from src.alice_personality.gemma_n0 import public_semantic_experiment as producer
from src.alice_personality.gemma_n0.semantic_readout import SemanticReadout
from tests.personality import test_public_feature_handoff as fixtures


class PublicCandidateOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.fixture = fixtures.PublicFeatureHandoffTests("test_fixture_scope_external_pins_and_create_only")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.export()
        self.root = self.fixture.root
        self.export = self.root / "run/semantic-readout.pt"
        self.midpoint = self.root / "run/midpoint.pt"
        self.export_sha = handoff._hash(self.export)
        self.midpoint_sha = handoff._hash(self.midpoint)

    def fit(self, name="candidate", **kwargs):
        return candidate.fit_public_candidate_only(self.fixture.package,
            expected_manifest_sha256=self.fixture.pins["manifest_sha256"],
            expected_closure_sha256=self.fixture.pins["closure_sha256"], semantic_export_path=self.export,
            expected_export_sha256=self.export_sha, original_midpoint_path=self.midpoint,
            expected_midpoint_sha256=self.midpoint_sha, output_directory=self.root / name,
            allow_mechanics_only=True, **kwargs)

    def test_real_fit_proof_reload_metrics_and_immutable_sources(self):
        original_files = {path.relative_to(self.fixture.package).as_posix(): handoff._hash(path)
            for path in self.fixture.package.rglob("*") if path.is_file()}
        original_export, original_midpoint = self.export.read_bytes(), self.midpoint.read_bytes()
        rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        with patch.object(producer, "_load_plan", side_effect=AssertionError("original source reopening forbidden")), \
                patch.object(producer, "_runtime", side_effect=AssertionError("publisher loading forbidden")):
            receipt = self.fit()
        self.fixture.provider_factory.assert_not_called()
        self.assertEqual(receipt["status"], "PUBLIC_MECHANICS_ONLY")
        self.assertEqual(receipt["qualification"], "UNQUALIFIED")
        self.assertEqual(receipt["optimizer_steps"], 2)
        self.assertTrue(receipt["first_party_public_training_performed"])
        for flag in ("publisher_checkpoint_loaded", "publisher_forward_performed", "upstream_gradient", "upstream_tensor_mutation",
            "private_identity_data", "final_payload_opened", "n0_approved", "personality_qualified", "repair_selected", "gemma_neutrality_established"):
            self.assertIs(receipt[flag], False)
        self.assertTrue(receipt["export"]["reload_exact"])
        self.assertEqual(receipt["export"]["schema"], candidate.EXPORT_SCHEMA)
        self.assertNotEqual(receipt["initial_model_state_tensor_sha256"], receipt["final_model_state_tensor_sha256"])
        self.assertEqual(receipt["original_metrics_reproduced_exact"]["learned"], self.fixture.experiment["learned"])
        for name in ("untrained_readout", "fixed_semantic_control"):
            self.assertEqual(receipt["original_metrics_reproduced_exact"][name], self.fixture.experiment["baselines"][name])
        self.assertTrue(receipt["candidate_only_ignores_actual_sources_and_styles_exact"])
        self.assertEqual(receipt["candidate_permutation"]["rows_checked"], 3)
        self.assertEqual(receipt["artificial_source"]["shape"], [3, 2, 8])
        self.assertTrue(receipt["artificial_source"]["artificial"])
        self.assertFalse(receipt["artificial_source"]["provider_produced"])
        self.assertFalse(receipt["artificial_source"]["real_source_tokens_or_features_used"])
        proof = receipt["sampler_proof"]
        generator = torch.Generator().manual_seed(self.fixture.recipe["seed"] + 1)
        expected = [int(torch.randint(2, (), generator=generator)) for _ in range(2)]
        self.assertEqual(proof["update_indices_metadata"], expected)
        self.assertTrue(proof["midpoint_rng_matches_original_checkpoint"])
        self.assertEqual(proof["observed_original_midpoint_step"], 1)
        self.assertEqual(sum(stats["gradient_observed_steps"] for stats in receipt["gradient_statistics"].values()),
            2 * len(receipt["gradient_statistics"]))
        self.assertTrue(any(stats["nonzero_gradient_steps"] for stats in receipt["gradient_statistics"].values()))
        self.assertTrue(any(stats["nonzero_gradient_steps"] == 0 for stats in receipt["gradient_statistics"].values()))
        self.assertTrue(any(stats["initial_to_final_parameter_update_l2_norm"] > 0 for stats in receipt["gradient_statistics"].values()))
        exposure = receipt["description_exposure"]
        self.assertEqual(exposure["static_train_pool_each_row_once"]["positive_slots"], 2)
        self.assertEqual(exposure["actual_sampler_selected_optimizer_updates"]["positive_slots"], 2)
        self.assertEqual(exposure["actual_sampler_selected_optimizer_updates"]["negative_slots"], 4)
        actual_positive = {key: value["positive_occurrences"] for key, value in exposure["actual_sampler_selected_optimizer_updates"]["by_description_input_sha256"].items()}
        plan_rows = [row for row in handoff._read(self.fixture.plan_file)["examples"] if row["split"] == "train"]
        for index, row in enumerate(plan_rows):
            target = candidate.diagnostic._digest({"text": row["descriptions"][row["target_index_metadata"]], "spans": {}})
            self.assertEqual(actual_positive[target], expected.count(index))
        for split, size in (("train", 2), ("dev", 1), ("sentence_unseen_dev", 1)):
            values = receipt["aggregates"][split]
            for scorer in ("original_learned", "fixed_semantic_control", "candidate_only"):
                metrics = values[scorer]["metrics"]
                self.assertEqual(metrics["rows"], size)
                self.assertGreaterEqual(metrics["mean_multiclass_brier_sum"], 0)
                self.assertLessEqual(metrics["mean_multiclass_brier_sum"], 2)
                self.assertEqual(set(values[scorer]["strata"]), {"candidate_count", "target_candidate_index_metadata", "target_description_seen_positive_in_train"})
            for pairs in values["paired_comparisons"].values():
                self.assertEqual(sum(pairs[key] for key in ("candidate_only_correct_baseline_correct",
                    "candidate_only_correct_baseline_wrong", "candidate_only_wrong_baseline_correct", "both_wrong")), size)
        for row in receipt["per_example"]:
            self.assertEqual(len(row["candidate_input_sha256"]), 3)
            for value in row["results"].values():
                probabilities = torch.tensor(value["probabilities"])
                target = torch.zeros_like(probabilities)
                target[row["target_candidate_index_metadata"]] = 1
                self.assertEqual(value["multiclass_brier_sum"], float(((probabilities - target) ** 2).sum()))
        plan = handoff._read(self.root / "candidate/candidate-only-plan.json")
        handoff._verify_seal(plan)
        self.assertEqual(plan["receipt_sha256"], receipt["binding"]["plan_content_sha256"])
        self.assertEqual(receipt, handoff._read(self.root / "candidate/candidate-only-fit.json"))
        self.assertEqual(original_files, {path.relative_to(self.fixture.package).as_posix(): handoff._hash(path)
            for path in self.fixture.package.rglob("*") if path.is_file()})
        self.assertEqual(self.export.read_bytes(), original_export)
        self.assertEqual(self.midpoint.read_bytes(), original_midpoint)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(torch.get_num_threads(), threads)

    def test_artificial_source_exact_zero_roles_detached_and_source_free(self):
        source = candidate.artificial_source({"state_count": 3, "hidden_size": 8, "width": 4}, torch)
        self.assertFalse(bool(source.layers.any()))
        self.assertEqual(source.layers.dtype, torch.bfloat16)
        self.assertEqual(source.input_ids.tolist(), [0, 0])
        self.assertEqual(source.head_mask.tolist(), [True, False])
        self.assertEqual(source.tail_mask.tolist(), [False, True])
        self.assertFalse(bool((source.head_mask & source.tail_mask).any()))
        self.assertFalse(source.layers.requires_grad)
        imported = self.fixture.admit()
        model = SemanticReadout(state_count=3, hidden_size=8, width=4)
        scorer = lambda ignored, candidates: model(source, candidates)
        banks = imported.examples[0]["candidates"]
        baseline = scorer(None, banks)
        self.assertTrue(torch.equal(baseline, scorer(imported.examples[1]["source"], banks)))
        baseline.sum().backward()
        self.assertIsNone(source.layers.grad)
        self.assertTrue(all(bank.layers.grad is None for bank in banks))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def checkpoint_context(self):
        value = torch.load(self.midpoint, weights_only=True, map_location="cpu")
        torch.manual_seed(self.fixture.recipe["seed"])
        model = SemanticReadout(state_count=3, hidden_size=8, width=4)
        rng = torch.get_rng_state().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.fixture.recipe["learning_rate"], weight_decay=self.fixture.recipe["weight_decay"])
        return value, {"binding": self.fixture.experiment["binding"], "model": model,
            "optimizer": optimizer, "step": 1, "initial_global_rng": rng, "torch": torch}

    def test_checkpoint_strict_fields_shapes_optimizer_and_rng(self):
        original, context = self.checkpoint_context()
        candidate._validate_midpoint(original, **context)
        cases = []
        value = deepcopy(original); value["step"] = True; cases.append(value)
        value = deepcopy(original); value["binding"]["recipe"]["width"] += 1; cases.append(value)
        value = deepcopy(original); value["geometry"]["hidden_size"] += 1; cases.append(value)
        value = deepcopy(original); value["unexpected"] = "fixture"; cases.append(value)
        value = deepcopy(original); value["model"]["layer_logits"] = torch.zeros(2); cases.append(value)
        value = deepcopy(original); value["model"]["layer_logits"][0] = torch.nan; cases.append(value)
        value = deepcopy(original); value["sampler_rng"] = value["sampler_rng"].float(); cases.append(value)
        value = deepcopy(original); value["torch_cpu_rng"][0] ^= 1; cases.append(value)
        value = deepcopy(original); value["optimizer"]["param_groups"][0]["lr"] *= 2; cases.append(value)
        value = deepcopy(original); value["optimizer"]["param_groups"][0]["amsgrad"] = 0; cases.append(value)
        value = deepcopy(original); value["optimizer"]["state"][0]["step"] += 1; cases.append(value)
        value = deepcopy(original); value["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1); cases.append(value)
        value = deepcopy(original); value["optimizer"]["state"][0]["surprise"] = True; cases.append(value)
        for index, value in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(candidate.CandidateOnlyError):
                candidate._validate_midpoint(value, **context)

    def test_midpoint_proves_actual_scalar_sequence_not_batched_draw(self):
        value, _ = self.checkpoint_context()
        actual = candidate._sampler_proof(value, recipe=self.fixture.recipe, train_count=2, torch=torch)
        wrong = deepcopy(value)
        wrong["sampler_rng"] = torch.Generator().manual_seed(self.fixture.recipe["seed"] + 2).get_state()
        with self.assertRaisesRegex(candidate.CandidateOnlyError, "sample sequence"):
            candidate._sampler_proof(wrong, recipe=self.fixture.recipe, train_count=2, torch=torch)
        self.assertEqual(len(actual["update_indices_metadata"]), 2)

    def test_wrong_external_midpoint_and_history_link_fail_before_output(self):
        self.midpoint.write_bytes(self.midpoint.read_bytes() + b"PUBLIC fixture mutation")
        with self.assertRaisesRegex(candidate.CandidateOnlyError, "midpoint external pin"):
            self.fit()
        self.midpoint_sha = handoff._hash(self.midpoint)
        with self.assertRaisesRegex(candidate.CandidateOnlyError, "original completed experiment"):
            self.fit()
        self.assertFalse((self.root / "candidate").exists())

    def test_production_rejects_fixture_and_cli_has_no_fixture_recipe_override(self):
        with self.assertRaisesRegex(handoff.HandoffError, "mechanics-only"):
            candidate.fit_public_candidate_only(self.fixture.package,
                expected_manifest_sha256=self.fixture.pins["manifest_sha256"], expected_closure_sha256=self.fixture.pins["closure_sha256"],
                semantic_export_path=self.export, expected_export_sha256=self.export_sha,
                original_midpoint_path=self.midpoint, expected_midpoint_sha256=self.midpoint_sha,
                output_directory=self.root / "false-production")
        args = ["--package-directory", str(self.fixture.package), "--manifest-sha256", "a" * 64,
            "--closure-sha256", "b" * 64, "--semantic-export", str(self.export), "--export-sha256", "c" * 64,
            "--original-midpoint", str(self.midpoint), "--midpoint-sha256", "d" * 64,
            "--output-directory", str(self.root / "output")]
        summary = {"schema": "fixture", "state": "unqualified", "status": "fixture", "receipt_sha256": "e" * 64,
            "optimizer_steps": 256, "n0_approved": False, "personality_qualified": False, "repair_selected": False}
        with patch.object(cli, "fit_public_candidate_only", return_value=summary) as call, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
        self.assertNotIn("allow_mechanics_only", call.call_args.kwargs)
        self.assertNotIn("recipe", call.call_args.kwargs)
        for extra in ("--allow-mechanics-only", "--width"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(args + [extra])

    def test_runtime_output_bounds_and_strict_types(self):
        with patch.object(torch, "__version__", "wrong"), self.assertRaisesRegex(candidate.CandidateOnlyError, "original Torch"):
            self.fit()
        for name in ("package", "package/nested", "run"):
            with self.subTest(path=name), self.assertRaisesRegex(candidate.CandidateOnlyError, "fresh and outside"):
                self.fit(name=name)
        with self.assertRaisesRegex(candidate.CandidateOnlyError, "types"):
            self.fit(cpu_threads=True)
        self.assertFalse((self.root / "package/nested").exists())

    def test_original_metrics_failure_keeps_plan_but_no_success(self):
        evaluate = producer._evaluate
        def mismatch(*args, **kwargs):
            value = evaluate(*args, **kwargs)
            value["mean_cross_entropy"] += .01
            return value
        with patch.object(producer, "_evaluate", side_effect=mismatch), self.assertRaisesRegex(candidate.CandidateOnlyError, "did not reproduce exactly"):
            self.fit()
        self.assertTrue((self.root / "candidate/candidate-only-plan.json").is_file())
        self.assertFalse((self.root / "candidate/candidate-only-fit.json").exists())

    def test_in_memory_features_mutation_refuses_success(self):
        captured = {}
        banks, train = candidate.diagnostic._banks, candidate._train
        def capture(imported):
            captured.update(banks(imported))
            return captured
        def mutate(*args, **kwargs):
            value = train(*args, **kwargs)
            next(iter(captured.values())).layers[0, 0, 0] += 1
            return value
        with patch.object(candidate.diagnostic, "_banks", side_effect=capture), patch.object(candidate, "_train", side_effect=mutate), \
                self.assertRaisesRegex(candidate.CandidateOnlyError, "in-memory frozen"):
            self.fit()
        self.assertFalse((self.root / "candidate/candidate-only-fit.json").exists())

    def test_package_and_original_artifact_mutations_during_fit_refuse(self):
        original_train = candidate._train
        path = next((self.fixture.package / "features").glob("*.json"))
        def mutate_package(*args, **kwargs):
            result = original_train(*args, **kwargs)
            path.write_bytes(path.read_bytes() + b" ")
            return result
        with patch.object(candidate, "_train", side_effect=mutate_package), self.assertRaisesRegex(handoff.HandoffError, "member size/hash differs"):
            self.fit()
        self.assertFalse((self.root / "candidate/candidate-only-fit.json").exists())

    def test_original_midpoint_mutation_after_proof_refuses_success(self):
        original_train = candidate._train
        def mutate(*args, **kwargs):
            result = original_train(*args, **kwargs)
            self.midpoint.write_bytes(self.midpoint.read_bytes() + b"PUBLIC mutation after proof")
            return result
        with patch.object(candidate, "_train", side_effect=mutate), self.assertRaisesRegex(candidate.CandidateOnlyError, "code, original artifacts"):
            self.fit()
        self.assertFalse((self.root / "candidate/candidate-only-fit.json").exists())

    def test_artificial_source_mutation_after_fit_refuses_success(self):
        original_train = candidate._train
        def mutate(model, source, *args):
            result = original_train(model, source, *args)
            source.layers[0, 0, 0] += 1
            return result
        with patch.object(candidate, "_train", side_effect=mutate), self.assertRaisesRegex(candidate.CandidateOnlyError, "in-memory frozen"):
            self.fit()
        self.assertFalse((self.root / "candidate/candidate-only-fit.json").exists())

    def test_explicit_fixture_option_cannot_override_production_recipe(self):
        imported = self.fixture.admit()
        imported.binding["mechanics_only"] = False
        with patch.object(handoff, "import_public_features", return_value=imported), \
                self.assertRaisesRegex(candidate.CandidateOnlyError, "production requires"):
            self.fit()
        self.assertFalse((self.root / "candidate").exists())

    def test_code_recheck_rejects_success_and_restores_runtime(self):
        code = candidate._own_code()
        rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        with patch.object(candidate, "_own_code", side_effect=[code, {**code, "public_candidate_only.py": "0" * 64}]), \
                self.assertRaisesRegex(candidate.CandidateOnlyError, "code, original artifacts"):
            self.fit()
        self.assertFalse((self.root / "candidate/candidate-only-fit.json").exists())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())

    def test_runtime_thread_change_during_fit_refuses_success(self):
        original_train = candidate._train
        threads = torch.get_num_threads()
        def change_threads(*args, **kwargs):
            result = original_train(*args, **kwargs)
            torch.set_num_threads(2)
            return result
        with patch.object(candidate, "_train", side_effect=change_threads), \
                self.assertRaisesRegex(candidate.CandidateOnlyError, "runtime or predeclared plan"):
            self.fit()
        self.assertFalse((self.root / "candidate/candidate-only-fit.json").exists())
        self.assertEqual(threads, torch.get_num_threads())

    def test_sbatch_binds_public_midpoint_and_uses_existing_single_session(self):
        raw = candidate._CODE["magnolia_fit_public_candidate_only.sbatch"].read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        text = raw.decode()
        self.assertIn(candidate.ORIGINAL_MIDPOINT_SHA256, text)
        self.assertEqual(text.count('"$UDOCKER" run'), 1)
        self.assertIn("rayan-n0-base", text)
        self.assertIn("--mem=16G", text)
        self.assertIn("--cpus-per-task=4", text)
        self.assertEqual(text.count("python -m unittest tests.personality.test_public_candidate_only -q"), 1)
        self.assertLess(text.index("python -m unittest"), text.index('python "$1/scripts/eipm/gemma_n0/fit_public_candidate_only.py"'))
        self.assertIn('export TMPDIR="$9/compatibility-tmp"', text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("--allow-mechanics-only", text)


if __name__ == "__main__":
    unittest.main()
