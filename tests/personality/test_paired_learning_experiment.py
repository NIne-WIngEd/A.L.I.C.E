"""Actual public tensor optimization; no private identity qualification."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.alice_personality.identity import IdentityModel, IdentityModelConfig
from src.alice_personality.identity.codec import fingerprint
from src.alice_personality.teaching import IdentityObjectives, TeachingError, TrainingRecipe
from tests.personality.test_identity_teaching import _batch, _frame, _target

try:
    api = importlib.import_module("scripts.eipm.gemma_n0.paired_learning_experiment")
except ModuleNotFoundError as error:
    if error.name != "scripts.eipm.gemma_n0.paired_learning_experiment":
        raise
    api = None


class PairedLearningExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.assertIsNotNone(api, "The bounded grouped-learning experiment is missing")
        torch.manual_seed(41)
        self.config = IdentityModelConfig(4, 3, learned_width=8, attention_heads=2,
                                          graph_layers=1, readout_layers=1)
        self.model = IdentityModel(self.config)
        self.objectives = IdentityObjectives(8)
        self.recipe = TrainingRecipe("n2", .003, 1., 41)
        frame = _frame(self.config)
        self.batches = {}
        for index in range(4):
            query = replace(frame.query, states=frame.query.states + index * .31)
            current = replace(frame, query=query)
            batch = _batch(self.model, current, name=f"public-{index}", full=True)
            targets = dict(batch.targets)
            targets["co_valid_probabilities"] = _target(
                [[1., 0., 0.]] if index % 2 else [[0., 1., 1.]])
            batch = replace(batch, targets=targets)
            self.batches[batch.fingerprint()] = batch
        self.digests = list(self.batches)
        self.groups = [self.digests[:2], self.digests[2:]]
        self.plan = api.make_plan(
            self.model, self.objectives, self.recipe,
            batch_sha256=self.digests, matched_groups=self.groups,
            control_permutation=[1, 0], controlled_slot=1,
            bindings={"source_class": "public_mechanical_fixture"})
        self.checked = []
        self.guards = 0

    def validate(self, batch):
        batch.validate(self.config)
        digest = batch.fingerprint()
        self.assertIn(digest, self.batches)
        self.checked.append(digest)
        return digest

    def guard(self):
        self.guards += 1

    def run_trial(self, **kwargs):
        return api.run_experiment(
            self.model, self.objectives, self.recipe, plan=self.plan,
            expected_plan_sha256=api.plan_sha256(self.plan),
            batch_loader=self.batches.__getitem__, validate_batch=self.validate,
            custody_guard=self.guard, **kwargs)

    def test_complete_fair_arms_preserve_original_frozen_stages_and_all_targets(self):
        before = fingerprint(self.model.state_dict())
        auxiliary = fingerprint(self.objectives.state_dict())
        result = self.run_trial()
        self.assertEqual(fingerprint(self.model.state_dict()), before)
        self.assertEqual(fingerprint(self.objectives.state_dict()), auxiliary)
        self.assertFalse(result["native_checkpoint"])
        self.assertFalse(result["resumable"])
        self.assertEqual(result["qualification"], "UNQUALIFIED")
        for name, arm in result["arms"].items():
            self.assertEqual(arm["updates"], 2)
            self.assertEqual(arm["endpoint_forwards"], 4)
            self.assertEqual(arm["batch_occurrences"], dict.fromkeys(self.digests, 1))
            self.assertEqual(len(arm["events"]), 2)
            for key, value in self.model.state_dict().items():
                if key.startswith(("n1.", "n3.")):
                    self.assertTrue(torch.equal(value, arm["model_state"][key]), key)
            self.assertTrue(any(not torch.equal(value, arm["model_state"][key])
                                for key, value in self.model.state_dict().items()
                                if key.startswith("n2.")))
            for event in arm["events"]:
                self.assertEqual(event["arm"], name)
                for endpoint in event["endpoints"]:
                    self.assertEqual(set(endpoint["objectives"]),
                                     set(self.batches[endpoint["batch_sha256"]].targets))
        for batch in self.batches.values():
            for bank in (batch.frame.query, batch.frame.sources, batch.frame.concepts,
                         batch.frame.candidates, batch.frame.relations, *batch.frame.labels.values()):
                self.assertIsNone(bank.states.grad)
        self.assertGreater(self.guards, 0)

    def test_group_step_matches_independent_native_mean_objective_adamw(self):
        # Joint and streamed backward add in different orders. In FP32,
        # ~1e-8 rounding near mathematically zero bias gradients can become
        # ~1e-4 AdamW differences. Use FP64 for this strict trajectory check;
        # the separate FP32 test checks the actual averaged gradients.
        self.model.double()
        self.objectives.double()
        self.plan = api.make_plan(
            self.model, self.objectives, self.recipe,
            batch_sha256=self.digests, matched_groups=self.groups,
            control_permutation=[1, 0], bindings=self.plan["bindings"])
        expected = deepcopy(self.model)
        expected.set_phase("n2")
        objectives = deepcopy(self.objectives)
        objectives.requires_grad_(False)
        params = list(expected.n2.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.recipe.learning_rate, betas=self.recipe.betas,
            eps=self.recipe.eps, weight_decay=self.recipe.weight_decay)
        expected_losses = []
        for group in self.groups:
            optimizer.zero_grad(set_to_none=True)
            losses = [objectives(expected(self.batches[digest].frame),
                                 self.batches[digest])[0] for digest in group]
            mean = torch.stack(losses).mean()
            expected_losses.append(float(mean.detach()))
            mean.backward()
            torch.nn.utils.clip_grad_norm_(params, self.recipe.max_grad_norm)
            optimizer.step()
        result = self.run_trial()
        matched = result["arms"]["matched"]
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(matched["model_state"][key], value, rtol=1e-5, atol=1e-7)
        for event, expected_loss in zip(matched["events"], expected_losses):
            self.assertAlmostEqual(event["mean_endpoint_loss"], expected_loss, places=6)

    def test_fp32_gradient_matches_independent_mean_before_first_update(self):
        expected = deepcopy(self.model)
        expected.set_phase("n2")
        objectives = deepcopy(self.objectives)
        objectives.requires_grad_(False)
        losses = [objectives(expected(self.batches[digest].frame),
                             self.batches[digest])[0] for digest in self.groups[0]]
        torch.stack(losses).mean().backward()
        params = list(expected.n2.parameters())
        torch.nn.utils.clip_grad_norm_(params, self.recipe.max_grad_norm)
        expected_gradients = [p.grad.clone() if p.grad is not None else None for p in params]

        class Captured(Exception):
            pass

        def compare(optimizer, *args, **kwargs):
            for reference, actual in zip(expected_gradients, optimizer.param_groups[0]["params"]):
                if reference is None:
                    self.assertIsNone(actual.grad)
                else:
                    torch.testing.assert_close(actual.grad, reference, rtol=1e-5, atol=1e-7)
            raise Captured

        with patch.object(torch.optim.AdamW, "step", compare), self.assertRaises(Captured):
            self.run_trial()

    def test_incomplete_duplicate_or_nonindependent_plans_refuse_before_loading(self):
        for change in (
            {"matched_groups": [self.groups[0], [self.digests[2]]]},
            {"matched_groups": [self.groups[0], [self.digests[2], self.digests[2]]]},
            {"control_permutation": [0, 1]},
            {"batch_sha256": self.digests[:-1]},
        ):
            with self.subTest(change=change):
                plan = dict(self.plan, **change)
                with self.assertRaises(TeachingError):
                    api.run_experiment(
                        self.model, self.objectives, self.recipe, plan=plan,
                        expected_plan_sha256=api.plan_sha256(plan),
                        batch_loader=lambda _: self.fail("invalid plan loaded a batch"),
                        validate_batch=self.validate, custody_guard=self.guard)

    def test_custody_failure_before_step_never_updates_or_emits_an_event(self):
        events = []
        before = fingerprint(self.model.state_dict())
        clipped = False
        native_clip = torch.nn.utils.clip_grad_norm_

        def clip(*args, **kwargs):
            nonlocal clipped
            result = native_clip(*args, **kwargs)
            clipped = True
            return result

        def guard():
            if clipped:
                raise TeachingError("deliberate public custody failure")

        with patch.object(torch.nn.utils, "clip_grad_norm_", clip), \
                patch.object(torch.optim.AdamW, "step") as step, \
                self.assertRaisesRegex(TeachingError, "custody"):
            api.run_experiment(
                self.model, self.objectives, self.recipe, plan=self.plan,
                expected_plan_sha256=api.plan_sha256(self.plan),
                batch_loader=self.batches.__getitem__, validate_batch=self.validate,
                custody_guard=guard, on_event=events.append)
        self.assertTrue(clipped)
        self.assertGreater(len(self.checked), len(self.digests))
        step.assert_not_called()
        self.assertEqual(events, [])
        self.assertEqual(fingerprint(self.model.state_dict()), before)

    def test_changed_external_pin_and_wrong_stage_refuse(self):
        with self.assertRaises(TeachingError):
            api.run_experiment(
                self.model, self.objectives, self.recipe, plan=self.plan,
                expected_plan_sha256="0" * 64, batch_loader=self.batches.__getitem__,
                validate_batch=self.validate, custody_guard=self.guard)
        with self.assertRaises(TeachingError):
            api.make_plan(
                self.model, self.objectives, replace(self.recipe, adapt_n1=True),
                batch_sha256=self.digests, matched_groups=self.groups,
                control_permutation=[1, 0], bindings={})

    def test_control_refuses_moving_a_case_between_different_mean_weights(self):
        inventory = self.digests + ["a" * 64, "b" * 64]
        with self.assertRaises(TeachingError):
            api.make_plan(
                self.model, self.objectives, self.recipe,
                batch_sha256=inventory,
                matched_groups=[inventory[:2], inventory[2:]],
                control_permutation=[1, 0], bindings={})

    def test_control_accepts_derangement_inside_each_group_size_stratum(self):
        inventory = [f"{index:064x}" for index in range(10)]
        groups = [inventory[:2], inventory[2:4], inventory[4:7], inventory[7:]]
        plan = api.make_plan(
            self.model, self.objectives, self.recipe,
            batch_sha256=inventory, matched_groups=groups,
            control_permutation=[1, 0, 3, 2], bindings={})
        self.assertEqual(plan["control_permutation"], [1, 0, 3, 2])

    def test_save_is_explicitly_experimental_and_contains_full_state_and_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "experiment"
            result = self.run_trial(destination=destination)
            saved = torch.load(destination / "experimental.pt", weights_only=True)
            self.assertEqual(saved["plan"], self.plan)
            self.assertEqual(saved["plan_sha256"], api.plan_sha256(self.plan))
            self.assertEqual(saved["model_config"], asdict(self.config))
            self.assertEqual(saved["recipe"], asdict(self.recipe))
            self.assertFalse(saved["native_checkpoint"])
            self.assertFalse(saved["resumable"])
            self.assertNotIn("resume.pt", {path.name for path in destination.iterdir()})
            for arm in saved["arms"].values():
                self.assertTrue(arm["optimizer_state"]["state"])
                self.assertEqual(set(arm["model_state"]), set(self.model.state_dict()))
                self.assertEqual(set(arm["objective_state"]), set(self.objectives.state_dict()))
            self.assertEqual(saved["schema"], result["schema"])


if __name__ == "__main__":
    unittest.main()
