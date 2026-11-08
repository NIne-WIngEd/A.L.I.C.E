"""Public native-tensor diagnostic mechanics, never personality qualification."""
from dataclasses import fields, replace
import importlib.util
import unittest
from unittest.mock import patch

import torch

from src.alice_personality.identity import IdentityModel, IdentityModelConfig, IdentityError
from src.alice_personality.identity.codec import fingerprint
from tests.personality.test_identity_teaching import _frame


NAME = 'scripts.eipm.gemma_n0.conditional_channel_probe'


class ConditionalChannelProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(NAME), 'conditional channel probe is missing')
        self.api = importlib.import_module(NAME)
        torch.manual_seed(53)
        self.model = IdentityModel(IdentityModelConfig(4, 3, learned_width=8, attention_heads=2,
                                                     graph_layers=1, readout_layers=1))
        self.model.set_phase('inference')
        self.frame = _frame(self.model.config)
        with torch.no_grad():
            self.representation, self.query = self.model.n1(self.frame)
            self.baseline = self.model(self.frame)

    def values(self):
        return {'concept_applicability': {('concept-0',): 1.},
                'concept_uncertainty': {('concept-0',): 0.},
                'concept_source_support': {('concept-0', 'source-0'): 1.},
                'concept_source_disconfirmation': {('concept-0', 'source-0'): 0.}}

    def test_same_value_sham_reproduces_exact_packet_without_n1_recomputation(self):
        values = self.values()
        for name, facts in values.items():
            for pointers in facts:
                index = (0, 0, 0) if len(pointers) == 2 else (0, 0)
                facts[pointers] = float(getattr(self.representation, name)[index])
        sham, metadata = self.api.patch_conditionals(self.representation, self.frame, values)
        self.assertEqual(metadata['applied_facts'], 4)
        self.assertEqual(metadata['changed_facts'], 0)
        self.assertEqual(fingerprint(sham), fingerprint(self.representation))
        original = self.model.n1.forward
        def forbidden(_frame):
            raise AssertionError('N1 was recomputed')
        with patch.object(self.model.n1, 'forward', forbidden):
            result = self.api.replay_with_representation(self.model, self.frame, sham, self.query)
            self.assertIs(self.model.n1.forward, forbidden)
        self.assertEqual(self.model.n1.forward, original)
        self.assertEqual(fingerprint(result), fingerprint(self.baseline))

    def test_binary_oracle_changes_real_n2_path_with_finite_explicit_audit_logits(self):
        state = fingerprint(self.model.state_dict())
        patched, metadata = self.api.patch_conditionals(self.representation, self.frame, self.values())
        self.assertEqual(metadata['binary_endpoint_facts'], 4)
        self.assertTrue(metadata['target_bearing_diagnostic'])
        self.assertFalse(metadata['independent_qualification'])
        self.assertEqual(float(patched.concept_applicability[0, 0]), 1.)
        self.assertEqual(float(patched.concept_uncertainty[0, 0]), 0.)
        self.assertEqual(float(patched.concept_source_support[0, 0, 0]), 1.)
        self.assertEqual(float(patched.concept_source_disconfirmation[0, 0, 0]), 0.)
        for name in self.values():
            self.assertTrue(bool(torch.isfinite(getattr(patched, name + '_logits')).all()))
        observed = []
        handle = self.model.n2.conditional_embedding.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0].detach().clone()))
        try:
            result = self.api.replay_with_representation(self.model, self.frame, patched, self.query)
        finally:
            handle.remove()
        result.validate()
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0, 0, :2].tolist(), [1., 0.])
        self.assertFalse(torch.equal(result.latent_candidates, self.baseline.latent_candidates))
        self.assertEqual(state, fingerprint(self.model.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))

    def test_sparse_targets_preserve_every_unselected_channel_coordinate_and_authority(self):
        before = fingerprint(self.representation)
        patched, metadata = self.api.patch_conditionals(self.representation, self.frame,
            {'concept_applicability': {('concept-0',): .25}})
        self.assertEqual(metadata['applied_facts'], 1)
        self.assertEqual(float(patched.concept_applicability[0, 0]), .25)
        torch.testing.assert_close(patched.concept_applicability_logits[0, 0], torch.tensor(-1.0986123))
        self.assertTrue(torch.equal(patched.concept_applicability[:, 1:], self.representation.concept_applicability[:, 1:]))
        self.assertTrue(torch.equal(patched.concept_applicability_logits[:, 1:], self.representation.concept_applicability_logits[:, 1:]))
        for field in fields(self.representation):
            if field.name not in ('concept_applicability', 'concept_applicability_logits'):
                self.assertIs(getattr(patched, field.name), getattr(self.representation, field.name))
        self.assertEqual(before, fingerprint(self.representation))

    def test_empty_unknown_coverage_is_exact_identity(self):
        patched, metadata = self.api.patch_conditionals(self.representation, self.frame, {})
        self.assertEqual(metadata['applied_facts'], 0)
        self.assertEqual(fingerprint(patched), fingerprint(self.representation))

    def test_bad_pointers_targets_or_outside_fields_refused(self):
        bad = [
            {'concept_applicability': {('missing',): 1.}},
            {'concept_applicability': {('concept-0', 'source-0'): 1.}},
            {'concept_source_support': {('concept-0', 'missing'): 1.}},
            {'concept_applicability': {('concept-0',): float('nan')}},
            {'concept_applicability': {('concept-0',): float('inf')}},
            {'concept_applicability': {('concept-0',): 1.01}},
            {'concept_applicability': {('concept-0',): True}},
            {'concept_applicability': {('concept-0',): None}},
            {'preferences': {('candidate-0',): 1.}},
        ]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(IdentityError):
                self.api.patch_conditionals(self.representation, self.frame, values)

    def test_forbidden_authority_is_not_granted_even_for_zero_target(self):
        self.assertFalse(bool(self.representation.concept_source_mask[0, 0, 2]))
        with self.assertRaises(IdentityError):
            self.api.patch_conditionals(self.representation, self.frame,
                {'concept_source_support': {('concept-0', 'source-2'): 0.}})
        mask = self.representation.concept_mask.clone()
        mask[0, 0] = False
        with self.assertRaises(IdentityError):
            self.api.patch_conditionals(replace(self.representation, concept_mask=mask), self.frame,
                {'concept_uncertainty': {('concept-0',): 1.}})

    def test_replay_refuses_training_or_trainable_state(self):
        self.model.set_phase('n2')
        with self.assertRaises(IdentityError):
            self.api.replay_with_representation(self.model, self.frame, self.representation, self.query)
        self.model.set_phase('inference')
        next(self.model.n2.parameters()).requires_grad_(True)
        with self.assertRaises(IdentityError):
            self.api.replay_with_representation(self.model, self.frame, self.representation, self.query)

    def test_downstream_failure_restores_n1_method_and_preserves_state(self):
        before = fingerprint(self.model.state_dict())
        self.assertNotIn('forward', self.model.n1.__dict__)
        def fail(_module, _args, _output):
            raise RuntimeError('public downstream failure')
        handle = self.model.n2.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'public downstream failure'):
                self.api.replay_with_representation(self.model, self.frame, self.representation, self.query)
        finally:
            handle.remove()
        self.assertNotIn('forward', self.model.n1.__dict__)
        self.assertEqual(before, fingerprint(self.model.state_dict()))
        self.assertEqual(fingerprint(self.model(self.frame)), fingerprint(self.baseline))


if __name__ == '__main__':
    unittest.main()
