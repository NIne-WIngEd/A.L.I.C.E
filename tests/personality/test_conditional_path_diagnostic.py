"""Check interventions against the real native N1/N2 path, not a fitted proxy."""
import importlib
import unittest

import torch

from src.alice_personality.identity.codec import fingerprint
from src.alice_personality.identity.model import IdentityModel
from src.alice_personality.identity.contracts import IdentityModelConfig
from tests.personality.test_identity_model import _frame

try:
    api = importlib.import_module('scripts.eipm.gemma_n0.conditional_path_diagnostic')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.conditional_path_diagnostic':
        raise
    api = None


class ConditionalPathDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(api, 'The measured conditional-input diagnostic is missing')
        torch.manual_seed(319)
        self.config = IdentityModelConfig(provider_width=8, provider_state_count=3,
            learned_width=8, attention_heads=2, graph_layers=1, readout_layers=1)
        self.model = IdentityModel(self.config)
        self.model.set_phase('inference')
        self.frame = _frame(self.config, batch=1)

    def test_baseline_matches_native_verdict_and_leaves_model_unchanged(self):
        before = fingerprint(self.model.state_dict())
        with torch.no_grad():
            native = self.model(self.frame)
        result = api.measure_conditional_path(self.model, self.frame, concept_id='concept-0')
        self.assertEqual(result['baseline']['co_valid_probabilities'], native.co_valid_probabilities.tolist())
        self.assertEqual(result['baseline']['contraindications'], native.contraindications.tolist())
        self.assertEqual(result['baseline']['preferences'], native.preferences.tolist())
        self.assertEqual(fingerprint(self.model.state_dict()), before)
        self.assertEqual(result['model_state_sha256'], before)
        self.assertFalse(result['authentic_response'])
        self.assertFalse(result['independent_qualification'])

    def test_applicability_clamp_changes_only_named_signal_and_not_source_authority(self):
        seen = []
        def capture(module, args):
            seen.append((args[1].concept_applicability.clone(),
                args[1].concept_uncertainty.clone(), args[1].source_states.clone(),
                args[1].concept_source_mask.clone()))
        hook = self.model.n2.register_forward_pre_hook(capture)
        try:
            result = api.measure_conditional_path(self.model, self.frame, concept_id='concept-0')
        finally:
            hook.remove()
        self.assertEqual(result['operations'], ['baseline','applicability_zero','applicability_one','conditional_messages_removed'])
        self.assertEqual(float(seen[1][0][0,0]), 0.)
        self.assertEqual(float(seen[2][0][0,0]), 1.)
        self.assertTrue(torch.equal(seen[0][0][:,1:], seen[1][0][:,1:]))
        self.assertTrue(torch.equal(seen[0][0][:,1:], seen[2][0][:,1:]))
        for observed in seen[1:]:
            for original, intervened in zip(seen[0][1:], observed[1:]):
                self.assertTrue(torch.equal(original, intervened))

    def test_query_swap_holds_candidates_sources_and_conditional_signals_fixed(self):
        donor = _frame(self.config, batch=1)
        seen = []
        def capture(module, args):
            seen.append((args[0], args[1], args[2].clone()))
        hook = self.model.n2.register_forward_pre_hook(capture)
        try:
            result = api.measure_conditional_path(self.model, self.frame,
                concept_id='concept-0', donor_frame=donor)
        finally:
            hook.remove()
        base, swapped = seen[0], seen[-1]
        self.assertEqual(result['operations'][-1], 'query_channel_swap')
        self.assertIs(swapped[0], self.frame)
        self.assertTrue(torch.equal(base[1].concept_applicability, swapped[1].concept_applicability))
        self.assertTrue(torch.equal(base[1].source_states, swapped[1].source_states))
        self.assertFalse(torch.equal(base[1].query_tokens, swapped[1].query_tokens))
        self.assertFalse(torch.equal(base[2], swapped[2]))
        self.assertEqual(result['modes']['query_channel_swap']['authority_frame_sha256'],
                         result['baseline']['authority_frame_sha256'])
