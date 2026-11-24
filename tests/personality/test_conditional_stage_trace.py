from dataclasses import replace
import importlib
import unittest
from unittest.mock import patch

import torch

from src.alice_personality.identity import IdentityModel, IdentityModelConfig, IdentityError
from src.alice_personality.identity.codec import fingerprint
from scripts.eipm.gemma_n0 import conditional_channel_probe as channel
from tests.personality.test_identity_teaching import _frame


class ConditionalStageTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.api = importlib.import_module('scripts.eipm.gemma_n0.conditional_stage_trace')
        torch.manual_seed(53)
        self.model = IdentityModel(IdentityModelConfig(4, 3, learned_width=8,
            attention_heads=2, graph_layers=1, readout_layers=2))
        self.model.set_phase('inference')
        self.frame = _frame(self.model.config)
        with torch.no_grad():
            self.rep, self.query = self.model.n1(self.frame)
            self.baseline = self.model(self.frame)

    def captured(self):
        return self.api.capture_stages(self.model, self.frame,
            lambda: channel.replay_with_representation(self.model, self.frame, self.rep, self.query))

    def test_passive_capture_and_cached_replay_preserve_exact_native_packet(self):
        actual, stages, cache = self.captured()
        self.assertEqual(fingerprint(actual), fingerprint(self.baseline))
        for name in ('conditional.signals', 'conditional.messages', 'memory.query', 'memory.source_tokens', 'memory.views',
                     'layer.0.context', 'layer.1.alternatives', 'layer.1.output_norm',
                     'scalar.linear', 'scalar.activation', 'scalar.output'):
            self.assertIn(name, stages)
        original = self.model.n2.readout.forward
        with self.api.replay_readout(self.model, cache):
            with patch.object(self.model.n2.readout.projection, 'forward',
                    side_effect=AssertionError('projection recomputed')):
                replay, observed, _ = self.captured()
        self.assertEqual(self.model.n2.readout.forward, original)
        self.assertEqual(fingerprint(actual), fingerprint(replay))
        self.assertTrue(all(v['changed_elements'] == 0 for v in self.api.compare_stages(stages, observed).values()))

    def test_real_conditional_patch_reaches_stages_without_mutating_state(self):
        _, before, cache = self.captured()
        state = fingerprint(self.model.state_dict())
        modified, _ = channel.patch_conditionals(self.rep, self.frame,
            {'concept_applicability': {('concept-0',): 1.},
             'concept_source_support': {('concept-0', 'source-0'): 0.}})
        with self.api.replay_readout(self.model, cache):
            _, after, _ = self.api.capture_stages(self.model, self.frame,
                lambda: channel.replay_with_representation(self.model, self.frame, modified, self.query))
        changes = self.api.compare_stages(before, after)
        self.assertGreater(changes['conditional.embedding']['changed_elements'], 0)
        self.assertGreater(changes['memory.views']['changed_elements'], 0)
        self.assertGreater(changes['layer.0.context']['changed_elements'], 0)
        self.assertEqual(changes['memory.query']['changed_elements'], 0)
        self.assertEqual(changes['memory.source_tokens']['changed_elements'], 0)
        self.assertEqual(changes['candidate.entry']['changed_elements'], 0)
        self.assertIn('candidate_centered_delta_rms', changes['scalar.output'])
        self.assertEqual(state, fingerprint(self.model.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))

    def test_cache_refuses_another_query_and_restores_method_after_failure(self):
        _, _, cache = self.captured()
        original = self.model.n2.readout.forward
        with self.assertRaisesRegex(IdentityError, 'query'):
            with self.api.replay_readout(self.model, cache):
                channel.replay_with_representation(self.model, self.frame, self.rep, self.query + 1)
        self.assertEqual(self.model.n2.readout.forward, original)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with self.api.replay_readout(self.model, cache):
                raise RuntimeError('injected')
        self.assertEqual(self.model.n2.readout.forward, original)

    def test_capture_removes_hooks_after_downstream_exception(self):
        before = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()]
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.api.capture_stages(self.model, self.frame,
                lambda: (_ for _ in ()).throw(RuntimeError('injected')))
        self.assertEqual(before, [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()])

    def test_cache_rejects_changed_feature_tensor_and_training_phase(self):
        _, _, cache = self.captured()
        self.frame.candidates.states.add_(1)
        with self.assertRaisesRegex(IdentityError, 'bank'):
            with self.api.replay_readout(self.model, cache):
                channel.replay_with_representation(self.model, self.frame, self.rep, self.query)
        self.model.set_phase('n2')
        with self.assertRaises(IdentityError):
            with self.api.replay_readout(self.model, cache): pass

    def test_effect_comparison_rejects_different_stage_geometry_or_masks(self):
        _, stages, _ = self.captured()
        altered = dict(stages)
        value, mask, centered = altered['candidate.entry']
        altered['candidate.entry'] = (value, ~mask, centered)
        with self.assertRaisesRegex(IdentityError, 'mask'):
            self.api.compare_stages(stages, altered)


if __name__ == '__main__': unittest.main()
