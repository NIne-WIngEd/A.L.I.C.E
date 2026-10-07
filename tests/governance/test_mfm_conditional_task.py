"""Parameter-free public task binding over real fresh specialist weights."""
import importlib.util
import io
import unittest

import torch
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel.formation_incremental_decoder import CachedFormationDecoder, greedy_tokens_cached
from scripts.mfm.run_v1_formation_specialist import greedy_tokens


class ConditionalTaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.model = FormationDecisionSpecialist(SpecialistConfig(
            12, 31, 16, 2, 4, 12, 0, 2, 1, 3)).eval()
        self.states = torch.randn(1, 5, 12)
        self.mask = torch.tensor([[1, 1, 0, 1, 0]])

    def bind(self, task='conditional_meaning_to_text'):
        self.assertIsNotNone(importlib.util.find_spec(
            'cognitive_kernel.formation_conditional_task'), 'conditional task binding is missing')
        from cognitive_kernel.formation_conditional_task import bind_conditional_task
        return bind_conditional_task(self.model, task=task)

    def test_binding_preserves_weights_positions_and_exact_prefix_logits(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        parameters = tuple(self.model.parameters())
        view = self.bind()
        self.assertNotIsInstance(view, torch.nn.Module)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in vars(view).values()))
        self.assertEqual(tuple(view.parameters()), parameters)
        self.assertEqual(set(view.state_dict()), set(before))
        for name in ('config', 'token_embedding', 'position_embedding', 'source_projection', 'decoder', 'output_bias'):
            self.assertIs(getattr(view, name), getattr(self.model, name))
        for length in range(1, 5):
            payload = dict(base_states=self.states, source_mask=self.mask,
                           input_ids=torch.tensor([[2, 7, 0, 9][:length]]))
            with torch.inference_mode():
                torch.testing.assert_close(view.forward(**payload), self.model(**payload), atol=0, rtol=0)
                torch.testing.assert_close(view.next_token_logits(**payload), self.model.next_token_logits(**payload), atol=0, rtol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_public_binding_rejects_other_task_names(self):
        for task in ('source_reconstruction', 'formation', 'unknown', None):
            with self.subTest(task=task), self.assertRaises(ValueError):
                self.bind(task)

    def test_view_training_loss_and_gradients_reach_original_weights(self):
        view = self.bind()
        payload = dict(base_states=self.states, source_mask=self.mask,
                       input_ids=torch.tensor([[2, 7, 9]]), labels=torch.tensor([[7, 9, 1]]))
        _, expected = self.model(**payload)
        expected.backward()
        gradients = {name: value.grad.clone() for name, value in self.model.named_parameters()}
        self.model.zero_grad(set_to_none=True)
        _, observed = view.forward(**payload)
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        observed.backward()
        for name, value in self.model.named_parameters():
            torch.testing.assert_close(value.grad, gradients[name], atol=0, rtol=0)

    def test_cached_and_reference_generation_use_same_specialist(self):
        view = self.bind()
        tokens = [2, 7, 0, 9]
        with torch.inference_mode():
            cache = CachedFormationDecoder(view, base_states=self.states, source_mask=self.mask)
            for length, token in enumerate(tokens, 1):
                expected = self.model.next_token_logits(base_states=self.states, source_mask=self.mask,
                    input_ids=torch.tensor([tokens[:length]]))
                torch.testing.assert_close(cache.step(token), expected, atol=2e-5, rtol=2e-5)
            expected = greedy_tokens(self.model, base_states=self.states, source_mask=self.mask, max_new_tokens=8)
            self.assertEqual(greedy_tokens(view, base_states=self.states, source_mask=self.mask, max_new_tokens=8), expected)
            self.assertEqual(greedy_tokens_cached(view, base_states=self.states, source_mask=self.mask, max_new_tokens=8), expected)
        self.model.train()
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, 'eval mode'):
            CachedFormationDecoder(view, base_states=self.states, source_mask=self.mask)

    def test_underlying_checkpoint_reloads_without_view_prefix(self):
        view = self.bind()
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = FormationDecisionSpecialist(self.model.config)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertEqual(set(restored.state_dict()), set(view.state_dict()))
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
