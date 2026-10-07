import unittest
import torch

from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist


class TaskSpecialistTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = SpecialistConfig(12, 29, 16, 2, 4, 128, 0, 1, 2, 8)
        torch.manual_seed(71)
        self.parent = FormationDecisionSpecialist(self.config).eval()
        self.payload = dict(base_states=torch.randn(1, 7, 12),
                            source_mask=torch.tensor([[1, 1, 1, 0, 1, 1, 1]]),
                            input_ids=torch.tensor([[1, 7, 11, 14, 5]]))

    def candidate(self):
        from cognitive_kernel.formation_sequence_specialist import FormationSequenceSpecialist
        torch.manual_seed(71)
        return FormationSequenceSpecialist(self.config)

    def test_migration_preserves_all_parent_parameters_and_formation_logits(self):
        model = self.candidate().eval()
        for key, value in self.parent.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        model.load_formation_parent(self.parent.state_dict())
        with torch.inference_mode():
            self.assertTrue(torch.equal(self.parent(**self.payload), model(**self.payload)))
        self.assertTrue(torch.equal(model.task_embedding.weight[0], torch.zeros(16)))

    def test_explicit_tasks_share_fresh_head_and_receive_gradients(self):
        model = self.candidate()
        losses = []
        for task in ('formation', 'source_reconstruction'):
            _, loss = model(**self.payload, labels=torch.tensor([[7, 11, 14, 5, 2]]), task=task)
            loss.backward()
            losses.append(float(loss.detach()))
        self.assertNotEqual(losses[0], losses[1])
        self.assertTrue(torch.all(model.task_embedding.weight.grad.norm(dim=1) > 0))
        self.assertGreater(float(model.token_embedding.weight.grad.norm()), 0)
        with self.assertRaises(ValueError):
            model(**self.payload, task='unregistered')

    def test_migration_rejects_missing_or_extra_parent_weights(self):
        model = self.candidate()
        missing = dict(self.parent.state_dict())
        missing.pop('output_bias')
        with self.assertRaises(ValueError):
            model.load_formation_parent(missing)
        with self.assertRaises(ValueError):
            model.load_formation_parent(model.state_dict())

    def test_cache_matches_long_prefix_after_both_task_embeddings_change(self):
        from cognitive_kernel.formation_incremental_decoder import CachedFormationDecoder
        model = self.candidate().eval()
        with torch.no_grad():
            model.task_embedding.weight.add_(torch.randn_like(model.task_embedding.weight) * .05)
        prefix = [1] + [3 + (i % 25) for i in range(69)]
        with torch.inference_mode():
            for task in ('formation', 'source_reconstruction'):
                cache = CachedFormationDecoder(model.bind_task(task),
                    base_states=self.payload['base_states'], source_mask=self.payload['source_mask'])
                for i, token in enumerate(prefix):
                    cached = cache.step(token)
                    if i in (0, 15, 31, 63, 69):
                        actual = model.next_token_logits(
                            base_states=self.payload['base_states'], source_mask=self.payload['source_mask'],
                            input_ids=torch.tensor([prefix[:i+1]]), task=task)
                        self.assertTrue(torch.allclose(cached, actual, atol=1e-6, rtol=1e-5))
                        self.assertTrue(torch.equal(cached.argmax(-1), actual.argmax(-1)))

    def test_task_bound_reference_runner_keeps_the_selected_task(self):
        model = self.candidate().eval()
        with torch.no_grad():
            model.task_embedding.weight[0].add_(.15)
        with torch.inference_mode():
            for task in ('formation', 'source_reconstruction'):
                view = model.bind_task(task)
                actual = model.next_token_logits(**self.payload, task=task)
                self.assertTrue(torch.equal(actual, view.next_token_logits(**self.payload)))
                self.assertTrue(torch.equal(model(**self.payload, task=task), view.forward(**self.payload)))

    def test_default_internal_decode_keeps_trained_formation_task(self):
        model = self.candidate().eval()
        with torch.no_grad():
            model.task_embedding.weight[0].add_(torch.randn(16) * .1)
        with torch.inference_mode():
            decoded = model._decode(**self.payload)
            projected = torch.nn.functional.linear(decoded, model.token_embedding.weight, model.output_bias)
            self.assertTrue(torch.equal(projected, model(**self.payload)))


if __name__ == '__main__':
    unittest.main()
