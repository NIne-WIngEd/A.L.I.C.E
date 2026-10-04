"""Analysis must measure the real objective without changing weights or targets."""
import unittest
import torch
from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
from scripts.mfm import analyze_v16_cached_fit as analysis


class FitAnalysisTests(unittest.TestCase):
    def test_full_chunked_analysis_matches_training_loss_and_preserves_weights(self):
        torch.manual_seed(14)
        model = FormationSpecialist(SpecialistConfig(12, 31, 16, 2, 4, 8, 0, 2, 1, 2)).eval()
        payload = dict(base_states=torch.randn(1, 5, 12).bfloat16(),
            source_mask=torch.ones(1, 5, dtype=torch.long),
            input_ids=torch.tensor([[2, 3, 4]]), labels=torch.tensor([[3, 4, 1]]))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            expected = model(**payload)[1].item()
        result = analysis.measure(model, payload, ['key', 'value', 'eos'])
        self.assertAlmostEqual(result['mean_loss'], expected, places=5)
        self.assertEqual(result['tokens'], 3)
        self.assertEqual(set(result['groups']), {'key', 'value', 'eos'})
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
