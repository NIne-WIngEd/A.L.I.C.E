"""N3 must have the information needed by its separately reviewed outcome task."""
import unittest
import torch
import torch.nn.functional as F

from src.alice_personality.identity import IdentityModel, IdentityModelConfig
from tests.personality.test_identity_teaching import _frame


class ContextualCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(407)
        torch.set_num_threads(1)
        self.config = IdentityModelConfig(8, 3, learned_width=16, attention_heads=4)
        self.model = IdentityModel(self.config)

    def test_equal_confidence_signals_can_have_distinct_contextual_fidelity_outcomes(self):
        scalar, margins = torch.zeros(1, 2, 6), torch.zeros(1, 2)
        context = torch.zeros(1, 2, 32)
        context[0, 0, 0], context[0, 1, 0] = -1, 1
        self.model.n3.requires_grad_(True)
        optimizer = torch.optim.AdamW(self.model.n3.parameters(), lr=0.025, weight_decay=0)
        target = torch.tensor([[0.0, 1.0]])
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            risk = self.model.n3.risks(scalar, margins, context=context)[..., 2]
            F.binary_cross_entropy_with_logits(risk, target).backward()
            optimizer.step()
        result = self.model.n3.risks(scalar, margins, context=context)[..., 2].sigmoid().detach()
        self.assertLess(float(result[0, 0]), .2)
        self.assertGreater(float(result[0, 1]), .8)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.n1.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.n2.parameters()))

    def test_untaught_n2_scalar_is_not_owner_fidelity_truth(self):
        frame = _frame(self.config)
        before = self.model(frame).owner_fidelity_uncertainty.detach()
        with torch.no_grad():
            self.model.n2.scalar[-1].bias[-1].add_(100)
        after = self.model(frame).owner_fidelity_uncertainty.detach()
        torch.testing.assert_close(before, after)
