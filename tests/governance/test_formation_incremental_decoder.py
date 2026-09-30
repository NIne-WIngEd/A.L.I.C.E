"""Numerical parity for an experimental specialist cache; no 12B receipt."""

import unittest

try:
    import torch
    from cognitive_kernel.formation_incremental_decoder import (
        CachedFormationDecoder, greedy_tokens_cached)
    from cognitive_kernel.formation_v1_specialist import (
        FormationSpecialist, SpecialistConfig)
    from scripts.mfm.run_v1_formation_specialist import greedy_tokens
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU PyTorch unavailable")
class CachedDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.specialist = FormationSpecialist(SpecialistConfig(
            12, 31, 16, 2, 4, 12, 0, 1, 2, 3)).eval()
        self.states = torch.randn(1, 7, 12)
        self.mask = torch.tensor([[1, 1, 1, 0, 1, 0, 0]], dtype=torch.long)

    def test_every_incremental_logit_matches_full_prefix(self):
        tokens = [1, 7, 0, 10, 9, 5]
        with torch.inference_mode():
            cached = CachedFormationDecoder(
                self.specialist, base_states=self.states, source_mask=self.mask)
            for length, token in enumerate(tokens, 1):
                observed = cached.step(token)
                expected = self.specialist.next_token_logits(
                    base_states=self.states, source_mask=self.mask,
                    input_ids=torch.tensor([tokens[:length]]))
                torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-5)

    def test_greedy_tokens_match_and_cache_refuses_training_mode(self):
        with torch.inference_mode():
            self.assertEqual(greedy_tokens_cached(
                self.specialist, base_states=self.states, source_mask=self.mask,
                max_new_tokens=8), greedy_tokens(
                self.specialist, base_states=self.states, source_mask=self.mask,
                max_new_tokens=8))
        self.specialist.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            CachedFormationDecoder(self.specialist, base_states=self.states,
                                   source_mask=self.mask)


if __name__ == "__main__":
    unittest.main()
