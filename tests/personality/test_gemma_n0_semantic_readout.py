"""Tiny CPU tensor mechanics, not natural semantics/personality competence."""

from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from src.alice_personality.gemma_n0.semantic_readout import (
    LayerTokenBank, ReadoutError, SemanticReadout, fixed_semantic_scores,
)


def bank(seed=1, *, source=False, count=3, width=8):
    generator = torch.Generator().manual_seed(seed)
    layers = torch.randn(count, 7, width, generator=generator).bfloat16()
    roles = ({"source_mask": torch.tensor([1, 1, 0, 0, 0, 0, 0]).bool(),
              "query_mask": torch.tensor([0, 0, 1, 1, 1, 1, 1]).bool(),
              "head_mask": torch.tensor([0, 0, 1, 0, 0, 0, 0]).bool(),
              "tail_mask": torch.tensor([0, 0, 0, 0, 1, 0, 0]).bool()} if source else {})
    return LayerTokenBank(layers, torch.arange(7), torch.ones(7, dtype=torch.bool), **roles)


class SemanticReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = SemanticReadout(state_count=3, hidden_size=8, width=6)
        self.source = bank(source=True)
        self.candidates = [bank(2), bank(3), bank(4)]

    def test_dynamic_description_permutation_scores_are_equivariant(self):
        original = self.model(self.source, self.candidates)
        reverse = self.model(self.source, list(reversed(self.candidates))).flip(0)
        self.assertTrue(torch.equal(original, reverse))
        self.assertEqual(self.model(self.source, self.candidates[:2]).shape, (2,))
        self.assertEqual(self.model(self.source, self.candidates[:1]).shape, (1,))

    def test_actual_fp32_optimizer_gradients_leave_bf16_features_unchanged(self):
        before = [value.layers.clone() for value in [self.source, *self.candidates]]
        parameters = {key: value.clone() for key, value in self.model.state_dict().items()}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        loss = torch.nn.functional.cross_entropy(self.model(self.source, self.candidates)[None], torch.tensor([1]))
        loss.backward()
        self.assertTrue(all(parameter.dtype == torch.float32 and parameter.grad is not None
                            and bool(torch.isfinite(parameter.grad).all()) for parameter in self.model.parameters()))
        self.assertGreater(float(self.model.layer_logits.grad.abs().sum()), 0)
        optimizer.step()
        self.assertTrue(any(not torch.equal(value, parameters[key]) for key, value in self.model.state_dict().items()))
        for original, frozen in zip(before, [self.source, *self.candidates]):
            self.assertTrue(torch.equal(original, frozen.layers))
            self.assertIsNone(frozen.layers.grad)
            self.assertFalse(frozen.layers.requires_grad)

    def test_masked_padding_does_not_supply_evidence(self):
        def padding(value):
            roles = {name: (None if getattr(value, name) is None else
                            torch.cat([getattr(value, name), torch.zeros(2, dtype=torch.bool)]))
                     for name in ("source_mask", "query_mask", "head_mask", "tail_mask")}
            return LayerTokenBank(torch.cat([value.layers, torch.full((3, 2, 8), 99, dtype=torch.bfloat16)], dim=1),
                                  torch.cat([value.input_ids, torch.zeros(2, dtype=torch.long)]),
                                  torch.cat([value.attention_mask, torch.zeros(2, dtype=torch.bool)]), **roles)
        original = self.model(self.source, self.candidates)
        changed = self.model(padding(self.source), [padding(value) for value in self.candidates])
        torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-6)

    def test_every_actual_layer_and_complete_ids_required(self):
        invalid = [replace(self.source, layers=self.source.layers[:2]),
                   replace(self.source, input_ids=self.source.input_ids[:-1]),
                   replace(self.source, attention_mask=torch.ones(7, dtype=torch.long)),
                   replace(self.source, query_mask=torch.zeros(7, dtype=torch.bool)),
                   replace(self.source, layers=self.source.layers.float()),
                   replace(self.source, layers=self.source.layers.clone().requires_grad_())]
        for value in invalid:
            with self.subTest(value=value.layers.shape), self.assertRaises(ReadoutError):
                self.model(value, self.candidates)

    def test_nonfinite_even_masked_states_and_overlap_roles_rejected(self):
        bad = self.source.layers.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(ReadoutError):
            self.model(replace(self.source, layers=bad), self.candidates)
        with self.assertRaisesRegex(ReadoutError, "overlap"):
            self.model(replace(self.source, tail_mask=self.source.head_mask), self.candidates)

    def test_labels_class_ids_and_candidate_order_embeddings_not_api_inputs(self):
        with self.assertRaises(TypeError):
            self.model(self.source, self.candidates, relation_ids=["P1", "P2", "P3"])
        self.assertFalse(any("class" in name or "identity" in name for name, _ in self.model.named_parameters()))
        with self.assertRaises(ReadoutError):
            self.model(self.source, [])

    def test_fp32_readout_contract_and_fixed_control_permutation(self):
        control = fixed_semantic_scores(self.source, self.candidates)
        self.assertTrue(torch.equal(control, fixed_semantic_scores(self.source, list(reversed(self.candidates))).flip(0)))
        self.assertFalse(control.requires_grad)
        self.model.bfloat16()
        with self.assertRaisesRegex(ReadoutError, "FP32"):
            self.model(self.source, self.candidates)


if __name__ == "__main__":
    unittest.main()
