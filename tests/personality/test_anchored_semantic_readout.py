"""PUBLIC mechanics: semantic initialization, learned correction and frozen inputs."""
import importlib
import unittest
import torch
from src.alice_personality.gemma_n0.semantic_readout import LayerTokenBank, fixed_semantic_scores


class AnchoredSemanticReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(74)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.source = self.bank(source=True)
        self.descriptions = [self.bank() for _ in range(3)]

    def model(self):
        try:
            module = importlib.import_module("src.alice_personality.gemma_n0.anchored_semantic_readout")
        except ModuleNotFoundError:
            self.fail("the semantic-anchored readout candidate is missing")
        return module.AnchoredSemanticReadout(state_count=3, hidden_size=8, width=4)

    def bank(self, source=False):
        layers = torch.randn(3, 4, 8).to(torch.bfloat16)
        ids = torch.tensor([2, 12, 16, 1])
        mask = torch.ones(4, dtype=torch.bool)
        return LayerTokenBank(layers, ids, mask,
            mask.clone() if source else None, mask.clone() if source else None,
            torch.tensor([False,True,False,False]) if source else None,
            torch.tensor([False,False,True,False]) if source else None)

    def test_initial_scores_preserve_fixed_semantics_for_actual_sources(self):
        model = self.model()
        for source in (self.source, self.bank(source=True)):
            self.assertTrue(torch.equal(model(source,self.descriptions), fixed_semantic_scores(source,self.descriptions)))

    def test_semantic_correction_learns_without_upstream_gradient_or_mutation(self):
        model = self.model()
        banks = [self.source,*self.descriptions]
        before = [b.layers.clone() for b in banks]
        initial = model(self.source,self.descriptions).detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(self.source,self.descriptions)[None],torch.tensor([1])).backward()
            optimizer.step()
        self.assertFalse(torch.equal(initial,model(self.source,self.descriptions)))
        for bank,original in zip(banks,before):
            self.assertTrue(torch.equal(bank.layers,original))
            self.assertIsNone(bank.layers.grad)
            self.assertFalse(bank.layers.requires_grad)

    def test_correction_is_candidate_permutation_equivariant(self):
        model = self.model()
        with torch.no_grad(): model.scorer[-1].weight.fill_(0.2)
        normal = model(self.source,self.descriptions)
        order = [2,0,1]
        shuffled = model(self.source,[self.descriptions[i] for i in order])
        self.assertTrue(torch.equal(shuffled,normal[order]))

    def test_all_bank_validation_applies_to_anchored_path(self):
        model = self.model()
        bad = LayerTokenBank(self.descriptions[0].layers.float(),self.descriptions[0].input_ids,self.descriptions[0].attention_mask)
        with self.assertRaises(ValueError):model(self.source,[bad])


if __name__ == "__main__":
    unittest.main()
