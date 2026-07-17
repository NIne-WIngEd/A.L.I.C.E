from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from alice_vault.claim_entailment_gate import load_claim_entailment_policy, score_claim_support

class FakeFeverNLI:
    def predict(self,pairs,batch_size=8,show_progress_bar=False):
        return [[4.0,0.2,0.1] for _ in pairs]

class FeverNLIVerifierTests(unittest.TestCase):
    def test_policy_uses_fact_verification_aware_model_and_correct_label_order(self):
        p=load_claim_entailment_policy(ROOT/'policies'/'claim_entailment_policy.json')
        self.assertEqual(p.model_id,'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
        self.assertEqual(p.label_order,('entailment','neutral','contradiction'))
        self.assertEqual(p.entailment_threshold,0.70)
    def test_dynamic_label_order_keeps_entailed_claim(self):
        p=load_claim_entailment_policy(ROOT/'policies'/'claim_entailment_policy.json')
        claim={'text':'The AFM project used a U-Net.','citations':['[S1]']}
        context={'evidence':[{'citation':'[S1]','owner_relation':'owner_self_record','context_text':'The AFM project used a U-Net for segmentation.'}]}
        result=score_claim_support(claim=claim,context_package=context,model=FakeFeverNLI(),policy=p)
        self.assertEqual(result['decision'],'keep_entailment')
        self.assertGreater(result['best_entailment_probability'],0.90)
if __name__=='__main__': unittest.main()
