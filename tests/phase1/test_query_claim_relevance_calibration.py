from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from alice_vault.query_claim_relevance_calibration import _rank_stratified_sample, evaluate_query_claim_relevance_calibration, load_query_claim_relevance_calibration_policy, save_relevance_human_label
class QueryClaimRelevanceCalibrationTests(unittest.TestCase):
    def test_policy_is_private_offline_and_excludes_regression_queries(self):
        p=load_query_claim_relevance_calibration_policy(); self.assertIn("personal-004",p.excluded_regression_query_ids); self.assertIn("personal-006",p.excluded_regression_query_ids); self.assertIn("personal-018",p.excluded_regression_query_ids); self.assertFalse(p.private_text_uploaded); self.assertFalse(p.memory_write_allowed); self.assertFalse(p.external_action_allowed); self.assertFalse(p.tool_calling_allowed); self.assertFalse(p.web_access_allowed)
    def test_rank_stratified_sample_spans_score_distribution(self):
        c=[{"item_id":str(i),"query_id":f"q{i%4}","relevance_score":float(i)} for i in range(20)]; s=_rank_stratified_sample(candidates=c,sample_size=8,seed="test"); scores=[x["relevance_score"] for x in s]; self.assertEqual(len(s),8); self.assertLess(min(scores),5.0); self.assertGreater(max(scores),14.0)
    def test_relevance_labels_are_separate_from_support_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"b.json"; path.write_text(json.dumps({"query_claim_relevance_calibration_bundle_schema_version":1,"items":[{"item_id":"x","human_label":"supported","relevance_human_label":""}]}),encoding="utf-8"); save_relevance_human_label(bundle_path=path,item_id="x",label="irrelevant"); b=json.loads(path.read_text()); self.assertEqual(b["items"][0]["human_label"],"supported"); self.assertEqual(b["items"][0]["relevance_human_label"],"irrelevant")
    def test_evaluation_selects_high_precision_threshold_without_production_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"manifests"/"calibration"/"pilot-v1").mkdir(parents=True); (root/"manifests"/"exports").mkdir(parents=True); labels=["irrelevant","irrelevant","partially_relevant","irrelevant","relevant","relevant","relevant","relevant"]; scores=[-10,-9,-8,-7,-3,-2,-1,0]; path=root/"b.json"; path.write_text(json.dumps({"query_claim_relevance_calibration_bundle_schema_version":1,"calibration_id":"cal","pilot_name":"pilot-v1","reranker":{"model_id":"test-model"},"items":[{"item_id":str(i),"query_id":f"q{i}","relevance_human_label":l,"relevance_score":s} for i,(l,s) in enumerate(zip(labels,scores))]}),encoding="utf-8"); r=evaluate_query_claim_relevance_calibration(vault_root=root,bundle_path=path); self.assertIsNotNone(r["best_high_precision_threshold"]); self.assertFalse(r["production_gate_changed"]); self.assertTrue(r["holdout_validation_required_before_production"])
if __name__=="__main__": unittest.main()
