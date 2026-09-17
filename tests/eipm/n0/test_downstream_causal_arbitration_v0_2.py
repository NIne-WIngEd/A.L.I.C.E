import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

DRIVER = Path(__file__).parents[3] / "scripts/eipm/n0/downstream_causal_arbitration_v0_2.py"
SPEC = importlib.util.spec_from_file_location("arb_v02", DRIVER)
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

class ArbitrationV02Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.evaluator = self.root / "eval.py"
        self.evaluator.write_text("import argparse,json\np=argparse.ArgumentParser(); p.add_argument('--graph'); p.add_argument('--latent-checkpoint'); p.add_argument('--config'); p.add_argument('--evaluation-set'); p.add_argument('--extra'); p.add_argument('--output'); a=p.parse_args(); g=json.load(open(a.graph)); json.dump({'score':g['score'],'loss':g['loss']},open(a.output,'w'))\n", encoding="utf-8")
        for name, text in {"latent.json":"{}", "config.json":"{}", "eval.json":"{}", "extra.json":"{}"}.items(): (self.root/name).write_text(text, encoding="utf-8")
        (self.root/"canonical.json").write_text('{"score":0.5,"loss":0.5}', encoding="utf-8")
        (self.root/"candidate.json").write_text('{"score":0.8,"loss":0.2}', encoding="utf-8")
    def tearDown(self): self.tmp.cleanup()
    def spec(self, name):
        p=self.root/name; return {"path":str(p),"sha256":MOD.sha256_file(p)}
    def manifest(self):
        return {"protocol_version":MOD.PROTOCOL_VERSION,"experiment_id":"unit-test","source_revision":"a"*40,
          "invariants":{"diagnostic_only":True,"training_enabled":False,"ratifies_repair":False,"scale_authorized":False,"promotion_authorized":False},
          "common_inputs":{"evaluator":self.spec("eval.py"),"latent_checkpoint":self.spec("latent.json"),"evaluation_config":self.spec("config.json"),"evaluation_set":self.spec("eval.json")},
          "arms":{"canonical":{"label":"canonical","graph":self.spec("canonical.json")},"candidate":{"label":"candidate","graph":self.spec("candidate.json")}},
          "evaluator_argv":[sys.executable,"{input:evaluator}","--graph","{graph}","--latent-checkpoint","{input:latent_checkpoint}","--config","{input:evaluation_config}","--evaluation-set","{input:evaluation_set}","--output","{output}"],
          "metrics":[{"path":"score","direction":"higher","atol":0.0,"rtol":0.0},{"path":"loss","direction":"lower","atol":0.0,"rtol":0.0}]}
    def write(self,m):
        p=self.root/"manifest.json"; p.write_text(json.dumps(m),encoding="utf-8"); return p
    def test_preflight_pins_shared_fingerprint(self):
        r=MOD.run_protocol(self.write(self.manifest()),self.root/"preflight.json",True)
        self.assertEqual(r["status"],"PREFLIGHT_OK"); self.assertEqual(r["arm_common_fingerprints"]["canonical"],r["arm_common_fingerprints"]["candidate"]); self.assertFalse(r["n0_complete"])
    def test_complete_improvement_never_ratifies(self):
        r=MOD.run_protocol(self.write(self.manifest()),self.root/"result.json")
        self.assertEqual(r["classification"],"IMPROVEMENT"); self.assertFalse(r["ratifies_repair"]); self.assertFalse(r["scale_authorized"]); self.assertFalse(r["promotion_authorized"])
    def test_equivalent(self):
        (self.root/"candidate.json").write_text('{"score":0.5,"loss":0.5,"variant":"candidate"}',encoding="utf-8")
        self.assertEqual(MOD.run_protocol(self.write(self.manifest()),self.root/"eq.json")["classification"],"EQUIVALENT")
    def test_harm_dominates(self):
        (self.root/"candidate.json").write_text('{"score":0.8,"loss":0.8}',encoding="utf-8")
        self.assertEqual(MOD.run_protocol(self.write(self.manifest()),self.root/"harm.json")["classification"],"HARM")
    def test_missing_each_mandatory_role_fails(self):
        for role in MOD.MANDATORY_COMMON_ROLES:
            m=self.manifest(); del m["common_inputs"][role]; p=self.write(m)
            with self.assertRaises(MOD.ProtocolError): MOD.validate_manifest(m,p)
    def test_bad_source_revision_fails(self):
        m=self.manifest(); m["source_revision"]="main"; p=self.write(m)
        with self.assertRaises(MOD.ProtocolError): MOD.validate_manifest(m,p)
    def test_unused_extra_input_fails(self):
        m=self.manifest(); m["common_inputs"]["extra"]=self.spec("extra.json"); p=self.write(m)
        with self.assertRaises(MOD.ProtocolError): MOD.validate_manifest(m,p)
    def test_referenced_extra_input_is_allowed(self):
        m=self.manifest(); m["common_inputs"]["extra"]=self.spec("extra.json"); m["evaluator_argv"][2:2]=["--extra","{input:extra}"]; p=self.write(m)
        self.assertIn("extra",MOD.validate_manifest(m,p)["common_inputs"])
    def test_unknown_placeholder_fails(self):
        m=self.manifest(); m["evaluator_argv"].append("{input:ghost}"); p=self.write(m)
        with self.assertRaises(MOD.ProtocolError): MOD.validate_manifest(m,p)
    def test_equals_form_training_arg_fails(self):
        m=self.manifest(); m["evaluator_argv"].append("--epochs=1"); p=self.write(m)
        with self.assertRaises(MOD.ProtocolError): MOD.validate_manifest(m,p)
    def test_hash_mismatch_fails(self):
        m=self.manifest(); m["common_inputs"]["evaluation_set"]["sha256"]="0"*64; p=self.write(m)
        with self.assertRaises(MOD.ProtocolError): MOD.validate_manifest(m,p)
    def test_identical_graph_hash_fails(self):
        m=self.manifest(); m["arms"]["candidate"]["graph"]=m["arms"]["canonical"]["graph"]; p=self.write(m)
        with self.assertRaises(MOD.ProtocolError): MOD.validate_manifest(m,p)
    def test_missing_metric_fails_closed_after_evaluation(self):
        m=self.manifest(); m["metrics"]=[{"path":"missing","direction":"higher"}]
        with self.assertRaises(MOD.ProtocolError): MOD.run_protocol(self.write(m),self.root/"badmetric.json")
    def test_refuses_overwrite(self):
        out=self.root/"exists.json"; out.write_text("{}",encoding="utf-8")
        with self.assertRaises(MOD.ProtocolError): MOD.run_protocol(self.write(self.manifest()),out,True)

if __name__ == "__main__": unittest.main()
