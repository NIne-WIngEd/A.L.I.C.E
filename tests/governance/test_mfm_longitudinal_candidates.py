from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_kernel.formation_gold import load_formation_gold


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mfm/generate_longitudinal_candidates.py"
spec = importlib.util.spec_from_file_location("mfm_longitudinal_candidates", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class LongitudinalCandidateTests(unittest.TestCase):
    def test_owner_authorized_candidates_reproduce_without_gold_admission(self):
        rows = module.generate(20260929, 15)
        self.assertEqual(module._json_bytes(rows), module._json_bytes(module.generate(20260929, 15)))
        self.assertEqual(len(rows), 150)
        self.assertEqual({r["generator_family"] for r in rows},
                         {"chatgpt-templated-diagnostic-v1"})
        self.assertEqual(len({r["scenario_family"] for r in rows}), 10)
        self.assertTrue(all(r["training_rights"] == "owner_authorized_chatgpt_codex_teaching_output"
                            for r in rows))
        self.assertTrue(all(r["admission"] == "candidate_only_unreviewed" for r in rows))
        self.assertTrue(all(r.get("decision_action", "propose") == "propose"
                            for r in rows))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "candidate.json"
            path.write_bytes(module._json_bytes(rows))
            compiled = load_formation_gold(path)
        self.assertEqual(len(compiled), len(rows))
        self.assertEqual(len({case.host_family for case in compiled}), 15)
        self.assertEqual(len([r for r in rows if any(
            d["action"] == "defer" for d in r.get("dispositions", []))]), 15)
        for row in rows:
            if "dispositions" in row:
                self.assertEqual(len(row["expected"]), 1)
                self.assertNotEqual(row["expected"][0]["kind"], row["critical_forbidden"][0][0])
                self.assertEqual(row["dispositions"][0]["action"], "propose")
                self.assertEqual(row["expected"][0]["disposition_scope_ref"],
                                 row["dispositions"][0]["scope_ref"])
            if row["case_id"].endswith("negotiated-norm"):
                self.assertEqual(row["expected"][0]["epistemic_status"], "owner_statement")
                self.assertEqual(len(row["expected"][0]["evidence_refs"]), 1)

    def test_no_hidden_final_or_real_person_payload(self):
        rows = module.generate(173, 4)
        self.assertFalse(any(r["split"] == "challenge" for r in rows))
        self.assertTrue(all(r["host_family"].startswith("fictional-mfm-") for r in rows))
        self.assertTrue(all(len(r["sources"]) > 1 for r in rows))
        with self.assertRaises(ValueError):
            module.generate(1, 0)


if __name__ == "__main__":
    unittest.main()
