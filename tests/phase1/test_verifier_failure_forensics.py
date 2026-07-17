from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alice_vault.verifier_failure_forensics import analyze_verifier_failures


class VerifierFailureForensicsTests(unittest.TestCase):
    def test_detects_shared_and_unanimous_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests" / "calibration" / "pilot-v1").mkdir(parents=True)
            (root / "manifests" / "exports").mkdir(parents=True)

            details = root / "details.json"
            holdout = root / "holdout.json"

            details.write_text(json.dumps({
                "hhem_holdout_evaluation_schema_version": 1,
                "run_id": "run",
                "holdout_id": "holdout",
                "items": [
                    {
                        "item_id": "1",
                        "query_id": "q1",
                        "human_label": "unsupported",
                        "hhem_score": 0.99,
                        "hhem_predicted_supported": True,
                        "qwen_verdict": "supported",
                        "fever_decision": "keep_entailment",
                    },
                    {
                        "item_id": "2",
                        "query_id": "q2",
                        "human_label": "supported",
                        "hhem_score": 0.99,
                        "hhem_predicted_supported": True,
                        "qwen_verdict": "supported",
                        "fever_decision": "keep_entailment",
                    },
                ],
            }), encoding="utf-8")

            holdout.write_text(json.dumps({
                "holdout_schema_version": 1,
                "items": [
                    {
                        "item_id": "1",
                        "claim_text": "bad claim",
                        "evidence_windows": [{"text": "weak evidence"}],
                    },
                    {
                        "item_id": "2",
                        "claim_text": "good claim",
                        "evidence_windows": [{"text": "strong evidence"}],
                    },
                ],
            }), encoding="utf-8")

            result = analyze_verifier_failures(
                holdout_details_path=details,
                holdout_bundle_path=holdout,
                vault_root=root,
            )

            self.assertEqual(result["qwen_false_positive_count"], 1)
            self.assertEqual(result["hhem_false_positive_count"], 1)
            self.assertEqual(result["unanimous_false_positive_count"], 1)
            self.assertFalse(result["production_gate_changed"])

            private = json.loads(
                Path(result["private_details_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                private["private_failure_cases"][0]["claim_text"],
                "bad claim",
            )


if __name__ == "__main__":
    unittest.main()
