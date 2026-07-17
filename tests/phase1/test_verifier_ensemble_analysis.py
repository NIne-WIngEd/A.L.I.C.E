from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alice_vault.verifier_ensemble_analysis import analyze_verifier_ensembles


class VerifierEnsembleAnalysisTests(unittest.TestCase):
    def test_fixed_boolean_ensemble_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            details_path = root / "holdout-details.json"

            items = [
                {
                    "item_id": "1",
                    "query_id": "q1",
                    "human_label": "supported",
                    "qwen_verdict": "supported",
                    "hhem_predicted_supported": True,
                    "fever_decision": "keep_entailment",
                },
                {
                    "item_id": "2",
                    "query_id": "q2",
                    "human_label": "unsupported",
                    "qwen_verdict": "supported",
                    "hhem_predicted_supported": False,
                    "fever_decision": "drop_neutral",
                },
                {
                    "item_id": "3",
                    "query_id": "q3",
                    "human_label": "partially_supported",
                    "qwen_verdict": "unsupported",
                    "hhem_predicted_supported": True,
                    "fever_decision": "drop_neutral",
                },
                {
                    "item_id": "4",
                    "query_id": "q4",
                    "human_label": "supported",
                    "qwen_verdict": "supported",
                    "hhem_predicted_supported": True,
                    "fever_decision": "drop_neutral",
                },
            ]

            details_path.write_text(
                json.dumps(
                    {
                        "hhem_holdout_evaluation_schema_version": 1,
                        "run_id": "run",
                        "holdout_id": "holdout",
                        "items": items,
                    }
                ),
                encoding="utf-8",
            )

            result = analyze_verifier_ensembles(details_path=details_path)

            self.assertEqual(
                result["rules"]["qwen_only"]["support_precision"],
                0.666667,
            )
            self.assertEqual(
                result["rules"]["qwen_and_hhem"]["support_precision"],
                1.0,
            )
            self.assertEqual(
                result["rules"]["all_three"]["support_precision"],
                1.0,
            )
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["production_gate_changed"])

    def test_partial_support_is_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "details.json"
            path.write_text(
                json.dumps(
                    {
                        "hhem_holdout_evaluation_schema_version": 1,
                        "items": [
                            {
                                "item_id": "1",
                                "query_id": "q1",
                                "human_label": "partially_supported",
                                "qwen_verdict": "supported",
                                "hhem_predicted_supported": True,
                                "fever_decision": "keep_entailment",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = analyze_verifier_ensembles(details_path=path)
            self.assertEqual(
                result["rules"]["all_three"]["false_positive"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
