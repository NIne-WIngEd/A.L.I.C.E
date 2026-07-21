from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alice_vault.query_claim_relevance_holdout import (
    evaluate_frozen_holdout,
    load_query_claim_relevance_holdout_policy,
    select_with_query_cap,
)


class QueryClaimRelevanceHoldoutTests(unittest.TestCase):
    def test_policy_freezes_v2_non_irrelevance_threshold(self):
        policy = load_query_claim_relevance_holdout_policy()
        self.assertEqual(policy.frozen_threshold, -9.76161)
        self.assertEqual(policy.objective, "non_irrelevance_filter")
        self.assertIn("relevant", policy.positive_labels)
        self.assertIn("partially_relevant", policy.positive_labels)
        self.assertEqual(policy.negative_labels, ("irrelevant",))
        self.assertFalse(policy.production_gate_changed if hasattr(policy, "production_gate_changed") else False)

    def test_query_cap_is_respected(self):
        candidates = []
        for i in range(6):
            candidates.append({
                "query_id": "q1",
                "relevance_score": float(i),
            })
        for i in range(4):
            candidates.append({
                "query_id": f"q{i+2}",
                "relevance_score": float(10 + i),
            })
        selected = select_with_query_cap(
            candidates,
            sample_size=6,
            max_per_query=2,
        )
        counts = {}
        for item in selected:
            counts[item["query_id"]] = counts.get(item["query_id"], 0) + 1
        self.assertTrue(all(count <= 2 for count in counts.values()))

    def test_holdout_uses_frozen_threshold_without_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests" / "calibration" / "pilot-v1").mkdir(parents=True)
            (root / "manifests" / "exports").mkdir(parents=True)

            items = [
                ("relevant", -3.0),
                ("partially_relevant", -8.0),
                ("partially_relevant", -9.0),
                ("irrelevant", -10.5),
                ("irrelevant", -11.0),
                ("irrelevant", -12.0),
            ]
            bundle = {
                "query_claim_relevance_holdout_bundle_schema_version": 1,
                "holdout_id": "test",
                "source_calibration_id": "cal",
                "threshold_frozen_before_human_review": True,
                "frozen_threshold": -9.76161,
                "items": [
                    {
                        "item_id": str(i),
                        "query_id": f"q{i}",
                        "relevance_human_label": label,
                        "relevance_score": score,
                    }
                    for i, (label, score) in enumerate(items)
                ],
            }
            path = root / "holdout.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")

            result = evaluate_frozen_holdout(
                vault_root=root,
                holdout_path=path,
            )

            self.assertEqual(result["frozen_threshold"], -9.76161)
            self.assertFalse(result["threshold_sweep_performed_on_holdout"])
            self.assertEqual(result["metrics"]["false_positive"], 0)
            self.assertTrue(result["passes_holdout_gate"])
            self.assertFalse(result["production_gate_changed"])


if __name__ == "__main__":
    unittest.main()
