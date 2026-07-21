from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FreshQueryClaimRelevanceCalibrationTests(unittest.TestCase):
    def test_query_cap_prevents_single_query_dominance(self):
        module = load_script("prepare_fresh_query_claim_relevance_calibration.py")
        candidates = []
        for i in range(12):
            candidates.append({
                "query_id": "dominant",
                "relevance_score": float(i),
            })
        for i in range(8):
            candidates.append({
                "query_id": f"q{i}",
                "relevance_score": float(i) - 20.0,
            })

        selected = module.select_rank_stratified_with_query_cap(
            candidates,
            sample_size=10,
            max_per_query=2,
        )
        dominant_count = sum(
            item["query_id"] == "dominant" for item in selected
        )
        self.assertLessEqual(dominant_count, 2)

    def test_v2_evaluator_can_treat_partial_as_positive_for_filtering(self):
        module = load_script("evaluate_query_claim_relevance_calibration_v2.py")
        items = [
            {"relevance_score": -11.0, "relevance_human_label": "irrelevant"},
            {"relevance_score": -10.0, "relevance_human_label": "irrelevant"},
            {"relevance_score": -9.0, "relevance_human_label": "partially_relevant"},
            {"relevance_score": -8.0, "relevance_human_label": "relevant"},
        ]
        strict = module.evaluate_objective(items, {"relevant"})
        broad = module.evaluate_objective(
            items,
            {"relevant", "partially_relevant"},
        )
        self.assertEqual(strict["positive_count"], 1)
        self.assertEqual(broad["positive_count"], 2)
        self.assertIsNotNone(broad["best_high_precision_threshold"])


if __name__ == "__main__":
    unittest.main()
