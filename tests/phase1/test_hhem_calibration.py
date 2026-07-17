from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.hhem_calibration import (
    _average_precision,
    _binary_metrics,
    _roc_auc,
    _threshold_sweep,
    build_hhem_premise,
    load_hhem_calibration_policy,
    score_hhem_pairs,
)


class FakeHHEM:
    def predict(self, pairs):
        return [
            0.95 if "supported fact" in hypothesis
            else 0.10
            for _, hypothesis in pairs
        ]


class HHEMCalibrationTests(unittest.TestCase):
    def test_policy_is_private_and_offline(self):
        policy = load_hhem_calibration_policy(
            ROOT / "policies" / "hhem_calibration_policy.json"
        )
        self.assertFalse(policy.private_text_uploaded)
        self.assertFalse(policy.web_access_allowed)
        self.assertEqual(
            policy.model_id,
            "vectara/hallucination_evaluation_model",
        )

    def test_premise_uses_review_evidence_and_deduplicates(self):
        item = {
            "evidence_windows": [
                {"text": "Evidence A"},
                {"text": "Evidence A"},
                {"text": "Evidence B"},
            ]
        }
        premise = build_hhem_premise(item)
        self.assertEqual(
            premise,
            "Evidence A\n\nEvidence B",
        )

    def test_scores_pairs_in_batches(self):
        scores = score_hhem_pairs(
            model=FakeHHEM(),
            pairs=[
                ("premise", "supported fact"),
                ("premise", "unsupported"),
            ],
            batch_size=1,
        )
        self.assertEqual(
            scores,
            [0.95, 0.10],
        )

    def test_auc_and_threshold_sweep_separate_supported_items(self):
        expected = [1, 1, 0, 0]
        scores = [0.95, 0.85, 0.20, 0.10]

        self.assertEqual(
            _roc_auc(
                expected=expected,
                scores=scores,
            ),
            1.0,
        )
        self.assertEqual(
            _average_precision(
                expected=expected,
                scores=scores,
            ),
            1.0,
        )

        sweep = _threshold_sweep(
            expected=expected,
            scores=scores,
            high_precision_target=0.90,
        )
        self.assertIsNotNone(
            sweep["best_high_precision"]
        )
        self.assertEqual(
            sweep["best_high_precision"]["support_precision"],
            1.0,
        )

    def test_binary_metrics_treat_only_full_support_as_positive(self):
        metrics = _binary_metrics(
            expected=[1, 0, 0],
            predicted=[1, 1, 0],
        )
        self.assertEqual(
            metrics["support_precision"],
            0.5,
        )
        self.assertEqual(
            metrics["support_recall"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
