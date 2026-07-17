from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.judge_calibration import (
    load_judge_calibration_policy,
    save_human_label,
    select_stratified_sample,
)


class JudgeCalibrationTests(unittest.TestCase):
    def test_stratified_sample_prioritizes_disagreement_buckets(self):
        candidates = [
            {
                "item_id": "a",
                "query_id": "q1",
                "stratum": "fever_high__qwen_not_supported",
            },
            {
                "item_id": "b",
                "query_id": "q2",
                "stratum": "fever_high__qwen_supported",
            },
            {
                "item_id": "c",
                "query_id": "q3",
                "stratum": "fever_borderline__qwen_not_supported",
            },
        ]
        selected = select_stratified_sample(
            candidates=candidates,
            sample_size=2,
            seed="test",
        )
        ids = {item["item_id"] for item in selected}
        self.assertIn("a", ids)
        self.assertIn("c", ids)

    def test_human_label_is_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle.json"
            path.write_text(
                """{
                  "items": [
                    {
                      "item_id": "x",
                      "human_label": "",
                      "human_labeled_at": ""
                    }
                  ]
                }""",
                encoding="utf-8",
            )
            result = save_human_label(
                bundle_path=path,
                item_id="x",
                label="supported",
            )
            self.assertEqual(result["labeled_count"], 1)

    def test_policy_is_loopback_and_private(self):
        policy = load_judge_calibration_policy(
            ROOT / "policies" / "judge_calibration_policy.json"
        )
        self.assertEqual(policy.review_host, "127.0.0.1")
        self.assertTrue(policy.blind_review)
        self.assertTrue(policy.private_output_only)


if __name__ == "__main__":
    unittest.main()
