from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alice_vault.hhem_holdout import (
    evaluate_hhem_holdout,
    load_hhem_holdout_policy,
)


class _FakeHHEM:
    def __init__(self, scores):
        self.scores = list(scores)

    def predict(self, pairs):
        count = len(pairs)
        result = self.scores[:count]
        self.scores = self.scores[count:]
        return result


class HHEMHoldoutTests(unittest.TestCase):
    def test_repository_policy_freezes_calibration_threshold(self):
        policy = load_hhem_holdout_policy()
        self.assertEqual(policy.frozen_threshold, 0.984156)
        self.assertFalse(policy.private_text_uploaded)
        self.assertFalse(policy.memory_write_allowed)
        self.assertFalse(policy.external_action_allowed)
        self.assertFalse(policy.tool_calling_allowed)
        self.assertFalse(policy.web_access_allowed)

    def test_evaluation_uses_frozen_threshold_without_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests" / "calibration" / "pilot-v1").mkdir(
                parents=True
            )
            (root / "manifests" / "exports").mkdir(parents=True)

            items = []
            labels = [
                "supported",
                "supported",
                "supported",
                "supported",
                "unsupported",
                "unsupported",
                "unsupported",
                "unsupported",
            ]
            for index, label in enumerate(labels):
                items.append(
                    {
                        "item_id": f"item-{index}",
                        "query_id": f"q-{index}",
                        "claim_text": f"claim {index}",
                        "evidence_windows": [{"text": f"evidence {index}"}],
                        "human_label": label,
                        "qwen_auditor": {
                            "verdict": "supported" if index < 3 else "unsupported"
                        },
                        "fever_nli": {
                            "decision": (
                                "keep_entailment"
                                if index < 4
                                else "drop_neutral"
                            ),
                            "best_entailment_probability": 0.99 if index < 4 else 0.1,
                        },
                    }
                )

            bundle_path = (
                root
                / "manifests"
                / "calibration"
                / "pilot-v1"
                / "judge-holdout-test.json"
            )
            bundle_path.write_text(
                json.dumps(
                    {
                        "judge_calibration_bundle_schema_version": 1,
                        "holdout_schema_version": 1,
                        "holdout_id": "test",
                        "threshold_frozen_before_human_holdout_review": True,
                        "frozen_hhem_threshold": 0.984156,
                        "items": items,
                    }
                ),
                encoding="utf-8",
            )

            def loader(**kwargs):
                return _FakeHHEM(
                    [0.99, 0.995, 0.999, 0.90, 0.2, 0.1, 0.4, 0.8]
                )

            result = evaluate_hhem_holdout(
                vault_root=root,
                holdout_bundle_path=bundle_path,
                model_loader=loader,
            )

            self.assertTrue(result["threshold_frozen"])
            self.assertFalse(result["threshold_sweep_performed_on_holdout"])
            self.assertEqual(result["frozen_hhem_threshold"], 0.984156)
            self.assertFalse(result["production_gate_changed"])
            self.assertEqual(result["hhem"]["false_positive"], 0)
            self.assertEqual(result["hhem"]["false_negative"], 1)


if __name__ == "__main__":
    unittest.main()
