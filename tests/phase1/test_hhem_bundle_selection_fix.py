from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.hhem_calibration import (
    _is_human_calibration_bundle,
    latest_calibration_bundle,
)


class HHEMBundleSelectionFixTests(unittest.TestCase):
    def test_derivative_evaluation_details_are_not_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            directory = (
                vault
                / "manifests"
                / "calibration"
                / "pilot-v1"
            )
            directory.mkdir(
                parents=True
            )

            raw_bundle = (
                directory
                / "judge-calibration-raw.json"
            )
            raw_bundle.write_text(
                json.dumps(
                    {
                        "judge_calibration_bundle_schema_version": 1,
                        "items": [
                            {
                                "claim_text": "Claim",
                                "evidence_windows": [
                                    {
                                        "text": "Evidence"
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            time.sleep(0.01)

            derivative = (
                directory
                / "judge-calibration-evaluation-details-newer.json"
            )
            derivative.write_text(
                json.dumps(
                    {
                        "judge_calibration_evaluation_schema_version": 1,
                        "items": [
                            {
                                "human_label": "supported"
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(
                _is_human_calibration_bundle(
                    derivative
                )
            )
            self.assertTrue(
                _is_human_calibration_bundle(
                    raw_bundle
                )
            )
            self.assertEqual(
                latest_calibration_bundle(
                    vault_root=vault,
                    pilot_name="pilot-v1",
                ),
                raw_bundle,
            )


if __name__ == "__main__":
    unittest.main()
