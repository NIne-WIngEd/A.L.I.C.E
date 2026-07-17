from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import load_grounded_response_policy


class GroundedResponseOutputBudgetTests(unittest.TestCase):
    def test_structured_output_budget_is_large_enough_for_full_json(self):
        policy = load_grounded_response_policy(
            ROOT / "policies" / "grounded_response_policy.json"
        )
        self.assertGreaterEqual(policy.maximum_output_tokens, 2048)
        self.assertEqual(policy.request_timeout_seconds, 600)
        self.assertTrue(policy.evidence_expansion["enabled"])


if __name__ == "__main__":
    unittest.main()
