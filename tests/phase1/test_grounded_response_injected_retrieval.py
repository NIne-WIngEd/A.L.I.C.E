from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import alice_vault.grounded_response_evaluation as module


class InjectedRetrievalBoundaryTests(unittest.TestCase):
    def test_custom_search_bypasses_production_evidence_expansion(self):
        source = inspect.getsource(
            module.evaluate_grounded_responses
        )
        self.assertIn(
            "and context_search_fn is None",
            source,
        )


if __name__ == "__main__":
    unittest.main()
