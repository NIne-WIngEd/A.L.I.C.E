from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_context import _label


class FalseContradictionLabelTests(unittest.TestCase):
    def test_false_like_values_are_not_contradiction_groups(self):
        for value in (
            "",
            None,
            "None",
            "null",
            "[none]",
            "No",
            "NO",
            "false",
            "False",
            "0",
            "N/A",
            "na",
            "not applicable",
            "not-applicable",
        ):
            with self.subTest(value=value):
                self.assertEqual(_label(value), "")

    def test_real_contradiction_label_is_preserved(self):
        self.assertEqual(
            _label("project-status"),
            "project-status",
        )


if __name__ == "__main__":
    unittest.main()
