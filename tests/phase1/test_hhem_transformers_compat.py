from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.hhem_calibration import (
    _transformers_major_version,
)


class HHEMTransformersCompatibilityTests(unittest.TestCase):
    def test_transformers_major_version_is_integer(self):
        fake_transformers = SimpleNamespace(
            __version__="4.57.6",
        )

        with patch.dict(
            sys.modules,
            {
                "transformers": fake_transformers,
            },
        ):
            major = _transformers_major_version()

        self.assertIsInstance(
            major,
            int,
        )
        self.assertEqual(
            major,
            4,
        )

    def test_requirements_pin_transformers_below_v5(self):
        requirements = (
            ROOT
            / "requirements-hhem-calibration.txt"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "transformers>=4.45,<5.0",
            requirements,
        )


if __name__ == "__main__":
    unittest.main()
