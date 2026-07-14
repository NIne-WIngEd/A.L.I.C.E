from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.parser_registry import load_registry


class ParserRegistryTests(unittest.TestCase):
    def test_registry_is_valid_and_xml_is_disabled(self) -> None:
        registry = load_registry(ROOT / "policies" / "parser_registry.json")
        self.assertEqual(registry.registry_id, "alice-parser-registry-v1")
        self.assertTrue(registry.by_family()["json"].enabled)
        self.assertFalse(registry.by_family()["xml"].enabled)
        self.assertEqual(len(registry.digest), 64)

    def test_registry_rejects_wrong_extension(self) -> None:
        registry = load_registry(ROOT / "policies" / "parser_registry.json")
        with self.assertRaises(ValueError):
            registry.select("json", Path("example.exe"))
        with self.assertRaises(ValueError):
            registry.select("xml", Path("example.xml"))


if __name__ == "__main__":
    unittest.main()
