from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import load_grounded_response_policy


class GroundedResponsePolicyCompatibilityTests(unittest.TestCase):
    def test_repository_policy_contains_resilience_and_expansion(self):
        policy = load_grounded_response_policy(
            ROOT / "policies" / "grounded_response_policy.json"
        )
        self.assertEqual(policy.request_timeout_seconds, 600)
        self.assertEqual(policy.request_retry_count, 2)
        self.assertEqual(policy.request_retry_backoff_seconds, 8.0)
        self.assertEqual(policy.keep_alive, "30m")
        self.assertEqual(policy.maximum_output_tokens, 2048)
        self.assertTrue(policy.evidence_expansion["enabled"])

    def test_legacy_policy_missing_resilience_keys_uses_safe_defaults(self):
        source = json.loads(
            (ROOT / "policies" / "grounded_response_policy.json")
            .read_text(encoding="utf-8")
        )
        for key in (
            "request_retry_count",
            "request_retry_backoff_seconds",
            "keep_alive",
            "maximum_output_tokens",
        ):
            source.pop(key, None)
        source["request_timeout_seconds"] = 180

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy-policy.json"
            path.write_text(
                json.dumps(source),
                encoding="utf-8",
            )
            policy = load_grounded_response_policy(path)

        self.assertEqual(policy.request_timeout_seconds, 180)
        self.assertEqual(policy.request_retry_count, 2)
        self.assertEqual(policy.request_retry_backoff_seconds, 8.0)
        self.assertEqual(policy.keep_alive, "30m")
        self.assertEqual(policy.maximum_output_tokens, 512)
        self.assertTrue(policy.evidence_expansion["enabled"])


if __name__ == "__main__":
    unittest.main()
