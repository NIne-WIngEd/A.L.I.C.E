from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from alice_vault.grounded_response import GroundedResponsePolicy, ollama_generate

class _Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def read(self): return json.dumps(self.payload).encode("utf-8")

class OllamaStructuredJSONRetryTests(unittest.TestCase):
    def policy(self):
        return GroundedResponsePolicy(
            policy_id="test", model="qwen3:8b", ollama_endpoint="http://127.0.0.1:11434/api/generate",
            request_timeout_seconds=30, request_retry_count=2, request_retry_backoff_seconds=0, keep_alive="30m",
            maximum_output_tokens=256, temperature=0.0, think=False, maximum_context_sources=6, maximum_answer_characters=5000,
            require_structured_output=True, require_citations_for_factual_claims=True, require_citations_for_inferences=True,
            allow_only_package_citations=True, surface_unresolved_contradictions=True, allow_general_knowledge_for_personal_facts=False,
            memory_write_allowed=False, external_action_allowed=False, tool_calling_allowed=False, web_access_allowed=False,
            private_output_only=True, evidence_expansion={}, minimum_verified_response_rate=1.0,
            minimum_expected_source_citation_rate=0.9, minimum_claim_citation_coverage=1.0, digest="test", source_path=None,
        )

    def test_invalid_structured_json_is_retried(self):
        bad = _Response({"response": "{\"answer_type\":\"grounded\",\"answer\":\"unterminated", "model": "qwen3:8b"})
        good_structured = {"answer_type":"insufficient_evidence","answer":"Not enough evidence.","claims":[],"uncertainty_notes":[],"contradiction_notes":[]}
        good = _Response({"response": json.dumps(good_structured), "model": "qwen3:8b", "done": True})
        with patch("alice_vault.grounded_response.urllib.request.urlopen", side_effect=[bad, good]) as mocked:
            result = ollama_generate(self.policy(), {"type":"object"}, "system", "user")
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["structured"], good_structured)
        self.assertEqual(result["ollama"]["attempt_count"], 2)

    def test_invalid_structured_json_fails_after_all_attempts(self):
        bad = _Response({"response":"{\"broken\":", "model":"qwen3:8b"})
        with patch("alice_vault.grounded_response.urllib.request.urlopen", return_value=bad):
            with self.assertRaisesRegex(RuntimeError, "invalid structured JSON after 3 attempt"):
                ollama_generate(self.policy(), {"type":"object"}, "system", "user")

if __name__ == "__main__": unittest.main()
