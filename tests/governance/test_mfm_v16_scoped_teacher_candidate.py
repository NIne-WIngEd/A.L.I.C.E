"""A scoped teacher sees three exact source fields, not surrounding records."""

from __future__ import annotations

from base64 import b64encode, b64decode
from hashlib import sha256
import json
import unittest

from scripts.mfm.prepare_v16_scoped_exercise_teacher_case import (
    exact_field_value, exact_property_spans, scoped_teacher_input,
)


class ScopedTeacherCandidateTests(unittest.TestCase):
    def test_claim_property_spans_exclude_unrelated_device_steps(self) -> None:
        raw = b'{"active_minutes": 32, "steps": 3253, "workout_detected": false}'
        spans = exact_property_spans(raw, ("active_minutes", "workout_detected"))
        self.assertEqual([name for name, _, _ in spans],
                         ["active_minutes", "workout_detected"])
        quoted = b" ".join(raw[start:end] for _, start, end in spans)
        self.assertIn(b'"active_minutes": 32', quoted)
        self.assertIn(b'"workout_detected": false', quoted)
        self.assertNotIn(b'"steps"', quoted)
        for name, start, end in spans:
            self.assertEqual(json.loads(b"{" + raw[start:end] + b"}")[name],
                             json.loads(raw)[name])
        with self.assertRaisesRegex(ValueError, "missing or repeated"):
            exact_property_spans(raw, ("active_minutes", "unavailable"))

    def test_exact_field_preserves_utf8_and_omits_neighbor(self) -> None:
        raw = ('{"date":"2026-01-01","exercise":{"type":"café walking",'
               '"duration_min":18},"secret_future":"never select"}').encode()
        selected, start, end = exact_field_value(raw, "exercise")
        self.assertEqual(selected, raw[start:end])
        self.assertEqual(json.loads(selected),
                         {"type": "café walking", "duration_min": 18})
        self.assertNotIn(b"secret_future", selected)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            exact_field_value(b'{"exercise":{},"exercise":{}}', "exercise")

    def test_three_field_view_contains_no_unrelated_source_payload(self) -> None:
        def item(kind: str, record: dict) -> dict:
            raw = json.dumps(record, sort_keys=True).encode()
            return {"source_id": f"source-{kind}", "source_type": kind,
                    "fragments": [{"payload_base64": b64encode(raw).decode(),
                                   "sha256": sha256(raw).hexdigest()}]}
        isolated = {"schema": "mfm-v16-isolated-source-packet-v1",
                    "training_admitted": False, "rights_status": "unverified",
                    "packet_id": "packet-opaque", "host_id": "host-opaque",
                    "window": {"start_day_index": 1, "end_day_index": 1},
                    "sources": [
                        item("planner", {"exercise_target": {"intended": True},
                                         "secret_future": "LEAK"}),
                        item("daily_self_report", {"exercise": {"duration_min": 18},
                                                   "mood": {"secret": "LEAK"}}),
                        item("objective_log", {"signals": {"unrelated": "LEAK"}}),
                        item("device_log", {"signals": {
                            "activity_tracker": {"active_minutes": 32},
                            "phone_usage": {"secret": "LEAK"}}})]}
        scoped = scoped_teacher_input(isolated)
        self.assertEqual([s["source_type"] for s in scoped["sources"]],
                         ["planner", "daily_self_report", "device_log"])
        self.assertEqual(len(scoped["sources"]), 3)
        for source in scoped["sources"]:
            raw = b64decode(source["payload_base64"])
            self.assertNotIn(b"LEAK", raw)
            self.assertEqual(sha256(raw).hexdigest(), source["field_sha256"])
        self.assertNotIn(b"LEAK", json.dumps(scoped).encode())
        self.assertFalse(scoped["training_admitted"])


if __name__ == "__main__":
    unittest.main()
