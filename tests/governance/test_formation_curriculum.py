from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mfm/assemble_formation_curriculum.py"
spec = importlib.util.spec_from_file_location("mfm_formation_curriculum", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def _complete_domains(streams):
    for item in streams["planner"]:
        item.update(work_target={"hours_limit": 6},
                    diet_target={"coffee_limit": 2},
                    social_target={"intent": True},
                    wellbeing_target={"mood_goal": 4})
    for item in streams["daily_self_report"]:
        item.update(work={"hours": 7}, diet={"coffee_cups": 1},
                    social={"activities": []}, mood={"overall": 3})
    return streams


class FormationCurriculumTests(unittest.TestCase):
    def test_plan_report_and_conflicting_tracker_do_not_conflate_outcome(self):
        date = "2026-01-03"
        streams = {
            "planner": [{"date": date, "day_index": 1,
                         "exercise_target": {"intended": True, "type": "walking",
                                             "duration_min": 30},
                         "sleep_target": {"duration_h": 7.5}}],
            "daily_self_report": [{"date": date, "day_index": 1,
                                   "exercise": {"did_exercise": True},
                                   "sleep": {"duration_h": 6.8}}],
            "device_log": [{"date": date, "day_index": 1, "available": True,
                            "signals": {"activity_tracker": {"workout_detected": False}}}],
        }
        rows = list(module.generate_rows("bench_test_01", _complete_domains(streams),
                                         {k: "a" * 64 for k in streams}))
        self.assertEqual(len(rows), 3)
        self.assertEqual([len(r["sources"]) for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["expected"][0]["kind"], "goal")
        self.assertEqual(rows[0]["dispositions"][1]["action"], "defer")
        self.assertEqual(rows[1]["expected"][1]["kind"], "host_observation")
        self.assertEqual(rows[-1]["expected"][2]["kind"], "uncertainty")
        self.assertEqual(next(d for d in rows[-1]["dispositions"] if
                              d["scope_ref"] == "verified_exercise_outcome")["action"],
                         "defer")
        self.assertEqual(len(rows[0]["expected"]), 6)
        self.assertEqual(len(rows[1]["expected"]), 12)
        for row in rows:
            self.assertEqual(row["authorization_id"], module.AUTHORIZATION)
            for proposal in row["expected"]:
                for anchor in proposal["anchors"]:
                    source = next(x for x in row["sources"] if x["ref_id"] == anchor["ref_id"])
                    raw = source["text"].encode("utf-8")
                    self.assertGreater(anchor["start_byte"], 0)
                    self.assertLess(anchor["end_byte"], len(raw))
            for source in row["sources"]:
                self.assertIsNone(source["recorded_at"])
                self.assertTrue(source["observed_at"].endswith("T00:00:00Z"))

    def test_missing_tracker_does_not_invent_device_evidence(self):
        date = "2026-01-03"
        streams = {
            "planner": [{"date": date, "day_index": 1,
                         "exercise_target": {"intended": False},
                         "sleep_target": {"duration_h": 8.0}}],
            "daily_self_report": [{"date": date, "day_index": 1,
                                   "exercise": {"did_exercise": False},
                                   "sleep": {"duration_h": 7.1}}],
            "device_log": [{"date": date, "day_index": 1, "available": False,
                            "signals": {}}],
        }
        rows = list(module.generate_rows("bench_test_02", _complete_domains(streams),
                                         {k: "b" * 64 for k in streams}))
        self.assertEqual(len(rows), 2)
        self.assertFalse(any("device-log" in str(r["sources"]) for r in rows))

    def test_month_history_is_bounded_and_day_15_cannot_see_future(self):
        from datetime import date, timedelta
        days = [(date(2026, 1, 3) + timedelta(days=i)).isoformat()
                for i in range(30)]
        streams = {"planner": [], "daily_self_report": [], "device_log": []}
        for index, day in enumerate(days, 1):
            streams["planner"].append({
                "date": day, "day_index": index,
                "exercise_target": {"intended": True, "type": "walk", "duration_min": 30},
                "sleep_target": {"duration_h": 8}, "work_target": {"hours_limit": 6},
                "diet_target": {"coffee_limit": 2}, "social_target": {"intent": True},
                "wellbeing_target": {"mood_goal": 4}})
            streams["daily_self_report"].append({
                "date": day, "day_index": index,
                "exercise": {"did_exercise": index <= 15}, "sleep": {"duration_h": 7},
                "work": {"hours": 7}, "diet": {"coffee_cups": 1},
                "social": {"activities": []}, "mood": {"overall": 3}})
            streams["device_log"].append({"date": day, "day_index": index,
                                           "available": False, "signals": {}})
        rows = list(module.generate_rows("bench_test_30", streams,
                                         {k: "c" * 64 for k in streams}))
        early, late = (next(r for r in rows if r["case_id"].endswith(suffix))
                       for suffix in ("asof-15", "asof-30"))
        self.assertEqual(len(early["sources"]), 15)
        self.assertEqual(len(late["sources"]), 30)
        self.assertEqual(len(late["expected"]), 2)
        self.assertIn("15 of 15", late["values"][late["expected"][0]["value_ref"]])
        self.assertIn("0 of 15", late["values"][late["expected"][1]["value_ref"]])
        self.assertEqual(late["expected"][0]["valid_from"], days[0] + "T00:00:00Z")
        self.assertEqual(late["expected"][0]["valid_to"], days[14] + "T00:00:00Z")
        self.assertEqual(late["expected"][1]["valid_from"], days[15] + "T00:00:00Z")
        self.assertEqual(late["expected"][1]["valid_to"], days[29] + "T00:00:00Z")
        from cognitive_kernel.formation_gold import compile_formation_case
        compiled = compile_formation_case(late)
        self.assertTrue(all(ref.temporal_granularity == "day"
                            and ref.recorded_at is None
                            for ref in compiled.gold.context.evidence))
        self.assertTrue(all(p.temporal_granularity == "day" and p.valid_to
                            for p in compiled.gold.expected))


if __name__ == "__main__":
    unittest.main()
