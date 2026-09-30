"""Tiny source-window fixture; no simulator QA or authored targets are inputs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.mfm.build_v16_source_review_packets import build_packets


def _raw(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class SourceReviewPacketTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.families = []
        for host in ("bench_alpha", "bench_beta"):
            home = self.root / host
            structural = home / "structural_sources"
            structural.mkdir(parents=True)
            (home / "event_table.json").write_text('"SIMULATOR_SECRET"')
            (home / "ground_truth.json").write_text('"QA_SECRET"')
            (structural / "generation_metadata.json").write_text('"KNOBS_SECRET"')
            (home / "v15_target.json").write_text('"AUTHORED_TARGET_SECRET"')
            sources = []
            common = {"schema_version": "v2.source_projection",
                      "persona_id": host, "persona_name": "Fixture"}
            records = {
                "planner": [{"day_index": day, "date": f"2026-01-0{day}",
                             "sleep_target": {}, "exercise_target": {},
                             "work_target": {}, "diet_target": {},
                             "social_target": {}, "wellbeing_target": {},
                             "persona_context": {"note": f"{host} plan {day}"}}
                            for day in (1, 2, 3)],
                "daily_self_report": [
                    {"day_index": day, "date": f"2026-01-0{day}",
                     "sleep": {}, "exercise": {}, "diet": {}, "work": {},
                     "social": {}, "mood": {"note": f"{host} observed {day}"}}
                    for day in (1, 2, 3)],
                "objective_log": [
                    {"day_index": day, "date": f"2026-01-0{day}",
                     "available": True, "signals": {"objective": day}}
                    for day in (1, 2, 3)],
                "device_log": [
                    {"day_index": day, "date": f"2026-01-0{day}",
                     "available": False, "signals": {}}
                    for day in (1, 2, 3)],
            }
            for kind, rows in records.items():
                sources.append(self._write_source(host, kind, {**common,
                                                              "source_type": kind,
                                                              "records": rows},
                                                  len(rows)))
            profile = {**common, "source_type": "profile_ltm",
                       "difficulty_type": "KNOBS_SECRET",
                       "anchor_window": {"start_day_index": 1,
                                         "end_day_index": 2,
                                         "total_days": 3, "staleness_days": 1},
                       "facts": {"identity": {"name": "Fixture"},
                                 "traits": {}, "routine_snapshot": {"value": 42}}}
            sources.append(self._write_source(host, "profile_ltm", profile, 3))
            self.families.append({"host_family": host, "persona_id": host,
                                  "generator_family": "generator-one",
                                  "ancestor_event_table_sha256": "a" * 64,
                                  "sources": sources})
        self._inventory()

    def _write_source(self, host: str, kind: str, doc: dict, units: int) -> dict:
        relative = f"{host}/structural_sources/{kind}.json"
        raw = _raw(doc)
        (self.root / relative).write_bytes(raw)
        return {"source_type": kind, "path": relative,
                "sha256": sha256(raw).hexdigest(), "units": units}

    def _inventory(self) -> None:
        inventory = {"schema": "mfm-source-annotation-inventory-v1",
                     "status": "source_candidate_only_no_formation_labels_or_review",
                     "training_admitted": False, "seed": 9, "upstream_commit": "commit-one",
                     "source_families": self.families,
                     "attribution": "INVENTORY_ONLY_SECRET"}
        self.inventory_path = self.root / "mfm_source_inventory.json"
        raw = _raw(inventory)
        self.inventory_path.write_bytes(raw)
        self.pin = sha256(raw).hexdigest()

    def build(self, **options) -> bytes:
        return build_packets(self.root, self.inventory_path,
                             expected_inventory_sha256=self.pin,
                             days=options.get("days", (1, 2, 3)),
                             hosts=options.get("hosts", ()))

    def test_cumulative_windows_bind_exact_sources_and_group_host_lineage(self):
        first = self.build(days=(3, 1, 2))
        self.assertEqual(first, self.build(days=(2, 3, 1)))
        rows = [json.loads(line) for line in first.splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual([(row["lineage"]["host_family"], row["window"]["end_day_index"])
                          for row in rows], [(host, day)
                                            for host in ("bench_alpha", "bench_beta")
                                            for day in (1, 2, 3)])
        self.assertEqual([len(row["sources"]) for row in rows], [4, 8, 13, 4, 8, 13])
        self.assertEqual(rows[0]["lineage"]["lineage_group"],
                         rows[2]["lineage"]["lineage_group"])
        self.assertNotEqual(rows[2]["lineage"]["lineage_group"],
                            rows[5]["lineage"]["lineage_group"])
        self.assertEqual(rows[2]["lineage"]["generator_family"],
                         rows[5]["lineage"]["generator_family"])
        self.assertTrue(all(row["status"] == "candidate_unreviewed" and
                            row["training_admitted"] is False for row in rows))
        self.assertEqual([e["observed_day_index"] for e in rows[1]["sources"]],
                         [1, 1, 1, 1, 2, 2, 2, 2])
        self.assertNotIn(b"observed 3", _raw(rows[0]))
        self.assertNotIn(b"observed 3", _raw(rows[1]))
        self.assertIn(b"observed 3", _raw(rows[2]))
        first_source = rows[0]["sources"][0]
        self.assertEqual(first_source["source_locator"], "/records/0")
        self.assertEqual(first_source["source_file_sha256"],
                         self.families[0]["sources"][0]["sha256"])
        self.assertEqual(first_source["unit_sha256"],
                         sha256(json.dumps(first_source["record"], sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(rows[2]["sources"][-1]["source_type"], "profile_ltm")
        self.assertEqual(rows[2]["sources"][-1]["anchor_locator"], "/anchor_window")
        for secret in (b"SIMULATOR_SECRET", b"QA_SECRET", b"KNOBS_SECRET",
                       b"AUTHORED_TARGET_SECRET", b"INVENTORY_ONLY_SECRET",
                       b"event_table", b"ground_truth", b"difficulty_type"):
            self.assertNotIn(secret, first)

    def test_frozen_inventory_and_file_bytes_cannot_drift(self):
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            build_packets(self.root, self.inventory_path,
                          expected_inventory_sha256="0" * 64, days=(1,))
        path = self.root / self.families[0]["sources"][0]["path"]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "source SHA-256 differs"):
            self.build(hosts=("bench_alpha",), days=(1,))

    def test_forbidden_record_fields_and_path_escape_fail_closed(self):
        source = self.families[0]["sources"][0]
        path = self.root / source["path"]
        doc = json.loads(path.read_bytes())
        doc["records"][0]["persona_context"]["ground_truth"] = "leak"
        source["sha256"] = sha256(_raw(doc)).hexdigest()
        path.write_bytes(_raw(doc))
        self._inventory()
        with self.assertRaisesRegex(ValueError, "forbidden simulator or label"):
            self.build(hosts=("bench_alpha",), days=(1,))
        source["path"] = "../outside.json"
        self._inventory()
        with self.assertRaisesRegex(ValueError, "outside the five structural"):
            self.build(hosts=("bench_alpha",), days=(1,))

    def test_future_day_and_duplicate_host_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds source horizon"):
            self.build(days=(1, 7), hosts=("bench_alpha",))
        with self.assertRaisesRegex(ValueError, "duplicated or absent"):
            self.build(days=(1,), hosts=("bench_alpha", "bench_alpha"))


if __name__ == "__main__":
    unittest.main()
