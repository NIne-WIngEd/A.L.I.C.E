"""Source packets may become incomplete drafts, never fabricated gold."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat
import tempfile
import unittest

from scripts.mfm.build_v16_source_review_packets import build_packets
from scripts.mfm.prepare_v16_annotation_drafts import prepare_drafts, write_private_new


def _raw(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class AnnotationDraftIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        families = []
        for host in ("bench_alpha", "bench_beta"):
            home = self.root / host / "structural_sources"
            home.mkdir(parents=True)
            sources = []
            for kind in ("planner", "daily_self_report", "objective_log", "device_log",
                         "profile_ltm"):
                common = {"schema_version": "v2.source_projection",
                          "persona_id": host, "source_type": kind}
                if kind == "profile_ltm":
                    document = {**common, "facts": {
                        "identity": {"name": host}, "traits": {},
                        "routine_snapshot": {}},
                        "anchor_window": {"start_day_index": 1,
                                          "end_day_index": 1,
                                          "total_days": 2, "staleness_days": 1}}
                    units = 3
                else:
                    records = []
                    for day in (1, 2):
                        entry = {"day_index": day, "date": f"2026-01-0{day}"}
                        if kind == "planner":
                            entry.update({name: {} for name in (
                                "sleep_target", "exercise_target", "work_target",
                                "diet_target", "social_target", "wellbeing_target")})
                            entry["persona_context"] = {"note": f"{host} plan {day}"}
                        elif kind == "daily_self_report":
                            entry.update({name: {} for name in (
                                "sleep", "exercise", "diet", "work", "social")})
                            entry["mood"] = {"note": f"{host} observed {day}"}
                        else:
                            entry.update({"available": False, "signals": {}})
                        records.append(entry)
                    document = {**common, "records": records}
                    units = len(records)
                relative = f"{host}/structural_sources/{kind}.json"
                payload = _raw(document)
                (self.root / relative).write_bytes(payload)
                sources.append({"source_type": kind, "path": relative,
                                "sha256": sha256(payload).hexdigest(), "units": units})
            families.append({"persona_id": host, "host_family": host,
                             "generator_family": "synthetic-family-one",
                             "sources": sources})
        inventory = {"schema": "mfm-source-annotation-inventory-v1",
                     "status": "source_candidate_only_no_formation_labels_or_review",
                     "training_admitted": False,
                     "seed": 7, "upstream_commit": "commit-seven",
                     "source_families": families}
        self.inventory = self.root / "inventory.json"
        self.inventory.write_bytes(_raw(inventory))
        self.inventory_sha = sha256(self.inventory.read_bytes()).hexdigest()
        self.packets = self.root / "packets.jsonl"
        self.packets.write_bytes(build_packets(self.root, self.inventory,
                                               expected_inventory_sha256=self.inventory_sha,
                                               days=(1, 2)))

    def _drafts(self) -> bytes:
        return prepare_drafts(self.packets,
                              packet_sha256=sha256(self.packets.read_bytes()).hexdigest(),
                              dataset_dir=self.root, inventory_path=self.inventory,
                              inventory_sha256=self.inventory_sha)

    def test_four_verified_windows_yield_only_incomplete_private_drafts(self) -> None:
        output = self._drafts()
        rows = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual([len(row["source_anchor_inventory"]) for row in rows],
                         [4, 9, 4, 9])
        self.assertEqual(rows[0]["provenance"]["packet_file_sha256"],
                         sha256(self.packets.read_bytes()).hexdigest())
        self.assertEqual(rows[1]["provenance"]["lineage"]["generator_family"],
                         "synthetic-family-one")
        for row in rows:
            self.assertFalse(row["training_admitted"])
            self.assertEqual(len(row["dimension_review"]), 10)
            self.assertEqual(set(row["dimension_review"].values()), {"unknown"})
            self.assertTrue(all(anchor["original_file_byte_anchor"] == {
                "start": None, "end": None} for anchor in row["source_anchor_inventory"]))
            self.assertFalse({"target", "split", "rights", "reviews", "signatures"} & set(row))
        self.assertNotIn(b"plan 1", output)
        self.assertNotIn(b"observed 2", output)

        target = self.root / "private-drafts.jsonl"
        write_private_new(target, output)
        self.assertEqual(target.read_bytes(), output)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            write_private_new(target, b"changed")
        self.assertEqual(target.read_bytes(), output)

    def test_changed_row_or_duplicate_cannot_gain_admission_by_rehashing(self) -> None:
        original = self.packets.read_bytes()
        first, *rest = original.splitlines()
        row = json.loads(first)
        row["sources"][0]["record"]["persona_context"]["note"] = "fabricated"
        self.packets.write_bytes(_raw(row) + b"\n".join(rest) + b"\n")
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            self._drafts()
        self.packets.write_bytes(original + first + b"\n")
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            self._drafts()

    def test_exact_pins_and_original_source_bytes_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            prepare_drafts(self.packets, packet_sha256="0" * 64,
                           dataset_dir=self.root, inventory_path=self.inventory,
                           inventory_sha256=self.inventory_sha)
        source = self.root / "bench_alpha/structural_sources/planner.json"
        source.write_bytes(source.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "source SHA-256 differs"):
            self._drafts()


if __name__ == "__main__":
    unittest.main()
