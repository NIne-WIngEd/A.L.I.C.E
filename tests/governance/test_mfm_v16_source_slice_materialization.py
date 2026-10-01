"""Exact original source slices cannot show future records or generator fields."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat
import tempfile
import unittest

from scripts.mfm.build_v16_source_review_packets import build_packets
from scripts.mfm.materialize_v16_source_slices import (
    isolated_packet, materialize, source_slice_locations,
)


def _raw(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class SourceSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dataset = self.root / "dataset"
        home = self.dataset / "bench_alpha/structural_sources"
        home.mkdir(parents=True)
        sources = []
        for kind in ("planner", "daily_self_report", "objective_log", "device_log",
                     "profile_ltm"):
            common = {"schema_version": "v2.source_projection", "persona_id": "bench_alpha",
                      "source_type": kind}
            if kind == "profile_ltm":
                doc = {**common, "facts": {
                    "identity": {"name": "Mónica"}, "traits": {}, "routine_snapshot": {}},
                    "anchor_window": {"start_day_index": 1, "end_day_index": 1,
                                      "total_days": 2, "staleness_days": 1},
                    "difficulty_type": "GENERATOR_ONLY_DO_NOT_SHOW"}
                units = 3
            else:
                records = []
                for day in (1, 2):
                    record = {"day_index": day, "date": f"2026-01-0{day}"}
                    if kind == "planner":
                        record.update({name: {} for name in (
                            "sleep_target", "exercise_target", "work_target",
                            "diet_target", "social_target", "wellbeing_target")})
                        record["persona_context"] = {"note": "café day one" if day == 1
                                                      else "FUTURE_SECRET_DAY_TWO"}
                    elif kind == "daily_self_report":
                        record.update({name: {} for name in (
                            "sleep", "exercise", "diet", "work", "social")})
                        record["mood"] = {"note": f"report day {day}"}
                    else:
                        record.update({"available": False, "signals": {}})
                    records.append(record)
                doc = {**common, "records": records}
                units = len(records)
            relative = f"bench_alpha/structural_sources/{kind}.json"
            content = _raw(doc)
            (self.dataset / relative).write_bytes(content)
            sources.append({"source_type": kind, "path": relative,
                            "sha256": sha256(content).hexdigest(), "units": units})
        inventory = {"schema": "mfm-source-annotation-inventory-v1",
                     "status": "source_candidate_only_no_formation_labels_or_review",
                     "training_admitted": False, "seed": 7, "upstream_commit": "commit-seven",
                     "source_families": [{"persona_id": "bench_alpha",
                                          "host_family": "bench_alpha",
                                          "generator_family": "synthetic-family-one",
                                          "sources": sources}]}
        self.inventory = self.dataset / "inventory.json"
        self.inventory.write_bytes(_raw(inventory))
        self.inventory_sha = sha256(self.inventory.read_bytes()).hexdigest()

    def _packets(self, days=(1,)) -> Path:
        packets = self.root / "packets.jsonl"
        packets.write_bytes(build_packets(self.dataset, self.inventory,
                                          expected_inventory_sha256=self.inventory_sha,
                                          days=days))
        return packets

    def _materialize(self, packets: Path, name="private-slices") -> tuple[Path, dict]:
        dest = self.root / name
        report = materialize(packets, packet_sha256=sha256(packets.read_bytes()).hexdigest(),
                             dataset_dir=self.dataset, inventory_path=self.inventory,
                             inventory_sha256=self.inventory_sha, output_dir=dest)
        return dest, report

    def test_one_day_materializes_verbatim_values_without_future_or_hidden_fields(self) -> None:
        dest, report = self._materialize(self._packets())
        self.assertEqual((report["packets"], report["source_references"],
                          report["unique_fragments"]), (1, 4, 4))
        self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((dest / "fragments.bin").stat().st_mode), 0o600)
        blob = (dest / "fragments.bin").read_bytes()
        self.assertIn("café day one".encode(), blob)
        self.assertNotIn(b"FUTURE_SECRET_DAY_TWO", blob)
        self.assertNotIn(b"GENERATOR_ONLY_DO_NOT_SHOW", blob)
        row = json.loads((dest / "index.jsonl").read_bytes())
        self.assertFalse(row["training_admitted"])
        self.assertEqual(row["rights_status"], "unverified")
        self.assertNotIn("target", row)
        self.assertEqual(len(row["sources"]), 4)
        for source in row["sources"]:
            fragment, = source["fragments"]
            self.assertEqual(fragment["source_locator"], "/records/0")
            orig = (self.dataset / source["parent_source_path"]).read_bytes()
            start = fragment["original_file_byte_anchor"]["start"]
            end = fragment["original_file_byte_anchor"]["end"]
            exact = blob[fragment["blob_offset"]:fragment["blob_offset"] +
                         fragment["byte_length"]]
            self.assertEqual(orig[start:end], exact)
            self.assertEqual(sha256(exact).hexdigest(), fragment["sha256"])

    def test_profile_is_two_exact_spans_and_not_generator_metadata(self) -> None:
        dest, report = self._materialize(self._packets(days=(1, 2)))
        self.assertEqual((report["packets"], report["source_references"],
                          report["unique_fragments"]), (2, 13, 10))
        rows = [json.loads(line) for line in (dest / "index.jsonl").read_bytes().splitlines()]
        self.assertFalse(any(s["source_type"] == "profile_ltm" for s in rows[0]["sources"]))
        profile = [s for s in rows[1]["sources"] if s["source_type"] == "profile_ltm"]
        self.assertEqual(len(profile), 1)
        self.assertEqual([f["source_locator"] for f in profile[0]["fragments"]],
                         ["/facts", "/anchor_window"])
        blob = (dest / "fragments.bin").read_bytes()
        self.assertNotIn(b"GENERATOR_ONLY_DO_NOT_SHOW", blob)
        original = (self.dataset / profile[0]["parent_source_path"]).read_bytes()
        for fragment in profile[0]["fragments"]:
            span = fragment["original_file_byte_anchor"]
            exact = blob[fragment["blob_offset"]:fragment["blob_offset"] +
                         fragment["byte_length"]]
            self.assertEqual(original[span["start"]:span["end"]], exact)
        with self.assertRaisesRegex(ValueError, "new output path"):
            self._materialize(self.root / "packets.jsonl")

    def test_receipt_pinned_single_packet_export_hides_future_global_blob(self) -> None:
        dest, _ = self._materialize(self._packets(days=(1, 2)))
        blob = (dest / "fragments.bin").read_bytes()
        self.assertIn(b"FUTURE_SECRET_DAY_TWO", blob)
        receipt_sha = sha256((dest / "receipt.json").read_bytes()).hexdigest()
        index = [json.loads(line) for line in (dest / "index.jsonl").read_bytes().splitlines()]
        packet = isolated_packet(dest, packet_id=index[0]["packet_id"],
                                 receipt_sha256=receipt_sha)
        teacher_input = json.dumps(packet, sort_keys=True).encode()
        self.assertEqual(len(packet["sources"]), 4)
        self.assertNotIn(b"FUTURE_SECRET_DAY_TWO", teacher_input)
        self.assertNotIn(b"GENERATOR_ONLY_DO_NOT_SHOW", teacher_input)
        self.assertNotIn(b"profile_ltm", teacher_input)
        self.assertNotIn("blob_offset", teacher_input.decode())
        self.assertTrue(packet["host_id"].startswith("host-"))
        self.assertTrue(packet["packet_id"].startswith("packet-"))
        self.assertTrue(all(s["source_id"].startswith("source-") for s in packet["sources"]))
        self.assertNotIn("bench_alpha", teacher_input.decode())
        for secret in ("generator_family", "generator_seed", "upstream_commit",
                       "lineage_group", "inventory_sha256", "packet_sha256",
                       "parent_source_file_sha256", "original_file_byte_anchor",
                       "source_locator", "blob_offset", "source_slice_receipt_sha256"):
            self.assertNotIn(secret, teacher_input.decode())
            self.assertNotIn(secret, packet)
        self.assertIn("generator_family", index[0]["lineage"])
        self.assertIn("parent_source_file_sha256", index[0]["sources"][0])
        self.assertIn("original_file_byte_anchor",
                      index[0]["sources"][0]["fragments"][0])
        self.assertEqual(set(packet["sources"][0]["fragments"][0]),
                         {"slice_handle", "sha256", "payload_base64"})
        with self.assertRaisesRegex(ValueError, "external SHA-256 pin"):
            isolated_packet(dest, packet_id=index[0]["packet_id"],
                            receipt_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            isolated_packet(dest, packet_id="missing", receipt_sha256=receipt_sha)
        (dest / "fragments.bin").write_bytes(blob + b"tamper")
        with self.assertRaisesRegex(ValueError, "differs from pinned receipt"):
            isolated_packet(dest, packet_id=index[0]["packet_id"],
                            receipt_sha256=receipt_sha)

    def test_pin_and_original_file_drift_are_rejected(self) -> None:
        packets = self._packets()
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            materialize(packets, packet_sha256="0" * 64, dataset_dir=self.dataset,
                        inventory_path=self.inventory, inventory_sha256=self.inventory_sha,
                        output_dir=self.root / "reject")
        original = (self.dataset / "bench_alpha/structural_sources/planner.json")
        original.write_bytes(original.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "source SHA-256 differs"):
            self._materialize(packets)

    def test_new_hash_does_not_authorize_a_modified_packet(self) -> None:
        packets = self._packets()
        row = json.loads(packets.read_bytes())
        row["sources"][0]["record"]["persona_context"]["note"] = "invented"
        packets.write_bytes((json.dumps(row, sort_keys=True, separators=(",", ":")) +
                             "\n").encode())
        with self.assertRaisesRegex(ValueError, "pinned reconstruction"):
            self._materialize(packets)

    def test_parser_offsets_survive_utf8_escapes_and_duplicate_key_is_rejected(self) -> None:
        raw = b'{"records":[ {"text":"caf\xc3\xa9 \\"quoted\\""} ],"metadata":{}}'
        offsets = source_slice_locations(raw)
        start, end = offsets["/records/0"]
        self.assertEqual(json.loads(raw[start:end]), {"text": 'café "quoted"'})
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            source_slice_locations(b'{"records":[{"a":1,"a":2}]}')


if __name__ == "__main__":
    unittest.main()
