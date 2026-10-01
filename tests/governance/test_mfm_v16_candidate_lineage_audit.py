"""Candidate lineage cannot silently become gold or a sealed split."""

from __future__ import annotations

from hashlib import sha256
import json
import unittest

from scripts.mfm.audit_v16_candidate_lineage import (
    audit_candidate_lineage, summarize_declared_lineage,
)
from tests.governance import test_mfm_v16_annotation_draft_intake as draft_fixture


class CandidateLineageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = draft_fixture.AnnotationDraftIntakeTests(
            "test_four_verified_windows_yield_only_incomplete_private_drafts")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _audit(self, digest: str | None = None) -> dict:
        fixture = self.fixture
        drafts = fixture.root / "drafts.jsonl"
        if not drafts.exists():
            drafts.write_bytes(fixture._drafts())
        return audit_candidate_lineage(
            drafts, drafts_sha256=digest or sha256(drafts.read_bytes()).hexdigest(),
            packet_path=fixture.packets,
            packet_sha256=sha256(fixture.packets.read_bytes()).hexdigest(),
            dataset_dir=fixture.root, inventory_path=fixture.inventory,
            inventory_sha256=fixture.inventory_sha)

    def test_replayed_drafts_have_one_connected_generator_component(self) -> None:
        report = self._audit()
        self.assertEqual(report["candidate_rows"], 4)
        self.assertEqual(report["host_families"], 2)
        self.assertEqual(report["source_anchor_inventory_entries"], 26)
        self.assertEqual(len(report["connected_declared_lineage_components"]), 1)
        self.assertEqual(report["declared_generator_families"], ["synthetic-family-one"])
        self.assertTrue(report["three_way_split_declared_lineage_blocked"])
        self.assertFalse(report["training_admitted"])
        self.assertFalse(report["sealed_final_created"])
        self.assertFalse(report["independent_family_provenance_established"])

    def test_rehashed_modified_draft_still_fails_source_replay(self) -> None:
        self._audit()
        drafts = self.fixture.root / "drafts.jsonl"
        rows = [json.loads(line) for line in drafts.read_bytes().splitlines()]
        rows[0]["dimension_review"]["episode"] = "present"
        drafts.write_bytes(b"".join(json.dumps(row, sort_keys=True,
                                             separators=(",", ":")).encode() + b"\n"
                                    for row in rows))
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            self._audit()

    def test_declared_shared_ancestor_joins_differently_named_families(self) -> None:
        def row(i: int, family: str, commit: str, source: str) -> dict:
            return {"packet_id": f"packet-{i}", "provenance": {"lineage": {
                "generator_family": family, "host_family": f"host-{i}",
                "lineage_group": f"{family}/host-{i}", "upstream_commit": commit}},
                "source_anchor_inventory": [{"source_id": f"source-{i}",
                                             "source_file_sha256": source}]}
        rows = [row(1, "family-a", "same-generator-code", "a" * 64),
                row(2, "family-b", "same-generator-code", "b" * 64),
                row(3, "family-c", "different-code", "b" * 64),
                row(4, "family-d", "independent-code", "d" * 64)]
        result = summarize_declared_lineage(rows)
        self.assertEqual(len(result["connected_declared_lineage_components"]), 2)
        self.assertTrue(result["three_way_split_declared_lineage_blocked"])
        rows.append(row(5, "family-e", "another-code", "e" * 64))
        result = summarize_declared_lineage(rows)
        self.assertEqual(len(result["connected_declared_lineage_components"]), 3)
        self.assertFalse(result["three_way_split_declared_lineage_blocked"])
        self.assertFalse(result["independent_family_provenance_established"])
        self.assertFalse(result["training_admitted"])


if __name__ == "__main__":
    unittest.main()
