"""Role curriculum source binding and counterfactual checks; no admission."""

from __future__ import annotations

from base64 import b64decode
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cognitive_kernel.formation_learning_v16 import (
    FULL_ROLE_DIMENSIONS, learning_example_v16_from_record,
    supervised_output_record_v16,
)
from scripts.mfm.build_v16_role_teacher_candidates import SPECS, build_one, render
from scripts.mfm.build_v16_role_teacher_candidates import ROOT


class RoleTeacherCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [json.loads(line) for line in render().splitlines()]
        cls.by_id = {row["case_id"]: row for row in cls.rows}

    def test_six_scoped_candidates_are_anchored_and_unqualified(self) -> None:
        self.assertEqual(len(self.rows), 6)
        self.assertEqual(len(self.by_id), 6)
        for row in self.rows:
            example = learning_example_v16_from_record(row)
            supervised_output_record_v16(example)
            self.assertFalse(row["status"]["training_admitted"])
            self.assertEqual(row["status"]["rights"], "unverified")
            self.assertEqual(row["split"], "train")
            self.assertEqual(set(row["target"]["adjudications"]), FULL_ROLE_DIMENSIONS)
            source_by_id = {item["ref_id"]: b64decode(item["content_b64"])
                            for item in row["sources"]}
            for evidence in row["context"]["base_context"]["evidence"]:
                self.assertEqual(sha256(source_by_id[evidence["ref_id"]]).hexdigest(),
                                 evidence["content_digest"])
            for proposal in row["target"]["proposals"]:
                for anchor in proposal["proposal"]["anchors"]:
                    raw = source_by_id[anchor["ref_id"]]
                    span = raw[anchor["start_byte"]:anchor["end_byte"]]
                    self.assertTrue(span)
                    self.assertEqual(raw.count(span), 1)

    def test_relationship_pair_changes_only_later_decisive_source(self) -> None:
        yes = self.by_id["role-norm-agreed"]
        no = self.by_id["role-norm-unsettled"]
        yes_e = yes["context"]["base_context"]["evidence"]
        no_e = no["context"]["base_context"]["evidence"]
        self.assertEqual(yes_e[0], no_e[0])
        self.assertNotEqual(yes_e[1]["ref_id"], no_e[1]["ref_id"])
        self.assertEqual(yes["sources"][0], no["sources"][0])
        self.assertNotEqual(yes["sources"][1], no["sources"][1])
        changed = deepcopy(no_e)
        for field in ("ref_id", "source_item_ref", "content_digest"):
            changed[1][field] = yes_e[1][field]
        self.assertEqual(yes_e, changed)
        self.assertEqual(yes["lineage"], no["lineage"])
        self.assertEqual(len(yes["target"]["proposals"]), 1)
        self.assertEqual(no["target"]["proposals"], [])
        self.assertEqual(yes["target"]["adjudications"]["relationship"], "present")
        self.assertEqual(no["target"]["adjudications"]["relationship"], "negative")
        ids = {row["ref_id"]: row["content_b64"]
               for case in self.rows for row in case["sources"]}
        for case in self.rows:
            for item in case["sources"]:
                self.assertEqual(ids[item["ref_id"]], item["content_b64"])

    def test_revocation_excludes_source_payload_and_no_completed_deletion(self) -> None:
        before = self.by_id["role-source-person-before-withdrawal"]
        after = self.by_id["role-source-person-revocation"]
        before_note = b64decode(before["sources"][0]["content_b64"])
        after_sources = [b64decode(item["content_b64"])
                         for item in after["sources"]]
        self.assertEqual(before["context"]["base_context"]["scope"],
                         after["context"]["base_context"]["scope"])
        self.assertNotEqual(before["sources"][0]["ref_id"],
                            after["sources"][0]["ref_id"])
        self.assertEqual(before["sources"][1], after["sources"][1])
        self.assertEqual(before["context"]["base_context"]["evidence"][0]
                         ["source_item_ref"], "mira-note-2023")
        self.assertEqual(after["context"]["base_context"]["evidence"][0]
                         ["source_item_ref"], "mira-note-2023")
        self.assertIn(b"tide chart before each boat trip", before_note)
        self.assertTrue(all(b"tide chart before each boat trip" not in source
                            for source in after_sources))
        self.assertIn(b"withheld", after_sources[0])
        self.assertEqual(before["target"]["adjudications"]["source_person"], "present")
        self.assertEqual(after["target"]["adjudications"]["source_person"], "negative")
        proposals = [p["proposal"] for p in after["target"]["proposals"]]
        self.assertEqual({p["kind"] for p in proposals},
                         {"host_observation", "revocation_request", "deletion_request"})
        revocation = next(p for p in proposals if p["kind"] == "revocation_request")
        self.assertEqual(revocation["target_refs"], ["mira-note-2023"])
        self.assertEqual(revocation["evidence_refs"], [after["sources"][2]["ref_id"]])
        self.assertEqual(len(revocation["anchors"]), 1)
        withdrawal_bytes = b64decode(after["sources"][2]["content_b64"])
        anchor = revocation["anchors"][0]
        self.assertIn(b"mira-note-2023",
                      withdrawal_bytes[anchor["start_byte"]:anchor["end_byte"]])
        self.assertFalse(any(p["subject_ref"].endswith("-mira") for p in proposals))
        actions = {(d["scope_ref"], d["action"])
                   for d in after["target"]["dispositions"]}
        self.assertIn(("mira_attribution", "abstain"), actions)
        self.assertIn(("removal_completed", "abstain"), actions)
        sensitivity_hints = {row["case_id"]: [p["sensitivity_hint"]
                              for p in row["target"]["proposals"]
                              if p["sensitivity_hint"] is not None]
                             for row in self.rows}
        self.assertEqual(sensitivity_hints["role-source-person-revocation"],
                         ["highly_sensitive", "highly_sensitive"])
        self.assertTrue(all(not hints for case_id, hints in sensitivity_hints.items()
                            if case_id != "role-source-person-revocation"))

    def test_correction_and_mission_do_not_promote_request_or_plan_to_result(self) -> None:
        correction = self.by_id["role-correct-old-review"]
        c = correction["target"]["proposals"][0]["proposal"]
        self.assertEqual(c["kind"], "correction_request")
        self.assertEqual(c["target_refs"], ["review-done-apr03"])
        self.assertEqual(c["contradicts"], ["review-done-apr03"])
        self.assertEqual(len(correction["target"]["proposals"]), 1)
        mission = self.by_id["role-mission-workspace-outcome"]
        m = [p["proposal"] for p in mission["target"]["proposals"]]
        self.assertEqual([p["kind"] for p in m], ["goal", "outcome", "goal"])
        self.assertEqual(m[1]["epistemic_status"], "observation")
        self.assertNotEqual(m[0]["evidence_refs"], m[1]["evidence_refs"])
        self.assertIn("not completed", m[1]["value_text"])
        self.assertEqual([d["action"] for d in mission["target"]["dispositions"]],
                         ["propose", "propose", "propose"])

    def test_missing_or_ambiguous_quote_fails_closed(self) -> None:
        case = deepcopy(SPECS[2])
        case["proposals"][0]["quotes"]["correction"] = "review was finished"
        with self.assertRaisesRegex(ValueError, "quote absent or ambiguous"):
            build_one(case)
        case = deepcopy(SPECS[2])
        case["sources"]["correction"]["text"] += " " + next(
            iter(case["proposals"][0]["quotes"].values()))
        with self.assertRaisesRegex(ValueError, "quote absent or ambiguous"):
            build_one(case)

    def test_invalid_scoped_adjudication_fails_closed(self) -> None:
        case = deepcopy(SPECS[0])
        case["adjudications"]["relationship"] = "negative"
        with self.assertRaisesRegex(Exception, "relationship"):
            build_one(case)

    def test_output_is_private_exclusive_and_checkable(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            custody = Path(parent) / "custody"
            custody.mkdir(mode=0o700)
            output = custody / "roles.jsonl"
            cmd = [sys.executable,
                   str(ROOT / "scripts/mfm/build_v16_role_teacher_candidates.py"),
                   "--output", str(output)]
            env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}")
            first = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                                   text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(output.read_bytes(), render())
            again = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                                   text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(output.read_bytes(), render())
            check = subprocess.run(cmd + ["--check"], cwd=ROOT, env=env,
                                   capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            public = Path(parent) / "public"
            public.mkdir(mode=0o755)
            refused = subprocess.run(cmd[:-1] + [str(public / "bad.jsonl")],
                                     cwd=ROOT, env=env, capture_output=True,
                                     text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse((public / "bad.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
