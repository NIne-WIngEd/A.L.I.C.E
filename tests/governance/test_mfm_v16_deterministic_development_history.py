"""Source and target controls for fictional deterministic development cases."""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_kernel.formation_learning_v16 import (
    FULL_ROLE_DIMENSIONS, learning_example_v16_from_record,
    supervised_output_record_v16,
)
from scripts.mfm.build_v16_development_history import (
    _materialized_files, render,
)


AUTHORIZATION = "owner-directed-mfm-v16-synthetic"


class DeterministicDevelopmentHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.target, cls.receipt = render(AUTHORIZATION)
        cls.packets = [json.loads(line) for line in cls.source.splitlines()]
        cls.rows = [json.loads(line) for line in cls.target.splitlines()]

    def test_exact_sources_and_targets_have_explicit_scoped_labels(self):
        self.assertEqual(len(self.rows), 9)
        self.assertEqual(len({r["case_id"] for r in self.rows}), 9)
        self.assertEqual({r["lineage"]["history_id"] for r in self.rows},
                         {"studio", "garden"})
        positives, kinds = set(), set()
        source_ids = set()
        for packet, row in zip(self.packets, self.rows, strict=True):
            self.assertEqual(packet["case_id"], row["case_id"])
            self.assertEqual(packet["rights_status"], "unverified")
            self.assertFalse(packet["training_admitted"])
            self.assertEqual(row["authorization_id"], AUTHORIZATION)
            example = learning_example_v16_from_record(row, split="development")
            supervised_output_record_v16(example)
            labels = row["target"]["adjudications"]
            self.assertEqual(set(labels), FULL_ROLE_DIMENSIONS)
            self.assertEqual(set(labels.values()), {"present", "negative"})
            positives.update(k for k, state in labels.items() if state == "present")
            kinds.update(p["proposal"]["kind"] for p in row["target"]["proposals"])
            for item in packet["sources"]:
                self.assertLessEqual(item["recorded_at"], packet["as_of"])
                raw = b64decode(item["content_b64"], validate=True)
                self.assertEqual(sha256(raw).hexdigest(), item["sha256"])
                self.assertNotIn(item["ref_id"], source_ids)
                source_ids.add(item["ref_id"])
        self.assertEqual(positives, FULL_ROLE_DIMENSIONS)
        self.assertTrue({"self_observation", "procedural_skill", "behavior_pattern",
                         "decision_rationale", "deletion_request",
                         "revocation_request"}.issubset(kinds))

    def test_withdrawn_content_is_not_reexposed_and_self_is_not_host(self):
        for history in ("studio", "garden"):
            before = next(r for r in self.rows if r["case_id"] == f"det-dev-{history}-plan")
            after = next(r for r in self.rows if r["case_id"] == f"det-dev-{history}-withdrawal")
            self.assertEqual(before["context"]["base_context"]["scope"],
                             after["context"]["base_context"]["scope"])
            earlier = {item["ref_id"]: b64decode(item["content_b64"])
                       for item in before["sources"]}
            later = {item["ref_id"]: b64decode(item["content_b64"])
                     for item in after["sources"]}
            self.assertTrue(any(b"catalog attributes to Ren" in x or
                                b"attributed to Ada" in x for x in earlier.values()))
            self.assertFalse(any(b"catalog attributes to Ren" in x or
                                 b"attributed to Ada" in x for x in later.values()))
            proposals = [p["proposal"] for p in after["target"]["proposals"]]
            self_prop = next(p for p in proposals if p["kind"] == "self_observation")
            self.assertEqual(self_prop["domain"], "self")
            self.assertEqual(self_prop["subject_ref"], f"det-{history}-self")
            self.assertFalse(any(p["kind"] == "source_person_evidence" for p in proposals))
            self.assertTrue(any(d["action"] == "abstain"
                                for d in after["target"]["dispositions"]))

    def test_materialized_bytes_match_assembler_contract_without_rights(self):
        files = _materialized_files(self.source, self.target, self.receipt)
        self.assertEqual(len([p for p in files if p.endswith("/candidate.json")]), 9)
        self.assertFalse(any(p.endswith("/rights.json") for p in files))
        index = json.loads(files["bundle-index.json"])
        self.assertEqual(index["status"], "unreviewed-unadmitted-source-rights-absent")
        for item in index["cases"]:
            for field in ("candidate", "prompt", "response", "converter",
                          "converter_provenance"):
                ref = item[field]
                self.assertEqual(sha256(files[ref["path"]]).hexdigest(), ref["sha256"])
            p = json.loads(files[item["converter_provenance"]["path"]])
            self.assertEqual(p["case_sha256"], item["candidate"]["sha256"])
            self.assertEqual(p["rendered_prompt_sha256"], item["prompt"]["sha256"])
            self.assertEqual(p["generator_output_sha256"], item["response"]["sha256"])
            for source in item["sources"]:
                self.assertEqual(sha256(files[source["path"]]).hexdigest(),
                                 source["sha256"])
                draft = json.loads(files[source["rights_draft"]["path"]])
                self.assertEqual(draft["schema"], "mfm-source-rights-unissued-draft-v1")
                self.assertEqual(draft["source_sha256"], source["sha256"])
                self.assertIsNone(draft["issuer_id"])
                self.assertIsNone(draft["formation_evaluation"])

    def test_authorization_changes_exact_candidate_and_receipt(self):
        other = render("other-owner-authorized-training")
        self.assertNotEqual(other[1], self.target)
        self.assertNotEqual(other[2], self.receipt)
        self.assertEqual(other[0], self.source)

    def test_public_synthetic_source_target_and_receipt_are_reproducible(self):
        root = Path(__file__).resolve().parents[2] / "benchmarks/mfm"
        self.assertEqual((root / "v16_deterministic_development_sources.jsonl").read_bytes(),
                         self.source)
        self.assertEqual((root / "v16_deterministic_development_targets.jsonl").read_bytes(),
                         self.target)
        self.assertEqual((root / "v16_deterministic_development_receipt.json").read_bytes(),
                         self.receipt)

    def test_counterfactual_changes_only_decisive_studio_observation(self):
        agreed = next(r for r in self.rows if r["case_id"] == "det-dev-studio-visit")
        unsettled = next(r for r in self.rows if r["case_id"] ==
                         "det-dev-studio-visit-unsettled")
        self.assertEqual(agreed["lineage"]["counterfactual_pair"],
                         unsettled["lineage"]["counterfactual_pair"])
        self.assertEqual(agreed["lineage"]["parent_case_ids"],
                         unsettled["lineage"]["parent_case_ids"])
        self.assertEqual(b64decode(agreed["sources"][0]["content_b64"]),
                         b64decode(unsettled["sources"][0]["content_b64"]))
        self.assertNotEqual(b64decode(agreed["sources"][1]["content_b64"]),
                            b64decode(unsettled["sources"][1]["content_b64"]))
        self.assertEqual(agreed["target"]["adjudications"]["relationship"], "present")
        self.assertEqual(unsettled["target"]["adjudications"]["relationship"], "negative")
        self.assertTrue(any(d["action"] == "defer"
                            for d in unsettled["target"]["dispositions"]))


if __name__ == "__main__":
    unittest.main()
