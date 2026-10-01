"""Contract-level checks for the train-only fictional chronology."""

from __future__ import annotations

from base64 import b64decode
from collections import Counter
import unittest

from scripts.mfm.build_v16_fictional_longitudinal_curriculum import (
    HOST_COUNT, SCENARIOS, build_one, spec,
)


class LongitudinalCurriculumTests(unittest.TestCase):
    def test_all_histories_are_grounded_and_train_only(self):
        dimensions = Counter()
        kinds = Counter()
        hosts = set()
        for i in range(HOST_COUNT):
            for scenario in SCENARIOS:
                row = build_one(spec(i, scenario))
                self.assertEqual(row["split"], "train")
                self.assertEqual(row["lineage"]["history_id"],
                                 row["context"]["base_context"]["scope"]["host_instance_id"])
                hosts.add(row["lineage"]["history_id"])
                dimensions.update(name for name, value in
                                  row["target"]["adjudications"].items()
                                  if value == "present")
                kinds.update(item["proposal"]["kind"] for item in row["target"]["proposals"])
        self.assertEqual(len(hosts), HOST_COUNT)
        self.assertEqual(len(dimensions), 10)
        self.assertGreaterEqual(len(kinds), 10)

    def test_counterfactual_norm_retains_early_evidence(self):
        yes = build_one(spec(0, "norm-agreed"))
        no = build_one(spec(0, "norm-unsettled"))
        self.assertEqual(yes["sources"][0], no["sources"][0])
        self.assertNotEqual(yes["sources"][1], no["sources"][1])
        self.assertEqual([p["proposal"]["kind"] for p in yes["target"]["proposals"]],
                         ["relationship_norm"])
        self.assertEqual(no["target"]["proposals"], [])

    def test_withdrawn_payload_absent_and_no_false_completion(self):
        row = build_one(spec(9, "revocation-deletion"))
        source_text = "\n".join(b64decode(item["content_b64"]).decode()
                                for item in row["sources"])
        self.assertIn("no payload is available", source_text)
        self.assertIn("I have not confirmed", source_text)
        self.assertEqual({item["proposal"]["kind"] for item in row["target"]["proposals"]},
                         {"revocation_request", "deletion_request"})
        self.assertIn("abstain", [item["action"] for item in row["target"]["dispositions"]])


if __name__ == "__main__":
    unittest.main()
