from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import (
    learning_example_v16_from_record, supervised_output_record_v16,
)


ROOT = Path(__file__).resolve().parents[2]


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "mfm" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


builder = module("mfm_v16_builder", "build_v16_authoring_seed.py")
auditor = module("mfm_v16_auditor", "audit_v16_authoring_corpus.py")
DATA = ROOT / "benchmarks" / "mfm" / "v16_authoring_seed.jsonl"
PIN = "8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22"


class MFMV16AuthoringSeedTests(unittest.TestCase):
    def test_exact_pack_has_complete_synthetic_labels_and_blocked_admission(self):
        self.assertEqual(sha256(DATA.read_bytes()).hexdigest(), PIN)
        self.assertEqual(builder.render(), DATA.read_bytes())
        report = auditor.audit(DATA, PIN)
        self.assertEqual(report["case_counts"], {"train": 15, "development": 6})
        self.assertEqual(report["full_role_admission"], "blocked")
        self.assertEqual(set(report["controlled_input_pairs"]),
                         {"temperament-evidence-flip", "social-boundary-dev-flip"})
        self.assertEqual(report["linked_history_cases"]["dove-writing-hours"],
                         ["v16-seed-19", "v16-seed-20"])
        self.assertEqual(report["cross_split_source_digest_count"], 0)
        for split in ("train", "development"):
            self.assertTrue(all(n > 0 for n in report["positive_case_counts"][split].values()))
        self.assertEqual(report["proposal_kind_counts"]["train"]["metacognitive_signal"], 1)
        self.assertEqual(report["proposal_kind_counts"]["train"]["procedural_skill"], 1)

    def test_pair_changes_only_decisive_source_bytes_not_metadata(self):
        rows = [json.loads(line) for line in DATA.read_text().splitlines()]
        pair = {row["case_id"]: row for row in rows if row["case_id"] in
                {"v16-seed-02", "v16-seed-03"}}
        decisive = auditor._pair_isolation("temperament-evidence-flip",
                                            pair["v16-seed-02"], pair["v16-seed-03"])
        self.assertTrue(decisive.endswith("-later"))
        altered = deepcopy(pair["v16-seed-03"])
        altered["context"]["base_context"]["evidence"][0]["observed_at"] = (
            "2026-01-15T09:00:00.000000Z")
        with self.assertRaisesRegex(CognitiveKernelContractError, "other input metadata"):
            auditor._pair_isolation("temperament-evidence-flip", pair["v16-seed-02"], altered)

    def test_absent_label_is_not_converted_to_negative(self):
        row = json.loads(DATA.read_text().splitlines()[0])
        row["target"]["adjudications"].pop("episode")
        with self.assertRaisesRegex(CognitiveKernelContractError, "adjudications"):
            learning_example_v16_from_record(row)
        row["target"]["adjudications"]["episode"] = "unknown"
        with self.assertRaisesRegex(CognitiveKernelContractError, "unadjudicated target"):
            learning_example_v16_from_record(row)
        row = json.loads(DATA.read_text().splitlines()[7])
        row["target"]["adjudications"]["episode"] = "unknown"
        example = learning_example_v16_from_record(row)
        with self.assertRaisesRegex(CognitiveKernelContractError, "unknown adjudication"):
            supervised_output_record_v16(example)


if __name__ == "__main__":
    unittest.main()
