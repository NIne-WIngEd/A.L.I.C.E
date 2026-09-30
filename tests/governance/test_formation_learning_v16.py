from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning import CURRICULUM_SCHEMA, TARGET_SCHEMA, output_record
from cognitive_kernel.formation_dataset_admission import (
    AdmittedCase, CorpusAdmission, PayloadRef,
)
from cognitive_kernel.formation_learning_v16 import (
    CURRICULUM_SCHEMA_V16, TARGET_SCHEMA_V16, admitted_rows_v16,
    learning_example_v16_from_record, model_input_sha256_v16,
    supervised_output_record_v16,
)
from cognitive_kernel.formation_learning_v16 import FULL_ROLE_DIMENSIONS
from tests.governance.test_formation_semantics_v16 import SOURCE, fixture


def case_record(*, split: str = "train") -> dict:
    context, bundle = fixture()
    output = bundle.output_record()
    output["proposals"][0]["proposal"]["disposition_scope_ref"] = "synthetic-scope"
    return {
        "schema": CURRICULUM_SCHEMA_V16,
        "case_id": "synthetic-codec-case",
        "split": split,
        "authorization_id": "owner-directed-mfm-v16-synthetic",
        "context": context.record(),
        "sources": [{"ref_id": "source-1", "content_b64": b64encode(SOURCE).decode("ascii")}],
        "target": {
            "schema": TARGET_SCHEMA_V16,
            "proposals": output["proposals"],
            "dispositions": [{"scope_ref": "synthetic-scope", "action": "propose",
                              "evidence_refs": ["source-1"], "target_refs": []}],
            "adjudications": {
                "sensitivity": "present", "episode": "present",
                "relationship": "negative", "mission": "present", "workspace": "present",
                "correction": "negative", "source_person": "negative",
                "contradiction": "negative", "outcome": "negative", "abstention": "negative",
            },
        },
        "status": {"fixture_only": True},
        "lineage": {"fixture": "synthetic-codec-unit-test"},
    }


class FormationLearningV16Tests(unittest.TestCase):
    def test_v16_roundtrip_and_exact_source_fingerprint(self):
        row = case_record()
        example = learning_example_v16_from_record(row)
        self.assertEqual(example.case_id, "synthetic-codec-case")
        self.assertEqual(example.opened_sources, (("source-1", SOURCE),))
        self.assertEqual(set(dict(example.adjudications)), FULL_ROLE_DIMENSIONS)
        self.assertEqual(supervised_output_record_v16(example)["schema"],
                         "mfm-formation-output-v1.6")
        self.assertEqual(len(model_input_sha256_v16(example)), 64)
        self.assertEqual(example.context.content_digest(), example.target.context_v16_digest)
        self.assertNotEqual(sha256(SOURCE).hexdigest(), model_input_sha256_v16(example))

    def test_explicit_unknown_never_becomes_an_absence_target(self):
        row = case_record()
        row["target"]["adjudications"]["relationship"] = "unknown"
        example = learning_example_v16_from_record(row)
        self.assertEqual(dict(example.adjudications)["relationship"], "unknown")
        with self.assertRaisesRegex(CognitiveKernelContractError, "unknown adjudication"):
            supervised_output_record_v16(example)

    def test_adjudication_presence_and_negative_are_consistent_with_output(self):
        row = case_record()
        row["target"]["adjudications"]["relationship"] = "present"
        with self.assertRaisesRegex(CognitiveKernelContractError, "marked present"):
            learning_example_v16_from_record(row)
        row = case_record()
        row["target"]["adjudications"]["episode"] = "negative"
        with self.assertRaisesRegex(CognitiveKernelContractError, "marked negative"):
            learning_example_v16_from_record(row)
        row = case_record()
        row["target"]["adjudications"]["mission"] = "unknown"
        with self.assertRaisesRegex(CognitiveKernelContractError, "unadjudicated target"):
            learning_example_v16_from_record(row)
        for name in ("correction", "source_person", "contradiction", "outcome", "abstention"):
            row = case_record()
            row["target"]["adjudications"][name] = "present"
            with self.subTest(name=name), self.assertRaisesRegex(
                    CognitiveKernelContractError, "marked present"):
                learning_example_v16_from_record(row)
        row = case_record()
        del row["target"]["adjudications"]["workspace"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "adjudications"):
            learning_example_v16_from_record(row)

    def test_true_abstention_and_negative_categories_are_distinct(self):
        row = case_record()
        row["target"]["proposals"] = []
        row["target"]["dispositions"][0]["action"] = "abstain"
        row["target"]["adjudications"] = {
            name: "present" if name == "abstention" else "negative"
            for name in FULL_ROLE_DIMENSIONS}
        example = learning_example_v16_from_record(row)
        output = supervised_output_record_v16(example)
        self.assertEqual(output["proposals"], [])
        self.assertEqual(output["dispositions"][0]["action"], "abstain")

    def test_version_and_final_boundaries(self):
        row = case_record()
        row["schema"] = CURRICULUM_SCHEMA
        with self.assertRaisesRegex(CognitiveKernelContractError, "schema or split"):
            learning_example_v16_from_record(row)
        row = case_record()
        row["target"]["schema"] = TARGET_SCHEMA
        with self.assertRaisesRegex(CognitiveKernelContractError, "target schema"):
            learning_example_v16_from_record(row)
        row = case_record(split="final")
        with self.assertRaisesRegex(CognitiveKernelContractError, "FINAL"):
            learning_example_v16_from_record(row, split="final")
        with self.assertRaisesRegex(CognitiveKernelContractError, "FINAL"):
            list(admitted_rows_v16(object(), split="final"))

    def test_source_bytes_order_canonical_encoding_and_grounding(self):
        row = case_record()
        row["sources"][0]["content_b64"] = b64encode(b"tampered").decode("ascii")
        with self.assertRaisesRegex(CognitiveKernelContractError, "digest mismatch"):
            learning_example_v16_from_record(row)
        row = case_record()
        row["sources"][0]["content_b64"] += "="
        with self.assertRaisesRegex(CognitiveKernelContractError, "base64"):
            learning_example_v16_from_record(row)
        row = case_record()
        row["sources"] = []
        with self.assertRaisesRegex(CognitiveKernelContractError, "evidence order"):
            learning_example_v16_from_record(row)
        row = case_record()
        row["target"]["proposals"][0]["mission_target_refs"] = ["unregistered"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "unregistered"):
            learning_example_v16_from_record(row)

    def test_v15_serializer_rejects_v16_target(self):
        example = learning_example_v16_from_record(case_record())
        with self.assertRaisesRegex(CognitiveKernelContractError, "v1.5 output serializer"):
            output_record(example.target)
        extra = case_record()
        extra["target"]["new_unknown"] = None
        with self.assertRaisesRegex(CognitiveKernelContractError, "unsupported or missing"):
            learning_example_v16_from_record(extra)

    def test_structural_admission_rechecks_bytes_and_never_opens_final(self):
        """Directly built unit fixture: no reviewer or rights authenticity claim."""
        row = case_record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_source = SOURCE
            target = {**row["target"], "context": row["context"],
                      "source_ids": ["source-1"]}
            target_bytes = json.dumps(target, sort_keys=True).encode()
            rights_bytes = json.dumps({"host_family": "owner-one",
                                       "source_id": "source-1"}).encode()
            (root / "source.bin").write_bytes(raw_source)
            (root / "target.json").write_bytes(target_bytes)
            (root / "rights.json").write_bytes(rights_bytes)
            case = AdmittedCase(
                "synthetic-codec-case", "train",
                (PayloadRef("source.bin", sha256(raw_source).hexdigest()),),
                PayloadRef("target.json", sha256(target_bytes).hexdigest()),
                (PayloadRef("rights.json", sha256(rights_bytes).hexdigest()),))
            final = AdmittedCase(
                "synthetic-sealed-case", "final",
                (PayloadRef("never-opened-final.bin", "a" * 64),),
                PayloadRef("never-opened-final.json", "b" * 64), ())
            admission = CorpusAdmission("unit-fixture", "f" * 64,
                                        (case,), (), (final,), root)
            self.assertEqual(len(list(admitted_rows_v16(admission, split="train"))), 1)
            original_audit = CorpusAdmission.audit_handoff
            def swap_after_audit(self, **kwargs):
                result = original_audit(self, **kwargs)
                (root / "target.json").write_bytes(b"changed after handoff")
                return result
            with patch.object(CorpusAdmission, "audit_handoff", swap_after_audit):
                with self.assertRaisesRegex(CognitiveKernelContractError,
                                            "consumed payload changed"):
                    list(admitted_rows_v16(admission, split="train"))
            (root / "target.json").write_bytes(target_bytes)
            (root / "source.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(CognitiveKernelContractError, "changed"):
                list(admitted_rows_v16(admission, split="train"))

    def test_structural_admission_rejects_cross_split_byte_reuse(self):
        """Even a hand-built admission object cannot reuse exact source bytes."""
        common = PayloadRef("source.bin", sha256(SOURCE).hexdigest())
        train = AdmittedCase("case-train", "train", (common,),
                             PayloadRef("train-target", "1" * 64), ())
        development = AdmittedCase("case-dev", "development", (
            PayloadRef("another-source", common.sha256),),
            PayloadRef("dev-target", "2" * 64), ())
        admission = CorpusAdmission("unit-fixture", "f" * 64, (train,),
                                    (development,), (), Path("/nonexistent"))
        with self.assertRaisesRegex(CognitiveKernelContractError, "share payload bytes"):
            list(admitted_rows_v16(admission, split="train"))


if __name__ == "__main__":
    unittest.main()
