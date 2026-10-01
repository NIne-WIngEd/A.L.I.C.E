"""Fictional-only regression for mixed teacher composition; no private corpus."""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_kernel.canonical import canonical_json_bytes
from scripts.mfm.build_v16_authoring_seed import render
from scripts.mfm.compose_v16_mixed_teacher_corpus import compose, AMI_CONVERTER
from tests.governance.test_mfm_v16_owner_teacher_assembly import (
    OwnerTeacherAssemblyTests, raw, write,
)


class MixedTeacherCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture = OwnerTeacherAssemblyTests("test_exact_source_target_and_provenance_admit")
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.synthetic = fixture.root
        self.synthetic_receipt = fixture.run_assembly()
        self.ami = self.root / "ami"
        (self.ami / "issued-rights").mkdir(parents=True)
        self.config = self.root / "ami-config.json"
        cases, receipt_refs = [], []
        selected = [row for row in map(json.loads, render().splitlines())
                    if row["case_id"] in {"v16-seed-02", "v16-seed-04"}]
        self.assertEqual(len(selected), 2)
        for index, candidate in enumerate(selected):
            case_id = f"ami-fixture-{index}-bcd"
            candidate["case_id"] = case_id
            candidate["authorization_id"] = "owner-directed-mfm-v16-public-teacher"
            candidate["split"] = "train"
            directory = self.ami / case_id
            directory.mkdir()
            candidate_ref = write(directory, "teacher-candidate.json", raw(candidate))
            host = candidate["context"]["base_context"]["scope"]["host_instance_id"]
            source_records = []
            for number, source in enumerate(candidate["sources"]):
                data = b64decode(source["content_b64"])
                source_ref = write(directory, f"source-{number}.txt", data)
                source_records.append({"ref_id": source["ref_id"], **source_ref})
                rights = write(self.ami / "issued-rights", f"{source['ref_id']}.json", raw({
                    "schema": "mfm-source-rights-v1",
                    "source_id": source["ref_id"], "source_sha256": source_ref["sha256"],
                    "host_family": host, "issuer_id": "alice-agent-under-owner-direction",
                    "authority_ref": "ami-manual-v162-ccby4-publisher-release",
                    "formation_training": True, "formation_evaluation": True,
                    "model_distribution": True, "revoked": False,
                }))
                receipt_refs.append(rights)
            envelope_ref = write(directory, "envelope.json", raw({
                "authorization_id": candidate["authorization_id"],
                "host_id": host, "source_family": "ami-manual-2017",
                "sources": source_records,
            }))
            prompt = write(directory, "prompt.txt", f"fixture prompt {index}".encode())
            response = write(directory, "teacher-response.json",
                             f"fixture response {index}".encode())
            provenance = write(directory, "teacher-provenance.json", raw({
                "case_id": case_id,
                "case_sha256": candidate_ref["sha256"],
                "converter_sha256": sha256(AMI_CONVERTER.read_bytes()).hexdigest(),
                "rendered_prompt_sha256": prompt["sha256"],
                "teacher_response_sha256": response["sha256"],
            }))
            cases.append({"case_id": case_id, "case_sha256": candidate_ref["sha256"],
                          "envelope_sha256": envelope_ref["sha256"],
                          "prompt_sha256": prompt["sha256"],
                          "response_sha256": response["sha256"],
                          "provenance_sha256": provenance["sha256"]})
        self.config_ref = write(self.root, "ami-config.json", raw({
            "schema": "mfm-v16-ami-public-teacher-expansion-v1", "cases": cases,
        }))
        self.issuance = write(self.ami / "issued-rights", "issuance-manifest.json", raw({
            "schema": "mfm-v16-public-license-issuance-v1",
            "source_count": len(receipt_refs), "receipts": receipt_refs,
        }))

    def compose(self, *, output_name="mixed", issuance_sha=None, config_sha=None):
        return compose(
            synthetic_root=self.synthetic,
            synthetic_intake_sha256=sha256((self.synthetic / "intake.json").read_bytes()).hexdigest(),
            synthetic_manifest_sha256=self.synthetic_receipt["manifest_sha256"],
            ami_root=self.ami, ami_config=self.config,
            ami_config_sha256=config_sha or self.config_ref["sha256"],
            ami_issuance_sha256=issuance_sha or self.issuance["sha256"],
            output_root=self.root / output_name,
            owner_authorization_ref="owner-directed-mfm-v16-mixed-test",
            check_role_coverage=False,
        )

    def test_composes_original_authorizations_replays_and_rejects_false_issuance(self):
        first = self.compose()
        second = self.compose(output_name="replay")
        self.assertEqual((first["train_cases"], first["development_cases"]), (3, 1))
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        manifest = json.loads(Path(first["manifest_path"]).read_bytes())
        self.assertEqual({case["authorization_id"] for case in manifest["cases"]}, {
            "fictional-owner-attestation", "owner-directed-mfm-v16-public-teacher"})
        self.assertEqual({case["authorization_id"] for case in manifest["cases"]
                          if case["case_id"].startswith("ami-")},
                         {"owner-directed-mfm-v16-public-teacher"})
        with self.assertRaisesRegex(ValueError, "pinned bytes differ"):
            self.compose(output_name="wrong-issuance", issuance_sha="0" * 64)
        self.assertFalse((self.root / "wrong-issuance").exists())

        # A newly pinned candidate and provenance still cannot relabel a
        # public-source case as synthetic ownership.
        config = json.loads(self.config.read_bytes())
        item = config["cases"][0]
        directory = self.ami / item["case_id"]
        candidate = json.loads((directory / "teacher-candidate.json").read_bytes())
        candidate["authorization_id"] = "fictional-owner-attestation"
        item["case_sha256"] = write(directory, "teacher-candidate.json", raw(candidate))["sha256"]
        provenance = json.loads((directory / "teacher-provenance.json").read_bytes())
        provenance["case_sha256"] = item["case_sha256"]
        item["provenance_sha256"] = write(directory, "teacher-provenance.json",
                                          raw(provenance))["sha256"]
        changed = canonical_json_bytes(config) + b"\n"
        self.config.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "original authorization"):
            self.compose(output_name="relabeled", config_sha=sha256(changed).hexdigest())
        self.assertFalse((self.root / "relabeled").exists())


if __name__ == "__main__":
    unittest.main()
