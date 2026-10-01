"""Fictional contract fixtures; no real owner rights or training corpus."""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes
from scripts.mfm.assemble_v16_owner_teacher_corpus import (
    assemble, INTAKE_SCHEMA, INTAKE_SCHEMA_V2, STATUS,
)
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus, TEACHER_SCHEMA_V2
from scripts.mfm.build_v16_authoring_seed import render


def raw(value: dict) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def write(root: Path, name: str, contents: bytes) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return {"path": name, "sha256": sha256(contents).hexdigest()}


class OwnerTeacherAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        selected = [row for row in map(json.loads, render().splitlines())
                    if row["case_id"] in {"v16-seed-01", "v16-seed-13"}]
        entries = []
        for row in selected:
            case_id = row["case_id"]
            split = row["split"]
            host = row["context"]["base_context"]["scope"]["host_instance_id"]
            row["authorization_id"] = "fictional-owner-attestation"
            candidate = write(self.root, f"{case_id}/candidate.json", raw(row))
            sources = []
            for index, item in enumerate(row["sources"]):
                content = b64decode(item["content_b64"])
                source = write(self.root, f"{case_id}/source-{index}.txt", content)
                rights = write(self.root, f"{case_id}/rights-{index}.json", raw({
                    "schema": "mfm-source-rights-v1", "issuer_id": "fictional-unit-owner",
                    "source_id": item["ref_id"], "source_sha256": source["sha256"],
                    "host_family": host, "authority_ref": "fictional-unit-only",
                    "revoked": False, "formation_training": True,
                    "formation_evaluation": True, "model_distribution": True,
                }))
                sources.append({"source_id": item["ref_id"], **source,
                                "rights_path": rights["path"],
                                "rights_sha256": rights["sha256"],
                                "parent_source_ids": []})
            prompt = write(self.root, f"{case_id}/prompt.txt", b"fictional prompt " + case_id.encode())
            response = write(self.root, f"{case_id}/response.json", b"fictional raw output " + case_id.encode())
            converter = write(self.root, f"{case_id}/converter.py", b"fictional converter " + case_id.encode())
            provenance = write(self.root, f"{case_id}/provenance.json", raw({
                "case_id": case_id, "case_sha256": candidate["sha256"],
                "rendered_prompt_sha256": prompt["sha256"],
                "teacher_response_sha256": response["sha256"],
                "converter_sha256": converter["sha256"],
            }))
            entries.append({
                "case_id": case_id, "split": split, "author_id": f"fictional-teacher-{split}",
                "host_family": host, "source_family": f"fictional-source-{split}",
                "generator_family": f"fictional-generator-{split}",
                "scenario_family": f"fictional-scenario-{split}",
                "duplicate_group": f"fictional-duplicate-{split}",
                "parent_case_ids": [], "candidate": candidate, "sources": sources,
                "target_origin": "owner-authorized-service-teacher",
                "producer_version": "fictional-unit-v1", "prompt": prompt,
                "response": response, "converter": converter,
                "converter_provenance": provenance,
            })
        self.intake = {"schema": INTAKE_SCHEMA, "status": STATUS,
                       "corpus_id": "fictional-unit-corpus",
                       "authorization_id": "fictional-owner-attestation",
                       "cases": entries}

    def run_assembly(self, *, check_role_coverage=False):
        intake_ref = write(self.root, "intake.json", raw(self.intake))
        return assemble(self.root, self.root / intake_ref["path"],
                        intake_sha256=intake_ref["sha256"],
                        check_role_coverage=check_role_coverage)

    def mixed_authorization(self):
        """A fictional stand-in for two distinct source-rights authorities."""
        train, development = self.intake["cases"]
        external_auth = "public-case-original-authorization"
        candidate_path = self.root / train["candidate"]["path"]
        candidate = json.loads(candidate_path.read_bytes())
        candidate["authorization_id"] = external_auth
        train["candidate"] = write(self.root, train["candidate"]["path"], raw(candidate))
        provenance_path = self.root / train["converter_provenance"]["path"]
        provenance = json.loads(provenance_path.read_bytes())
        provenance["case_sha256"] = train["candidate"]["sha256"]
        train["converter_provenance"] = write(
            self.root, train["converter_provenance"]["path"], raw(provenance))
        for source in train["sources"]:
            path = self.root / source["rights_path"]
            rights = json.loads(path.read_bytes())
            rights["authority_ref"] = "public-case-license-authority"
            rights["issuer_id"] = "project-license-record-issuer"
            source["rights_sha256"] = write(self.root, source["rights_path"], raw(rights))["sha256"]
        train["authorization_id"] = external_auth
        development["authorization_id"] = self.intake["authorization_id"]
        self.intake["schema"] = INTAKE_SCHEMA_V2
        self.intake["authorizations"] = [
            {"authorization_id": external_auth, "source_family": train["source_family"],
             "rights_authority_ref": "public-case-license-authority",
             "rights_issuer_id": "project-license-record-issuer"},
            {"authorization_id": self.intake["authorization_id"],
             "source_family": development["source_family"],
             "rights_authority_ref": "fictional-unit-only",
             "rights_issuer_id": "fictional-unit-owner"},
        ]

    def test_exact_source_target_and_provenance_admit(self):
        receipt = self.run_assembly()
        self.assertEqual((receipt["train_cases"], receipt["development_cases"]), (1, 1))
        manifest = json.loads(Path(receipt["manifest_path"]).read_bytes())
        self.assertEqual({item["split"] for item in manifest["cases"]},
                         {"train", "development"})
        target = json.loads((self.root / manifest["cases"][0]["target"]["path"]).read_bytes())
        self.assertEqual(target["schema"], "mfm-formation-target-v1.6")
        self.assertEqual(target["source_ids"],
                         [ref["source_id"] for ref in manifest["cases"][0]["sources"]])
        self.assertEqual(manifest["cases"][0]["reviews"], [])
        self.assertFalse(receipt["qualified_for_product"])
        with self.assertRaises(FileExistsError):
            self.run_assembly()

    def test_v2_preserves_original_case_authorizations_and_rights_authorities(self):
        self.mixed_authorization()
        receipt = self.run_assembly()
        manifest = json.loads(Path(receipt["manifest_path"]).read_bytes())
        self.assertEqual(manifest["schema"], TEACHER_SCHEMA_V2)
        self.assertEqual(manifest["authorization_id"], "fictional-owner-attestation")
        self.assertEqual([case["authorization_id"] for case in manifest["cases"]],
                         ["public-case-original-authorization", "fictional-owner-attestation"])
        admission = admit_formation_corpus(
            receipt["manifest_path"], expected_sha256=receipt["manifest_sha256"],
            teacher_training=True, owner_authorization_ref="fictional-owner-attestation")
        self.assertEqual((len(admission.train), len(admission.development)), (1, 1))
        with self.assertRaisesRegex(CognitiveKernelContractError, "pinned owner authorization"):
            admit_formation_corpus(receipt["manifest_path"],
                                   expected_sha256=receipt["manifest_sha256"],
                                   teacher_training=True, owner_authorization_ref="foreign-owner")

    def test_v2_rejects_foreign_case_authorization_and_mismatched_rights(self):
        self.mixed_authorization()
        self.intake["cases"][0]["authorization_id"] = "foreign-case-authorization"
        with self.assertRaisesRegex(ValueError, "unregistered authorization"):
            self.run_assembly()
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())

        self.intake["cases"][0]["authorization_id"] = "public-case-original-authorization"
        self.intake["authorizations"][0]["rights_authority_ref"] = "foreign-license"
        with self.assertRaisesRegex(ValueError, "rights differ"):
            self.run_assembly()
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())

    def test_v2_rejects_candidate_auth_relabel_and_registry_ambiguity(self):
        self.mixed_authorization()
        self.intake["cases"][0]["authorization_id"] = "fictional-owner-attestation"
        self.intake["authorizations"].append({
            **self.intake["authorizations"][0],
            "authorization_id": "fictional-owner-attestation"})
        with self.assertRaisesRegex(ValueError, "candidate schema, case, split or authorization"):
            self.run_assembly()
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())

        self.intake["cases"][0]["authorization_id"] = "public-case-original-authorization"
        self.intake["authorizations"].append(dict(self.intake["authorizations"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate authorization binding"):
            self.run_assembly()

    def test_v2_admission_rechecks_case_authority_after_assembly(self):
        self.mixed_authorization()
        receipt = self.run_assembly()
        manifest_path = Path(receipt["manifest_path"])
        manifest = json.loads(manifest_path.read_bytes())
        manifest["authorizations"][0]["rights_authority_ref"] = "foreign-license"
        changed = raw(manifest)
        manifest_path.write_bytes(changed)
        with self.assertRaisesRegex(CognitiveKernelContractError, "rights differ"):
            admit_formation_corpus(manifest_path, expected_sha256=sha256(changed).hexdigest(),
                                   teacher_training=True,
                                   owner_authorization_ref="fictional-owner-attestation")

    def test_mismatched_candidate_source_and_raw_teacher_provenance_fail(self):
        entry = self.intake["cases"][0]
        source_path = self.root / entry["sources"][0]["path"]
        source_path.write_bytes(b"different source")
        with self.assertRaisesRegex(ValueError, "externally pinned bytes"):
            self.run_assembly()
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())
        source_path.write_bytes(b64decode(json.loads(
            (self.root / entry["candidate"]["path"]).read_bytes())["sources"][0]["content_b64"]))
        response_path = self.root / entry["response"]["path"]
        response_path.write_bytes(b"different teacher output")
        entry["response"]["sha256"] = sha256(response_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "wrong teacher_response_sha256"):
            self.run_assembly()

    def test_cannot_rename_one_generator_across_splits(self):
        self.intake["cases"][1]["generator_family"] = self.intake["cases"][0]["generator_family"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "lineage leaks"):
            self.run_assembly()
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())
        self.assertFalse((self.root / "targets/v16-seed-01.json").exists())

    def test_rights_cannot_be_invented_or_revoked(self):
        entry = self.intake["cases"][0]
        receipt = entry["sources"][0]
        rights_path = self.root / receipt["rights_path"]
        rights = json.loads(rights_path.read_bytes())
        rights["revoked"] = True
        rights_path.write_bytes(raw(rights))
        receipt["rights_sha256"] = sha256(rights_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(CognitiveKernelContractError, "revoked"):
            self.run_assembly()
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())

    def test_deterministic_generator_output_must_be_exact_target(self):
        entry = self.intake["cases"][1]
        candidate = json.loads((self.root / entry["candidate"]["path"]).read_bytes())
        target = {"schema": "mfm-formation-target-v1.6", "context": candidate["context"],
                  "source_ids": [source["ref_id"] for source in candidate["sources"]],
                  **{key: candidate["target"][key] for key in (
                      "proposals", "dispositions", "adjudications")}}
        entry["target_origin"] = "licensed-deterministic-generator"
        entry["response"] = write(self.root, entry["response"]["path"], raw(target))
        provenance = {"case_id": entry["case_id"],
                      "case_sha256": entry["candidate"]["sha256"],
                      "rendered_prompt_sha256": entry["prompt"]["sha256"],
                      "generator_output_sha256": entry["response"]["sha256"],
                      "converter_sha256": entry["converter"]["sha256"]}
        entry["converter_provenance"] = write(
            self.root, entry["converter_provenance"]["path"], raw(provenance))
        receipt = self.run_assembly()
        manifest = json.loads(Path(receipt["manifest_path"]).read_bytes())
        development = next(row for row in manifest["cases"] if row["split"] == "development")
        self.assertEqual(development["target"], entry["response"])
        self.assertEqual(development["target_provenance"]["output"], entry["response"])

    def test_deterministic_output_with_extra_byte_fails(self):
        entry = self.intake["cases"][1]
        entry["target_origin"] = "licensed-deterministic-generator"
        provenance = json.loads((self.root / entry["converter_provenance"]["path"]).read_bytes())
        provenance["generator_output_sha256"] = provenance.pop("teacher_response_sha256")
        entry["converter_provenance"] = write(
            self.root, entry["converter_provenance"]["path"], raw(provenance))
        with self.assertRaisesRegex(ValueError, "exact target bytes"):
            self.run_assembly()
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())

    def test_role_coverage_gate_not_inferred_from_two_cases(self):
        with self.assertRaisesRegex(CognitiveKernelContractError, "positive supervised"):
            self.run_assembly(check_role_coverage=True)
        self.assertFalse((self.root / "owner-teacher-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
