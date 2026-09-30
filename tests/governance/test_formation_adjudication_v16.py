"""Signature and sealed-FINAL checks for an externally pinned review roster."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes
from cognitive_kernel.formation_adjudication_v16 import (
    COMPLETE_SCOPE, DOMAINS, ROSTER_SCHEMA, verify_adjudicated_corpus_v16,
)
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus


NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)
EXPIRY = "2027-01-01T00:00:00Z"


def _hash(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _pub(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(serialization.Encoding.Raw,
                                         serialization.PublicFormat.Raw)
    return b64encode(raw).decode("ascii")


def _signed(row: dict, key: Ed25519PrivateKey, domain: str) -> dict:
    signature = key.sign(DOMAINS[domain] + canonical_json_bytes(row))
    return {**row, "signature_v16": b64encode(signature).decode("ascii")}


def _write(path: Path, value: dict) -> str:
    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _hash(raw)


class FormationAdjudicationV16Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.keys = {name: Ed25519PrivateKey.generate()
                     for name in ("author", "reviewer-a", "reviewer-b", "issuer")}
        self.roster = {"schema": ROSTER_SCHEMA, "valid_until": EXPIRY,
                       "revoked_ids": [],
                       "authors": {"author": _pub(self.keys["author"])},
                       "reviewers": {name: _pub(self.keys[name])
                                     for name in ("reviewer-a", "reviewer-b")},
                       "rights_issuers": {"issuer": _pub(self.keys["issuer"])}}
        self.roster_sha = _write(self.root / "trust.json", self.roster)
        self.rows = []
        for split in ("train", "development", "final"):
            case_id = f"case-{split}"
            source_raw = f"source {split}".encode()
            target_raw = f"target {split}".encode()
            source_sha, target_sha = _hash(source_raw), _hash(target_raw)
            source_path, target_path = f"{split}/source", f"{split}/target"
            rights_path = f"{split}/rights.json"
            if split != "final":
                (self.root / split).mkdir()
                (self.root / source_path).write_bytes(source_raw)
                (self.root / target_path).write_bytes(target_raw)
                rights = _signed({"schema": "mfm-source-rights-v1",
                                  "issuer_id": "issuer", "source_id": f"source-{split}",
                                  "authority_ref": f"consent-{split}",
                                  "source_sha256": source_sha, "host_family": f"host-{split}",
                                  "formation_training": True,
                                  "formation_evaluation": True,
                                  "model_distribution": True, "revoked": False,
                                  "valid_until": EXPIRY}, self.keys["issuer"], "rights")
                rights_sha = _write(self.root / rights_path, rights)
            else:
                # There is deliberately no FINAL source, target, or rights file.
                rights_sha = "f" * 64
            source = {"source_id": f"source-{split}", "path": source_path,
                      "sha256": source_sha, "rights_path": rights_path,
                      "rights_sha256": rights_sha, "parent_source_ids": []}
            if split == "final":
                source["final_rights_attestation_v16"] = _signed({
                    "issuer_id": "issuer", "case_id": case_id,
                    "source_id": source["source_id"], "source_sha256": source_sha,
                    "rights_sha256": rights_sha, "host_family": f"host-{split}",
                    "formation_evaluation": True, "valid_until": EXPIRY,
                }, self.keys["issuer"], "final_rights")
            row = {"case_id": case_id, "split": split, "author_id": "author",
                   "host_family": f"host-{split}", "source_family": f"source-family-{split}",
                   "generator_family": f"generator-{split}",
                   "scenario_family": f"scenario-{split}",
                   "duplicate_group": f"group-{split}", "parent_case_ids": [],
                   "sources": [source], "target": {"path": target_path,
                                                    "sha256": target_sha}}
            lineage = _hash(canonical_json_bytes({
                field: row[field] for field in ("host_family", "source_family",
                    "generator_family", "scenario_family", "duplicate_group",
                    "parent_case_ids")}))
            row["reviews"] = [_signed({
                "reviewer_id": reviewer, "case_id": case_id, "split": split,
                "author_id": "author", "target_sha256": target_sha,
                "source_sha256s": [source_sha],
                "source_rights_sha256s": [rights_sha],
                "blind": True, "decision": "accept",
                "complete_target_review_v16": True,
                "coverage_scope": list(COMPLETE_SCOPE),
                "lineage_sha256": lineage, "valid_until": EXPIRY,
            }, self.keys[reviewer], "review") for reviewer in
                ("reviewer-a", "reviewer-b")]
            self.rows.append(row)
        self._manifest()

    def _manifest(self) -> None:
        digest = _write(self.root / "manifest.json", {
            "schema": "mfm-formation-corpus-v1", "corpus_id": "corpus-v16",
            "cases": self.rows})
        self.admission = admit_formation_corpus(self.root / "manifest.json",
                                                 expected_sha256=digest)

    def _check(self):
        return verify_adjudicated_corpus_v16(
            self.admission, self.root / "manifest.json", self.root / "trust.json",
            expected_roster_sha256=self.roster_sha, as_of=NOW)

    def test_valid_signatures_bind_complete_review_and_keep_final_sealed(self):
        receipt = self._check()
        self.assertEqual(receipt["case_counts"], {"train": 1,
                     "development": 1, "final": 1})
        self.assertFalse(receipt["final_payloads_opened"])
        self.assertFalse(receipt["external_identity_and_semantic_truth_verified"])
        self.assertFalse((self.root / "final/target").exists())

    def test_changed_review_or_lineage_fails_even_with_new_manifest_hash(self):
        self.rows[0]["reviews"][0]["complete_target_review_v16"] = False
        self._manifest()
        with self.assertRaises(CognitiveKernelContractError):
            self._check()

    def test_revoked_review_key_fails_even_when_roster_is_rehashed(self):
        self.roster["revoked_ids"] = ["reviewer-a"]
        self.roster_sha = _write(self.root / "trust.json", self.roster)
        with self.assertRaises(CognitiveKernelContractError):
            self._check()

    def test_caller_constructed_admission_cannot_bypass_base_rights_audit(self):
        forged = replace(self.admission, train=())
        with self.assertRaisesRegex(CognitiveKernelContractError,
                                    "differs from base admission"):
            verify_adjudicated_corpus_v16(
                forged, self.root / "manifest.json", self.root / "trust.json",
                expected_roster_sha256=self.roster_sha, as_of=NOW)

    def test_sealed_final_rights_cannot_be_relabelled_to_another_host(self):
        self.rows[2]["host_family"] = "other-host"
        lineage = _hash(canonical_json_bytes({
            field: self.rows[2][field] for field in ("host_family", "source_family",
                "generator_family", "scenario_family", "duplicate_group",
                "parent_case_ids")}))
        for index, reviewer in enumerate(("reviewer-a", "reviewer-b")):
            unsigned = {key: value for key, value in self.rows[2]["reviews"][index].items()
                        if key != "signature_v16"}
            unsigned["lineage_sha256"] = lineage
            self.rows[2]["reviews"][index] = _signed(
                unsigned, self.keys[reviewer], "review")
        self._manifest()
        with self.assertRaisesRegex(CognitiveKernelContractError,
                                    "FINAL rights attestation changed source"):
            self._check()


if __name__ == "__main__":
    unittest.main()
