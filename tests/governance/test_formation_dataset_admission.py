from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def write(root: Path, name: str, payload: bytes) -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": name, "sha256": digest(payload)}


def case(root: Path, name: str, split: str) -> dict:
    source_bytes = f"{name} says the appointment moved to June.".encode()
    target_bytes = json.dumps({"action": "propose", "subject": name}).encode()
    source_name = f"{name}/source.txt"
    target_name = f"{name}/target.json"
    if split != "final":
        source = write(root, source_name, source_bytes)
        target = write(root, target_name, target_bytes)
        rights = {
            "schema": "mfm-source-rights-v1", "issuer_id": f"issuer-{name}",
            "authority_ref": f"consent-{name}", "host_family": f"host-{name}",
            "source_id": f"source-{name}", "source_sha256": source["sha256"],
            "formation_training": True, "formation_evaluation": True,
            "model_distribution": True, "revoked": False,
        }
        receipt = write(root, f"{name}/rights.json", json.dumps(rights).encode())
    else:
        # Sealed FINAL content and rights receipt must never be opened here.
        source = {"path": source_name, "sha256": digest(source_bytes)}
        target = {"path": target_name, "sha256": digest(target_bytes)}
        receipt = {"path": f"{name}/rights.json", "sha256": "1" * 64}
    return {
        "case_id": name, "split": split, "host_family": f"host-{name}",
        "source_family": f"family-{name}", "generator_family": f"generator-{name}",
        "scenario_family": f"scenario-{name}", "duplicate_group": f"dup-{name}",
        "parent_case_ids": [], "author_id": f"author-{name}",
        "sources": [{"source_id": f"source-{name}", **source,
                     "rights_path": receipt["path"], "rights_sha256": receipt["sha256"],
                     "parent_source_ids": []}],
        "target": target,
        "reviews": [{"reviewer_id": f"reviewer-{name}-{index}", "blind": True,
                     "decision": "accept", "target_sha256": target["sha256"],
                     "source_sha256s": [source["sha256"]]} for index in (1, 2)],
    }


class FormationDatasetAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [case(self.root, name, split) for name, split in
                     (("aria", "train"), ("ben", "development"), ("cy", "final"))]

    def admit(self, rows: list[dict] | None = None):
        manifest = {"schema": "mfm-formation-corpus-v1", "corpus_id": "separate-histories",
                    "cases": rows if rows is not None else self.rows}
        raw = json.dumps(manifest, sort_keys=True).encode()
        (self.root / "manifest.json").write_bytes(raw)
        return admit_formation_corpus(self.root / "manifest.json", expected_sha256=digest(raw))

    def test_unopened_final_and_exact_train_dev_handoff(self):
        admitted = self.admit()
        self.assertFalse((self.root / "cy/source.txt").exists())
        handoff = admitted.audit_handoff(
            gradient_paths=("aria/source.txt", "aria/target.json"),
            development_paths=("ben/source.txt", "ben/target.json"))
        self.assertEqual(handoff["corpus_manifest_sha256"], admitted.manifest_sha256)
        self.assertEqual(handoff["final_exclusion_paths"], ["cy/source.txt", "cy/target.json"])
        with self.assertRaisesRegex(CognitiveKernelContractError, "unadmitted or FINAL"):
            admitted.audit_handoff(gradient_paths=("cy/source.txt",), development_paths=())
        with self.assertRaisesRegex(CognitiveKernelContractError, "unadmitted or FINAL"):
            admitted.audit_handoff(gradient_paths=("ben/source.txt",), development_paths=())

    def test_rights_receipt_binds_exact_source_and_authorized_use(self):
        admitted = self.admit()
        rows = deepcopy(self.rows)
        rights_path = self.root / rows[0]["sources"][0]["rights_path"]
        rights = json.loads(rights_path.read_text())
        rights["model_distribution"] = False
        raw = json.dumps(rights).encode()
        rights_path.write_bytes(raw)
        rows[0]["sources"][0]["rights_sha256"] = digest(raw)
        with self.assertRaisesRegex(CognitiveKernelContractError, "rights do not permit"):
            self.admit(rows)
        rows[0]["sources"][0]["rights_sha256"] = "0" * 64
        with self.assertRaisesRegex(CognitiveKernelContractError, "differ from manifest"):
            self.admit(rows)
        with self.assertRaisesRegex(CognitiveKernelContractError, "differ from manifest"):
            admitted.audit_handoff(gradient_paths=("aria/source.txt",), development_paths=())

    def test_adjudication_requires_two_distinct_blind_source_bound_reviewers(self):
        rows = deepcopy(self.rows)
        rows[0]["reviews"][1]["reviewer_id"] = rows[0]["author_id"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "independent"):
            self.admit(rows)
        rows = deepcopy(self.rows)
        rows[0]["reviews"][1]["target_sha256"] = "0" * 64
        with self.assertRaisesRegex(CognitiveKernelContractError, "exact target"):
            self.admit(rows)
        rows = deepcopy(self.rows)
        rows[0]["reviews"][0]["source_sha256s"] = []
        with self.assertRaisesRegex(CognitiveKernelContractError, "exact target"):
            self.admit(rows)

    def test_generator_and_parent_lineage_cannot_cross_splits(self):
        rows = deepcopy(self.rows)
        rows[1]["generator_family"] = rows[0]["generator_family"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "leaks across splits"):
            self.admit(rows)
        rows = deepcopy(self.rows)
        rows[1]["parent_case_ids"] = [rows[0]["case_id"]]
        with self.assertRaisesRegex(CognitiveKernelContractError, "leaks across splits"):
            self.admit(rows)
        rows = deepcopy(self.rows)
        rows[1]["sources"][0]["parent_source_ids"] = [rows[0]["sources"][0]["source_id"]]
        with self.assertRaisesRegex(CognitiveKernelContractError, "leaks across splits"):
            self.admit(rows)

    def test_parent_cycle_and_unknown_ancestry_rejected(self):
        rows = deepcopy(self.rows)
        rows[0]["parent_case_ids"] = ["ben"]
        rows[1]["parent_case_ids"] = ["aria"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "lineage cycle"):
            self.admit(rows)
        rows = deepcopy(self.rows)
        rows[0]["sources"][0]["parent_source_ids"] = ["missing-source"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "unregistered parent source"):
            self.admit(rows)

    def test_source_hash_duplicate_and_split_path_audit(self):
        rows = deepcopy(self.rows)
        rows[1]["sources"][0]["sha256"] = rows[0]["sources"][0]["sha256"]
        with self.assertRaisesRegex(CognitiveKernelContractError, "differ from manifest"):
            self.admit(rows)
        admitted = self.admit()
        (self.root / "aria/source.txt").write_bytes(b"tampered after admission")
        with self.assertRaisesRegex(CognitiveKernelContractError, "payload changed"):
            admitted.audit_handoff(gradient_paths=("aria/source.txt",), development_paths=())
        (self.root / "aria/source.txt").write_bytes(b"aria says the appointment moved to June.")
        rows = deepcopy(self.rows)
        rows[2]["sources"][0]["sha256"] = rows[0]["sources"][0]["sha256"]
        for review in rows[2]["reviews"]:
            review["source_sha256s"] = [rows[2]["sources"][0]["sha256"]]
        with self.assertRaisesRegex(CognitiveKernelContractError, "leaks across splits"):
            self.admit(rows)

    def test_frozen_manifest_and_path_escape(self):
        self.admit()
        with self.assertRaisesRegex(CognitiveKernelContractError, "frozen digest"):
            admit_formation_corpus(self.root / "manifest.json", expected_sha256="0" * 64)
        rows = deepcopy(self.rows)
        rows[0]["sources"][0]["path"] = "../secret.txt"
        with self.assertRaisesRegex(CognitiveKernelContractError, "escapes"):
            self.admit(rows)


if __name__ == "__main__":
    unittest.main()
