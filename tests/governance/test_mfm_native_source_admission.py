from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.mfm.native_source_admission import admit_native_sources


CANDIDATES = (Path(__file__).resolve().parents[2] / "configs/mfm/"
              "native_foundation_source_candidates_v1.json")
CANDIDATE_SHA = sha256(CANDIDATES.read_bytes()).hexdigest()
CANDIDATE = "common_pile_arxiv_abstracts_filtered"
CANDIDATE_URL = "https://huggingface.co/datasets/common-pile/arxiv_abstracts_filtered"


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def write(root: Path, path: str, content: bytes) -> dict[str, str]:
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return {"path": path, "sha256": digest(content)}


def item(root: Path, name: str, split: str) -> dict:
    upstream = write(root, f"{name}/upstream.parquet", f"upstream-{name}".encode())
    payload = write(root, f"{name}/payload.txt", f"source-{name}".encode())
    terms = write(root, f"{name}/terms.txt", b"CC BY 4.0 source terms")
    evidence = write(root, f"{name}/review.txt", b"Separate source and privacy review evidence")
    source_id = f"source-{name}"
    source_url = f"https://example.org/items/{name}"
    revision = "rev-2026-09-29"
    upstream_item_ref = f"row-{name}"
    rights = {
        "schema": "mfm-native-public-item-rights-v1",
        "item_id": name, "source_id": source_id,
        "candidate_id": CANDIDATE, "payload_sha256": payload["sha256"],
        "upstream_file_sha256": upstream["sha256"],
        "source_url": source_url, "upstream_revision": revision,
        "upstream_item_ref": upstream_item_ref, "data_class": "public_source",
        "license_id": "CC-BY-4.0", "author": "Public author",
        "attribution": "Public author, example item", "reviewer_id": "rights-reviewer",
        "authority_ref": "authority-record-1", "reviewed_at": "2026-09-29",
        "permissions": {"commercial_training": True, "model_distribution": True},
        "privacy": {"decision": "approved", "private_owner_data_present": False,
                    "reviewer_id": "privacy-reviewer", "reviewed_at": "2026-09-29"},
        "removal": {"state": "active", "checked_at": "2026-09-29"},
        "source_terms": terms, "review_evidence": evidence,
    }
    rights_ref = write(root, f"{name}/rights.json", json.dumps(rights).encode())
    return {
        "item_id": name, "source_id": source_id, "candidate_id": CANDIDATE,
        "candidate_source_url": CANDIDATE_URL, "source_url": source_url,
        "upstream_revision": revision, "upstream_item_ref": upstream_item_ref,
        "extraction_process_sha256": "1" * 64, "modality": "text", "split": split,
        "upstream_file": upstream, "payload": payload, "rights": rights_ref,
        "lineage_group": f"lineage-{name}", "duplicate_group": f"duplicate-{name}",
        "parent_item_ids": [],
    }


class NativeSourceAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.rows = [item(self.root, "alpha", "train"),
                     item(self.root, "beta", "development")]
        final = item(self.root, "gamma", "final")
        # FINAL content must stay sealed: the admission reads metadata only.
        for path in ("gamma/upstream.parquet", "gamma/payload.txt", "gamma/rights.json",
                     "gamma/terms.txt", "gamma/review.txt"):
            (self.root / path).unlink()
        self.rows.append(final)

    def admission(self, rows: list[dict] | None = None, exclusions=None):
        manifest = {
            "schema": "mfm-native-foundation-admission-v1", "pack_id": "example-pack",
            "candidate_inventory_sha256": CANDIDATE_SHA,
            "exclusions": exclusions if exclusions is not None else {
                "source_ids": ["benchmark-source"],
                "sha256s": ["f" * 64], "lineage_groups": ["benchmark-family"],
                "reserved_evaluations": ["LongMemEval-V2", "LoCoMo", "CareCall"]},
            "items": rows if rows is not None else self.rows,
        }
        raw = json.dumps(manifest, sort_keys=True).encode()
        path = self.root / "admission.json"
        path.write_bytes(raw)
        return admit_native_sources(CANDIDATES, path,
                                    expected_candidate_sha256=CANDIDATE_SHA,
                                    expected_manifest_sha256=digest(raw))

    def test_exact_public_pack_and_sealed_final_handoff(self):
        admitted = self.admission()
        self.assertFalse((self.root / "gamma/payload.txt").exists())
        self.assertEqual(admitted.train_items[0].payload_path,
                         self.root / "alpha/payload.txt")
        handoff = admitted.audit_handoff(
            gradient_paths=("alpha/payload.txt",),
            development_paths=("beta/payload.txt",))
        self.assertEqual(handoff["gradient_inputs"][0]["sha256"],
                         self.rows[0]["payload"]["sha256"])
        self.assertFalse(handoff["foundation_trained"])
        self.assertFalse(handoff["full_capability_qualified"])
        self.assertEqual(handoff["final_exclusion_paths"], ["gamma/payload.txt"])
        with self.assertRaisesRegex(ValueError, "unadmitted, excluded or FINAL"):
            admitted.audit_handoff(gradient_paths=("beta/payload.txt",),
                                   development_paths=())
        with self.assertRaisesRegex(ValueError, "unadmitted, excluded or FINAL"):
            admitted.audit_handoff(gradient_paths=("gamma/payload.txt",),
                                   development_paths=())

    def test_candidate_inventory_is_never_an_admission(self):
        with self.assertRaisesRegex(ValueError, "separate native foundation admission"):
            admit_native_sources(CANDIDATES, CANDIDATES,
                                 expected_candidate_sha256=CANDIDATE_SHA,
                                 expected_manifest_sha256=CANDIDATE_SHA)
        rows = deepcopy(self.rows)
        rows[0]["candidate_id"] = "made-up-dataset"
        with self.assertRaisesRegex(ValueError, "absent from candidate inventory"):
            self.admission(rows)
        with self.assertRaisesRegex(ValueError, "frozen digest"):
            admit_native_sources(CANDIDATES, self.root / "admission.json",
                                 expected_candidate_sha256="0" * 64,
                                 expected_manifest_sha256="0" * 64)

    def test_source_terms_rights_and_removal_are_rechecked(self):
        admitted = self.admission()
        (self.root / "alpha/terms.txt").write_text("changed terms")
        with self.assertRaisesRegex(ValueError, "source terms bytes differ"):
            admitted.audit_handoff(gradient_paths=("alpha/payload.txt",),
                                   development_paths=())
        rows = deepcopy(self.rows)
        rights_path = self.root / rows[0]["rights"]["path"]
        rights = json.loads(rights_path.read_text())
        rights["permissions"]["model_distribution"] = False
        rows[0]["rights"] = write(self.root, rows[0]["rights"]["path"],
                                  json.dumps(rights).encode())
        with self.assertRaisesRegex(ValueError, "per-item rights do not permit"):
            self.admission(rows)
        rights["permissions"]["model_distribution"] = True
        rights["removal"]["state"] = "withdrawn"
        rows[0]["rights"] = write(self.root, rows[0]["rights"]["path"],
                                  json.dumps(rights).encode())
        with self.assertRaisesRegex(ValueError, "removed or removal state"):
            self.admission(rows)

    def test_private_owner_data_and_unreviewed_privacy_fail(self):
        rows = deepcopy(self.rows)
        rights_path = self.root / rows[0]["rights"]["path"]
        rights = json.loads(rights_path.read_text())
        rights["data_class"] = "owner_history"
        rows[0]["rights"] = write(self.root, rows[0]["rights"]["path"],
                                  json.dumps(rights).encode())
        with self.assertRaisesRegex(ValueError, "private owner data"):
            self.admission(rows)
        rights["data_class"] = "public_source"
        rights["privacy"]["private_owner_data_present"] = True
        rows[0]["rights"] = write(self.root, rows[0]["rights"]["path"],
                                  json.dumps(rights).encode())
        with self.assertRaisesRegex(ValueError, "privacy review is not approved"):
            self.admission(rows)

    def test_source_bytes_and_split_lineage_are_not_transferable(self):
        admitted = self.admission()
        (self.root / "alpha/payload.txt").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "payload bytes differ"):
            admitted.audit_handoff(gradient_paths=("alpha/payload.txt",),
                                   development_paths=())
        (self.root / "alpha/payload.txt").write_bytes(b"source-alpha")
        rows = deepcopy(self.rows)
        rows[1]["duplicate_group"] = rows[0]["duplicate_group"]
        with self.assertRaisesRegex(ValueError, "lineage leaks across splits"):
            self.admission(rows)
        rows = deepcopy(self.rows)
        rows[1]["parent_item_ids"] = [rows[0]["item_id"]]
        with self.assertRaisesRegex(ValueError, "lineage leaks across splits"):
            self.admission(rows)

    def test_exclusion_and_path_escape_fail_closed(self):
        rows = deepcopy(self.rows)
        rows[0]["payload"]["path"] = "../unregistered.txt"
        with self.assertRaisesRegex(ValueError, "escapes admission root"):
            self.admission(rows)
        excluded = {"source_ids": ["source-alpha"], "sha256s": [],
                    "lineage_groups": [],
                    "reserved_evaluations": ["LongMemEval-V2", "LoCoMo", "CareCall"]}
        with self.assertRaisesRegex(ValueError, "frozen evaluation or removal"):
            self.admission(exclusions=excluded)


if __name__ == "__main__":
    unittest.main()
