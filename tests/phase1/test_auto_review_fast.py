from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.auto_review import auto_review_pilot
from alice_vault.semantic_review import SemanticReview


class BatchFakeClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        self.request_count = 0

    def verify_model(self):
        return None

    def review_batch(self, *, items, private_profile=""):
        type(self).calls += 1
        self.request_count += 1
        return {
            item["item_id"]: SemanticReview(
                relevant_to_alice=True,
                relevance_score=0.97,
                recommended_decision="approve",
                document_category="research_project",
                sensitivity="private",
                contains_identity_document=False,
                contains_credentials_or_secrets=False,
                contains_third_party_private_data=False,
                contradiction_topic="",
                summary="Relevant project record",
                reason="Relevant to the owner",
            )
            for item in items
        }

    def metrics(self):
        return {"request_count": self.request_count}


class FastAutoReviewTests(unittest.TestCase):
    def _make_vault(self, base: Path, count: int = 13):
        vault = base / "vault"
        exports = vault / "manifests" / "exports"
        exports.mkdir(parents=True)
        database = vault / "manifests" / "inventory.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TABLE files("
            "file_id TEXT PRIMARY KEY, original_path TEXT, "
            "size_bytes INTEGER, sha256 TEXT)"
        )
        fields = [
            "item_index", "file_id", "content_key", "relative_path",
            "filename", "size_bytes", "sha256", "role", "family",
            "source_bucket", "year_hint", "duplicate_control_group",
            "selection_reason", "decision", "review_notes",
            "known_contradiction_group",
            "contains_identity_document",
            "contains_credentials_or_secrets",
        ]
        rows = []
        for index in range(count):
            file_id = f"f-{index}"
            path = base / f"{file_id}.txt"
            path.write_text(
                f"Useful project information number {index}",
                encoding="utf-8",
            )
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            connection.execute(
                "INSERT INTO files VALUES(?,?,?,?)",
                (file_id, str(path), len(raw), digest),
            )
            rows.append(
                {
                    "item_index": str(index + 1),
                    "file_id": file_id,
                    "content_key": f"{digest}:{len(raw)}",
                    "relative_path": path.name,
                    "filename": path.name,
                    "size_bytes": str(len(raw)),
                    "sha256": digest,
                    "role": "primary",
                    "family": "text",
                    "source_bucket": "test",
                    "year_hint": "2026",
                    "duplicate_control_group": "",
                    "selection_reason": "test",
                    "decision": "pending",
                    "review_notes": "",
                    "known_contradiction_group": "",
                    "contains_identity_document": "",
                    "contains_credentials_or_secrets": "",
                }
            )
        connection.commit()
        connection.close()

        proposal_id = "proposal-fast"
        (exports / f"pilot-proposal-summary-{proposal_id}.json").write_text(
            json.dumps({"proposal_id": proposal_id}),
            encoding="utf-8",
        )
        with (
            exports / f"pilot-proposal-{proposal_id}.csv"
        ).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return vault

    def test_batching_and_resume_reduce_model_calls(self):
        import alice_vault.auto_review as module

        original = module.OllamaLocalClient
        module.OllamaLocalClient = BatchFakeClient
        BatchFakeClient.calls = 0
        try:
            with tempfile.TemporaryDirectory() as temp:
                vault = self._make_vault(Path(temp), count=13)
                first = auto_review_pilot(
                    vault_root=vault,
                    model="fake",
                    batch_size=6,
                    max_chars=1000,
                )
                self.assertEqual(BatchFakeClient.calls, 3)
                self.assertEqual(first["auto_approved"], 13)
                self.assertTrue(first["canonical_review_updated"])

                BatchFakeClient.calls = 0
                second = auto_review_pilot(
                    vault_root=vault,
                    model="fake",
                    batch_size=6,
                    max_chars=1000,
                )
                self.assertEqual(BatchFakeClient.calls, 0)
                self.assertEqual(second["resumed_unique_contents"], 13)
        finally:
            module.OllamaLocalClient = original

    def test_run_specific_output_survives_canonical_lock(self):
        import alice_vault.auto_review as module

        original_client = module.OllamaLocalClient
        original_promote = module._promote_canonical
        module.OllamaLocalClient = BatchFakeClient
        module._promote_canonical = lambda **kwargs: (
            False,
            "simulated lock",
        )
        try:
            with tempfile.TemporaryDirectory() as temp:
                vault = self._make_vault(Path(temp), count=2)
                summary = auto_review_pilot(
                    vault_root=vault,
                    model="fake",
                    batch_size=2,
                    max_chars=1000,
                )
                self.assertFalse(summary["canonical_review_updated"])
                self.assertTrue(
                    Path(summary["run_review_csv_path"]).is_file()
                )
                self.assertTrue(Path(summary["summary_path"]).is_file())
        finally:
            module.OllamaLocalClient = original_client
            module._promote_canonical = original_promote


if __name__ == "__main__":
    unittest.main()
