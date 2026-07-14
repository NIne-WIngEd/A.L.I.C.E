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
from alice_vault.semantic_review import SemanticReview, validate_semantic_review


class SplitClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def verify_model(self):
        return None

    def review_batch(self, *, items, private_profile=""):
        type(self).calls += 1
        if len(items) > 1:
            raise TimeoutError("simulated batch timeout")
        item = items[0]
        return {
            item["item_id"]: SemanticReview(
                True,
                0.96,
                "approve",
                "research_project",
                "private",
                False,
                False,
                False,
                "",
                "Relevant",
                "Relevant",
            )
        }

    def metrics(self):
        return {"request_attempt_count": type(self).calls}


class AlwaysFailClient:
    def __init__(self, *args, **kwargs):
        pass

    def verify_model(self):
        return None

    def review_batch(self, *, items, private_profile=""):
        raise TimeoutError("simulated persistent timeout")

    def metrics(self):
        return {}


class SuccessClient(SplitClient):
    def review_batch(self, *, items, private_profile=""):
        return {
            item["item_id"]: SemanticReview(
                True,
                0.96,
                "approve",
                "research_project",
                "private",
                False,
                False,
                False,
                "",
                "Relevant",
                "Relevant",
            )
            for item in items
        }


class ResilienceTests(unittest.TestCase):
    def _make_vault(self, base: Path, count: int = 3) -> Path:
        vault = base / "vault"
        exports = vault / "manifests" / "exports"
        exports.mkdir(parents=True)
        db = vault / "manifests" / "inventory.sqlite3"
        connection = sqlite3.connect(db)
        connection.execute(
            "CREATE TABLE files(file_id TEXT PRIMARY KEY, "
            "original_path TEXT, size_bytes INTEGER, sha256 TEXT)"
        )
        fields = [
            "item_index", "file_id", "content_key", "relative_path",
            "filename", "size_bytes", "sha256", "role", "family",
            "source_bucket", "year_hint", "duplicate_control_group",
            "selection_reason", "decision", "review_notes",
            "known_contradiction_group", "contains_identity_document",
            "contains_credentials_or_secrets",
        ]
        rows = []
        for index in range(count):
            path = base / f"f-{index}.txt"
            path.write_text(f"Useful project record {index}", encoding="utf-8")
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            file_id = f"f-{index}"
            connection.execute(
                "INSERT INTO files VALUES(?,?,?,?)",
                (file_id, str(path), len(raw), digest),
            )
            rows.append({
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
            })
        connection.commit()
        connection.close()
        proposal_id = "resilience"
        (exports / f"pilot-proposal-summary-{proposal_id}.json").write_text(
            json.dumps({"proposal_id": proposal_id}), encoding="utf-8"
        )
        with (exports / f"pilot-proposal-{proposal_id}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return vault

    def test_failed_batch_splits_to_single_items(self):
        import alice_vault.auto_review as module

        original = module.OllamaLocalClient
        module.OllamaLocalClient = SplitClient
        SplitClient.calls = 0
        try:
            with tempfile.TemporaryDirectory() as temp:
                vault = self._make_vault(Path(temp), 3)
                summary = auto_review_pilot(
                    vault_root=vault,
                    model="fake",
                    batch_size=3,
                    max_chars=1000,
                )
                self.assertEqual(summary["model_error_count"], 0)
                self.assertEqual(summary["auto_approved"], 3)
                self.assertGreaterEqual(SplitClient.calls, 4)
        finally:
            module.OllamaLocalClient = original

    def test_failed_items_are_not_resumed_as_completed(self):
        import alice_vault.auto_review as module

        original = module.OllamaLocalClient
        try:
            with tempfile.TemporaryDirectory() as temp:
                vault = self._make_vault(Path(temp), 1)
                module.OllamaLocalClient = AlwaysFailClient
                first = auto_review_pilot(
                    vault_root=vault,
                    model="fake",
                    batch_size=1,
                    max_chars=1000,
                )
                self.assertEqual(first["model_error_count"], 1)

                module.OllamaLocalClient = SuccessClient
                second = auto_review_pilot(
                    vault_root=vault,
                    model="fake",
                    batch_size=1,
                    max_chars=1000,
                )
                self.assertEqual(second["resumed_unique_contents"], 0)
                self.assertEqual(second["model_error_count"], 0)
                self.assertEqual(second["auto_approved"], 1)
        finally:
            module.OllamaLocalClient = original

    def test_score_is_clamped_instead_of_rejecting_whole_batch(self):
        review = validate_semantic_review({
            "id": "i1",
            "rel": True,
            "score": 104,
            "decision": "approve",
            "category": "research_project",
            "sensitivity": "private",
            "identity": False,
            "secrets": False,
            "third_party": False,
            "contradiction": "",
            "reason": "Relevant",
        })
        self.assertEqual(review.relevance_score, 1.0)


if __name__ == "__main__":
    unittest.main()
