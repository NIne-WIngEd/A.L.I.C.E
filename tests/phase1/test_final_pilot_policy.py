from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.final_pilot_policy import (
    apply_final_pilot_policy,
    decide_for_pilot,
)


class FinalPolicyTests(unittest.TestCase):
    def test_high_confidence_positive_approval_ignores_conflicting_boolean(self):
        result = decide_for_pilot(
            {
                "decision": "pending",
                "reason": "Useful research notes",
                "category": "research_project",
                "confidence": 0.95,
                "semantic_recommended_decision": "approve",
                "semantic_relevant_to_alice": False,
                "semantic_contains_third_party_private_data": False,
                "extraction_status": "ok",
                "extraction_truncated": False,
                "identity_flag": "no",
                "credential_flag": "no",
            }
        )
        self.assertEqual(result["decision"], "approve")

    def test_sensitive_and_ambiguous_records_are_excluded_not_pending(self):
        sensitive = decide_for_pilot(
            {
                "decision": "pending",
                "reason": "Sensitive category requires human review: financial",
                "category": "financial",
                "confidence": 0.99,
                "semantic_recommended_decision": "approve",
                "extraction_status": "ok",
                "identity_flag": "no",
                "credential_flag": "no",
            }
        )
        ambiguous = decide_for_pilot(
            {
                "decision": "pending",
                "reason": "unclear",
                "category": "education",
                "confidence": 0.60,
                "semantic_recommended_decision": "manual",
                "extraction_status": "ok",
                "identity_flag": "no",
                "credential_flag": "no",
            }
        )
        self.assertEqual(sensitive["decision"], "reject")
        self.assertEqual(ambiguous["decision"], "reject")

    def test_policy_writes_zero_pending_and_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            exports = vault / "manifests" / "exports"
            exports.mkdir(parents=True)

            proposal_id = "proposal-1"
            (exports / f"pilot-proposal-summary-{proposal_id}.json").write_text(
                json.dumps({"proposal_id": proposal_id}),
                encoding="utf-8",
            )
            fields = [
                "item_index",
                "file_id",
                "content_key",
                "relative_path",
                "filename",
                "size_bytes",
                "sha256",
                "role",
                "family",
                "source_bucket",
                "year_hint",
                "duplicate_control_group",
                "selection_reason",
                "decision",
                "review_notes",
                "known_contradiction_group",
                "contains_identity_document",
                "contains_credentials_or_secrets",
            ]
            rows = []
            details = {}
            for index in range(1, 7):
                content_key = f"k{index}"
                duplicate = "dup-1" if index in {5, 6} else ""
                rows.append(
                    {
                        "item_index": str(index),
                        "file_id": f"f{index}",
                        "content_key": content_key,
                        "relative_path": f"f{index}.json",
                        "filename": f"f{index}.json",
                        "size_bytes": "10",
                        "sha256": "x",
                        "role": "duplicate_control" if duplicate else "primary",
                        "family": "json",
                        "source_bucket": f"b{index}",
                        "year_hint": str(2020 + index),
                        "duplicate_control_group": duplicate,
                        "selection_reason": "test",
                        "decision": "pending",
                        "review_notes": "",
                        "known_contradiction_group": "",
                        "contains_identity_document": "",
                        "contains_credentials_or_secrets": "",
                    }
                )
                details[content_key] = {
                    "decision": "pending",
                    "reason": "Useful life record",
                    "category": "life_event",
                    "confidence": 0.95,
                    "semantic_recommended_decision": "approve",
                    "semantic_relevant_to_alice": False,
                    "semantic_contains_third_party_private_data": False,
                    "extraction_status": "ok",
                    "extraction_truncated": False,
                    "identity_flag": "no",
                    "credential_flag": "no",
                    "contradiction_topic": (
                        "plan-change" if index in {1, 2} else ""
                    ),
                }

            proposal_path = exports / f"pilot-proposal-{proposal_id}.csv"
            with proposal_path.open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            (exports / "pilot-policy-calibration-details-run.json").write_text(
                json.dumps({"content_results": details}),
                encoding="utf-8",
            )

            summary = apply_final_pilot_policy(
                vault_root=vault,
                audit_approved=2,
                audit_rejected=0,
            )
            self.assertEqual(summary["auto_approved"], 6)
            self.assertEqual(summary["manual_review_required"], 0)
            self.assertTrue(summary["canonical_review_updated"])
            self.assertTrue(Path(summary["audit_csv_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
