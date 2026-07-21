from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.owner_attribution import (
    annotate_context_owner_relation,
    classify_owner_relation,
    initialize_owner_identity,
    load_owner_identity,
)


class OwnerAttributionTests(unittest.TestCase):
    def test_named_resume_is_high_confidence_self_record(self):
        identity = {
            "primary_name": "Alex Example",
            "aliases": ["Alex Example", "A. Example"],
        }
        evidence = {
            "context_text": (
                "Research Experience: Built an AFM segmentation pipeline."
            ),
            "provenance": [
                {
                    "filename": "Alex_Example_Resume.pdf",
                    "original_relative_path": "Portfolio/Resume.pdf",
                    "source_bucket": "Portfolio",
                }
            ],
        }
        result = classify_owner_relation(
            evidence=evidence,
            identity=identity,
        )
        self.assertEqual(
            result["owner_relation"],
            "owner_self_record",
        )
        self.assertEqual(
            result["owner_relation_confidence"],
            "high",
        )

    def test_generic_work_export_is_not_silently_self_attributed(self):
        identity = {
            "primary_name": "Alex Example",
            "aliases": ["Alex Example"],
        }
        evidence = {
            "context_text": "A generic work activity record.",
            "provenance": [
                {
                    "filename": "work.html",
                    "original_relative_path": "export/work.html",
                    "source_bucket": "export",
                }
            ],
        }
        result = classify_owner_relation(
            evidence=evidence,
            identity=identity,
        )
        self.assertNotEqual(
            result["owner_relation"],
            "owner_self_record",
        )

    def test_private_identity_initialization_and_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp) / "vault"
            vault.mkdir()
            initialize_owner_identity(
                vault_root=vault,
                primary_name="Alex Example",
                aliases=["A. Example"],
            )
            identity = load_owner_identity(
                vault_root=vault,
            )
            self.assertEqual(
                identity["primary_name"],
                "Alex Example",
            )

            package = {
                "package_id": "p1",
                "query": "What research experience do I have?",
                "evidence": [
                    {
                        "citation": "[S1]",
                        "context_text": (
                            "Research Experience: AFM analysis."
                        ),
                        "provenance": [
                            {
                                "filename": "Alex_Example_Resume.pdf",
                                "original_relative_path": "resume.pdf",
                                "source_bucket": "portfolio",
                            }
                        ],
                    }
                ],
                "contradiction_groups": [],
                "guardrails": {},
            }

            # Stub fingerprint-compatible keys are enough for the real helper;
            # it only recomputes the package fingerprint.
            package["pilot_name"] = "pilot-v1"
            annotated = annotate_context_owner_relation(
                vault_root=vault,
                context_package=package,
            )
            self.assertTrue(
                annotated[
                    "owner_identity_context"
                ]["available"]
            )
            self.assertEqual(
                annotated["evidence"][0][
                    "owner_relation"
                ],
                "owner_self_record",
            )


if __name__ == "__main__":
    unittest.main()
