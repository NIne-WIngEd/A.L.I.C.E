from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.auto_review import _decide
from alice_vault.content_extraction import ExtractionResult
from alice_vault.privacy_scan import (
    PrivacyScanResult,
    presidio_blocking_entities,
)
from alice_vault.review_calibration import _existing_policy_decision
from alice_vault.semantic_review import SemanticReview


def extraction(*, truncated: bool = False) -> ExtractionResult:
    return ExtractionResult(
        status="ok",
        text="A useful record about research and education.",
        chars=46,
        truncated=truncated,
        parser="test",
    )


def privacy() -> PrivacyScanResult:
    return PrivacyScanResult(
        secret_types=[],
        identity_document_types=[],
        prompt_injection_types=[],
        pii_counts={},
        sensitive_topics=[],
        presidio_counts={},
        presidio_max_scores={},
    )


def semantic(
    *,
    category: str = "research_project",
    sensitivity: str = "private",
    score: float = 0.90,
    recommendation: str = "approve",
    contradiction: str = "",
) -> SemanticReview:
    return SemanticReview(
        relevant_to_alice=True,
        relevance_score=score,
        recommended_decision=recommendation,
        document_category=category,
        sensitivity=sensitivity,
        contains_identity_document=False,
        contains_credentials_or_secrets=False,
        contains_third_party_private_data=False,
        contradiction_topic=contradiction,
        summary="Relevant record",
        reason="Relevant to the owner",
    )


class PolicyCalibrationTests(unittest.TestCase):
    def test_highly_sensitive_safe_category_is_not_blanket_blocked(self):
        decision = _decide(
            extraction(),
            privacy(),
            semantic(sensitivity="highly_sensitive"),
            approve_threshold=0.85,
            reject_threshold=0.85,
        )
        self.assertEqual(decision[0], "approve")

    def test_contradiction_topic_is_annotation_not_blanket_blocker(self):
        decision = _decide(
            extraction(),
            privacy(),
            semantic(contradiction="project_status"),
            approve_threshold=0.85,
            reject_threshold=0.85,
        )
        self.assertEqual(decision[0], "approve")

    def test_truncated_preview_requires_stronger_approval(self):
        high = _decide(
            extraction(truncated=True),
            privacy(),
            semantic(score=0.95),
            approve_threshold=0.85,
            reject_threshold=0.85,
        )
        lower = _decide(
            extraction(truncated=True),
            privacy(),
            semantic(score=0.90),
            approve_threshold=0.85,
            reject_threshold=0.85,
        )
        self.assertEqual(high[0], "approve")
        self.assertEqual(lower[0], "pending")

    def test_presidio_high_risk_requires_score_and_context(self):
        result = PrivacyScanResult(
            secret_types=[],
            identity_document_types=[],
            prompt_injection_types=[],
            pii_counts={},
            sensitive_topics=[],
            presidio_counts={"US_DRIVER_LICENSE": 1},
            presidio_max_scores={"US_DRIVER_LICENSE": 0.96},
        )
        self.assertEqual(
            presidio_blocking_entities(result, "random identifier ABC123"),
            set(),
        )
        self.assertEqual(
            presidio_blocking_entities(
                result,
                "Driver license number ABC123",
            ),
            {"US_DRIVER_LICENSE"},
        )

    def test_existing_high_confidence_safe_category_is_recalibrated(self):
        result = _existing_policy_decision(
            {
                "decision": "pending",
                "confidence": 0.96,
                "reason": "Highly sensitive content requires human review",
                "category": "education",
                "sensitivity": "highly_sensitive",
            },
            approve_threshold=0.85,
            reject_threshold=0.85,
        )
        self.assertEqual(result["decision"], "approve")


if __name__ == "__main__":
    unittest.main()
