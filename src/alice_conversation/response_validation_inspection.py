"""Metadata-safe inspection for P3.6 response-validation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .response_validation import (
    ConversationResponseValidationReport,
    conversation_response_validation_report_sha256,
)


@dataclass(frozen=True)
class ConversationResponseValidationInspection:
    policy_version: str
    request_id: str
    response_sha256: str
    grounding_packet_id: str | None
    grounding_packet_sha256: str | None
    outcome: str
    issue_count: int
    issue_codes: tuple[str, ...]
    cited_claim_count: int
    cited_claim_ids: tuple[str, ...]
    cited_token_count: int
    cited_token_sha256: tuple[str, ...]
    report_sha256: str


def inspect_conversation_response_validation(
    report: ConversationResponseValidationReport,
) -> ConversationResponseValidationInspection:
    """Return report metadata without response text or raw citation tokens."""

    report.validate()
    return ConversationResponseValidationInspection(
        policy_version=report.policy_version,
        request_id=report.request_id,
        response_sha256=report.response_sha256,
        grounding_packet_id=report.grounding_packet_id,
        grounding_packet_sha256=report.grounding_packet_sha256,
        outcome=report.outcome,
        issue_count=len(report.issues),
        issue_codes=tuple(issue.code for issue in report.issues),
        cited_claim_count=len(report.cited_claim_ids),
        cited_claim_ids=report.cited_claim_ids,
        cited_token_count=len(report.cited_token_sha256),
        cited_token_sha256=report.cited_token_sha256,
        report_sha256=conversation_response_validation_report_sha256(report),
    )


def render_conversation_response_validation_inspection(
    inspection: ConversationResponseValidationInspection,
) -> str:
    """Render a deterministic metadata-only JSON inspection record."""

    payload = {
        "policy_version": inspection.policy_version,
        "request_id": inspection.request_id,
        "response_sha256": inspection.response_sha256,
        "grounding_packet_id": inspection.grounding_packet_id,
        "grounding_packet_sha256": inspection.grounding_packet_sha256,
        "outcome": inspection.outcome,
        "issue_count": inspection.issue_count,
        "issue_codes": list(inspection.issue_codes),
        "cited_claim_count": inspection.cited_claim_count,
        "cited_claim_ids": list(inspection.cited_claim_ids),
        "cited_token_count": inspection.cited_token_count,
        "cited_token_sha256": list(inspection.cited_token_sha256),
        "report_sha256": inspection.report_sha256,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
