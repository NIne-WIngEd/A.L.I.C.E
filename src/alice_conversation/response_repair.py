"""Deterministic sanitized response-repair requests for A.L.I.C.E. P3.9."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .contracts import (
    ConversationContractError,
    ModelRequest,
    ModelResponse,
    sha256_text,
)
from .grounding_bridge import conversation_grounding_packet_sha256
from .repair_policy import ConversationResponseRepairPolicy
from .response_validation import (
    ConversationResponseValidationReport,
    conversation_response_validation_report_sha256,
)


class ConversationResponseRepairError(ConversationContractError):
    """Raised when a controlled repair request cannot be built safely."""


@dataclass(frozen=True)
class ConversationResponseRepairRequest:
    """One sanitized repair generation plus metadata-safe digests."""

    policy_version: str
    issue_codes: tuple[str, ...]
    original_response_sha256: str
    validation_report_sha256: str
    context_sha256: str
    grounding_packet_sha256: str | None
    repair_request_sha256: str
    generation_id: str
    request: ModelRequest

    def validate(
        self,
        *,
        original_request: ModelRequest,
        policy: ConversationResponseRepairPolicy,
    ) -> None:
        policy.validate()
        original_request.validate()
        self.request.validate()
        if not policy.enabled:
            raise ConversationResponseRepairError(
                "Response repair is not enabled by policy."
            )
        if self.policy_version != policy.version:
            raise ConversationResponseRepairError(
                "Repair request policy version does not match."
            )
        if not self.issue_codes or tuple(sorted(set(self.issue_codes))) != self.issue_codes:
            raise ConversationResponseRepairError(
                "Repair issue codes must be sorted, unique, and non-empty."
            )
        if len(self.issue_codes) > policy.max_issue_codes:
            raise ConversationResponseRepairError(
                "Repair request exceeds the issue-code limit."
            )
        for value in (
            self.original_response_sha256,
            self.validation_report_sha256,
            self.context_sha256,
            self.repair_request_sha256,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ConversationResponseRepairError(
                    "Repair metadata requires lower-case SHA-256 digests."
                )
        if self.grounding_packet_sha256 is not None and (
            len(self.grounding_packet_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.grounding_packet_sha256)
        ):
            raise ConversationResponseRepairError(
                "Repair grounding metadata requires a SHA-256 digest."
            )
        if self.request.session_id != original_request.session_id:
            raise ConversationResponseRepairError("Repair request changed session identity.")
        if self.request.turn_id != original_request.turn_id:
            raise ConversationResponseRepairError("Repair request changed turn identity.")
        if self.request.messages != original_request.messages:
            raise ConversationResponseRepairError(
                "Repair request changed governed conversation context."
            )
        if self.request.grounding != original_request.grounding:
            raise ConversationResponseRepairError(
                "Repair request changed the grounding packet."
            )
        if self.request.capabilities != original_request.capabilities:
            raise ConversationResponseRepairError(
                "Repair request changed the capability boundary."
            )
        if self.request.temperature != original_request.temperature:
            raise ConversationResponseRepairError(
                "Repair request changed deterministic sampling settings."
            )
        if self.request.max_output_tokens > policy.max_repair_output_tokens:
            raise ConversationResponseRepairError(
                "Repair request exceeds the repair output-token limit."
            )
        if not self.request.request_id.startswith("repair-request:"):
            raise ConversationResponseRepairError(
                "Repair request ID is not derived from the repair digest."
            )
        if self.request.request_id.split(":", 1)[1] != self.repair_request_sha256:
            raise ConversationResponseRepairError(
                "Repair request ID does not match its deterministic digest."
            )
        if self.generation_id != f"repair-generation:{self.repair_request_sha256}":
            raise ConversationResponseRepairError(
                "Repair generation ID does not match the deterministic repair digest."
            )
        expected = _repair_digest(
            policy=policy,
            original_request=original_request,
            original_response_sha256=self.original_response_sha256,
            validation_report_sha256=self.validation_report_sha256,
            issue_codes=self.issue_codes,
            context_sha256=self.context_sha256,
            grounding_packet_sha256=self.grounding_packet_sha256,
        )
        if expected != self.repair_request_sha256:
            raise ConversationResponseRepairError(
                "Repair request digest does not match its governed inputs."
            )


def build_conversation_response_repair_request(
    *,
    original_request: ModelRequest,
    rejected_response: ModelResponse,
    validation_report: ConversationResponseValidationReport,
    policy: ConversationResponseRepairPolicy,
    context_sha256: str,
) -> ConversationResponseRepairRequest:
    """Build exactly one repair request without including rejected response text."""

    policy.validate()
    original_request.validate()
    rejected_response.validate()
    validation_report.validate()
    if not policy.enabled:
        raise ConversationResponseRepairError("Response repair is disabled by policy.")
    if validation_report.outcome != "rejected":
        raise ConversationResponseRepairError(
            "Response repair requires a rejected validation report."
        )
    if rejected_response.request_id != original_request.request_id:
        raise ConversationResponseRepairError(
            "Rejected response does not belong to the original request."
        )
    original_response_sha256 = sha256_text(rejected_response.content)
    if validation_report.response_sha256 != original_response_sha256:
        raise ConversationResponseRepairError(
            "Rejected response digest does not match its validation report."
        )
    issue_codes = tuple(sorted({issue.code for issue in validation_report.issues}))
    if not issue_codes or len(issue_codes) > policy.max_issue_codes:
        raise ConversationResponseRepairError(
            "Rejected response has no policy-approved repair issue set."
        )
    grounding_sha256 = (
        conversation_grounding_packet_sha256(original_request.grounding)
        if original_request.grounding is not None
        else None
    )
    report_sha256 = conversation_response_validation_report_sha256(validation_report)
    repair_sha256 = _repair_digest(
        policy=policy,
        original_request=original_request,
        original_response_sha256=original_response_sha256,
        validation_report_sha256=report_sha256,
        issue_codes=issue_codes,
        context_sha256=context_sha256,
        grounding_packet_sha256=grounding_sha256,
    )
    directive = _repair_directive(issue_codes)
    if len(directive) > policy.max_repair_prompt_chars:
        raise ConversationResponseRepairError(
            "Sanitized response-repair directive exceeds its character limit."
        )
    if rejected_response.content in directive:
        raise ConversationResponseRepairError(
            "Rejected response text cannot be copied into a repair directive."
        )
    repair_request = ModelRequest(
        request_id=f"repair-request:{repair_sha256}",
        session_id=original_request.session_id,
        turn_id=original_request.turn_id,
        system_contract_version=(
            f"{original_request.system_contract_version}+repair:{policy.version}"
        ),
        system_contract=f"{original_request.system_contract}\n\n{directive}",
        messages=original_request.messages,
        grounding=original_request.grounding,
        capabilities=original_request.capabilities,
        max_output_tokens=min(
            original_request.max_output_tokens,
            policy.max_repair_output_tokens,
        ),
        temperature=original_request.temperature,
    )
    result = ConversationResponseRepairRequest(
        policy_version=policy.version,
        issue_codes=issue_codes,
        original_response_sha256=original_response_sha256,
        validation_report_sha256=report_sha256,
        context_sha256=context_sha256,
        grounding_packet_sha256=grounding_sha256,
        repair_request_sha256=repair_sha256,
        generation_id=f"repair-generation:{repair_sha256}",
        request=repair_request,
    )
    result.validate(original_request=original_request, policy=policy)
    return result


def _repair_directive(issue_codes: tuple[str, ...]) -> str:
    rendered = "\n".join(f"- {code}" for code in issue_codes)
    return (
        "BEGIN CONTROLLED RESPONSE REPAIR\n"
        "The previous model response was rejected by deterministic validation.\n"
        "Generate one complete replacement answer to the same user request.\n"
        "Use the unchanged conversation context and unchanged grounding data.\n"
        "Do not claim tools, web access, external actions, memory writes, or hidden reasoning.\n"
        "The rejected response text is unavailable and must not be reconstructed.\n"
        "Correct only the following sanitized validation issue codes:\n"
        f"{rendered}\n"
        "END CONTROLLED RESPONSE REPAIR"
    )


def _repair_digest(
    *,
    policy: ConversationResponseRepairPolicy,
    original_request: ModelRequest,
    original_response_sha256: str,
    validation_report_sha256: str,
    issue_codes: tuple[str, ...],
    context_sha256: str,
    grounding_packet_sha256: str | None,
) -> str:
    payload = {
        "policy_version": policy.version,
        "original_request": {
            "request_id_sha256": sha256_text(original_request.request_id),
            "session_id_sha256": sha256_text(original_request.session_id),
            "turn_id_sha256": sha256_text(original_request.turn_id),
            "system_contract_version": original_request.system_contract_version,
            "system_contract_sha256": sha256_text(original_request.system_contract),
            "messages": [
                {
                    "role": message.role,
                    "content_sha256": message.content_sha256,
                    "data_classification": message.data_classification,
                }
                for message in original_request.messages
            ],
            "max_output_tokens": original_request.max_output_tokens,
            "temperature": original_request.temperature,
        },
        "original_response_sha256": original_response_sha256,
        "validation_report_sha256": validation_report_sha256,
        "issue_codes": list(issue_codes),
        "context_sha256": context_sha256,
        "grounding_packet_sha256": grounding_packet_sha256,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
