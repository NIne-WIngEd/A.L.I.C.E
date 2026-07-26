"""Deterministic generated-response validation for A.L.I.C.E. P3.6."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable

from .contracts import (
    ConversationContractError,
    ConversationGroundingPacket,
    ModelResponse,
)
from .grounding_bridge import conversation_grounding_packet_sha256
from .response_validation_policy import ConversationResponseValidationPolicy


_VALIDATION_OUTCOMES = ("accepted", "abstained", "rejected")
_ISSUE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,127}$")
_CITATION_CANDIDATE = re.compile(
    r"\[[A-Za-z][A-Za-z0-9_.-]{0,63}:[^\]\r\n]{1,256}\]"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_WORD = re.compile(r"[A-Za-z0-9']+")
_PERSONAL_ASSERTION = re.compile(
    r"\b(?:Rayan\s+(?:is|was|has|had|prefers|likes|lives|works|studies|owns|"
    r"said|did|wants|needs)|you\s+(?:are|were|have|had|prefer|like|live|work|"
    r"study|own|said|did|want|need)|your\s+[A-Za-z][A-Za-z -]{0,48}\s+"
    r"(?:is|was|has|had))\b",
    re.IGNORECASE,
)
_FACTUAL_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|does|did|contains|includes|requires|"
    r"uses|supports|completed|passed|failed|occurred|happened|will|can)\b",
    re.IGNORECASE,
)
_NUMBER_FACT = re.compile(
    r"(?:\$\s?\d|\b\d+(?:\.\d+)?\s?(?:%|percent|years?|months?|days?|hours?|"
    r"tests?|files?|commits?|pull requests?)\b|\b20\d{2}\b)",
    re.IGNORECASE,
)
_ACTION_COMPLETION = re.compile(
    r"\b(?:I|A\.L\.I\.C\.E\.|Alice)\s+(?:have\s+|already\s+)?"
    r"(?:sent|emailed|forwarded|booked|scheduled|deleted|archived|updated|"
    r"created|submitted|purchased|ordered|called|posted|uploaded|downloaded|"
    r"installed|ran|executed|completed|changed|modified|contacted|applied)\b|"
    r"\b(?:has|have)\s+been\s+(?:sent|emailed|booked|scheduled|deleted|"
    r"updated|submitted|purchased|ordered|posted|uploaded|installed|executed)\b",
    re.IGNORECASE,
)
_CAPABILITY_CLAIM = re.compile(
    r"\bI\s+(?:searched|browsed)\s+(?:the\s+)?web\b|"
    r"\bI\s+(?:used|called|ran)\s+(?:a\s+)?tool\b|"
    r"\bI\s+(?:accessed|opened|read)\s+your\s+(?:email|calendar|drive|files?)\b|"
    r"\bI\s+(?:saved|wrote|added|updated|promoted)\s+(?:this\s+)?(?:to\s+)?memory\b",
    re.IGNORECASE,
)
_DEPENDENCY_LANGUAGE = re.compile(
    r"\bonly I understand you\b|\byou (?:do not|don't) need anyone else\b|"
    r"\bdo not tell anyone\b|\bdon't tell anyone\b|\bstay with me instead\b|"
    r"\byou only need me\b|\bno one else understands you\b",
    re.IGNORECASE,
)
_HIDDEN_REASONING = re.compile(
    r"\bmy chain[- ]of[- ]thought\b|\bmy hidden reasoning\b|"
    r"\binternal reasoning steps\b|\bprivate reasoning trace\b",
    re.IGNORECASE,
)
_CERTAINTY = re.compile(
    r"\b(?:definitely|certainly|unquestionably|conclusively|without doubt|"
    r"proves that|guaranteed)\b",
    re.IGNORECASE,
)
_CONFLICT = re.compile(
    r"\b(?:conflict|conflicting|contradiction|contradictory|disagree|"
    r"inconsistent|disputed|sources differ|evidence differs)\b",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"\b(?:uncertain|unclear|not certain|cannot confirm|can't confirm|may|might|"
    r"appears|suggests|possibly|insufficient confidence)\b",
    re.IGNORECASE,
)
_INSUFFICIENT = re.compile(
    r"\b(?:insufficient evidence|not enough evidence|cannot determine|can't determine|"
    r"unable to determine|do not have enough evidence|don't have enough evidence)\b",
    re.IGNORECASE,
)
_DENIAL = re.compile(
    r"\b(?:cannot comply|can't comply|cannot help with|can't help with|decline|"
    r"not permitted|request is denied|I must refuse)\b",
    re.IGNORECASE,
)
_NOT_APPLICABLE = re.compile(
    r"\b(?:not applicable|does not apply|doesn't apply)\b",
    re.IGNORECASE,
)
_SAFE_FACTUAL_PREFIXES = (
    "based on the provided evidence",
    "based on the available evidence",
    "the available evidence is insufficient",
    "the evidence is insufficient",
    "the sources conflict",
    "the evidence conflicts",
    "i cannot determine",
    "i can't determine",
    "i am uncertain",
    "i'm uncertain",
    "this request is denied",
    "this is not applicable",
    "you can ",
    "consider ",
    "try ",
)


class ConversationResponseValidationError(ConversationContractError):
    """Raised when validator inputs or report integrity are invalid."""


@dataclass(frozen=True)
class ConversationResponseValidationIssue:
    code: str
    sentence_index: int | None = None

    def validate(self) -> None:
        if not isinstance(self.code, str) or not _ISSUE_PATTERN.fullmatch(self.code):
            raise ConversationResponseValidationError(
                "Validation issue code must be a safe lower-case identifier."
            )
        if self.sentence_index is not None and (
            isinstance(self.sentence_index, bool)
            or not isinstance(self.sentence_index, int)
            or self.sentence_index < 0
        ):
            raise ConversationResponseValidationError(
                "Validation issue sentence_index must be a non-negative integer."
            )


@dataclass(frozen=True)
class ConversationResponseValidationReport:
    policy_version: str
    request_id: str
    response_sha256: str
    grounding_packet_id: str | None
    grounding_packet_sha256: str | None
    outcome: str
    issues: tuple[ConversationResponseValidationIssue, ...]
    cited_claim_ids: tuple[str, ...]
    cited_token_sha256: tuple[str, ...]

    def validate(self) -> None:
        for field_name in ("policy_version", "request_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConversationResponseValidationError(
                    f"{field_name} must be non-empty text."
                )
        _require_digest(self.response_sha256, field="response_sha256")
        if (self.grounding_packet_id is None) != (
            self.grounding_packet_sha256 is None
        ):
            raise ConversationResponseValidationError(
                "Grounding packet identity and digest must be paired."
            )
        if self.grounding_packet_id is not None:
            if not self.grounding_packet_id.strip():
                raise ConversationResponseValidationError(
                    "grounding_packet_id must be non-empty when present."
                )
            _require_digest(
                self.grounding_packet_sha256 or "",
                field="grounding_packet_sha256",
            )
        if self.outcome not in _VALIDATION_OUTCOMES:
            raise ConversationResponseValidationError(
                f"Unsupported response-validation outcome: {self.outcome!r}"
            )
        for issue in self.issues:
            issue.validate()
        if len(self.issues) != len(
            {(issue.code, issue.sentence_index) for issue in self.issues}
        ):
            raise ConversationResponseValidationError(
                "Response-validation issues cannot contain duplicates."
            )
        if self.outcome == "rejected" and not self.issues:
            raise ConversationResponseValidationError(
                "Rejected validation reports require at least one issue."
            )
        if self.outcome != "rejected" and self.issues:
            raise ConversationResponseValidationError(
                "Accepted or abstained reports cannot contain issues."
            )
        if tuple(sorted(set(self.cited_claim_ids))) != self.cited_claim_ids:
            raise ConversationResponseValidationError(
                "cited_claim_ids must be sorted and unique."
            )
        if tuple(sorted(set(self.cited_token_sha256))) != self.cited_token_sha256:
            raise ConversationResponseValidationError(
                "cited_token_sha256 must be sorted and unique."
            )
        for digest in self.cited_token_sha256:
            _require_digest(digest, field="cited_token_sha256 item")


class ConversationResponseRejectedError(RuntimeError):
    """Raised by orchestration when a deterministic validation report rejects output."""

    def __init__(self, report: ConversationResponseValidationReport) -> None:
        report.validate()
        if report.outcome != "rejected":
            raise ConversationResponseValidationError(
                "ConversationResponseRejectedError requires a rejected report."
            )
        self.report = report
        super().__init__("The generated response failed deterministic validation.")


def validate_conversation_response(
    *,
    response: ModelResponse,
    grounding: ConversationGroundingPacket | None,
    policy: ConversationResponseValidationPolicy,
) -> ConversationResponseValidationReport:
    """Validate one model response without mutating state or repairing content."""

    if not isinstance(policy, ConversationResponseValidationPolicy):
        raise ConversationResponseValidationError(
            "A validated P3.6 response-validation policy is required."
        )
    response.validate()
    if grounding is not None:
        grounding.validate()

    issues: list[ConversationResponseValidationIssue] = []

    def add(code: str, sentence_index: int | None = None) -> None:
        issue = ConversationResponseValidationIssue(code, sentence_index)
        issue.validate()
        key = (issue.code, issue.sentence_index)
        if key not in {(item.code, item.sentence_index) for item in issues}:
            if len(issues) < policy.max_issues:
                issues.append(issue)

    text = response.content
    lowered = text.lower()
    if len(text) > policy.max_response_chars:
        add("response_too_long")
    if policy.safety_rule("reject_truncated_responses") and response.finish_reason == "length":
        add("truncated_response")
    if policy.safety_rule("reject_action_completion_claims") and _ACTION_COMPLETION.search(text):
        add("fabricated_action_completion")
    if policy.safety_rule("reject_capability_claims") and _CAPABILITY_CLAIM.search(text):
        add("fabricated_capability_claim")
    if policy.safety_rule("reject_dependency_language") and _DEPENDENCY_LANGUAGE.search(text):
        add("dependency_building_language")
    if policy.safety_rule("reject_hidden_reasoning_disclosure") and _HIDDEN_REASONING.search(text):
        add("hidden_reasoning_disclosure")

    token_to_claims: dict[str, set[str]] = {}
    claim_by_id = {}
    known_tokens: set[str] = set()
    if grounding is not None:
        for claim in grounding.claims:
            claim_by_id[claim.claim_id] = claim
            for citation in claim.citations:
                known_tokens.add(citation.token)
                token_to_claims.setdefault(citation.token, set()).add(claim.claim_id)

    candidates = tuple(_CITATION_CANDIDATE.findall(text))
    used_known_tokens = {token for token in known_tokens if token in text}
    unknown_tokens = {token for token in candidates if token not in known_tokens}
    if policy.citation_rule("reject_unknown_tokens") and unknown_tokens:
        add("unknown_citation_token")
    if grounding is None and candidates:
        add("citation_without_grounding")

    cited_claim_ids = sorted(
        {
            claim_id
            for token in used_known_tokens
            for claim_id in token_to_claims.get(token, set())
        }
    )

    sentences = _sentences_with_trailing_citations(text)
    for index, sentence in enumerate(sentences):
        sentence_tokens = tuple(
            token for token in used_known_tokens if token in sentence
        )
        sentence_has_known_token = bool(sentence_tokens)
        if sentence_tokens and grounding is not None:
            cited_claims = tuple(
                claim_by_id[claim_id]
                for token in sentence_tokens
                for claim_id in sorted(token_to_claims.get(token, set()))
            )
            if not _sentence_supported_by_cited_claims(sentence, cited_claims):
                add("citation_claim_mismatch", index)
        if _PERSONAL_ASSERTION.search(sentence) and not sentence_has_known_token:
            if grounding is None and policy.safety_rule("reject_invented_personal_facts"):
                add("invented_personal_fact", index)
            elif policy.citation_rule("require_grounded_personal_claims"):
                add("ungrounded_personal_claim", index)
        if (
            grounding is not None
            and grounding.outcome in {"answerable", "conflict", "uncertain"}
            and policy.citation_rule("require_supported_factual_claims")
            and not sentence_has_known_token
            and _looks_like_unsupported_factual_assertion(sentence, grounding)
        ):
            add("unsupported_factual_claim", index)

    if grounding is not None:
        if grounding.outcome == "answerable":
            if len(cited_claim_ids) < policy.minimum_answerable_claims_cited:
                add("missing_required_citation")
            if _INSUFFICIENT.search(text) or _DENIAL.search(text):
                add("unexpected_abstention")
        elif grounding.outcome == "conflict":
            if policy.epistemic_rule("preserve_conflict") and not _CONFLICT.search(text):
                add("conflict_not_preserved")
            if len(cited_claim_ids) < policy.minimum_conflict_claims_cited:
                add("insufficient_conflict_citations")
            if (
                policy.epistemic_rule("reject_certainty_language_for_conflict")
                and _CERTAINTY.search(text)
            ):
                add("false_certainty_on_conflict")
        elif grounding.outcome == "uncertain":
            if policy.epistemic_rule("preserve_uncertainty") and not _UNCERTAINTY.search(text):
                add("uncertainty_not_preserved")
            if len(cited_claim_ids) < policy.minimum_answerable_claims_cited:
                add("missing_required_citation")
            if (
                policy.epistemic_rule("reject_certainty_language_for_uncertainty")
                and _CERTAINTY.search(text)
            ):
                add("false_certainty_on_uncertainty")
        elif grounding.outcome == "insufficient_evidence":
            if (
                policy.epistemic_rule("require_abstention_on_insufficient_evidence")
                and not _INSUFFICIENT.search(text)
            ):
                add("missing_insufficient_evidence_abstention")
            if candidates:
                add("citation_on_empty_grounding")
        elif grounding.outcome == "denied":
            if policy.epistemic_rule("require_abstention_on_denied") and not _DENIAL.search(text):
                add("missing_denial_abstention")
            if candidates:
                add("citation_on_empty_grounding")
        elif grounding.outcome == "not_applicable":
            if (
                policy.epistemic_rule("require_abstention_on_not_applicable")
                and not _NOT_APPLICABLE.search(text)
            ):
                add("missing_not_applicable_abstention")
            if candidates:
                add("citation_on_empty_grounding")

    outcome = "rejected" if issues else _accepted_outcome(grounding)
    report = ConversationResponseValidationReport(
        policy_version=policy.version,
        request_id=response.request_id,
        response_sha256=_sha256_text(response.content),
        grounding_packet_id=grounding.packet_id if grounding is not None else None,
        grounding_packet_sha256=(
            conversation_grounding_packet_sha256(grounding)
            if grounding is not None
            else None
        ),
        outcome=outcome,
        issues=tuple(issues),
        cited_claim_ids=tuple(cited_claim_ids),
        cited_token_sha256=tuple(
            sorted(_sha256_text(token) for token in used_known_tokens)
        ),
    )
    report.validate()
    return report


def conversation_response_validation_report_sha256(
    report: ConversationResponseValidationReport,
) -> str:
    report.validate()
    payload = {
        "policy_version": report.policy_version,
        "request_id": report.request_id,
        "response_sha256": report.response_sha256,
        "grounding_packet_id": report.grounding_packet_id,
        "grounding_packet_sha256": report.grounding_packet_sha256,
        "outcome": report.outcome,
        "issues": [
            {"code": issue.code, "sentence_index": issue.sentence_index}
            for issue in report.issues
        ],
        "cited_claim_ids": list(report.cited_claim_ids),
        "cited_token_sha256": list(report.cited_token_sha256),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _accepted_outcome(grounding: ConversationGroundingPacket | None) -> str:
    if grounding is not None and grounding.outcome in {
        "insufficient_evidence",
        "denied",
        "not_applicable",
    }:
        return "abstained"
    return "accepted"


def _looks_like_unsupported_factual_assertion(
    sentence: str,
    grounding: ConversationGroundingPacket,
) -> bool:
    stripped = sentence.strip()
    lowered = stripped.lower()
    if not stripped or stripped.endswith("?"):
        return False
    if any(lowered.startswith(prefix) for prefix in _SAFE_FACTUAL_PREFIXES):
        return False
    if _PERSONAL_ASSERTION.search(stripped) or _NUMBER_FACT.search(stripped):
        return True
    sentence_words = _content_words(stripped)
    for claim in grounding.claims:
        claim_words = _content_words(claim.text)
        overlap = sentence_words & claim_words
        if len(overlap) >= 3 and len(overlap) / max(1, min(len(sentence_words), len(claim_words))) >= 0.5:
            return True
    return bool(_FACTUAL_VERB.search(stripped))




def _sentences_with_trailing_citations(text: str) -> tuple[str, ...]:
    raw = [item.strip() for item in _SENTENCE_SPLIT.split(text) if item.strip()]
    grouped: list[str] = []
    leading = re.compile(
        r"^(?P<tokens>(?:\[[A-Za-z][A-Za-z0-9_.-]{0,63}:[^\]\r\n]{1,256}\]\s*)+)(?P<rest>.*)$"
    )
    for item in raw:
        match = leading.match(item)
        if match and grouped:
            grouped[-1] = f"{grouped[-1]} {match.group('tokens').strip()}"
            rest = match.group("rest").strip()
            if rest:
                grouped.append(rest)
            continue
        grouped.append(item)
    return tuple(grouped)

def _sentence_supported_by_cited_claims(sentence: str, claims: Iterable[object]) -> bool:
    cleaned = _CITATION_CANDIDATE.sub("", sentence)
    sentence_words = _content_words(cleaned)
    if not sentence_words:
        return False
    for claim in claims:
        claim_words = _content_words(getattr(claim, "text", ""))
        overlap = sentence_words & claim_words
        if len(overlap) >= 3:
            return True
        if len(overlap) >= 2 and len(overlap) / max(1, min(len(sentence_words), len(claim_words))) >= 0.4:
            return True
    return False

def _content_words(value: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
    return {word.lower() for word in _WORD.findall(value) if word.lower() not in stop}


def _require_digest(value: str, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ConversationResponseValidationError(f"{field} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConversationResponseValidationError(
            f"{field} must contain hexadecimal SHA-256 text."
        ) from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
