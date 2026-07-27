"""Deterministic freshness and temporal reasoning for Phase 4 P4.4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re
import unicodedata

from .contracts import InformationContractError, InformationQuery, sha256_text
from .freshness_policy import (
    ALLOWED_FRESHNESS_VERDICTS,
    ALLOWED_TEMPORAL_INTENTS,
    InformationFreshnessPolicy,
)
from .injection_firewall import InformationInspectedSource
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy

TEMPORAL_BASES = ("updated_at", "published_at", "none")
_ERROR_MESSAGES = {
    "freshness_insufficient": "Retrieved source does not satisfy the requested temporal requirement.",
    "temporal_intent_invalid": "Temporal intent validation failed.",
    "temporal_metadata_invalid": "Retrieved source temporal metadata is invalid.",
    "temporal_binding_invalid": "Freshness assessment binding validation failed.",
}

_TEMPORAL_SIGNAL_PATTERNS = {
    "latest": re.compile(r"\b(?:latest|newest|most\s+recent)\b", re.IGNORECASE),
    "current": re.compile(r"\b(?:current|currently|today|right\s+now)\b", re.IGNORECASE),
    "recent": re.compile(r"(?:\brecently\b|(?<!most\s)\brecent\b|\b(?:past|last|this)\s+(?:week|month|quarter)\b|\byesterday\b)", re.IGNORECASE),
    "historical": re.compile(r"\b(?:historical|history|last\s+year|in\s+(?:19|20)\d{2}|as\s+of\s+(?:19|20)\d{2}|between\s+(?:19|20)\d{2}\s+and\s+(?:19|20)\d{2}|before\s+(?:19|20)\d{2}|after\s+(?:19|20)\d{2})\b", re.IGNORECASE),
}
_UNSUPPORTED_FUTURE_PATTERN = re.compile(
    r"\b(?:tomorrow|next\s+(?:week|month|quarter|year)|upcoming|future)\b",
    re.IGNORECASE,
)


class InformationFreshnessError(InformationContractError):
    """Sanitized P4.4 temporal failure with an approved code."""

    def __init__(self, code: str):
        if code not in _ERROR_MESSAGES:
            code = "temporal_binding_invalid"
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


def freshness_failure(code: str) -> InformationFreshnessError:
    return InformationFreshnessError(code)


def _detect_temporal_kind(text: str) -> str:
    detection_view = unicodedata.normalize("NFKC", text).casefold()
    for character in detection_view:
        if unicodedata.category(character) in {"Cf", "Cc"} and character not in "\n\r\t":
            raise freshness_failure("temporal_intent_invalid")
    if _UNSUPPORTED_FUTURE_PATTERN.search(detection_view):
        raise freshness_failure("temporal_intent_invalid")
    matched = tuple(
        kind
        for kind, pattern in _TEMPORAL_SIGNAL_PATTERNS.items()
        if pattern.search(detection_view)
    )
    if len(matched) > 1:
        raise freshness_failure("temporal_intent_invalid")
    return matched[0] if matched else "time_insensitive"


def _parse_timestamp(value: str | None, *, required: bool = True) -> datetime | None:
    if value is None:
        if required:
            raise freshness_failure("temporal_metadata_invalid")
        return None
    if not isinstance(value, str) or not value.strip():
        raise freshness_failure("temporal_metadata_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise freshness_failure("temporal_metadata_invalid") from exc
    if parsed.tzinfo is None:
        raise freshness_failure("temporal_metadata_invalid")
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    """Return canonical UTC text without discarding fractional precision."""

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise freshness_failure("temporal_binding_invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise freshness_failure("temporal_binding_invalid") from exc
    return value.lower()


@dataclass(frozen=True)
class InformationTemporalIntent:
    """Explicit, query-bound temporal requirement for one research request."""

    intent_id: str
    query_id: str
    query_content_sha256: str
    kind: str
    reference_time: str
    window_start: str | None = None
    window_end: str | None = None

    @classmethod
    def create(
        cls,
        *,
        intent_id: str,
        query: InformationQuery,
        kind: str,
        reference_time: str,
        window_start: str | None = None,
        window_end: str | None = None,
        policy: InformationFreshnessPolicy,
    ) -> "InformationTemporalIntent":
        intent = cls(
            intent_id=intent_id,
            query_id=query.query_id,
            query_content_sha256=query.content_sha256,
            kind=kind,
            reference_time=reference_time,
            window_start=window_start,
            window_end=window_end,
        )
        intent.validate(query=query, policy=policy)
        return intent

    def validate(self, *, query: InformationQuery, policy: InformationFreshnessPolicy) -> None:
        query.validate()
        policy.validate()
        if not isinstance(self.intent_id, str) or not self.intent_id.strip():
            raise freshness_failure("temporal_intent_invalid")
        if self.query_id != query.query_id:
            raise freshness_failure("temporal_intent_invalid")
        if _require_digest(self.query_content_sha256) != query.content_sha256:
            raise freshness_failure("temporal_intent_invalid")
        if self.kind not in ALLOWED_TEMPORAL_INTENTS or self.kind not in policy.allowed_intents:
            raise freshness_failure("temporal_intent_invalid")
        if policy.deterministic_query_classification_required:
            if _detect_temporal_kind(query.text) != self.kind:
                raise freshness_failure("temporal_intent_invalid")
        reference = _parse_timestamp(self.reference_time)
        query_created = _parse_timestamp(query.created_at)
        assert reference is not None and query_created is not None
        skew = timedelta(seconds=policy.max_clock_skew_seconds)
        if abs((reference - query_created).total_seconds()) > skew.total_seconds():
            raise freshness_failure("temporal_intent_invalid")
        start = _parse_timestamp(self.window_start, required=False)
        end = _parse_timestamp(self.window_end, required=False)
        if self.kind == "historical":
            if start is None or end is None:
                raise freshness_failure("temporal_intent_invalid")
            if start > end:
                raise freshness_failure("temporal_intent_invalid")
            if end > reference:
                raise freshness_failure("temporal_intent_invalid")
        elif start is not None or end is not None:
            raise freshness_failure("temporal_intent_invalid")


@dataclass(frozen=True)
class DeterministicInformationTemporalClassifier:
    """Conservative, model-free temporal classifier for explicit query language."""

    policy: InformationFreshnessPolicy

    def __post_init__(self) -> None:
        self.policy.validate()

    def classify(
        self,
        query: InformationQuery,
        *,
        reference_time: str,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> InformationTemporalIntent:
        query.validate()
        self.policy.validate()
        kind = _detect_temporal_kind(query.text)
        seed = "\n".join(
            (
                query.query_id,
                query.content_sha256,
                kind,
                reference_time,
                window_start or "none",
                window_end or "none",
                self.policy.version,
            )
        )
        return InformationTemporalIntent.create(
            intent_id=f"temporal-{sha256_text(seed)[:16]}",
            query=query,
            kind=kind,
            reference_time=reference_time,
            window_start=window_start,
            window_end=window_end,
            policy=self.policy,
        )


@dataclass(frozen=True)
class InformationFreshnessAssessment:
    """Reproducible, digest-bound temporal assessment for one source version."""

    source_id: str
    source_content_sha256: str
    source_metadata_sha256: str
    intent_id: str
    intent_sha256: str
    query_content_sha256: str
    policy_version: str
    intent_kind: str
    reference_time: str
    verdict: str
    temporal_basis: str
    effective_source_time: str | None
    age_seconds: int | None
    supports_claim: bool
    assessed_at: str

    def validate(
        self,
        *,
        inspected_source: InformationInspectedSource,
        intent: InformationTemporalIntent,
        query: InformationQuery,
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
    ) -> None:
        expected = _derive_assessment(
            inspected_source=inspected_source,
            intent=intent,
            query=query,
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
        )
        if self != expected:
            raise freshness_failure("temporal_binding_invalid")


@dataclass(frozen=True)
class InformationTemporallyQualifiedSource:
    """Clear source plus an exact P4.4 temporal assessment."""

    inspected_source: InformationInspectedSource
    intent: InformationTemporalIntent
    assessment: InformationFreshnessAssessment

    def validate(
        self,
        *,
        query: InformationQuery,
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
    ) -> None:
        self.assessment.validate(
            inspected_source=self.inspected_source,
            intent=self.intent,
            query=query,
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
        )

    def render_for_model(
        self,
        *,
        query: InformationQuery,
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
    ) -> str:
        self.validate(
            query=query,
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
        )
        if not self.assessment.supports_claim:
            raise freshness_failure("freshness_insufficient")
        inner = self.inspected_source.render_for_model(policy=firewall_policy)
        boundary_seed = "\n".join(
            (
                self.assessment.source_content_sha256,
                self.assessment.intent_id,
                self.assessment.reference_time,
                self.assessment.verdict,
            )
        )
        boundary = f"ALICE-FRESHNESS-ASSESSMENT-{sha256_text(boundary_seed).upper()}"
        return "\n".join(
            (
                f"BEGIN VERIFIED SOURCE FRESHNESS {boundary}",
                f"Freshness policy: {freshness_policy.policy_name}@{freshness_policy.version}",
                f"Temporal intent: {self.assessment.intent_kind}",
                f"Freshness verdict: {self.assessment.verdict}",
                f"Temporal basis: {self.assessment.temporal_basis}",
                f"Effective source time: {self.assessment.effective_source_time or 'unknown'}",
                f"Reference time: {self.assessment.reference_time}",
                f"Age seconds: {self.assessment.age_seconds if self.assessment.age_seconds is not None else 'unknown'}",
                inner,
                f"END VERIFIED SOURCE FRESHNESS {boundary}",
            )
        )


@dataclass(frozen=True)
class DeterministicInformationFreshnessEvaluator:
    """Model-free freshness evaluator operating only on clear inspected sources."""

    information_policy: InformationPolicy
    firewall_policy: InformationInjectionFirewallPolicy
    freshness_policy: InformationFreshnessPolicy

    def __post_init__(self) -> None:
        self.freshness_policy.validate(
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
        )

    def assess(
        self,
        inspected_source: InformationInspectedSource,
        *,
        intent: InformationTemporalIntent,
        query: InformationQuery,
    ) -> InformationTemporallyQualifiedSource:
        assessment = _derive_assessment(
            inspected_source=inspected_source,
            intent=intent,
            query=query,
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
        )
        return InformationTemporallyQualifiedSource(
            inspected_source=inspected_source,
            intent=intent,
            assessment=assessment,
        )


def _intent_sha256(intent: InformationTemporalIntent) -> str:
    return sha256_text(
        "\n".join(
            (
                intent.intent_id,
                intent.query_id,
                intent.query_content_sha256,
                intent.kind,
                intent.reference_time,
                intent.window_start or "none",
                intent.window_end or "none",
            )
        )
    )


def _source_metadata_sha256(inspected_source: InformationInspectedSource) -> str:
    source = inspected_source.source
    return sha256_text(
        "\n".join(
            (
                source.source_id,
                source.canonical_url,
                source.title,
                source.retrieved_at,
                source.published_at or "unknown",
                source.updated_at or "unknown",
                source.content_sha256,
            )
        )
    )


def _derive_assessment(
    *,
    inspected_source: InformationInspectedSource,
    intent: InformationTemporalIntent,
    query: InformationQuery,
    information_policy: InformationPolicy,
    firewall_policy: InformationInjectionFirewallPolicy,
    freshness_policy: InformationFreshnessPolicy,
) -> InformationFreshnessAssessment:
    freshness_policy.validate(
        information_policy=information_policy,
        firewall_policy=firewall_policy,
    )
    inspected_source.validate(policy=firewall_policy)
    if inspected_source.inspection.verdict != "clear":
        raise freshness_failure("freshness_insufficient")
    intent.validate(query=query, policy=freshness_policy)
    source = inspected_source.source
    retrieved = _parse_timestamp(source.retrieved_at)
    published = _parse_timestamp(source.published_at, required=False)
    updated = _parse_timestamp(source.updated_at, required=False)
    reference = _parse_timestamp(intent.reference_time)
    assert retrieved is not None and reference is not None
    skew = timedelta(seconds=freshness_policy.max_clock_skew_seconds)
    if retrieved > reference + skew:
        raise freshness_failure("temporal_metadata_invalid")
    for source_time in (published, updated):
        if source_time is not None and source_time > retrieved + skew:
            raise freshness_failure("temporal_metadata_invalid")
        if source_time is not None and source_time > reference + skew:
            raise freshness_failure("temporal_metadata_invalid")
    if published is not None and updated is not None and updated + skew < published:
        raise freshness_failure("temporal_metadata_invalid")
    if updated is not None:
        basis = "updated_at"
        effective = updated
    elif published is not None:
        basis = "published_at"
        effective = published
    else:
        basis = "none"
        effective = None
    verdict: str
    supports_claim: bool
    age_seconds: int | None
    if effective is None:
        age_seconds = None
        age = None
    else:
        age = max(timedelta(0), reference - effective)
        age_seconds = math.ceil(age.total_seconds())
    if intent.kind == "time_insensitive":
        verdict = "time_insensitive"
        supports_claim = True
    elif intent.kind == "historical":
        if effective is None:
            verdict = "unknown"
            supports_claim = False
        else:
            start = _parse_timestamp(intent.window_start)
            end = _parse_timestamp(intent.window_end)
            assert start is not None and end is not None
            supports_claim = start <= effective <= end
            verdict = "historical_match" if supports_claim else "historical_mismatch"
    else:
        if effective is None:
            verdict = "unknown"
            supports_claim = False
        else:
            assert age is not None
            supports_claim = age <= timedelta(
                seconds=freshness_policy.max_age_seconds(intent.kind)
            )
            verdict = "fresh" if supports_claim else "stale"
    if verdict not in ALLOWED_FRESHNESS_VERDICTS:
        raise freshness_failure("temporal_binding_invalid")
    canonical_reference = _timestamp_text(reference)
    assessment = InformationFreshnessAssessment(
        source_id=source.source_id,
        source_content_sha256=_require_digest(source.content_sha256),
        source_metadata_sha256=_source_metadata_sha256(inspected_source),
        intent_id=intent.intent_id,
        intent_sha256=_intent_sha256(intent),
        query_content_sha256=_require_digest(query.content_sha256),
        policy_version=freshness_policy.version,
        intent_kind=intent.kind,
        reference_time=canonical_reference,
        verdict=verdict,
        temporal_basis=basis,
        effective_source_time=_timestamp_text(effective) if effective is not None else None,
        age_seconds=age_seconds,
        supports_claim=supports_claim,
        assessed_at=canonical_reference,
    )
    if assessment.temporal_basis not in TEMPORAL_BASES:
        raise freshness_failure("temporal_binding_invalid")
    return assessment
