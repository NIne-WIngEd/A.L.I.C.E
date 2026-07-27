"""Deterministic temporal-metadata evidence and conflict handling for P4.4b."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime, parsedate_to_datetime
from typing import TYPE_CHECKING
import re
import unicodedata
from urllib.parse import urlsplit

from .contracts import InformationContractError, InformationSourceDocument, sha256_text
from .freshness_policy import InformationFreshnessPolicy
from .policy import InformationPolicy
from .temporal_metadata_policy import (
    ALLOWED_TEMPORAL_CONSENSUS_VERDICTS,
    ALLOWED_TEMPORAL_METADATA_KINDS,
    ALLOWED_TEMPORAL_METADATA_ORIGINS,
    ALLOWED_TEMPORAL_METADATA_VERDICTS,
    InformationTemporalMetadataPolicy,
    APPROVED_MAX_RAW_VALUE_CHARACTERS,
)

if TYPE_CHECKING:
    from .retrieval import InformationRetrievedResource


_ERROR_MESSAGES = {
    "temporal_candidate_invalid": "Retrieved temporal metadata candidate was invalid.",
    "temporal_resolution_invalid": "Retrieved temporal metadata resolution was invalid.",
    "temporal_metadata_conflict": "Retrieved temporal metadata contains an unresolved conflict.",
    "temporal_subject_invalid": "Cross-source temporal subject binding was invalid.",
    "temporal_consensus_invalid": "Cross-source temporal consensus validation failed.",
}


_RFC3339_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_IMF_FIXDATE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2} "
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"\d{4} \d{2}:\d{2}:\d{2} GMT$"
)

_ORIGIN_KIND = {
    "html_meta_article_published_time": "published_at",
    "html_meta_article_modified_time": "updated_at",
    "html_meta_date_published": "published_at",
    "html_meta_date_modified": "updated_at",
    "html_time_date_published": "published_at",
    "html_time_date_modified": "updated_at",
    "http_last_modified": "updated_at",
}


class InformationTemporalMetadataError(InformationContractError):
    """Sanitized P4.4b temporal-metadata failure."""

    def __init__(self, code: str):
        if code not in _ERROR_MESSAGES:
            code = "temporal_resolution_invalid"
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


def temporal_metadata_failure(code: str) -> InformationTemporalMetadataError:
    return InformationTemporalMetadataError(code)


def _require_digest(value: str, *, code: str = "temporal_resolution_invalid") -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise temporal_metadata_failure(code)
    try:
        int(value, 16)
    except ValueError as exc:
        raise temporal_metadata_failure(code) from exc
    return value.lower()


def _canonical_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_candidate_timestamp(raw_value: str, *, origin: str) -> str | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    if len(raw_value) > APPROVED_MAX_RAW_VALUE_CHARACTERS:
        return None
    for character in raw_value:
        if unicodedata.category(character) in {"Cf", "Cc"}:
            return None
    stripped = raw_value.strip()
    try:
        if origin == "http_last_modified":
            if not _IMF_FIXDATE.fullmatch(stripped):
                return None
            parsed = parsedate_to_datetime(stripped)
            if parsed.tzinfo is None or format_datetime(parsed, usegmt=True) != stripped:
                return None
        else:
            if not _RFC3339_TIMESTAMP.fullmatch(stripped):
                return None
            if stripped.endswith("-00:00"):
                return None
            parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    return _canonical_timestamp(parsed)


@dataclass(frozen=True)
class InformationTemporalMetadataCandidate:
    """One recognized metadata value preserved for deterministic re-derivation."""

    kind: str
    origin: str
    raw_value: str = field(repr=False)
    raw_value_sha256: str
    normalized_timestamp: str | None
    valid: bool

    @classmethod
    def create(
        cls,
        *,
        origin: str,
        raw_value: str,
    ) -> "InformationTemporalMetadataCandidate":
        kind = _ORIGIN_KIND.get(origin)
        if kind is None:
            raise temporal_metadata_failure("temporal_candidate_invalid")
        if not isinstance(raw_value, str):
            raise temporal_metadata_failure("temporal_candidate_invalid")
        normalized = _parse_candidate_timestamp(raw_value, origin=origin)
        candidate = cls(
            kind=kind,
            origin=origin,
            raw_value=raw_value,
            raw_value_sha256=sha256_text(raw_value),
            normalized_timestamp=normalized,
            valid=normalized is not None,
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if self.kind not in ALLOWED_TEMPORAL_METADATA_KINDS:
            raise temporal_metadata_failure("temporal_candidate_invalid")
        if self.origin not in ALLOWED_TEMPORAL_METADATA_ORIGINS:
            raise temporal_metadata_failure("temporal_candidate_invalid")
        if _ORIGIN_KIND[self.origin] != self.kind:
            raise temporal_metadata_failure("temporal_candidate_invalid")
        if (
            not isinstance(self.raw_value, str)
            or len(self.raw_value) > APPROVED_MAX_RAW_VALUE_CHARACTERS
        ):
            raise temporal_metadata_failure("temporal_candidate_invalid")
        expected = InformationTemporalMetadataCandidate.create_unchecked(
            origin=self.origin,
            raw_value=self.raw_value,
        )
        if self != expected:
            raise temporal_metadata_failure("temporal_candidate_invalid")

    @classmethod
    def create_unchecked(
        cls,
        *,
        origin: str,
        raw_value: str,
    ) -> "InformationTemporalMetadataCandidate":
        kind = _ORIGIN_KIND[origin]
        normalized = _parse_candidate_timestamp(raw_value, origin=origin)
        return cls(
            kind=kind,
            origin=origin,
            raw_value=raw_value,
            raw_value_sha256=sha256_text(raw_value),
            normalized_timestamp=normalized,
            valid=normalized is not None,
        )

    def metadata_record(self) -> dict[str, str | bool | None]:
        """Return a log-safe record without the raw metadata value."""

        self.validate()
        return {
            "kind": self.kind,
            "origin": self.origin,
            "raw_value_sha256": self.raw_value_sha256,
            "normalized_timestamp": self.normalized_timestamp,
            "valid": self.valid,
        }


def temporal_candidate_set_sha256(
    candidates: tuple[InformationTemporalMetadataCandidate, ...],
) -> str:
    parts: list[str] = []
    for candidate in candidates:
        candidate.validate()
        parts.append(
            "\n".join(
                (
                    candidate.kind,
                    candidate.origin,
                    candidate.raw_value_sha256,
                    candidate.normalized_timestamp or "invalid",
                    "valid" if candidate.valid else "invalid",
                )
            )
        )
    return sha256_text("\n---\n".join(parts))


@dataclass(frozen=True)
class InformationTemporalMetadataResolution:
    """Digest-bound resolution of one retrieved resource's date evidence."""

    source_url: str
    source_content_sha256: str
    candidate_set_sha256: str
    policy_version: str
    verdict: str
    published_at: str | None
    updated_at: str | None
    published_origins: tuple[str, ...]
    updated_origins: tuple[str, ...]
    supports_temporal_claims: bool
    resolution_sha256: str

    def validate(
        self,
        *,
        resource: "InformationRetrievedResource",
        policy: InformationTemporalMetadataPolicy,
        freshness_policy: InformationFreshnessPolicy,
    ) -> None:
        expected = _derive_resolution(
            resource=resource,
            policy=policy,
            freshness_policy=freshness_policy,
        )
        if self != expected:
            raise temporal_metadata_failure("temporal_resolution_invalid")


@dataclass(frozen=True)
class InformationResolvedTemporalResource:
    """Retrieved resource plus its exact P4.4b temporal resolution."""

    resource: "InformationRetrievedResource"
    resolution: InformationTemporalMetadataResolution

    def validate(
        self,
        *,
        policy: InformationTemporalMetadataPolicy,
        freshness_policy: InformationFreshnessPolicy,
    ) -> None:
        self.resolution.validate(
            resource=self.resource,
            policy=policy,
            freshness_policy=freshness_policy,
        )

    def to_source_document(
        self,
        *,
        source_id: str,
        provider: str,
        retrieved_at: str,
        policy: InformationTemporalMetadataPolicy,
        freshness_policy: InformationFreshnessPolicy,
    ) -> InformationSourceDocument:
        self.validate(policy=policy, freshness_policy=freshness_policy)
        if self.resolution.verdict in {"invalid", "conflict"}:
            raise temporal_metadata_failure("temporal_metadata_conflict")
        title = (
            self.resource.title
            or urlsplit(self.resource.final_url).hostname
            or "External source"
        )
        return InformationSourceDocument.create(
            source_id=source_id,
            provider=provider,
            url=self.resource.final_url,
            title=title,
            normalized_text=self.resource.normalized_text,
            retrieved_at=retrieved_at,
            published_at=self.resolution.published_at,
            updated_at=self.resolution.updated_at,
        )


@dataclass(frozen=True)
class DeterministicInformationTemporalMetadataResolver:
    """Resolve extracted metadata without natural-language or model inference."""

    information_policy: InformationPolicy
    freshness_policy: InformationFreshnessPolicy
    temporal_metadata_policy: InformationTemporalMetadataPolicy

    def __post_init__(self) -> None:
        self.temporal_metadata_policy.validate(
            information_policy=self.information_policy,
            freshness_policy=self.freshness_policy,
        )

    def resolve(
        self,
        resource: "InformationRetrievedResource",
    ) -> InformationResolvedTemporalResource:
        resolution = _derive_resolution(
            resource=resource,
            policy=self.temporal_metadata_policy,
            freshness_policy=self.freshness_policy,
        )
        return InformationResolvedTemporalResource(
            resource=resource,
            resolution=resolution,
        )


def _resolution_digest(
    *,
    source_url: str,
    source_content_sha256: str,
    candidate_set_sha256: str,
    policy_version: str,
    verdict: str,
    published_at: str | None,
    updated_at: str | None,
    published_origins: tuple[str, ...],
    updated_origins: tuple[str, ...],
    supports_temporal_claims: bool,
) -> str:
    return sha256_text(
        "\n".join(
            (
                source_url,
                source_content_sha256,
                candidate_set_sha256,
                policy_version,
                verdict,
                published_at or "unknown",
                updated_at or "unknown",
                ",".join(published_origins) or "none",
                ",".join(updated_origins) or "none",
                "supported" if supports_temporal_claims else "unsupported",
            )
        )
    )


def _derive_resolution(
    *,
    resource: "InformationRetrievedResource",
    policy: InformationTemporalMetadataPolicy,
    freshness_policy: InformationFreshnessPolicy,
) -> InformationTemporalMetadataResolution:
    policy.validate(freshness_policy=freshness_policy)
    resource.validate()
    candidates = resource.temporal_metadata_candidates
    if len(candidates) > policy.max_candidates:
        raise temporal_metadata_failure("temporal_resolution_invalid")
    candidate_digest = temporal_candidate_set_sha256(candidates)
    invalid = any(not candidate.valid for candidate in candidates)
    published_values = {
        candidate.normalized_timestamp
        for candidate in candidates
        if candidate.valid and candidate.kind == "published_at"
    }
    explicit_updated_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.kind == "updated_at" and candidate.origin != "http_last_modified"
    )
    selected_updated_candidates = (
        explicit_updated_candidates
        if explicit_updated_candidates
        else tuple(
            candidate
            for candidate in candidates
            if candidate.kind == "updated_at"
        )
    )
    updated_values = {
        candidate.normalized_timestamp
        for candidate in selected_updated_candidates
        if candidate.valid
    }
    published_values.discard(None)
    updated_values.discard(None)
    published_origins = tuple(
        sorted(
            {
                candidate.origin
                for candidate in candidates
                if candidate.valid and candidate.kind == "published_at"
            }
        )
    )
    updated_origins = tuple(
        sorted(
            {
                candidate.origin
                for candidate in selected_updated_candidates
                if candidate.valid
            }
        )
    )
    published_at = next(iter(published_values)) if len(published_values) == 1 else None
    updated_at = next(iter(updated_values)) if len(updated_values) == 1 else None
    conflict = len(published_values) > 1 or len(updated_values) > 1
    if published_at is not None and updated_at is not None:
        published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        skew = timedelta(seconds=freshness_policy.max_clock_skew_seconds)
        if updated_dt + skew < published_dt:
            conflict = True
    if invalid:
        verdict = "invalid"
    elif conflict:
        verdict = "conflict"
    elif not candidates:
        verdict = "undated"
    else:
        verdict = "resolved"
    if verdict not in ALLOWED_TEMPORAL_METADATA_VERDICTS:
        raise temporal_metadata_failure("temporal_resolution_invalid")
    if verdict != "resolved":
        published_at = None
        updated_at = None
    supports = verdict == "resolved" and (
        published_at is not None or updated_at is not None
    )
    digest = _resolution_digest(
        source_url=resource.final_url,
        source_content_sha256=resource.content_sha256,
        candidate_set_sha256=candidate_digest,
        policy_version=policy.version,
        verdict=verdict,
        published_at=published_at,
        updated_at=updated_at,
        published_origins=published_origins,
        updated_origins=updated_origins,
        supports_temporal_claims=supports,
    )
    return InformationTemporalMetadataResolution(
        source_url=resource.final_url,
        source_content_sha256=resource.content_sha256,
        candidate_set_sha256=candidate_digest,
        policy_version=policy.version,
        verdict=verdict,
        published_at=published_at,
        updated_at=updated_at,
        published_origins=published_origins,
        updated_origins=updated_origins,
        supports_temporal_claims=supports,
        resolution_sha256=digest,
    )


@dataclass(frozen=True)
class InformationCrossSourceTemporalConsensus:
    """Conflict-preserving aggregation for one explicitly bound temporal fact."""

    subject_sha256: str
    resolution_sha256s: tuple[str, ...]
    policy_version: str
    verdict: str
    published_at: str | None
    updated_at: str | None
    consensus_sha256: str

    def validate(
        self,
        *,
        observations: tuple[InformationResolvedTemporalResource, ...],
        policy: InformationTemporalMetadataPolicy,
        freshness_policy: InformationFreshnessPolicy,
    ) -> None:
        expected = _derive_consensus(
            subject_sha256=self.subject_sha256,
            observations=observations,
            policy=policy,
            freshness_policy=freshness_policy,
        )
        if self != expected:
            raise temporal_metadata_failure("temporal_consensus_invalid")


@dataclass(frozen=True)
class DeterministicInformationTemporalMetadataAggregator:
    """Aggregate only caller-declared same-fact observations; never infer grouping."""

    policy: InformationTemporalMetadataPolicy
    freshness_policy: InformationFreshnessPolicy

    def __post_init__(self) -> None:
        self.policy.validate(freshness_policy=self.freshness_policy)

    def aggregate(
        self,
        *,
        subject_sha256: str,
        observations: tuple[InformationResolvedTemporalResource, ...],
    ) -> InformationCrossSourceTemporalConsensus:
        return _derive_consensus(
            subject_sha256=subject_sha256,
            observations=observations,
            policy=self.policy,
            freshness_policy=self.freshness_policy,
        )


def _consensus_digest(
    *,
    subject_sha256: str,
    resolution_sha256s: tuple[str, ...],
    policy_version: str,
    verdict: str,
    published_at: str | None,
    updated_at: str | None,
) -> str:
    return sha256_text(
        "\n".join(
            (
                subject_sha256,
                ",".join(resolution_sha256s),
                policy_version,
                verdict,
                published_at or "unknown",
                updated_at or "unknown",
            )
        )
    )


def _derive_consensus(
    *,
    subject_sha256: str,
    observations: tuple[InformationResolvedTemporalResource, ...],
    policy: InformationTemporalMetadataPolicy,
    freshness_policy: InformationFreshnessPolicy,
) -> InformationCrossSourceTemporalConsensus:
    policy.validate(freshness_policy=freshness_policy)
    subject = _require_digest(subject_sha256, code="temporal_subject_invalid")
    if (
        not isinstance(observations, tuple)
        or not policy.min_cross_source_observations
        <= len(observations)
        <= policy.max_cross_source_observations
    ):
        raise temporal_metadata_failure("temporal_consensus_invalid")
    for observation in observations:
        if type(observation) is not InformationResolvedTemporalResource:
            raise temporal_metadata_failure("temporal_consensus_invalid")
        observation.validate(policy=policy, freshness_policy=freshness_policy)
    source_urls = tuple(observation.resource.final_url for observation in observations)
    if len(set(source_urls)) != len(source_urls):
        raise temporal_metadata_failure("temporal_consensus_invalid")
    resolution_sha256s = tuple(
        sorted(observation.resolution.resolution_sha256 for observation in observations)
    )
    if len(set(resolution_sha256s)) != len(resolution_sha256s):
        raise temporal_metadata_failure("temporal_consensus_invalid")
    verdicts = {observation.resolution.verdict for observation in observations}
    published_values = {
        observation.resolution.published_at
        for observation in observations
        if observation.resolution.published_at is not None
    }
    updated_values = {
        observation.resolution.updated_at
        for observation in observations
        if observation.resolution.updated_at is not None
    }
    conflict = bool(verdicts & {"invalid", "conflict"})
    conflict = conflict or len(published_values) > 1 or len(updated_values) > 1
    if conflict:
        verdict = "conflict"
        published_at = None
        updated_at = None
    elif not published_values and not updated_values:
        verdict = "insufficient"
        published_at = None
        updated_at = None
    else:
        verdict = "consistent"
        published_at = next(iter(published_values)) if published_values else None
        updated_at = next(iter(updated_values)) if updated_values else None
    if verdict not in ALLOWED_TEMPORAL_CONSENSUS_VERDICTS:
        raise temporal_metadata_failure("temporal_consensus_invalid")
    digest = _consensus_digest(
        subject_sha256=subject,
        resolution_sha256s=resolution_sha256s,
        policy_version=policy.version,
        verdict=verdict,
        published_at=published_at,
        updated_at=updated_at,
    )
    return InformationCrossSourceTemporalConsensus(
        subject_sha256=subject,
        resolution_sha256s=resolution_sha256s,
        policy_version=policy.version,
        verdict=verdict,
        published_at=published_at,
        updated_at=updated_at,
        consensus_sha256=digest,
    )
