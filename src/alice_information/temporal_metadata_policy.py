"""Versioned temporal-metadata evidence policy for Phase 4 P4.4b."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InformationContractError
from .freshness_policy import InformationFreshnessPolicy
from .policy import InformationPolicy

DEFAULT_TEMPORAL_METADATA_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_temporal_metadata_policy.json"
)

ALLOWED_TEMPORAL_METADATA_KINDS = ("published_at", "updated_at")
ALLOWED_TEMPORAL_METADATA_ORIGINS = (
    "html_meta_article_published_time",
    "html_meta_article_modified_time",
    "html_meta_date_published",
    "html_meta_date_modified",
    "html_time_date_published",
    "html_time_date_modified",
    "http_last_modified",
)
ALLOWED_TEMPORAL_METADATA_VERDICTS = (
    "resolved",
    "undated",
    "invalid",
    "conflict",
)
ALLOWED_TEMPORAL_CONSENSUS_VERDICTS = (
    "consistent",
    "insufficient",
    "conflict",
)
APPROVED_MAX_CANDIDATES = 32
APPROVED_MAX_RAW_VALUE_CHARACTERS = 256
APPROVED_MIN_CROSS_SOURCE_OBSERVATIONS = 2
APPROVED_MAX_CROSS_SOURCE_OBSERVATIONS = 16


class InformationTemporalMetadataPolicyError(InformationContractError):
    """Raised when the public P4.4b temporal-metadata policy is invalid."""


@dataclass(frozen=True)
class InformationTemporalMetadataPolicy:
    """Fail-closed policy for extracted date evidence and conflict handling."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    allowed_candidate_kinds: tuple[str, ...]
    allowed_candidate_origins: tuple[str, ...]
    max_candidates: int
    max_raw_value_characters: int
    min_cross_source_observations: int
    max_cross_source_observations: int
    deterministic_html_metadata_extraction_required: bool
    http_last_modified_allowed: bool
    visible_text_date_inference_allowed: bool
    model_date_extraction_allowed: bool
    invalid_candidates_fail_closed: bool
    conflicting_candidates_fail_closed: bool
    conflict_winner_selection_allowed: bool
    raw_temporal_metadata_logging_allowed: bool
    source_digest_binding_required: bool
    explicit_subject_digest_required: bool
    cross_source_conflicts_preserved: bool

    def validate(
        self,
        *,
        information_policy: InformationPolicy | None = None,
        freshness_policy: InformationFreshnessPolicy | None = None,
    ) -> None:
        if self.policy_name != "alice_information_temporal_metadata_policy":
            raise InformationTemporalMetadataPolicyError(
                "Unexpected P4.4b temporal-metadata policy name."
            )
        if self.version != "1.0.0":
            raise InformationTemporalMetadataPolicyError(
                "P4.4b temporal-metadata policy version must be 1.0.0."
            )
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.4b",
            "deterministic_temporal_metadata_evidence",
        ):
            raise InformationTemporalMetadataPolicyError(
                "Temporal-metadata policy milestone binding is invalid."
            )
        if self.permission_id != "web.search":
            raise InformationTemporalMetadataPolicyError(
                "P4.4b must remain bound to web.search."
            )
        if self.allowed_candidate_kinds != ALLOWED_TEMPORAL_METADATA_KINDS:
            raise InformationTemporalMetadataPolicyError(
                "P4.4b temporal candidate kinds changed."
            )
        if self.allowed_candidate_origins != ALLOWED_TEMPORAL_METADATA_ORIGINS:
            raise InformationTemporalMetadataPolicyError(
                "P4.4b temporal candidate origins changed."
            )
        if self.max_candidates != APPROVED_MAX_CANDIDATES:
            raise InformationTemporalMetadataPolicyError(
                "P4.4b candidate limit changed without a policy-version change."
            )
        if self.max_raw_value_characters != APPROVED_MAX_RAW_VALUE_CHARACTERS:
            raise InformationTemporalMetadataPolicyError(
                "P4.4b raw-value limit changed without a policy-version change."
            )
        if (
            self.min_cross_source_observations
            != APPROVED_MIN_CROSS_SOURCE_OBSERVATIONS
        ):
            raise InformationTemporalMetadataPolicyError(
                "P4.4b minimum observation limit changed without a policy-version change."
            )
        if (
            self.max_cross_source_observations
            != APPROVED_MAX_CROSS_SOURCE_OBSERVATIONS
        ):
            raise InformationTemporalMetadataPolicyError(
                "P4.4b observation limit changed without a policy-version change."
            )
        required_true = (
            self.deterministic_html_metadata_extraction_required,
            self.http_last_modified_allowed,
            self.invalid_candidates_fail_closed,
            self.conflicting_candidates_fail_closed,
            self.source_digest_binding_required,
            self.explicit_subject_digest_required,
            self.cross_source_conflicts_preserved,
        )
        if not all(value is True for value in required_true):
            raise InformationTemporalMetadataPolicyError(
                "Required P4.4b controls must remain enabled."
            )
        required_false = (
            self.visible_text_date_inference_allowed,
            self.model_date_extraction_allowed,
            self.conflict_winner_selection_allowed,
            self.raw_temporal_metadata_logging_allowed,
        )
        if not all(value is False for value in required_false):
            raise InformationTemporalMetadataPolicyError(
                "Prohibited P4.4b capabilities must remain disabled."
            )
        if information_policy is not None:
            information_policy.validate()
            if information_policy.raw_content_logging_allowed is not False:
                raise InformationTemporalMetadataPolicyError(
                    "Base policy must prohibit raw source logging."
                )
            if information_policy.capabilities.external_action_allowed is not False:
                raise InformationTemporalMetadataPolicyError(
                    "External actions must remain disabled."
                )
            if information_policy.capabilities.memory_write_allowed is not False:
                raise InformationTemporalMetadataPolicyError(
                    "Memory writes must remain disabled."
                )
        if freshness_policy is not None:
            freshness_policy.validate(information_policy=information_policy)
            if freshness_policy.raw_temporal_metadata_logging_allowed is not False:
                raise InformationTemporalMetadataPolicyError(
                    "Freshness policy must prohibit raw temporal metadata logging."
                )
            if freshness_policy.model_temporal_inference_allowed is not False:
                raise InformationTemporalMetadataPolicyError(
                    "Freshness policy must prohibit model temporal inference."
                )


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InformationTemporalMetadataPolicyError(f"{field} must be an object.")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise InformationTemporalMetadataPolicyError(
            f"{field} contains missing or unknown keys."
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationTemporalMetadataPolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _strict_bool(value: Any, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationTemporalMetadataPolicyError(
            f"{field} must remain {str(expected).lower()}."
        )
    return expected


def _exact_int(value: Any, field: str, expected: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise InformationTemporalMetadataPolicyError(
            f"{field} must equal the approved value {expected}."
        )
    return value


def parse_information_temporal_metadata_policy(
    payload: dict[str, Any],
    *,
    information_policy: InformationPolicy | None = None,
    freshness_policy: InformationFreshnessPolicy | None = None,
) -> InformationTemporalMetadataPolicy:
    """Validate and project one decoded P4.4b temporal-metadata policy."""

    if not isinstance(payload, dict):
        raise InformationTemporalMetadataPolicyError(
            "Temporal-metadata policy root must be an object."
        )
    expected_root = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "permission_id",
        "allowed_candidate_kinds",
        "allowed_candidate_origins",
        "limits",
        "deterministic_html_metadata_extraction_required",
        "http_last_modified_allowed",
        "visible_text_date_inference_allowed",
        "model_date_extraction_allowed",
        "invalid_candidates_fail_closed",
        "conflicting_candidates_fail_closed",
        "conflict_winner_selection_allowed",
        "raw_temporal_metadata_logging_allowed",
        "source_digest_binding_required",
        "explicit_subject_digest_required",
        "cross_source_conflicts_preserved",
    }
    _exact_keys(payload, expected_root, "policy")
    limits = _mapping(payload["limits"], "limits")
    _exact_keys(
        limits,
        {
            "max_candidates",
            "max_raw_value_characters",
            "min_cross_source_observations",
            "max_cross_source_observations",
        },
        "limits",
    )
    if payload["allowed_candidate_kinds"] != list(
        ALLOWED_TEMPORAL_METADATA_KINDS
    ):
        raise InformationTemporalMetadataPolicyError(
            "allowed_candidate_kinds must match the approved vocabulary."
        )
    if payload["allowed_candidate_origins"] != list(
        ALLOWED_TEMPORAL_METADATA_ORIGINS
    ):
        raise InformationTemporalMetadataPolicyError(
            "allowed_candidate_origins must match the approved vocabulary."
        )
    policy = InformationTemporalMetadataPolicy(
        policy_name=_text(payload["policy_name"], "policy_name"),
        version=_text(payload["version"], "version"),
        phase=_text(payload["phase"], "phase"),
        milestone=_text(payload["milestone"], "milestone"),
        status=_text(payload["status"], "status"),
        permission_id=_text(payload["permission_id"], "permission_id"),
        allowed_candidate_kinds=tuple(payload["allowed_candidate_kinds"]),
        allowed_candidate_origins=tuple(payload["allowed_candidate_origins"]),
        max_candidates=_exact_int(
            limits["max_candidates"],
            "limits.max_candidates",
            APPROVED_MAX_CANDIDATES,
        ),
        max_raw_value_characters=_exact_int(
            limits["max_raw_value_characters"],
            "limits.max_raw_value_characters",
            APPROVED_MAX_RAW_VALUE_CHARACTERS,
        ),
        min_cross_source_observations=_exact_int(
            limits["min_cross_source_observations"],
            "limits.min_cross_source_observations",
            APPROVED_MIN_CROSS_SOURCE_OBSERVATIONS,
        ),
        max_cross_source_observations=_exact_int(
            limits["max_cross_source_observations"],
            "limits.max_cross_source_observations",
            APPROVED_MAX_CROSS_SOURCE_OBSERVATIONS,
        ),
        deterministic_html_metadata_extraction_required=_strict_bool(
            payload["deterministic_html_metadata_extraction_required"],
            "deterministic_html_metadata_extraction_required",
            True,
        ),
        http_last_modified_allowed=_strict_bool(
            payload["http_last_modified_allowed"],
            "http_last_modified_allowed",
            True,
        ),
        visible_text_date_inference_allowed=_strict_bool(
            payload["visible_text_date_inference_allowed"],
            "visible_text_date_inference_allowed",
            False,
        ),
        model_date_extraction_allowed=_strict_bool(
            payload["model_date_extraction_allowed"],
            "model_date_extraction_allowed",
            False,
        ),
        invalid_candidates_fail_closed=_strict_bool(
            payload["invalid_candidates_fail_closed"],
            "invalid_candidates_fail_closed",
            True,
        ),
        conflicting_candidates_fail_closed=_strict_bool(
            payload["conflicting_candidates_fail_closed"],
            "conflicting_candidates_fail_closed",
            True,
        ),
        conflict_winner_selection_allowed=_strict_bool(
            payload["conflict_winner_selection_allowed"],
            "conflict_winner_selection_allowed",
            False,
        ),
        raw_temporal_metadata_logging_allowed=_strict_bool(
            payload["raw_temporal_metadata_logging_allowed"],
            "raw_temporal_metadata_logging_allowed",
            False,
        ),
        source_digest_binding_required=_strict_bool(
            payload["source_digest_binding_required"],
            "source_digest_binding_required",
            True,
        ),
        explicit_subject_digest_required=_strict_bool(
            payload["explicit_subject_digest_required"],
            "explicit_subject_digest_required",
            True,
        ),
        cross_source_conflicts_preserved=_strict_bool(
            payload["cross_source_conflicts_preserved"],
            "cross_source_conflicts_preserved",
            True,
        ),
    )
    policy.validate(
        information_policy=information_policy,
        freshness_policy=freshness_policy,
    )
    return policy


def load_information_temporal_metadata_policy(
    path: Path | str = DEFAULT_TEMPORAL_METADATA_POLICY_PATH,
    *,
    information_policy: InformationPolicy | None = None,
    freshness_policy: InformationFreshnessPolicy | None = None,
) -> InformationTemporalMetadataPolicy:
    """Load the exact public P4.4b temporal-metadata policy from disk."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationTemporalMetadataPolicyError(
            "Could not load the P4.4b temporal-metadata policy."
        ) from exc
    return parse_information_temporal_metadata_policy(
        payload,
        information_policy=information_policy,
        freshness_policy=freshness_policy,
    )
