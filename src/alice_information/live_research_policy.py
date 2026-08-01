"""Exact additive P4.10b live governed-research policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class InformationLiveResearchPolicyError(ValueError):
    """Raised when the exact P4.10b policy is malformed or changed."""


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InformationLiveResearchPolicyError(
                f"Duplicate JSON key is not allowed: {key}."
            )
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


_EXACT_POLICY: dict[str, object] = {
    "information_live_research_policy_schema_version": 1,
    "policy_name": "alice_information_live_research_policy",
    "version": "1.0.0",
    "phase": "4",
    "milestone": "P4.10b",
    "status": "live_governed_research_execution",
    "permission_id": "web.search",
    "required_mode": "research",
    "required_availability": "available",
    "required_operations": ["search", "fetch"],
    "search_provider": "brave-search-v1",
    "fetch_provider": "controlled-live-http-v1",
    "maximum_search_calls": 1,
    "maximum_fetch_calls": 5,
    "maximum_sources": 5,
    "maximum_grounded_sources": 2,
    "skippable_fetch_failure_codes": [
        "http_status_rejected",
        "response_header_invalid",
    ],
    "required_path": [
        "explicit_research_mode",
        "live_search",
        "controlled_https_fetch",
        "injection_inspection",
        "temporal_and_freshness_analysis",
        "exact_extractive_grounding",
        "phase3_projection",
        "p36_pre_commit_validation",
        "p45b_citation_validation",
    ],
    "controls": {
        "public_queries_only": True,
        "foreground_only": True,
        "exact_provider_selection": True,
        "no_silent_fallback": True,
        "no_retry": True,
        "no_recursive_browsing": True,
        "no_source_body_persistence": True,
        "no_phase5_storage": True,
        "no_memory_write": True,
        "no_external_action": True,
        "no_background_execution": True,
        "continue_after_skippable_source_failure": True,
        "pre_commit_validation_required": True,
        "web_citation_validation_required": True,
    },
    "capability_ceiling": False,
}


@dataclass(frozen=True)
class InformationLiveResearchPolicy:
    policy_name: str
    version: str
    permission_id: str
    search_provider: str
    fetch_provider: str
    maximum_search_calls: int
    maximum_fetch_calls: int
    maximum_sources: int
    skippable_fetch_failure_codes: tuple[str, ...]
    required_mode: str
    required_availability: str
    required_operations: tuple[str, ...]
    maximum_grounded_sources: int
    raw: Mapping[str, object]
    policy_sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "InformationLiveResearchPolicy":
        raw = dict(value)
        if raw != _EXACT_POLICY:
            raise InformationLiveResearchPolicyError(
                "The exact additive P4.10b policy changed."
            )
        return cls(
            policy_name=str(raw["policy_name"]),
            version=str(raw["version"]),
            permission_id=str(raw["permission_id"]),
            search_provider=str(raw["search_provider"]),
            fetch_provider=str(raw["fetch_provider"]),
            maximum_search_calls=int(raw["maximum_search_calls"]),
            maximum_fetch_calls=int(raw["maximum_fetch_calls"]),
            maximum_sources=int(raw["maximum_sources"]),
            skippable_fetch_failure_codes=tuple(
                str(item) for item in raw["skippable_fetch_failure_codes"]
            ),
            required_mode=str(raw["required_mode"]),
            required_availability=str(raw["required_availability"]),
            required_operations=tuple(raw["required_operations"]),
            maximum_grounded_sources=int(raw["maximum_grounded_sources"]),
            raw=MappingProxyType(raw),
            policy_sha256=hashlib.sha256(_canonical(raw)).hexdigest(),
        )

    @classmethod
    def load(cls, path: str | Path) -> "InformationLiveResearchPolicy":
        try:
            value = json.loads(
                Path(path).read_text(encoding="utf-8-sig"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InformationLiveResearchPolicyError(
                "P4.10b policy could not be loaded."
            ) from exc
        if not isinstance(value, dict):
            raise InformationLiveResearchPolicyError(
                "P4.10b policy root must be an object."
            )
        return cls.from_mapping(value)

    def validate(self, *, provider_policy: object | None = None) -> None:
        reconstructed = type(self).from_mapping(self.raw)
        if reconstructed != self:
            raise InformationLiveResearchPolicyError(
                "P4.10b policy fields or digest changed."
            )
        if provider_policy is not None:
            validator = getattr(provider_policy, "validate", None)
            if not callable(validator):
                raise InformationLiveResearchPolicyError(
                    "P4.10b requires the exact P4.10a provider policy."
                )
            validator()
            if (
                getattr(provider_policy, "search_provider_id", None)
                != self.search_provider
                or getattr(provider_policy, "fetch_provider_id", None)
                != self.fetch_provider
            ):
                raise InformationLiveResearchPolicyError(
                    "P4.10b provider bindings changed."
                )

    @property
    def binding(self) -> str:
        return f"{self.policy_name}@{self.version}:{self.policy_sha256}"
