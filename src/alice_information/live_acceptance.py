"""P4.10c private live acceptance and exact-commit closure.

The record produced by this module is metadata only. The single real PUBLIC
research turn is supplied by a private runtime factory outside Git. Public CI
runs deterministic tests only and never requires credentials or live network.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .brave_search import BraveInformationSearchProvider
from .brave_search_live import StrictBraveSearchHttpsTransport
from .live_fetch_provider import LiveControlledInformationFetchProvider
from .retrieval import LiveControlledInformationHttpRetriever
from .live_provider_contracts import canonical_sha256
from .live_provider_registry import ExactInformationLiveProviderRegistry
from .live_research import (
    InformationLiveResearchReceipt,
    InformationLiveResearchTurnResult,
    InformationLiveSourceOutcome,
    LiveInformationResearchExecutor,
)

_SHA256 = frozenset("0123456789abcdef")
_EXACT_TOP_LEVEL = {
    "information_live_acceptance_release_policy_schema_version",
    "policy_name",
    "version",
    "phase",
    "milestone",
    "status",
    "permission_id",
    "package_version",
    "required_provider_policy",
    "required_research_policy",
    "required_search_provider",
    "required_fetch_provider",
    "required_live_path",
    "required_acceptance_domains",
    "required_live_result",
    "required_repository_state",
    "required_private_record",
    "required_regression",
    "execution_controls",
    "capability_ceiling",
}



_EXACT_ACCEPTANCE_DOMAINS = (
    "provider_availability",
    "public_network_boundary",
    "live_source_fetch",
    "injection_and_freshness_gates",
    "exact_grounding",
    "phase3_projection",
    "pre_commit_validation",
    "citation_validation",
    "privacy_and_egress",
    "deterministic_regression",
    "exact_commit_and_rollback",
)

_LIVE_ACCEPTANCE_RECORD_KEYS = {
    "approved", "audit_version", "release_id", "repository_commit",
    "repository_head_commit", "rollback_commit", "repository_clean",
    "repository_snapshot_before_sha256", "repository_snapshot_after_sha256",
    "package_version", "evaluated_at", "policy_id", "policy_sha256",
    "benchmark_id", "benchmark_sha256", "provider_policy_binding",
    "research_policy_binding", "deterministic_test_collected",
    "deterministic_test_passed", "deterministic_test_skipped",
    "deterministic_test_output_sha256",
    "repository_regression_collected", "repository_regression_passed",
    "repository_regression_skipped", "repository_regression_subtests_passed",
    "repository_regression_output_sha256", "live_research_receipt",
    "source_outcomes", "acceptance_domains", "decision_reasons",
    "boundaries", "record_sha256",
}

_EXACT_RECORD_BOUNDARIES = {
    "public_queries_only": True,
    "foreground_only": True,
    "fallback_allowed": False,
    "retry_allowed": False,
    "recursive_browsing_allowed": False,
    "source_body_persistence_allowed": False,
    "phase5_storage_allowed": False,
    "memory_write_allowed": False,
    "external_action_allowed": False,
    "background_execution_allowed": False,
    "private_record_only": True,
}

_ALLOWED_DECISION_REASONS = frozenset({
    "live_result_not_answerable",
    "live_search_returned_no_results",
    "live_fetch_not_proven",
    "live_grounding_not_proven",
    "p36_precommit_not_proven",
    "p45b_validation_not_accepted",
})

_EXACT_BENCHMARK = {
    "benchmark_schema_version": 1,
    "benchmark_id": "phase4-live-public-information-acceptance-v1",
    "version": "1.0.0",
    "phase": "4",
    "milestone": "P4.10c",
    "description": "Private metadata-only acceptance of one exact live PUBLIC research path plus deterministic regression evidence.",
    "live_case": {
        "case_id": "live-public-research-001",
        "classification": "PUBLIC",
        "required_mode": "research",
        "required_availability": "available",
        "required_outcome": "answerable",
        "minimum_search_results": 1,
        "minimum_fetches": 1,
        "minimum_grounded_sources": 1,
        "required_pre_commit_validation_count": 1,
        "required_citation_validation_outcome": "accepted",
    },
    "deterministic_domains": [
        "credential_redaction",
        "private_configuration_boundary",
        "brave_request_exactness",
        "dns_and_peer_publicness",
        "redirect_and_ssrf_rejection",
        "timeout_and_cancellation",
        "rate_limit_and_quota_failures",
        "response_size_and_protocol_failures",
        "injection_blocking",
        "freshness_rejection",
        "exact_extractive_grounding",
        "phase3_pre_commit_binding",
        "p45b_citation_validation",
        "record_tamper_detection",
    ],
    "boundaries": {
        "live_network_in_public_ci": False,
        "source_body_persistence": False,
        "phase5_storage": False,
        "memory_write": False,
        "external_action": False,
        "recursive_browsing": False,
        "retry": False,
        "fallback": False,
        "background_execution": False,
    },
}


class InformationLiveAcceptanceError(RuntimeError):
    """Sanitized P4.10c acceptance failure."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InformationLiveAcceptanceError(
                f"Duplicate JSON key is not allowed: {key}.",
                code="live_acceptance_input_invalid",
            )
        result[key] = value
    return result


def _text(value: object, field: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise InformationLiveAcceptanceError(
            f"{field} is invalid.", code="live_acceptance_binding_invalid"
        )
    return value.strip()


def _digest(value: object, field: str) -> str:
    text = _text(value, field, maximum=64).lower()
    if len(text) != 64 or any(character not in _SHA256 for character in text):
        raise InformationLiveAcceptanceError(
            f"{field} is invalid.", code="live_acceptance_binding_invalid"
        )
    return text


def _git_commit(value: object, field: str) -> str:
    text = _text(value, field, maximum=128).lower()
    if len(text) != 40 or any(character not in _SHA256 for character in text):
        raise InformationLiveAcceptanceError(
            f"{field} must be a 40-character Git commit.",
            code="live_acceptance_binding_invalid",
        )
    return text


def _timestamp(value: object, field: str) -> str:
    text = _text(value, field, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InformationLiveAcceptanceError(
            f"{field} is invalid.", code="live_acceptance_binding_invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise InformationLiveAcceptanceError(
            f"{field} is invalid.", code="live_acceptance_binding_invalid"
        )
    return text


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InformationLiveAcceptanceError(
            "P4.10c JSON input could not be loaded.",
            code="live_acceptance_input_invalid",
        ) from exc
    if not isinstance(value, dict):
        raise InformationLiveAcceptanceError(
            "P4.10c JSON input root must be an object.",
            code="live_acceptance_input_invalid",
        )
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git(repository: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and completed.returncode != 0:
        raise InformationLiveAcceptanceError(
            "Repository verification failed.", code="live_acceptance_repository_invalid"
        )
    return completed.stdout.strip()


def repository_snapshot_sha256(repository: str | Path) -> str:
    """Hash tracked file paths, modes, and exact bytes without modifying Git."""

    root = Path(repository).resolve(strict=True)
    files = _git(root, "ls-files", "-z").split("\0")
    digest = hashlib.sha256()
    for relative in sorted(item for item in files if item):
        path = root / relative
        if not path.is_file():
            raise InformationLiveAcceptanceError(
                "Tracked repository file is missing.",
                code="live_acceptance_repository_invalid",
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class InformationLiveAcceptancePolicy:
    policy_name: str
    version: str
    package_version: str
    required_provider_policy: str
    required_research_policy: str
    required_search_provider: str
    required_fetch_provider: str
    required_live_path: tuple[str, ...]
    required_acceptance_domains: tuple[str, ...]
    raw: Mapping[str, object]
    policy_sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "InformationLiveAcceptancePolicy":
        raw = dict(value)
        if set(raw) != _EXACT_TOP_LEVEL:
            raise InformationLiveAcceptanceError(
                "P4.10c policy fields changed.",
                code="live_acceptance_policy_invalid",
            )
        expected = {
            "information_live_acceptance_release_policy_schema_version": 1,
            "policy_name": "phase4-live-information-acceptance-v1",
            "version": "1.0.0",
            "phase": "4",
            "milestone": "P4.10c",
            "status": "private_live_acceptance_and_exact_commit_closure",
            "permission_id": "web.search",
            "package_version": "0.18.0",
            "required_provider_policy": "alice_information_live_provider_runtime_policy@1.0.0",
            "required_research_policy": "alice_information_live_research_policy@1.0.0",
            "required_search_provider": "brave-search-v1",
            "required_fetch_provider": "controlled-live-http-v1",
            "capability_ceiling": False,
        }
        for key, expected_value in expected.items():
            if raw.get(key) != expected_value:
                raise InformationLiveAcceptanceError(
                    f"P4.10c policy field {key} changed.",
                    code="live_acceptance_policy_invalid",
                )
        required_path = tuple(raw.get("required_live_path", ()))
        if required_path != (
            "explicit_research_mode",
            "live_search",
            "controlled_https_fetch",
            "injection_inspection",
            "temporal_and_freshness_analysis",
            "exact_extractive_grounding",
            "phase3_projection",
            "p36_pre_commit_validation",
            "p45b_citation_validation",
        ):
            raise InformationLiveAcceptanceError(
                "P4.10c required live path changed.",
                code="live_acceptance_policy_invalid",
            )
        domains = tuple(raw.get("required_acceptance_domains", ()))
        if domains != _EXACT_ACCEPTANCE_DOMAINS:
            raise InformationLiveAcceptanceError(
                "P4.10c acceptance domains changed.",
                code="live_acceptance_policy_invalid",
            )
        live = raw.get("required_live_result")
        if live != {
            "outcome": "answerable",
            "minimum_search_results": 1,
            "minimum_fetches": 1,
            "minimum_grounded_sources": 1,
            "minimum_p36_pre_commit_validations": 1,
            "p45b_outcome": "accepted",
        }:
            raise InformationLiveAcceptanceError(
                "P4.10c live-result thresholds changed.",
                code="live_acceptance_policy_invalid",
            )
        repository = raw.get("required_repository_state")
        if repository != {
            "clean": True,
            "head_matches_requested_commit": True,
            "rollback_must_be_distinct_ancestor": True,
            "repository_unchanged_after_audit": True,
        }:
            raise InformationLiveAcceptanceError(
                "P4.10c repository controls changed.",
                code="live_acceptance_policy_invalid",
            )
        private = raw.get("required_private_record")
        if private != {
            "outside_repository": True,
            "raw_query_allowed": False,
            "source_body_allowed": False,
            "credential_allowed": False,
            "provider_response_body_allowed": False,
            "model_prompt_allowed": False,
            "metadata_only": True,
        }:
            raise InformationLiveAcceptanceError(
                "P4.10c private-record controls changed.",
                code="live_acceptance_policy_invalid",
            )
        regression = raw.get("required_regression")
        if regression != {
            "p410_targeted_passed": 101,
            "p410_targeted_skipped": 0,
            "minimum_repository_passed": 2124,
            "maximum_repository_skipped": 2,
            "required_repository_subtests_passed": 14,
        }:
            raise InformationLiveAcceptanceError(
                "P4.10c regression thresholds changed.",
                code="live_acceptance_policy_invalid",
            )
        controls = raw.get("execution_controls")
        if controls != {
            "foreground_only": True,
            "public_queries_only": True,
            "exact_provider_selection": True,
            "no_fallback": True,
            "no_retry": True,
            "no_recursive_browsing": True,
            "no_source_body_persistence": True,
            "no_phase5_storage": True,
            "no_memory_write": True,
            "no_external_action": True,
            "no_background_execution": True,
        }:
            raise InformationLiveAcceptanceError(
                "P4.10c execution controls changed.",
                code="live_acceptance_policy_invalid",
            )
        return cls(
            policy_name=str(raw["policy_name"]),
            version=str(raw["version"]),
            package_version=str(raw["package_version"]),
            required_provider_policy=str(raw["required_provider_policy"]),
            required_research_policy=str(raw["required_research_policy"]),
            required_search_provider=str(raw["required_search_provider"]),
            required_fetch_provider=str(raw["required_fetch_provider"]),
            required_live_path=required_path,
            required_acceptance_domains=domains,
            raw=MappingProxyType(raw),
            policy_sha256=canonical_sha256(raw),
        )

    @classmethod
    def load(cls, path: str | Path) -> "InformationLiveAcceptancePolicy":
        return cls.from_mapping(_read_json(Path(path)))

    def validate(self) -> None:
        if type(self).from_mapping(self.raw) != self:
            raise InformationLiveAcceptanceError(
                "P4.10c policy binding changed.",
                code="live_acceptance_policy_invalid",
            )

    @property
    def binding(self) -> str:
        return f"{self.policy_name}@{self.version}:{self.policy_sha256}"


@dataclass(frozen=True)
class Phase4LiveAcceptanceRuntime:
    """Private exact runtime inputs for the single live acceptance turn."""

    executor: LiveInformationResearchExecutor
    command: object
    request: object
    reference_time: str
    created_at: str
    window_start: str | None = None
    window_end: str | None = None
    cancellation: object | None = None

    def validate(self) -> None:
        if type(self.executor) is not LiveInformationResearchExecutor:
            raise InformationLiveAcceptanceError(
                "P4.10c requires the exact live research executor.",
                code="live_acceptance_runtime_invalid",
            )
        registry = self.executor.registry
        if type(registry) is not ExactInformationLiveProviderRegistry:
            raise InformationLiveAcceptanceError(
                "P4.10c live registry was substituted.",
                code="live_acceptance_runtime_invalid",
            )
        if type(registry.search_provider) is not BraveInformationSearchProvider:
            raise InformationLiveAcceptanceError(
                "P4.10c Brave provider was substituted.",
                code="live_acceptance_runtime_invalid",
            )
        if type(registry.search_provider.transport) is not StrictBraveSearchHttpsTransport:
            raise InformationLiveAcceptanceError(
                "P4.10c Brave transport was substituted.",
                code="live_acceptance_runtime_invalid",
            )
        if type(registry.fetch_provider) is not LiveControlledInformationFetchProvider:
            raise InformationLiveAcceptanceError(
                "P4.10c fetch provider was substituted.",
                code="live_acceptance_runtime_invalid",
            )
        if type(registry.fetch_provider.retriever) is not LiveControlledInformationHttpRetriever:
            raise InformationLiveAcceptanceError(
                "P4.10c controlled retriever was substituted.",
                code="live_acceptance_runtime_invalid",
            )
        self.executor.validate_operational_boundary()
        registry.search_provider.transport.validate_live_boundary()
        registry.fetch_provider.retriever._validate_runtime_components()
        command_validator = getattr(self.command, "validate", None)
        request_validator = getattr(self.request, "validate", None)
        if not callable(command_validator) or not callable(request_validator):
            raise InformationLiveAcceptanceError(
                "P4.10c command or request is invalid.",
                code="live_acceptance_runtime_invalid",
            )
        command_validator()
        request_validator()
        if getattr(self.command, "grounding", None) is not None:
            raise InformationLiveAcceptanceError(
                "P4.10c command must begin without grounding.",
                code="live_acceptance_runtime_invalid",
            )
        query = getattr(self.request, "query", None)
        if getattr(query, "data_classification", None) != "PUBLIC":
            raise InformationLiveAcceptanceError(
                "P4.10c accepts PUBLIC queries only.",
                code="live_acceptance_runtime_invalid",
            )
        _timestamp(self.reference_time, "reference_time")
        _timestamp(self.created_at, "created_at")
        if self.window_start is not None:
            _timestamp(self.window_start, "window_start")
        if self.window_end is not None:
            _timestamp(self.window_end, "window_end")


def _live_receipt_from_mapping(value: Mapping[str, object]) -> InformationLiveResearchReceipt:
    expected = {
        "receipt_id", "policy_version", "request_id", "query_id", "query_sha256",
        "outcome", "search_result_count", "search_receipt_sha256",
        "fetch_attempt_count", "fetch_attempt_sequence_sha256",
        "fetch_receipt_sha256s", "fetch_failure_sha256s",
        "temporal_resolution_sha256s",
        "source_outcome_sha256", "grounded_source_sha256s",
        "grounding_sha256", "projection_sha256",
        "conversation_packet_sha256", "response_sha256",
        "validation_sha256", "citation_validation_outcome",
        "pre_commit_validation_count", "policy_bindings", "created_at",
        "receipt_sha256",
    }
    raw = dict(value)
    if set(raw) != expected:
        raise InformationLiveAcceptanceError(
            "P4.10c live receipt fields changed.",
            code="live_acceptance_record_invalid",
        )
    try:
        receipt = InformationLiveResearchReceipt(
            receipt_id=raw["receipt_id"],
            policy_version=raw["policy_version"],
            request_id=raw["request_id"],
            query_id=raw["query_id"],
            query_sha256=raw["query_sha256"],
            outcome=raw["outcome"],
            search_result_count=raw["search_result_count"],
            search_receipt_sha256=raw["search_receipt_sha256"],
            fetch_attempt_count=raw["fetch_attempt_count"],
            fetch_attempt_sequence_sha256=raw[
                "fetch_attempt_sequence_sha256"
            ],
            fetch_receipt_sha256s=tuple(raw["fetch_receipt_sha256s"]),
            fetch_failure_sha256s=tuple(raw["fetch_failure_sha256s"]),
            temporal_resolution_sha256s=tuple(raw["temporal_resolution_sha256s"]),
            source_outcome_sha256=raw["source_outcome_sha256"],
            grounded_source_sha256s=tuple(raw["grounded_source_sha256s"]),
            grounding_sha256=raw["grounding_sha256"],
            projection_sha256=raw["projection_sha256"],
            conversation_packet_sha256=raw["conversation_packet_sha256"],
            response_sha256=raw["response_sha256"],
            validation_sha256=raw["validation_sha256"],
            citation_validation_outcome=raw["citation_validation_outcome"],
            pre_commit_validation_count=raw["pre_commit_validation_count"],
            policy_bindings=tuple(raw["policy_bindings"]),
            created_at=raw["created_at"],
            receipt_sha256=raw["receipt_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InformationLiveAcceptanceError(
            "P4.10c live receipt structure is invalid.",
            code="live_acceptance_record_invalid",
        ) from exc
    try:
        receipt.validate()
    except Exception as exc:
        raise InformationLiveAcceptanceError(
            "P4.10c embedded live receipt failed validation.",
            code="live_acceptance_record_invalid",
        ) from exc
    return receipt


def _source_outcome_from_mapping(value: Mapping[str, object]) -> InformationLiveSourceOutcome:
    expected = {
        "source_id", "canonical_url", "source_content_sha256",
        "temporal_verdict", "inspection_verdict", "freshness_verdict",
        "supports_claim", "disposition", "reason_code",
    }
    raw = dict(value)
    if set(raw) != expected:
        raise InformationLiveAcceptanceError(
            "P4.10c source-outcome fields changed.",
            code="live_acceptance_record_invalid",
        )
    try:
        outcome = InformationLiveSourceOutcome(**raw)
    except (TypeError, ValueError) as exc:
        raise InformationLiveAcceptanceError(
            "P4.10c source outcome is invalid.",
            code="live_acceptance_record_invalid",
        ) from exc
    try:
        outcome.validate()
    except Exception as exc:
        raise InformationLiveAcceptanceError(
            "P4.10c embedded source outcome failed validation.",
            code="live_acceptance_record_invalid",
        ) from exc
    return outcome


@dataclass(frozen=True)
class InformationLiveAcceptanceRecord:
    """Tamper-evident private metadata-only P4.10c release record."""

    approved: bool
    audit_version: str
    release_id: str
    repository_commit: str
    repository_head_commit: str
    rollback_commit: str
    repository_clean: bool
    repository_snapshot_before_sha256: str
    repository_snapshot_after_sha256: str
    package_version: str
    evaluated_at: str
    policy_id: str
    policy_sha256: str
    benchmark_id: str
    benchmark_sha256: str
    provider_policy_binding: str
    research_policy_binding: str
    deterministic_test_collected: int
    deterministic_test_passed: int
    deterministic_test_skipped: int
    deterministic_test_output_sha256: str
    repository_regression_collected: int
    repository_regression_passed: int
    repository_regression_skipped: int
    repository_regression_subtests_passed: int
    repository_regression_output_sha256: str
    live_research_receipt: Mapping[str, object]
    source_outcomes: tuple[Mapping[str, object], ...]
    acceptance_domains: tuple[str, ...]
    decision_reasons: tuple[str, ...]
    boundaries: Mapping[str, bool]
    record_sha256: str

    @classmethod
    def create(cls, **values: object) -> "InformationLiveAcceptanceRecord":
        draft = cls(
            release_id="p410-live-pending",
            record_sha256="0" * 64,
            **values,
        )  # type: ignore[arg-type]
        identified = replace(
            draft,
            release_id=f"p410-live-{canonical_sha256(draft._payload(include_id=False))[:24]}",
        )
        record = replace(
            identified,
            record_sha256=canonical_sha256(identified._payload(include_id=True)),
        )
        record.validate()
        return record

    def _payload(self, *, include_id: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "approved": self.approved,
            "audit_version": self.audit_version,
            "release_id": self.release_id,
            "repository_commit": self.repository_commit,
            "repository_head_commit": self.repository_head_commit,
            "rollback_commit": self.rollback_commit,
            "repository_clean": self.repository_clean,
            "repository_snapshot_before_sha256": self.repository_snapshot_before_sha256,
            "repository_snapshot_after_sha256": self.repository_snapshot_after_sha256,
            "package_version": self.package_version,
            "evaluated_at": self.evaluated_at,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "benchmark_id": self.benchmark_id,
            "benchmark_sha256": self.benchmark_sha256,
            "provider_policy_binding": self.provider_policy_binding,
            "research_policy_binding": self.research_policy_binding,
            "deterministic_test_collected": self.deterministic_test_collected,
            "deterministic_test_passed": self.deterministic_test_passed,
            "deterministic_test_skipped": self.deterministic_test_skipped,
            "deterministic_test_output_sha256": self.deterministic_test_output_sha256,
            "repository_regression_collected": self.repository_regression_collected,
            "repository_regression_passed": self.repository_regression_passed,
            "repository_regression_skipped": self.repository_regression_skipped,
            "repository_regression_subtests_passed": self.repository_regression_subtests_passed,
            "repository_regression_output_sha256": self.repository_regression_output_sha256,
            "live_research_receipt": dict(self.live_research_receipt),
            "source_outcomes": [dict(item) for item in self.source_outcomes],
            "acceptance_domains": list(self.acceptance_domains),
            "decision_reasons": list(self.decision_reasons),
            "boundaries": dict(self.boundaries),
        }
        if not include_id:
            payload.pop("release_id")
        return payload

    def validate(self) -> None:
        if not isinstance(self.approved, bool) or self.audit_version != "p4.10c-v1":
            raise InformationLiveAcceptanceError(
                "P4.10c record identity is invalid.",
                code="live_acceptance_record_invalid",
            )
        for value, field in (
            (self.repository_commit, "repository_commit"),
            (self.repository_head_commit, "repository_head_commit"),
            (self.rollback_commit, "rollback_commit"),
        ):
            _git_commit(value, field)
        for value, field in (
            (self.repository_snapshot_before_sha256, "repository_snapshot_before_sha256"),
            (self.repository_snapshot_after_sha256, "repository_snapshot_after_sha256"),
            (self.policy_sha256, "policy_sha256"),
            (self.benchmark_sha256, "benchmark_sha256"),
            (self.deterministic_test_output_sha256, "deterministic_test_output_sha256"),
            (self.repository_regression_output_sha256, "repository_regression_output_sha256"),
            (self.record_sha256, "record_sha256"),
        ):
            _digest(value, field)
        if self.repository_commit != self.repository_head_commit:
            raise InformationLiveAcceptanceError(
                "P4.10c record is not bound to repository HEAD.",
                code="live_acceptance_record_invalid",
            )
        if self.repository_commit == self.rollback_commit:
            raise InformationLiveAcceptanceError(
                "P4.10c rollback must be a distinct ancestor.",
                code="live_acceptance_record_invalid",
            )
        if (
            self.repository_snapshot_before_sha256
            != self.repository_snapshot_after_sha256
        ):
            raise InformationLiveAcceptanceError(
                "Repository changed during P4.10c acceptance.",
                code="live_acceptance_record_invalid",
            )
        if self.package_version != "0.18.0":
            raise InformationLiveAcceptanceError(
                "P4.10c package version changed.",
                code="live_acceptance_record_invalid",
            )
        _timestamp(self.evaluated_at, "evaluated_at")
        if self.policy_id != "phase4-live-information-acceptance-v1":
            raise InformationLiveAcceptanceError(
                "P4.10c policy identity changed.",
                code="live_acceptance_record_invalid",
            )
        if self.benchmark_id != "phase4-live-public-information-acceptance-v1":
            raise InformationLiveAcceptanceError(
                "P4.10c benchmark identity changed.",
                code="live_acceptance_record_invalid",
            )
        if not _text(self.provider_policy_binding, "provider_policy_binding").startswith(
            "alice_information_live_provider_runtime_policy@1.0.0:"
        ):
            raise InformationLiveAcceptanceError(
                "P4.10c provider binding is invalid.",
                code="live_acceptance_record_invalid",
            )
        if not _text(self.research_policy_binding, "research_policy_binding").startswith(
            "alice_information_live_research_policy@1.0.0:"
        ):
            raise InformationLiveAcceptanceError(
                "P4.10c research binding is invalid.",
                code="live_acceptance_record_invalid",
            )
        for value, field in (
            (self.deterministic_test_collected, "deterministic_test_collected"),
            (self.deterministic_test_passed, "deterministic_test_passed"),
            (self.deterministic_test_skipped, "deterministic_test_skipped"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InformationLiveAcceptanceError(
                    f"{field} is invalid.", code="live_acceptance_record_invalid"
                )
        if self.deterministic_test_passed + self.deterministic_test_skipped != self.deterministic_test_collected:
            raise InformationLiveAcceptanceError(
                "P4.10c deterministic test totals are inconsistent.",
                code="live_acceptance_record_invalid",
            )
        for value, field in (
            (self.repository_regression_collected, "repository_regression_collected"),
            (self.repository_regression_passed, "repository_regression_passed"),
            (self.repository_regression_skipped, "repository_regression_skipped"),
            (self.repository_regression_subtests_passed, "repository_regression_subtests_passed"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InformationLiveAcceptanceError(
                    f"{field} is invalid.", code="live_acceptance_record_invalid"
                )
        if self.repository_regression_passed + self.repository_regression_skipped != self.repository_regression_collected:
            raise InformationLiveAcceptanceError(
                "P4.10c repository regression totals are inconsistent.",
                code="live_acceptance_record_invalid",
            )
        live_receipt = _live_receipt_from_mapping(self.live_research_receipt)
        live = live_receipt.to_metadata_record()
        for item in self.source_outcomes:
            _source_outcome_from_mapping(item)
        if canonical_sha256([dict(item) for item in self.source_outcomes]) != live_receipt.source_outcome_sha256:
            raise InformationLiveAcceptanceError(
                "P4.10c source outcomes do not match the live receipt.",
                code="live_acceptance_record_invalid",
            )
        if self.acceptance_domains != _EXACT_ACCEPTANCE_DOMAINS:
            raise InformationLiveAcceptanceError(
                "P4.10c acceptance domains are invalid.",
                code="live_acceptance_record_invalid",
            )
        if len(set(self.decision_reasons)) != len(self.decision_reasons):
            raise InformationLiveAcceptanceError(
                "P4.10c decision reasons contain duplicates.",
                code="live_acceptance_record_invalid",
            )
        for reason in self.decision_reasons:
            if _text(reason, "decision_reason", maximum=256) not in _ALLOWED_DECISION_REASONS:
                raise InformationLiveAcceptanceError(
                    "P4.10c decision reason is not approved.",
                    code="live_acceptance_record_invalid",
                )
        if dict(self.boundaries) != _EXACT_RECORD_BOUNDARIES:
            raise InformationLiveAcceptanceError(
                "P4.10c boundary record is invalid.",
                code="live_acceptance_record_invalid",
            )
        expected_approved = (
            self.repository_clean
            and not self.decision_reasons
            and self.deterministic_test_collected > 0
            and self.deterministic_test_passed == self.deterministic_test_collected
            and self.deterministic_test_skipped == 0
            and self.repository_regression_passed >= 2124
            and self.repository_regression_skipped <= 2
            and self.repository_regression_subtests_passed == 14
            and live_receipt.outcome == "answerable"
            and live_receipt.search_result_count >= 1
            and len(live_receipt.fetch_receipt_sha256s) >= 1
            and len(live_receipt.grounded_source_sha256s) >= 1
            and live_receipt.pre_commit_validation_count == 1
            and live_receipt.citation_validation_outcome == "accepted"
        )
        if self.approved is not expected_approved:
            raise InformationLiveAcceptanceError(
                "P4.10c approval decision is inconsistent.",
                code="live_acceptance_record_invalid",
            )
        expected_id = f"p410-live-{canonical_sha256(self._payload(include_id=False))[:24]}"
        if self.release_id != expected_id:
            raise InformationLiveAcceptanceError(
                "P4.10c release ID changed.",
                code="live_acceptance_record_invalid",
            )
        if self.record_sha256 != canonical_sha256(self._payload(include_id=True)):
            raise InformationLiveAcceptanceError(
                "P4.10c record digest changed.",
                code="live_acceptance_record_invalid",
            )

    def to_mapping(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(include_id=True), "record_sha256": self.record_sha256}


def validate_repository_release_state(
    repository: str | Path,
    *,
    repository_commit: str,
    rollback_commit: str,
) -> tuple[str, bool]:
    root = Path(repository).resolve(strict=True)
    requested = _git_commit(repository_commit, "repository_commit")
    rollback = _git_commit(rollback_commit, "rollback_commit")
    head = _git(root, "rev-parse", "HEAD")
    _git_commit(head, "repository_head_commit")
    if head != requested:
        raise InformationLiveAcceptanceError(
            "Requested acceptance commit is not repository HEAD.",
            code="live_acceptance_repository_invalid",
        )
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise InformationLiveAcceptanceError(
            "Repository must be clean before P4.10c acceptance.",
            code="live_acceptance_repository_invalid",
        )
    if requested == rollback:
        raise InformationLiveAcceptanceError(
            "Rollback commit must be distinct.",
            code="live_acceptance_repository_invalid",
        )
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", rollback, requested],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise InformationLiveAcceptanceError(
            "Rollback commit is not an ancestor of the release commit.",
            code="live_acceptance_repository_invalid",
        )
    return head, True


def run_deterministic_acceptance_tests(
    repository: str | Path,
    *,
    target_files: Sequence[str],
) -> tuple[int, int, int, str]:
    root = Path(repository).resolve(strict=True)
    if not target_files:
        raise InformationLiveAcceptanceError(
            "P4.10c deterministic test set is empty.",
            code="live_acceptance_tests_invalid",
        )
    for relative in target_files:
        path = root / relative
        if not path.is_file() or not _inside(path.resolve(), root):
            raise InformationLiveAcceptanceError(
                "P4.10c deterministic test target is missing.",
                code="live_acceptance_tests_invalid",
            )
    environment = os.environ.copy()
    environment.pop("ALICE_BRAVE_SEARCH_API_KEY", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
    environment["ALICE_P410_PUBLIC_CI"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *target_files, "-q"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode != 0:
        raise InformationLiveAcceptanceError(
            "P4.10c deterministic acceptance tests failed.",
            code="live_acceptance_tests_failed",
        )
    import re

    passed = 0
    skipped = 0
    for match in re.finditer(r"(\d+) passed", output):
        passed = int(match.group(1))
    for match in re.finditer(r"(\d+) skipped", output):
        skipped = int(match.group(1))
    collected = passed + skipped
    if passed != 101 or collected != 101 or skipped != 0:
        raise InformationLiveAcceptanceError(
            "P4.10c deterministic tests did not produce the exact 101-test all-pass result.",
            code="live_acceptance_tests_failed",
        )
    return collected, passed, skipped, hashlib.sha256(output.encode("utf-8")).hexdigest()


def run_repository_regression_tests(
    repository: str | Path,
) -> tuple[int, int, int, int, str]:
    """Run the exact full repository regression with live credentials removed."""

    root = Path(repository).resolve(strict=True)
    environment = os.environ.copy()
    environment.pop("ALICE_BRAVE_SEARCH_API_KEY", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
    environment["ALICE_P410_PUBLIC_CI"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode != 0:
        raise InformationLiveAcceptanceError(
            "Full repository regression failed during P4.10c acceptance.",
            code="live_acceptance_tests_failed",
        )
    import re

    passed_matches = list(re.finditer(r"(?<!subtests )(\d+) passed", output))
    skipped_matches = list(re.finditer(r"(\d+) skipped", output))
    subtest_matches = list(re.finditer(r"(\d+) subtests passed", output))
    passed = int(passed_matches[-1].group(1)) if passed_matches else 0
    skipped = int(skipped_matches[-1].group(1)) if skipped_matches else 0
    subtests = int(subtest_matches[-1].group(1)) if subtest_matches else 0
    collected = passed + skipped
    if passed < 2124 or skipped > 2 or subtests != 14:
        raise InformationLiveAcceptanceError(
            "Full repository regression did not meet the P4.10c release floor.",
            code="live_acceptance_tests_failed",
        )
    return (
        collected,
        passed,
        skipped,
        subtests,
        hashlib.sha256(output.encode("utf-8")).hexdigest(),
    )


def build_live_acceptance_record(
    *,
    repository: str | Path,
    repository_commit: str,
    rollback_commit: str,
    evaluated_at: str,
    package_version: str,
    policy: InformationLiveAcceptancePolicy,
    benchmark_path: str | Path,
    runtime: Phase4LiveAcceptanceRuntime,
    deterministic_test_result: tuple[int, int, int, str],
    repository_regression_result: tuple[int, int, int, int, str],
    snapshot_before_sha256: str,
) -> InformationLiveAcceptanceRecord:
    root = Path(repository).resolve(strict=True)
    policy.validate()
    runtime.validate()
    head, clean = validate_repository_release_state(
        root,
        repository_commit=repository_commit,
        rollback_commit=rollback_commit,
    )
    benchmark = _read_json(Path(benchmark_path))
    if benchmark != _EXACT_BENCHMARK:
        raise InformationLiveAcceptanceError(
            "P4.10c benchmark content changed.",
            code="live_acceptance_input_invalid",
        )
    result: InformationLiveResearchTurnResult = runtime.executor.run_turn(
        runtime.command,
        mode="research",
        availability="available",
        request=runtime.request,
        reference_time=runtime.reference_time,
        created_at=runtime.created_at,
        window_start=runtime.window_start,
        window_end=runtime.window_end,
        cancellation=runtime.cancellation,
    )
    result.validate(executor=runtime.executor)
    reasons: list[str] = []
    if result.receipt.outcome != "answerable":
        reasons.append("live_result_not_answerable")
    if len(result.evidence.search_response.results) < 1:
        reasons.append("live_search_returned_no_results")
    if len(result.evidence.fetch_responses) < 1:
        reasons.append("live_fetch_not_proven")
    if len(result.evidence.grounded_sources) < 1:
        reasons.append("live_grounding_not_proven")
    if result.receipt.pre_commit_validation_count != 1:
        reasons.append("p36_precommit_not_proven")
    if result.response_validation.report.outcome != "accepted":
        reasons.append("p45b_validation_not_accepted")
    collected, passed, skipped, output_sha = deterministic_test_result
    (
        regression_collected,
        regression_passed,
        regression_skipped,
        regression_subtests,
        regression_output_sha,
    ) = repository_regression_result
    boundaries = MappingProxyType(dict(_EXACT_RECORD_BOUNDARIES))
    snapshot_after_sha256 = repository_snapshot_sha256(root)
    validate_repository_release_state(
        root,
        repository_commit=repository_commit,
        rollback_commit=rollback_commit,
    )
    approved = (
        clean
        and not reasons
        and collected == passed
        and skipped == 0
        and regression_passed >= 2124
        and regression_skipped <= 2
        and regression_subtests == 14
    )
    record = InformationLiveAcceptanceRecord.create(
        approved=approved,
        audit_version="p4.10c-v1",
        repository_commit=repository_commit,
        repository_head_commit=head,
        rollback_commit=rollback_commit,
        repository_clean=clean,
        repository_snapshot_before_sha256=_digest(
            snapshot_before_sha256, "repository_snapshot_before_sha256"
        ),
        repository_snapshot_after_sha256=_digest(
            snapshot_after_sha256, "repository_snapshot_after_sha256"
        ),
        package_version=package_version,
        evaluated_at=_timestamp(evaluated_at, "evaluated_at"),
        policy_id=policy.policy_name,
        policy_sha256=policy.policy_sha256,
        benchmark_id=str(benchmark["benchmark_id"]),
        benchmark_sha256=canonical_sha256(benchmark),
        provider_policy_binding=runtime.executor.registry.policy.binding,
        research_policy_binding=runtime.executor.research_policy.binding,
        deterministic_test_collected=collected,
        deterministic_test_passed=passed,
        deterministic_test_skipped=skipped,
        deterministic_test_output_sha256=output_sha,
        repository_regression_collected=regression_collected,
        repository_regression_passed=regression_passed,
        repository_regression_skipped=regression_skipped,
        repository_regression_subtests_passed=regression_subtests,
        repository_regression_output_sha256=regression_output_sha,
        live_research_receipt=MappingProxyType(result.receipt.to_metadata_record()),
        source_outcomes=tuple(
            MappingProxyType(item.metadata_record())
            for item in result.evidence.source_outcomes
        ),
        acceptance_domains=policy.required_acceptance_domains,
        decision_reasons=tuple(reasons),
        boundaries=boundaries,
    )
    serialized = json.dumps(
        record.to_mapping(), sort_keys=True, ensure_ascii=False
    )
    raw_query = runtime.request.query.text
    secret = runtime.executor.registry.search_provider.configuration.credential.reveal_for_exact_header()
    source_bodies = [
        item.source_document.normalized_text for item in result.evidence.fetch_responses
    ]
    if raw_query in serialized or secret in serialized or any(
        body and body in serialized for body in source_bodies
    ):
        raise InformationLiveAcceptanceError(
            "Private P4.10c record contains prohibited content.",
            code="live_acceptance_privacy_failed",
        )
    return record


def write_live_acceptance_record(
    record: InformationLiveAcceptanceRecord,
    path: str | Path,
    *,
    repository_root: str | Path,
    private_root: str | Path,
) -> Path:
    record.validate()
    repository = Path(repository_root).resolve(strict=True)
    private = Path(private_root).resolve(strict=True)
    output = Path(path).resolve()
    if _inside(private, repository) or _inside(output, repository):
        raise InformationLiveAcceptanceError(
            "P4.10c record must remain outside Git.",
            code="live_acceptance_output_invalid",
        )
    if not _inside(output, private):
        raise InformationLiveAcceptanceError(
            "P4.10c record must remain under the private vault root.",
            code="live_acceptance_output_invalid",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record.to_mapping(), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    loaded = load_live_acceptance_record(output)
    if loaded != record:
        raise InformationLiveAcceptanceError(
            "P4.10c record reload verification failed.",
            code="live_acceptance_record_invalid",
        )
    return output


def load_live_acceptance_record(path: str | Path) -> InformationLiveAcceptanceRecord:
    value = _read_json(Path(path))
    if set(value) != _LIVE_ACCEPTANCE_RECORD_KEYS:
        raise InformationLiveAcceptanceError(
            "P4.10c record fields changed.",
            code="live_acceptance_record_invalid",
        )
    try:
        record = InformationLiveAcceptanceRecord(
            approved=value["approved"],
            audit_version=value["audit_version"],
            release_id=value["release_id"],
            repository_commit=value["repository_commit"],
            repository_head_commit=value["repository_head_commit"],
            rollback_commit=value["rollback_commit"],
            repository_clean=value["repository_clean"],
            repository_snapshot_before_sha256=value[
                "repository_snapshot_before_sha256"
            ],
            repository_snapshot_after_sha256=value[
                "repository_snapshot_after_sha256"
            ],
            package_version=value["package_version"],
            evaluated_at=value["evaluated_at"],
            policy_id=value["policy_id"],
            policy_sha256=value["policy_sha256"],
            benchmark_id=value["benchmark_id"],
            benchmark_sha256=value["benchmark_sha256"],
            provider_policy_binding=value["provider_policy_binding"],
            research_policy_binding=value["research_policy_binding"],
            deterministic_test_collected=value["deterministic_test_collected"],
            deterministic_test_passed=value["deterministic_test_passed"],
            deterministic_test_skipped=value["deterministic_test_skipped"],
            deterministic_test_output_sha256=value[
                "deterministic_test_output_sha256"
            ],
            repository_regression_collected=value["repository_regression_collected"],
            repository_regression_passed=value["repository_regression_passed"],
            repository_regression_skipped=value["repository_regression_skipped"],
            repository_regression_subtests_passed=value["repository_regression_subtests_passed"],
            repository_regression_output_sha256=value["repository_regression_output_sha256"],
            live_research_receipt=MappingProxyType(
                dict(value["live_research_receipt"])
            ),
            source_outcomes=tuple(
                MappingProxyType(dict(item)) for item in value["source_outcomes"]
            ),
            acceptance_domains=tuple(value["acceptance_domains"]),
            decision_reasons=tuple(value["decision_reasons"]),
            boundaries=MappingProxyType(dict(value["boundaries"])),
            record_sha256=value["record_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InformationLiveAcceptanceError(
            "P4.10c record structure is invalid.",
            code="live_acceptance_record_invalid",
        ) from exc
    record.validate()
    return record
