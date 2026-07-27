"""Exact allowlist policy for A.L.I.C.E. Phase 4 P4.1 providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import (
    INFORMATION_PROVIDER_TYPES,
    InformationProviderConfigurationError,
    validate_provider_identity,
)

DEFAULT_PROVIDER_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_provider_policy.json"
)


class InformationProviderPolicyError(InformationProviderConfigurationError):
    """Raised when the P4.1 provider policy is invalid."""


@dataclass(frozen=True)
class ApprovedInformationProvider:
    """One exact provider identity and its approved read-only operations."""

    provider: str
    provider_type: str
    operations: tuple[str, ...]

    def validate(self) -> None:
        validate_provider_identity(self.provider)
        if self.provider_type not in INFORMATION_PROVIDER_TYPES:
            raise InformationProviderPolicyError(
                "Approved provider type is not recognized."
            )
        if not self.operations:
            raise InformationProviderPolicyError(
                "Approved providers require at least one operation."
            )
        if len(set(self.operations)) != len(self.operations):
            raise InformationProviderPolicyError(
                "Approved provider operations cannot contain duplicates."
            )
        if any(operation not in {"search", "fetch"} for operation in self.operations):
            raise InformationProviderPolicyError(
                "Approved provider operations must be search or fetch."
            )


@dataclass(frozen=True)
class InformationProviderPolicy:
    """Validated P4.1 provider allowlist and no-fallback policy."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    live_network_access_allowed: bool
    provider_fallback_allowed: bool
    sanitized_failures_required: bool
    raw_query_in_failures_allowed: bool
    raw_source_content_in_failures_allowed: bool
    approved_providers: tuple[ApprovedInformationProvider, ...]

    def validate(self) -> None:
        if self.policy_name != "alice_information_provider_policy":
            raise InformationProviderPolicyError(
                "P4.1 provider policy_name must be alice_information_provider_policy."
            )
        if self.version != "1.0.0":
            raise InformationProviderPolicyError(
                "P4.1 provider policy version must be 1.0.0."
            )
        if self.phase != "4" or self.milestone != "P4.1":
            raise InformationProviderPolicyError(
                "Provider policy must be bound to Phase 4 milestone P4.1."
            )
        if self.status != "provider_abstraction":
            raise InformationProviderPolicyError(
                "P4.1 provider policy status must be provider_abstraction."
            )
        if self.live_network_access_allowed is not False:
            raise InformationProviderPolicyError(
                "P4.1 live network access must remain false."
            )
        if self.provider_fallback_allowed is not False:
            raise InformationProviderPolicyError(
                "P4.1 provider fallback must remain false."
            )
        if self.sanitized_failures_required is not True:
            raise InformationProviderPolicyError(
                "P4.1 sanitized provider failures must remain required."
            )
        if self.raw_query_in_failures_allowed is not False:
            raise InformationProviderPolicyError(
                "P4.1 raw query text in failures must remain false."
            )
        if self.raw_source_content_in_failures_allowed is not False:
            raise InformationProviderPolicyError(
                "P4.1 raw source content in failures must remain false."
            )
        if not self.approved_providers:
            raise InformationProviderPolicyError(
                "P4.1 requires at least one deterministic fixture provider."
            )
        identities: set[str] = set()
        for provider in self.approved_providers:
            provider.validate()
            if provider.provider in identities:
                raise InformationProviderPolicyError(
                    "Approved provider identities must be unique."
                )
            identities.add(provider.provider)
            if provider.provider_type != "deterministic_fixture":
                raise InformationProviderPolicyError(
                    "P4.1 approves deterministic fixture providers only."
                )

    def allows(self, *, provider: str, provider_type: str, operation: str) -> bool:
        """Return whether one exact provider and operation is approved."""

        exact_provider = validate_provider_identity(provider)
        for entry in self.approved_providers:
            if (
                entry.provider == exact_provider
                and entry.provider_type == provider_type
                and operation in entry.operations
            ):
                return True
        return False


def _strict_bool(value: Any, *, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationProviderPolicyError(
            f"{field} must remain {str(expected).lower()} in P4.1."
        )
    return expected


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationProviderPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def parse_information_provider_policy(
    payload: dict[str, Any],
) -> InformationProviderPolicy:
    """Validate and project one decoded P4.1 provider policy."""

    raw_providers = payload.get("approved_providers")
    if not isinstance(raw_providers, list):
        raise InformationProviderPolicyError(
            "approved_providers must be a list."
        )
    providers: list[ApprovedInformationProvider] = []
    for index, raw in enumerate(raw_providers):
        if not isinstance(raw, dict):
            raise InformationProviderPolicyError(
                f"approved_providers[{index}] must be an object."
            )
        raw_operations = raw.get("operations")
        if not isinstance(raw_operations, list) or any(
            not isinstance(item, str) for item in raw_operations
        ):
            raise InformationProviderPolicyError(
                f"approved_providers[{index}].operations must be a text list."
            )
        providers.append(
            ApprovedInformationProvider(
                provider=_text(
                    raw.get("provider"),
                    field=f"approved_providers[{index}].provider",
                ),
                provider_type=_text(
                    raw.get("provider_type"),
                    field=f"approved_providers[{index}].provider_type",
                ),
                operations=tuple(raw_operations),
            )
        )
    policy = InformationProviderPolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        live_network_access_allowed=_strict_bool(
            payload.get("live_network_access_allowed"),
            field="live_network_access_allowed",
            expected=False,
        ),
        provider_fallback_allowed=_strict_bool(
            payload.get("provider_fallback_allowed"),
            field="provider_fallback_allowed",
            expected=False,
        ),
        sanitized_failures_required=_strict_bool(
            payload.get("sanitized_failures_required"),
            field="sanitized_failures_required",
            expected=True,
        ),
        raw_query_in_failures_allowed=_strict_bool(
            payload.get("raw_query_in_failures_allowed"),
            field="raw_query_in_failures_allowed",
            expected=False,
        ),
        raw_source_content_in_failures_allowed=_strict_bool(
            payload.get("raw_source_content_in_failures_allowed"),
            field="raw_source_content_in_failures_allowed",
            expected=False,
        ),
        approved_providers=tuple(providers),
    )
    policy.validate()
    return policy


def load_information_provider_policy(
    path: str | Path = DEFAULT_PROVIDER_POLICY_PATH,
) -> InformationProviderPolicy:
    """Load and validate the versioned public P4.1 provider policy."""

    policy_path = Path(path)
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationProviderPolicyError(
            f"Unable to load information provider policy: {policy_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise InformationProviderPolicyError(
            "Information provider policy root must be a JSON object."
        )
    return parse_information_provider_policy(payload)
