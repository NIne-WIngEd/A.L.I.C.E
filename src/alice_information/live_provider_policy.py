"""Exact additive policy for the P4.10a live-provider foundation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class InformationLiveProviderPolicyError(ValueError):
    """Raised when the P4.10a runtime policy is malformed or weakened."""


_REQUIRED_TOP_LEVEL = {
    "information_live_provider_runtime_policy_schema_version",
    "policy_name",
    "version",
    "phase",
    "milestone",
    "status",
    "permission_id",
    "query_classifications",
    "search_provider",
    "fetch_provider",
    "execution_controls",
    "private_configuration",
    "capability_ceiling",
}


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InformationLiveProviderPolicyError(
                f"Duplicate JSON key is not allowed: {key}."
            )
        result[key] = value
    return result


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InformationLiveProviderPolicyError(f"{field} must be an object.")
    return dict(value)


def _exact_bool(mapping: Mapping[str, object], key: str, expected: bool) -> None:
    if mapping.get(key) is not expected:
        raise InformationLiveProviderPolicyError(
            f"{key} must remain exactly {str(expected).lower()}."
        )


@dataclass(frozen=True)
class InformationLiveProviderRuntimePolicy:
    """Canonical no-fallback P4.10a provider selection and egress policy."""

    policy_name: str
    version: str
    permission_id: str
    search_provider_id: str
    fetch_provider_id: str
    search_host: str
    search_path: str
    credential_environment_variable: str
    credential_header: str
    max_query_characters: int
    max_query_words: int
    max_results: int
    max_response_bytes: int
    private_configuration_path: str
    raw: Mapping[str, object]
    policy_sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "InformationLiveProviderRuntimePolicy":
        raw = dict(value)
        if set(raw) != _REQUIRED_TOP_LEVEL:
            missing = sorted(_REQUIRED_TOP_LEVEL - set(raw))
            extra = sorted(set(raw) - _REQUIRED_TOP_LEVEL)
            raise InformationLiveProviderPolicyError(
                f"P4.10a policy fields changed; missing={missing}, extra={extra}."
            )
        expected_scalars = {
            "information_live_provider_runtime_policy_schema_version": 1,
            "policy_name": "alice_information_live_provider_runtime_policy",
            "version": "1.0.0",
            "phase": "4",
            "milestone": "P4.10a",
            "status": "live_provider_foundation",
            "permission_id": "web.search",
            "capability_ceiling": False,
        }
        for key, expected in expected_scalars.items():
            if raw.get(key) != expected:
                raise InformationLiveProviderPolicyError(
                    f"{key} must remain exactly {expected!r}."
                )
        if raw.get("query_classifications") != ["PUBLIC"]:
            raise InformationLiveProviderPolicyError(
                "The initial P4.10 profile accepts PUBLIC queries only."
            )

        search = _exact_mapping(raw["search_provider"], "search_provider")
        expected_search = {
            "provider_id": "brave-search-v1",
            "provider_type": "live",
            "method": "GET",
            "scheme": "https",
            "host": "api.search.brave.com",
            "port": 443,
            "path": "/res/v1/web/search",
            "credential_environment_variable": "ALICE_BRAVE_SEARCH_API_KEY",
            "credential_header": "X-Subscription-Token",
            "maximum_query_characters": 400,
            "maximum_query_words": 50,
            "maximum_results": 5,
            "maximum_response_bytes": 1048576,
            "result_filter": "web",
            "offset": 0,
            "spellcheck": False,
            "text_decorations": False,
            "extra_snippets": False,
            "summary": False,
            "rich_callback": False,
            "fetch_metadata": False,
        }
        if search != expected_search:
            raise InformationLiveProviderPolicyError(
                "The exact Brave Search API profile changed."
            )

        fetch = _exact_mapping(raw["fetch_provider"], "fetch_provider")
        expected_fetch = {
            "provider_id": "controlled-live-http-v1",
            "provider_type": "live",
            "implementation": "LiveControlledInformationHttpRetriever",
            "credential_headers_allowed": False,
        }
        if fetch != expected_fetch:
            raise InformationLiveProviderPolicyError(
                "The exact controlled live fetch profile changed."
            )

        controls = _exact_mapping(raw["execution_controls"], "execution_controls")
        true_controls = {
            "foreground_only",
            "exact_provider_selection",
            "network_egress_metadata_required",
            "sanitized_failure_only",
        }
        false_controls = {
            "provider_fallback_allowed",
            "retry_allowed",
            "pagination_allowed",
            "recursive_browsing_allowed",
            "proxy_allowed",
            "cookie_allowed",
            "redirect_allowed_for_search_api",
            "source_body_persistence_allowed",
            "phase5_storage_allowed",
            "memory_write_allowed",
            "external_action_allowed",
            "background_execution_allowed",
            "private_query_allowed",
        }
        if set(controls) != true_controls | false_controls:
            raise InformationLiveProviderPolicyError(
                "The P4.10a execution-control vocabulary changed."
            )
        for key in true_controls:
            _exact_bool(controls, key, True)
        for key in false_controls:
            _exact_bool(controls, key, False)

        private = _exact_mapping(raw["private_configuration"], "private_configuration")
        expected_private = {
            "default_path": "C:\\ALICE_Vault\\config\\phase4-live-provider.json",
            "must_be_outside_repository": True,
            "credentials_in_configuration_allowed": False,
        }
        if private != expected_private:
            raise InformationLiveProviderPolicyError(
                "Private provider-configuration controls changed."
            )
        frozen = MappingProxyType(raw)
        result = cls(
            policy_name=str(raw["policy_name"]),
            version=str(raw["version"]),
            permission_id=str(raw["permission_id"]),
            search_provider_id=str(search["provider_id"]),
            fetch_provider_id=str(fetch["provider_id"]),
            search_host=str(search["host"]),
            search_path=str(search["path"]),
            credential_environment_variable=str(
                search["credential_environment_variable"]
            ),
            credential_header=str(search["credential_header"]),
            max_query_characters=int(search["maximum_query_characters"]),
            max_query_words=int(search["maximum_query_words"]),
            max_results=int(search["maximum_results"]),
            max_response_bytes=int(search["maximum_response_bytes"]),
            private_configuration_path=str(private["default_path"]),
            raw=frozen,
            policy_sha256=canonical_sha256(raw),
        )
        return result

    @classmethod
    def load(cls, path: str | Path) -> "InformationLiveProviderRuntimePolicy":
        policy_path = Path(path)
        try:
            value = json.loads(
                policy_path.read_text(encoding="utf-8-sig"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InformationLiveProviderPolicyError(
                "P4.10a runtime policy could not be loaded."
            ) from exc
        if not isinstance(value, dict):
            raise InformationLiveProviderPolicyError(
                "P4.10a runtime policy root must be an object."
            )
        return cls.from_mapping(value)

    def validate(self) -> None:
        reconstructed = InformationLiveProviderRuntimePolicy.from_mapping(self.raw)
        if reconstructed.policy_sha256 != self.policy_sha256:
            raise InformationLiveProviderPolicyError(
                "P4.10a runtime policy digest does not match canonical content."
            )
        for field in (
            self.policy_name,
            self.version,
            self.permission_id,
            self.search_provider_id,
            self.fetch_provider_id,
            self.search_host,
            self.search_path,
            self.credential_environment_variable,
            self.credential_header,
            self.private_configuration_path,
        ):
            if not field:
                raise InformationLiveProviderPolicyError(
                    "P4.10a policy identity fields must be non-empty."
                )

    @property
    def binding(self) -> str:
        return f"{self.policy_name}@{self.version}:{self.policy_sha256}"
