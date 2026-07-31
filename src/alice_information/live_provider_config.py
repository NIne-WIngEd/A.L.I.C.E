"""Private P4.10 live-provider configuration and secret handling."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .live_provider_policy import (
    InformationLiveProviderRuntimePolicy,
    canonical_json_bytes,
)


class InformationLiveProviderConfigurationError(ValueError):
    """Raised when live configuration or credential custody is invalid."""


_SECRET_WORDS = ("key", "token", "secret", "password", "credential")
_ALLOWED_FIELDS = {"provider", "country", "search_lang", "ui_lang", "safesearch"}
_SAFESEARCH = {"off", "moderate", "strict"}


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InformationLiveProviderConfigurationError(
                f"Duplicate JSON key is not allowed: {key}."
            )
        result[key] = value
    return result


class InformationSecretValue:
    """Opaque credential wrapper that cannot be serialized or represented raw."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise InformationLiveProviderConfigurationError(
                "Live provider credential is missing."
            )
        if any(ord(character) < 33 or ord(character) == 127 for character in value):
            raise InformationLiveProviderConfigurationError(
                "Live provider credential contains invalid characters."
            )
        self._value = value

    def reveal_for_exact_header(self) -> str:
        return self._value

    def __str__(self) -> str:
        return "<redacted>"

    def __repr__(self) -> str:
        return "InformationSecretValue(<redacted>)"

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("InformationSecretValue cannot be serialized.")

    def __getstate__(self) -> object:
        raise TypeError("InformationSecretValue cannot be serialized.")


@dataclass(frozen=True)
class InformationLiveProviderConfiguration:
    """Nonsecret private configuration plus an environment-held credential."""

    provider: str
    country: str
    search_lang: str
    ui_lang: str
    safesearch: str
    configuration_path: str
    configuration_sha256: str
    credential: InformationSecretValue = field(repr=False)
    metadata: Mapping[str, str] = field(repr=False)

    def validate(self, *, policy: InformationLiveProviderRuntimePolicy) -> None:
        if self.provider != policy.search_provider_id:
            raise InformationLiveProviderConfigurationError(
                "Configured live provider does not match the exact P4.10a policy."
            )
        if not (len(self.country) == 2 and self.country.isalpha() and self.country.isupper()):
            raise InformationLiveProviderConfigurationError(
                "country must be a two-letter uppercase code."
            )
        if not self.search_lang or len(self.search_lang) > 16:
            raise InformationLiveProviderConfigurationError("search_lang is invalid.")
        if not self.ui_lang or len(self.ui_lang) > 32:
            raise InformationLiveProviderConfigurationError("ui_lang is invalid.")
        if self.safesearch not in _SAFESEARCH:
            raise InformationLiveProviderConfigurationError("safesearch is invalid.")
        expected = {
            "provider": self.provider,
            "country": self.country,
            "search_lang": self.search_lang,
            "ui_lang": self.ui_lang,
            "safesearch": self.safesearch,
        }
        digest = hashlib.sha256(canonical_json_bytes(expected)).hexdigest()
        if digest != self.configuration_sha256:
            raise InformationLiveProviderConfigurationError(
                "Live provider configuration digest does not match."
            )
        if dict(self.metadata) != expected:
            raise InformationLiveProviderConfigurationError(
                "Live provider metadata changed after loading."
            )
        if not isinstance(self.credential, InformationSecretValue):
            raise InformationLiveProviderConfigurationError(
                "Live provider credential wrapper was substituted."
            )

    def to_metadata_record(self) -> dict[str, str]:
        return {
            **dict(self.metadata),
            "configuration_path": self.configuration_path,
            "configuration_sha256": self.configuration_sha256,
        }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_live_provider_configuration(
    path: str | Path,
    *,
    repository_root: str | Path,
    policy: InformationLiveProviderRuntimePolicy,
    environment: Mapping[str, str] | None = None,
) -> InformationLiveProviderConfiguration:
    """Load one private nonsecret configuration and exact environment credential."""

    policy.validate()
    repository = Path(repository_root).resolve(strict=True)
    config_path = Path(path).resolve(strict=True)
    if _is_within(config_path, repository):
        raise InformationLiveProviderConfigurationError(
            "Live provider configuration must remain outside the repository."
        )
    try:
        raw = config_path.read_text(encoding="utf-8-sig")
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InformationLiveProviderConfigurationError(
            "Live provider configuration could not be loaded."
        ) from exc
    if not isinstance(value, dict):
        raise InformationLiveProviderConfigurationError(
            "Live provider configuration root must be an object."
        )
    if set(value) != _ALLOWED_FIELDS:
        raise InformationLiveProviderConfigurationError(
            "Live provider configuration fields do not match the approved schema."
        )
    for key in value:
        if any(word in key.casefold() for word in _SECRET_WORDS):
            raise InformationLiveProviderConfigurationError(
                "Credentials cannot be stored in the provider configuration file."
            )
    if not all(isinstance(item, str) and item.strip() for item in value.values()):
        raise InformationLiveProviderConfigurationError(
            "Live provider configuration values must be non-empty text."
        )
    metadata = {
        "provider": value["provider"].strip(),
        "country": value["country"].strip().upper(),
        "search_lang": value["search_lang"].strip(),
        "ui_lang": value["ui_lang"].strip(),
        "safesearch": value["safesearch"].strip().lower(),
    }
    source = os.environ if environment is None else environment
    token = source.get(policy.credential_environment_variable, "")
    configuration = InformationLiveProviderConfiguration(
        provider=metadata["provider"],
        country=metadata["country"],
        search_lang=metadata["search_lang"],
        ui_lang=metadata["ui_lang"],
        safesearch=metadata["safesearch"],
        configuration_path=str(config_path),
        configuration_sha256=hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(),
        credential=InformationSecretValue(token),
        metadata=MappingProxyType(metadata),
    )
    configuration.validate(policy=policy)
    return configuration
