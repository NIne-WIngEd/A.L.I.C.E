"""Shared capability-profile registry for evolvable A.L.I.C.E. runtimes.

A profile describes a deployed or experimental configuration. A false value in one
profile is not a permanent system prohibition. Permanent authority-kernel rules live
in policies/authority_kernel_policy.json, not in feature modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_PROFILE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "policies" / "capability_profiles.json"
)


class CapabilityProfileError(ValueError):
    """Raised when a capability profile or activation request is invalid."""


@dataclass(frozen=True)
class CapabilityProfile:
    profile_id: str
    domain: str
    scope_kind: str
    capability_ceiling: bool
    payload: Mapping[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.payload.get(name, {})
        if not isinstance(value, dict):
            raise CapabilityProfileError(
                f"Profile {self.profile_id!r} section {name!r} must be an object."
            )
        return dict(value)


def _read_registry(path: str | Path = DEFAULT_PROFILE_REGISTRY_PATH) -> dict[str, Any]:
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityProfileError(
            f"Unable to load capability profile registry: {selected}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        raise CapabilityProfileError("Capability profile registry requires a profiles object.")
    return payload


def load_capability_profile(
    profile_id: str,
    *,
    expected_domain: str | None = None,
    path: str | Path = DEFAULT_PROFILE_REGISTRY_PATH,
) -> CapabilityProfile:
    registry = _read_registry(path)
    raw = registry["profiles"].get(profile_id)
    if not isinstance(raw, dict):
        raise CapabilityProfileError(f"Unknown capability profile: {profile_id!r}")
    domain = raw.get("domain")
    scope_kind = raw.get("scope_kind")
    ceiling = raw.get("capability_ceiling")
    if not isinstance(domain, str) or not domain:
        raise CapabilityProfileError(f"Profile {profile_id!r} requires a domain.")
    if expected_domain is not None and domain != expected_domain:
        raise CapabilityProfileError(
            f"Profile {profile_id!r} belongs to {domain!r}, not {expected_domain!r}."
        )
    if not isinstance(scope_kind, str) or not scope_kind:
        raise CapabilityProfileError(f"Profile {profile_id!r} requires scope_kind.")
    if ceiling is not False:
        raise CapabilityProfileError(
            f"Profile {profile_id!r} must declare capability_ceiling=false."
        )
    return CapabilityProfile(
        profile_id=profile_id,
        domain=domain,
        scope_kind=scope_kind,
        capability_ceiling=False,
        payload=dict(raw),
    )


def authorize_boolean_map(
    *,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    field: str,
) -> dict[str, bool]:
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise CapabilityProfileError(
            f"{field} keys differ from profile; missing={missing}, extra={extra}."
        )
    result: dict[str, bool] = {}
    for name, expected_value in expected.items():
        value = actual[name]
        if not isinstance(value, bool):
            raise CapabilityProfileError(f"{field}.{name} must be boolean.")
        if value is not expected_value:
            raise CapabilityProfileError(
                f"{field}.{name}={value!r} is not authorized by the selected profile."
            )
        result[name] = value
    return result


def require_profile_value(*, actual: Any, expected: Any, field: str) -> Any:
    if actual != expected:
        raise CapabilityProfileError(
            f"{field}={actual!r} is not authorized by the selected capability profile."
        )
    return actual
