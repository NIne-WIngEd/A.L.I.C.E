"""Mission-scoped, profile-driven capability activation.

This is the successor path for release modules that intentionally hard-code a narrow
Phase 1-4 behavior. It does not assume a closed universe of capabilities, tools,
models, providers, data classes, or resource dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from alice_capability_profiles import (
    CapabilityProfile,
    CapabilityProfileError,
    load_capability_profile,
)

AUTONOMY_ORDER = {f"A{index}": index for index in range(7)}


class RuntimeActivationError(ValueError):
    """Raised when a profile activation exceeds mission authority or is malformed."""


@dataclass(frozen=True)
class MissionBudget:
    """Open-ended resource budget.

    Keys may include money, tokens, GPU-hours, wall-clock seconds, network bytes,
    storage bytes, API calls, energy, robot travel, or future resource dimensions.
    Missing keys are not interpreted as universal zero; policy decides whether the
    mission may infer a reasonable amount or must request an extension.
    """

    limits: Mapping[str, float | int] = field(default_factory=dict)

    def permits(self, resource: str, amount: float | int) -> bool:
        limit = self.limits.get(resource)
        return limit is None or amount <= limit


@dataclass(frozen=True)
class MissionAuthority:
    mission_id: str
    autonomy_class: str
    profile_ids: tuple[str, ...]
    allowed_external_targets: tuple[str, ...] = ()
    allowed_data_classes: tuple[str, ...] = ()
    budgets: MissionBudget = field(default_factory=MissionBudget)
    owner_ratified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.mission_id.strip():
            raise RuntimeActivationError("mission_id must be non-empty")
        if self.autonomy_class not in AUTONOMY_ORDER:
            raise RuntimeActivationError(
                f"Unknown autonomy class: {self.autonomy_class!r}"
            )
        if self.autonomy_class == "A6" and not self.owner_ratified:
            raise RuntimeActivationError(
                "A6 authority-kernel activation requires owner ratification."
            )


@dataclass(frozen=True)
class CapabilityDecision:
    capability: str
    allowed: bool
    profile_id: str
    mission_id: str
    reason: str
    required_autonomy_class: str | None = None


class CapabilityRuntime:
    """Activate named capability profiles inside a mission.

    Profiles can be added without editing this class. Unknown capability names are
    therefore not rejected merely because an older release did not anticipate them.
    """

    def __init__(self, mission: MissionAuthority):
        mission.validate()
        self.mission = mission
        self._profiles: dict[str, CapabilityProfile] = {}
        for profile_id in mission.profile_ids:
            try:
                profile = load_capability_profile(profile_id)
            except CapabilityProfileError as exc:
                raise RuntimeActivationError(str(exc)) from exc
            self._profiles[profile_id] = profile

    @property
    def profiles(self) -> tuple[CapabilityProfile, ...]:
        return tuple(self._profiles.values())

    def profile(self, profile_id: str) -> CapabilityProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise RuntimeActivationError(
                f"Profile {profile_id!r} is not active for mission {self.mission.mission_id!r}."
            ) from exc

    def capability_map(self, profile_id: str) -> dict[str, bool]:
        profile = self.profile(profile_id)
        for section_name in ("capabilities", "boundaries"):
            section = profile.payload.get(section_name)
            if isinstance(section, dict):
                result: dict[str, bool] = {}
                for name, value in section.items():
                    if isinstance(value, bool):
                        result[name] = value
                return result
        return {}

    def decide(
        self,
        *,
        profile_id: str,
        capability: str,
        required_autonomy_class: str | None = None,
        default_for_unknown: bool = False,
    ) -> CapabilityDecision:
        profile = self.profile(profile_id)
        if required_autonomy_class is not None:
            required = AUTONOMY_ORDER.get(required_autonomy_class)
            actual = AUTONOMY_ORDER[self.mission.autonomy_class]
            if required is None:
                raise RuntimeActivationError(
                    f"Unknown required autonomy class: {required_autonomy_class!r}"
                )
            if actual < required:
                return CapabilityDecision(
                    capability=capability,
                    allowed=False,
                    profile_id=profile_id,
                    mission_id=self.mission.mission_id,
                    required_autonomy_class=required_autonomy_class,
                    reason=(
                        f"Mission autonomy {self.mission.autonomy_class} is below "
                        f"required {required_autonomy_class}."
                    ),
                )
        mapping = self.capability_map(profile_id)
        allowed = mapping.get(capability, default_for_unknown)
        reason = (
            "Enabled by the selected mission profile."
            if allowed
            else "Not enabled by the selected mission profile."
        )
        return CapabilityDecision(
            capability=capability,
            allowed=allowed,
            profile_id=profile.profile_id,
            mission_id=self.mission.mission_id,
            required_autonomy_class=required_autonomy_class,
            reason=reason,
        )

    def resource_permitted(self, resource: str, amount: float | int) -> bool:
        return self.mission.budgets.permits(resource, amount)

    def describe(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission.mission_id,
            "autonomy_class": self.mission.autonomy_class,
            "owner_ratified": self.mission.owner_ratified,
            "profiles": [profile.profile_id for profile in self.profiles],
            "capabilities": {
                profile.profile_id: self.capability_map(profile.profile_id)
                for profile in self.profiles
            },
            "budget_limits": dict(self.mission.budgets.limits),
        }
