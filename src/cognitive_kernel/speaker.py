"""Metadata-only speaker-context contracts for the Personal Cognitive Kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_confidence,
    require_identifier,
    require_schema_version,
)
from .contracts import ProductHostScope

SPEAKER_STATES = frozenset(
    {"recognized", "uncertain", "unknown", "multiple_speakers"}
)
SPEAKER_TRUST_STATES = frozenset(
    {
        "speaker_recognized",
        "speaker_uncertain",
        "host_context_recognized",
        "host_privilege_verified",
        "guest_session",
        "delegated_guest_session",
        "multiple_speakers",
    }
)
SPEAKER_EVIDENCE_CLASSES = frozenset(
    {
        "unknown",
        "diarization_metadata",
        "local_voice_match",
        "host_session_context",
        "stronger_authentication",
        "explicit_guest_grant",
    }
)
SPEAKER_AUTHORITY_CEILINGS = (
    "none",
    "guest_scoped",
    "host_context",
    "host_verified",
)


def _enum(value: object, field: str, allowed: Iterable[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


def _speaker_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CognitiveKernelContractError("speaker_count must be an integer")
    if value < 1 or value > 32:
        raise CognitiveKernelContractError("speaker_count must be between 1 and 32")
    return value


@dataclass(frozen=True)
class SpeakerContext:
    """Tamper-evident metadata about who may be speaking and what is known."""

    schema_version: str
    context_id: str
    context_key: str
    scope: ProductHostScope
    observed_at: str
    speaker_state: str
    trust_state: str
    evidence_class: str
    confidence: float
    speaker_reference_id: str | None
    session_reference_id: str | None
    speaker_count: int
    stronger_authentication_verified: bool
    raw_audio_included: bool
    voice_profile_included: bool
    authority_ceiling: str
    reason_codes: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    context_sha256: str

    @classmethod
    def create(
        cls,
        *,
        context_key: object,
        scope: ProductHostScope,
        observed_at: object,
        speaker_state: object,
        trust_state: object,
        evidence_class: object,
        confidence: object,
        speaker_count: object = 1,
        speaker_reference_id: object | None = None,
        session_reference_id: object | None = None,
        stronger_authentication_verified: object = False,
        raw_audio_included: object = False,
        voice_profile_included: object = False,
        authority_ceiling: object = "none",
        reason_codes: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "SpeakerContext":
        scope.validate()
        normalized_confidence = require_confidence(confidence)
        if normalized_confidence is None:
            raise CognitiveKernelContractError("confidence is required")
        if not isinstance(stronger_authentication_verified, bool):
            raise CognitiveKernelContractError(
                "stronger_authentication_verified must be boolean"
            )
        if not isinstance(raw_audio_included, bool) or not isinstance(
            voice_profile_included, bool
        ):
            raise CognitiveKernelContractError(
                "payload inclusion flags must be boolean"
            )
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "context_key": require_identifier(context_key, "context_key"),
            "observed_at": normalize_timestamp(observed_at, "observed_at"),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            context_id=f"speaker-context-{canonical_sha256(identity)[:32]}",
            context_key=identity["context_key"],
            scope=scope,
            observed_at=identity["observed_at"],
            speaker_state=_enum(speaker_state, "speaker_state", SPEAKER_STATES),
            trust_state=_enum(
                trust_state, "trust_state", SPEAKER_TRUST_STATES
            ),
            evidence_class=_enum(
                evidence_class,
                "evidence_class",
                SPEAKER_EVIDENCE_CLASSES,
            ),
            confidence=normalized_confidence,
            speaker_reference_id=(
                require_identifier(
                    speaker_reference_id, "speaker_reference_id"
                )
                if speaker_reference_id is not None
                else None
            ),
            session_reference_id=(
                require_identifier(
                    session_reference_id, "session_reference_id"
                )
                if session_reference_id is not None
                else None
            ),
            speaker_count=_speaker_count(speaker_count),
            stronger_authentication_verified=stronger_authentication_verified,
            raw_audio_included=raw_audio_included,
            voice_profile_included=voice_profile_included,
            authority_ceiling=_enum(
                authority_ceiling,
                "authority_ceiling",
                SPEAKER_AUTHORITY_CEILINGS,
            ),
            reason_codes=normalize_identifier_sequence(
                reason_codes, "reason_codes"
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            context_sha256="0" * 64,
        )
        provisional._validate_material()
        context = cls(
            **{
                **provisional.__dict__,
                "context_sha256": canonical_sha256(
                    provisional.material_record()
                ),
            }
        )
        context.validate()
        return context

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "context_key": self.context_key,
            "observed_at": self.observed_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.context_key, "context_key")
        normalize_timestamp(self.observed_at, "observed_at")
        state = _enum(self.speaker_state, "speaker_state", SPEAKER_STATES)
        trust = _enum(
            self.trust_state, "trust_state", SPEAKER_TRUST_STATES
        )
        evidence = _enum(
            self.evidence_class,
            "evidence_class",
            SPEAKER_EVIDENCE_CLASSES,
        )
        confidence = require_confidence(self.confidence)
        if confidence is None:
            raise CognitiveKernelContractError("confidence is required")
        _speaker_count(self.speaker_count)
        if self.speaker_reference_id is not None:
            require_identifier(
                self.speaker_reference_id, "speaker_reference_id"
            )
        if self.session_reference_id is not None:
            require_identifier(
                self.session_reference_id, "session_reference_id"
            )
        if not isinstance(self.stronger_authentication_verified, bool):
            raise CognitiveKernelContractError(
                "stronger_authentication_verified must be boolean"
            )
        if self.raw_audio_included or self.voice_profile_included:
            raise CognitiveKernelContractError(
                "speaker context must not contain raw audio or voice profiles"
            )
        ceiling = _enum(
            self.authority_ceiling,
            "authority_ceiling",
            SPEAKER_AUTHORITY_CEILINGS,
        )
        normalize_identifier_sequence(self.reason_codes, "reason_codes")
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )

        if state == "multiple_speakers":
            if self.speaker_count < 2 or trust != "multiple_speakers":
                raise CognitiveKernelContractError(
                    "multiple-speaker state requires count >= 2 and matching trust state"
                )
        elif self.speaker_count != 1:
            raise CognitiveKernelContractError(
                "single-speaker states require speaker_count=1"
            )
        if trust == "speaker_uncertain" and state not in {
            "uncertain",
            "unknown",
        }:
            raise CognitiveKernelContractError(
                "speaker_uncertain trust requires uncertain or unknown state"
            )
        if trust == "host_privilege_verified":
            if not self.stronger_authentication_verified:
                raise CognitiveKernelContractError(
                    "host privilege requires stronger authentication"
                )
            if evidence != "stronger_authentication" or ceiling != "host_verified":
                raise CognitiveKernelContractError(
                    "host privilege requires stronger-auth evidence and host_verified ceiling"
                )
        if evidence == "local_voice_match" and ceiling == "host_verified":
            raise CognitiveKernelContractError(
                "voice matching alone may not establish privileged authority"
            )
        if trust in {"guest_session", "delegated_guest_session"}:
            if evidence != "explicit_guest_grant":
                raise CognitiveKernelContractError(
                    "guest trust requires explicit grant evidence"
                )
            if ceiling != "guest_scoped" or self.session_reference_id is None:
                raise CognitiveKernelContractError(
                    "guest trust requires a scoped session and guest authority ceiling"
                )
        if trust == "multiple_speakers" and ceiling != "none":
            raise CognitiveKernelContractError(
                "multiple-speaker context may not carry authority"
            )

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "context_id": self.context_id,
            "speaker_state": self.speaker_state,
            "trust_state": self.trust_state,
            "evidence_class": self.evidence_class,
            "confidence": self.confidence,
            "speaker_reference_id": self.speaker_reference_id,
            "session_reference_id": self.session_reference_id,
            "speaker_count": self.speaker_count,
            "stronger_authentication_verified": self.stronger_authentication_verified,
            "raw_audio_included": self.raw_audio_included,
            "voice_profile_included": self.voice_profile_included,
            "authority_ceiling": self.authority_ceiling,
            "reason_codes": list(self.reason_codes),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        return {**self.material_record(), "context_sha256": self.context_sha256}

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"speaker-context-{canonical_sha256(self.identity_record())[:32]}"
        if self.context_id != expected_id:
            raise CognitiveKernelContractError("speaker context identity is invalid")
        expected_digest = canonical_sha256(self.material_record())
        if self.context_sha256 != expected_digest:
            raise CognitiveKernelContractError("speaker context digest mismatch")
