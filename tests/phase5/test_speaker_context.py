from dataclasses import replace

import pytest

from cognitive_kernel import CognitiveKernelContractError, SpeakerContext
from speaker_guest_authority_helpers import host_context, scope


def test_host_privilege_context_is_metadata_only_and_tamper_evident():
    context = host_context()
    assert context.authority_ceiling == "host_verified"
    assert context.raw_audio_included is False
    assert context.voice_profile_included is False
    assert context.context_sha256
    with pytest.raises(CognitiveKernelContractError):
        replace(context, confidence=0.2).validate()


def test_voice_match_alone_cannot_establish_host_verified_authority():
    with pytest.raises(CognitiveKernelContractError):
        SpeakerContext.create(
            context_key="voice-only",
            scope=scope(),
            observed_at="2026-08-01T20:00:00Z",
            speaker_state="recognized",
            trust_state="host_context_recognized",
            evidence_class="local_voice_match",
            confidence=0.95,
            authority_ceiling="host_verified",
        )


def test_multiple_speakers_carry_no_authority():
    context = SpeakerContext.create(
        context_key="room",
        scope=scope(),
        observed_at="2026-08-01T20:00:00Z",
        speaker_state="multiple_speakers",
        trust_state="multiple_speakers",
        evidence_class="diarization_metadata",
        confidence=0.8,
        speaker_count=3,
        authority_ceiling="none",
    )
    assert context.speaker_count == 3


def test_raw_audio_and_voice_profiles_are_rejected():
    with pytest.raises(CognitiveKernelContractError):
        host_context(raw_audio_included=True)
    with pytest.raises(CognitiveKernelContractError):
        host_context(voice_profile_included=True)
