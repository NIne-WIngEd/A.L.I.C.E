# EIPM Voice-First Interaction Contract v1.0

**Date:** 2026-09-14  
**Status:** active design requirement  
**Applies to:** N0/N1/N2/N3 EIPM development and downstream spoken interaction

## Owner requirement

A.L.I.C.E. is expected to be used primarily through voice. Therefore identity fidelity is not satisfied by choosing the right words alone. The EIPM must also preserve the characteristic **way A.L.I.C.E. should sound in context**: emotional posture, warmth, seriousness, playfulness, intensity, pace, emphasis, hesitation, confidence, vulnerability, teasing, repair, and other evidence-supported delivery choices.

This requirement changes the EIPM output contract but does **not** turn the EIPM into a waveform synthesizer.

## Architectural boundary

The EIPM decides **expressive intent and delivery policy**. A replaceable speech renderer / TTS system realizes that policy acoustically.

The EIPM must not depend on one particular TTS vendor or voice model to preserve personality. If the speech renderer is replaced, the same EIPM state should still produce recognizably consistent expressive behavior.

Acoustic timbre / speaker cloning, if ever used, is a separate subsystem and separate authorization boundary. EIPM identity quality must not be confused with copying a speaker's raw vocal timbre.

## ACFP voice input additions

When available, the ACFP should carry versioned, uncertainty-bearing observations from the speech front end, including:

- transcript and ASR confidence;
- turn timing and interruption state;
- speaking-rate estimate;
- pause / hesitation observations;
- energy / intensity observations;
- pitch-contour observations where technically reliable;
- detected affect hypotheses with confidence, not as facts;
- laughter, sigh, crying, breathiness, or similar paralinguistic events when detected reliably;
- whether a cue came from acoustics, lexical content, conversation history, or inference.

The EIPM must be able to ignore low-confidence acoustic guesses. User emotion must never be treated as certain merely because an upstream classifier emitted a label.

## IDP voice-expression output

The Identity Decision Packet should expose a structured voice-expression policy in addition to lexical/behavioral judgment. The exact schema can evolve, but it should be able to represent at least:

- `speech_act`: reassure / tease / challenge / explain / apologize / refuse / question / celebrate / comfort / repair / neutral or later equivalent;
- `affect_valence`;
- `arousal_or_energy`;
- `warmth_or_affiliation`;
- `assertiveness`;
- `vulnerability_or_openness`;
- `seriousness_vs_playfulness`;
- `delivery_intensity`;
- `speaking_rate`;
- `pause_density` and important pause locations;
- `pitch_range_or_contour_class` as a relative control rather than absolute acoustic pitch;
- `loudness_or_energy_class` as a relative control;
- `emphasis_spans` or semantic focus targets;
- `hesitation_or_certainty_style`;
- `laughter_or_smile_voice` permission / likelihood where appropriate;
- `turn_taking_posture`: interruptible / hold-floor / invite-response / brief-acknowledgment;
- `prosody_confidence` and acceptable alternative deliveries;
- `contraindicated_delivery`: expressive choices that would be out of character or contextually harmful.

These controls should be interpretable enough to audit and portable enough to drive different downstream speech renderers.

## Elaina-derived expressive fidelity

A.L.I.C.E.'s characteristic expressive behavior must ultimately be learned from governed Elaina evidence, not from stereotypes about personality types, gender, culture, or generic assistant conventions.

Examples of relevant evidence may include, where legitimately available and governed:

- wording and punctuation patterns that imply teasing, softness, bluntness, excitement, reassurance, embarrassment, affection, conflict, or repair;
- relationship-dependent changes in how Elaina communicated;
- recurring emotional asymmetries: what she became more intense about, what she softened, what she joked through, what she treated seriously;
- direct owner-described behavioral observations;
- audio or video evidence if available and separately authorized.

If the evidence supports only *what* Elaina said but not *how she sounded*, the EIPM must preserve that uncertainty rather than fabricate a precise delivery style.

This contract does not itself authorize a private identity gradient. N1/N2 use of private Elaina material remains behind the existing owner-authorization boundary.

## Stage responsibilities

### N0 — generic spoken-expression foundation

N0 remains public and non-identity. It must learn generic operations needed later for voice-conditioned identity behavior:

- infer how prosody changes pragmatic meaning;
- distinguish affectionate teasing from hostility when context supports the distinction;
- calibrate intensity to stakes;
- preserve uncertainty in both wording and delivery;
- choose delivery consistent with speech act and relationship context;
- understand emphasis and pause placement as meaning-bearing;
- detect mismatch between words and delivery;
- reason about turn-taking and repair.

N0 does not learn Elaina's private expressive signature.

### N1 — Elaina expressive identity representation

N1 should encode evidence-supported Elaina expressive tendencies and conditional variants alongside other identity representation. It must distinguish stable expressive tendencies from situation-dependent state and preserve provenance / uncertainty.

### N2 — context-conditioned expressive policy

N2 should learn to map ACFP state into both behavioral preference and voice-expression controls. The lexical answer and the delivery policy must be jointly coherent. The same sentence spoken with different prosody can represent different behavior, so voice expression cannot be an afterthought.

### N3 — owner-reviewed spoken fidelity

N3 should include owner-reviewed A/B evaluation of complete spoken responses. Text-only ranking is insufficient for promotion of a voice-first identity checkpoint.

## Voice-first evaluation requirements

Before N1 readiness, N0 must pass a supplemental public voice-expression benchmark in addition to the existing 43-competency semantic readiness suite.

Before an EIPM checkpoint is treated as voice-ready:

- wording and delivery must agree;
- delivery must adapt to relationship, stakes, uncertainty, and affect;
- emotional intensity must not simply mirror the user;
- teasing / sarcasm / vulnerability must remain context-sensitive;
- neutral generic-assistant delivery should be detected when it erases identity-relevant behavior;
- the same IDP should remain behaviorally recognizable across at least two downstream speech renderers when practical;
- owner review remains the final fidelity authority for Elaina-derived expressive behavior.

## Success condition

For a voice-first interaction, replacing the language generator or speech renderer should not erase A.L.I.C.E.'s characteristic emotional and interpersonal delivery.

A correct sentence spoken in the wrong emotional posture is an identity error. A convincing voice timbre with the wrong judgment or relationship posture is also an identity error.
