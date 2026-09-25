# Fable Personal Development Architecture

**Status:** canonical destination architecture  
**Scope:** host-neutral user-model development, assistant-self development, relationship development, and native judgment integration

## Purpose

A Fable instance is not complete merely because it has a memory system, a personality prompt, or stored user projections.

The destination capability is a developing personal intelligence that can form and revise a model of its user, develop and revise its own post-activation state, and let those learned states causally affect judgment.

## Subject separation

Every Fable instance keeps at least two personal subjects distinct:

- the user/host;
- the Fable instance itself.

A.L.I.C.E. additionally preserves a separate source-person axis:

- Rayan — owner/host;
- Mehejabin Elaina — source-person identity foundation;
- A.L.I.C.E. — current assistant/entity self.

A.L.I.C.E. must not silently become an idealized Rayan. A general Fable must not treat a user's declaration of an ideal or "true" self as canonical identity authority.

User statements are evidence. They may be revised, contradicted, superseded, or reinterpreted when later authorized evidence warrants it.

## Required causal loop

The capability must eventually implement and behaviorally verify:

```text
authorized experience / observation
        ↓
subject-bound user / relationship / assistant-self state
        ↓
native personal judgment
        ↓
action / response / decision
        ↓
observed outcome and later evidence
        ↓
governed revision
        ↓
changed future judgment when relevant
```

Storage is not sufficient evidence of this loop.

A fixed system prompt that says "challenge the user when needed" is not proof of learned personal judgment.

A generic downstream model refusing, disagreeing, or proposing an alternative is also not proof that Fable's learned personal state caused the behavior.

## User/host model

The user model should support:

- preferences;
- goals;
- projects;
- habits;
- changing beliefs;
- routines;
- interaction patterns;
- relationship expectations;
- evidence-grounded predictions;
- uncertainty;
- conflict;
- temporal change;
- correction and supersession;
- provenance.

The user model remains revisable.

User approval or disapproval is evidence about the interaction. It is not automatically ground truth about the world, the user, or the assistant's judgment.

## Assistant-self model

The assistant-self model represents the Fable instance's own post-activation continuity and learned behavioral state.

It may include:

- learned preferences;
- recurring judgments;
- interaction posture;
- relationship-specific expectations;
- evaluations of prior outcomes;
- stable or changing behavioral tendencies;
- self-consistency and conflict;
- learned boundaries and uncertainty;
- post-activation continuity.

This state must remain distinct from the user model.

For product-neutral projection contracts, the subject type is `assistant_self`. A.L.I.C.E.-specific compatibility records may retain `alice_self`.

## Relationship state

Relationship state captures information that belongs to neither subject alone.

Examples include:

- shared history;
- interaction norms;
- recurring coordination patterns;
- expectations;
- trust and uncertainty state;
- shared projects;
- shared habits;
- relationship-specific behavioral adaptation.

Relationship development must not silently rewrite the user's personal history or the assistant's source identity.

## Native judgment integration

Personal state must eventually enter native judgment before the downstream response is generated.

The target boundary is:

```text
versioned personal-state projections
        ↓
ACFP / identity-relevant context assembly
        ↓
EIPM native judgment
        ↓
identity decision packet
        ↓
downstream reasoning / response generation
```

The downstream model should not be the sole owner of personality, disagreement, preference, or relationship behavior.

## First-release conversation boundary (2026-09-25)

The destination loop also controls the *expression* of a judgment. Conversation is a first-party local capability under the personal foundation, even when a replaceable external language engine proposes words. The five starting personal roles and existing memory infrastructure participate in evaluating each draft; this is not a sixth memory model or an external judge.

The native identity decision packet must specify stance, evidence and uncertainty, reasons and disagreement intent, subject/relationship context, and enough expression obligations to tell whether the reply behaves as that particular entity should. A packet saying only “agree” or “disagree” cannot qualify.

The conversation architecture:

1. assembles authorized personal state and computes the native verdict before invoking a downstream generator;
2. compiles the verdict into a first-pass language request through a local privacy and egress boundary;
3. receives a candidate from a replaceable GPT/Claude-class feature service or another qualified engine;
4. compares the candidate against the verdict with the existing personal models, state, and governed response rules, including voice and relationship behavior, then restores locally held private references;
5. accepts a matching draft or issues a targeted, bounded correction; on exhaustion it uses an appropriate local response or exposes the limitation;
6. records any later outcome for governed learning.

The intended ordinary path is one external generation attempt, not repeated redrafting. Latency, first-pass acceptance, provider portability, false acceptance of generic-assistant phrasing, and context-specific identity behavior require measurable release gates. A third-party generator's apparent disagreement is not evidence that EIPM caused it.

A local encoder/decoder and egress gateway can redact identifiers and minimize disclosed context, but cannot guarantee zero semantic leakage to an external text API. The gateway must check source permissions and data classes, keep private mappings local, disclose destination and task, and decline or seek explicit user authorization when an exact task requires private material. No unfiltered memory store or raw identity packet becomes an API prompt. Offline behavior and remote-processing limits must be represented truthfully.

This is a destination design requirement, not a statement that Phase 3's existing local Qwen adapter or current EIPM runtime implements it. Preserve A.L.I.C.E.'s Elaina source-person, Rayan host, and assistant-self separation when prototyping. Generalize only the capability to Fable.

## Outcome-driven development

The architecture must support learning from outcomes.

A later decision should change when:

- the user model changed for a relevant reason;
- assistant-self state changed from a relevant experience;
- relationship state changed;
- prior judgment produced a meaningful outcome;
- new evidence corrected an earlier hypothesis.

Unrelated behavior should remain stable when the changed state is irrelevant.

## Qualification requirements

Capability claims require behavioral interventions, not only schema tests.

At minimum:

1. change only user-model state and verify relevant judgment changes;
2. change only assistant-self state and verify relevant judgment changes;
3. change only relationship state and verify relationship-specific behavior changes;
4. provide conflicting user evidence and verify evidence-governed revision;
5. provide outcome evidence and verify later judgment revision;
6. swap the downstream reasoning/generation model and verify characteristic personal judgment remains attributable to the personal-state/EIPM path;
7. stale or remove personal state and verify the system does not reconstruct the same behavior from a hidden prompt and call it learned development;
8. hold the conclusion fixed while changing relevant personal voice or relationship state, and verify that an otherwise fluent generic response is rejected or corrected;
9. measure first-pass acceptance, corrective-call count, end-to-end latency, provider-swap stability, and privacy egress across realistic and adversarial multi-turn scenarios.

## Authority boundary

Continuing personal development does not authorize unrestricted self-modification.

The existing provenance, memory, authority, correction, deletion, revocation, rollback, and model-promotion rules remain active.

The system may learn and revise personal state only through governed mechanisms whose lineage can be inspected and reversed when required.

## N0 boundary

N0 remains the host-neutral semantic/evidential/relational/candidate-comparison foundation.

It should be able to consume semantically described personal-state views in the future.

N0 does not become:

- the user model;
- the assistant-self model;
- the relationship store;
- source-person authority;
- memory authority;
- the continual-development engine.

This separation preserves N0's generality while allowing later EIPM stages to make personal state causally control judgment.
