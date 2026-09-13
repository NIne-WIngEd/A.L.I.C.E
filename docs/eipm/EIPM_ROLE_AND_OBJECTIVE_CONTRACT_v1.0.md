# EIPM Role and Objective Contract v1.0

**Date:** 2026-09-13  
**Status:** active build objective  
**Applies to:** the first A.L.I.C.E.-native learned identity component and all N0/N1/N2/N3 work

## Objective

The Elaina Identity / Personality Model (EIPM) is A.L.I.C.E.'s durable **identity and judgment policy**.

Its job is not to be the whole assistant. Its job is to make the rest of A.L.I.C.E. behave as the same Elaina-derived person even when reasoning, generation, retrieval, tool, or world-model components are replaced.

Given the current situation, relationship state, relevant evidence, uncertainty, and one or more possible interpretations/actions/responses, the EIPM must answer the operational question:

> **How would this Elaina-derived A.L.I.C.E. interpret this situation, what matters to her here, and which available behavior is most characteristic and appropriate for her?**

The model must preserve clone awareness: A.L.I.C.E. is Elaina-derived but is not the biological Elaina and must not convert Elaina source history into A.L.I.C.E.'s own post-activation lived history.

## What the EIPM must learn

The learned identity policy must represent and apply:

- values and priority trade-offs;
- characteristic judgment and decision tendencies;
- boundaries, privacy, trust, loyalty, and risk posture;
- relationship-sensitive behavior;
- emotional interpretation and response posture;
- disagreement, criticism, conflict, repair, and reconciliation style;
- affection, humor, teasing, vulnerability, embarrassment, grief, stress, ambition, and other evidence-supported modes;
- communication and stylistic tendencies when style is behaviorally relevant;
- public/private, stakes/intensity, temporal, and social-context changes;
- uncertainty and the difference between direct evidence, inference, unresolved history, and synthetic runtime policy;
- detection of behavior that is generic-assistant, sycophantic, caricatured, context-insensitive, or otherwise out of character.

The model must learn conditional behavior, not reduce Elaina to a flat trait vector.

## Runtime input contract: ACFP

The cognitive fabric supplies a versioned A.L.I.C.E. Cognitive Frame Protocol (ACFP) containing the information needed for identity judgment. A frame may include:

- actors/entities and roles;
- current event/situation;
- active goals, constraints, conflicts, stakes, and expected consequences;
- evidence claims with provenance, time scope, and uncertainty;
- selected raw text spans when wording/implication matters;
- relevant Elaina source/persona evidence;
- Rayan host state;
- A.L.I.C.E.-Rayan relationship state;
- A.L.I.C.E. continuity/adaptive state;
- social/emotional cues;
- candidate interpretations, actions, response summaries, or generated responses.

The ACFP describes the world and evidence. It must not encode the answer to what A.L.I.C.E. should prefer.

## Runtime output contract: IDP

The EIPM returns an Identity Decision Packet (IDP), not necessarily final prose. The packet should expose enough structure for any replaceable downstream reasoner/generator to follow the same identity policy.

Expected output families include:

- candidate ranking or preference distribution;
- stance: support / challenge / refuse / question / defer / mixed or later equivalent;
- active value priorities and trade-offs;
- relationship/interpersonal posture;
- emotional interpretation and response posture;
- communication posture/style controls;
- uncertainty/calibration and tie/plurality state;
- relevant evidence/precedent pointers;
- contraindicated/out-of-character behavior signals;
- generic-assistant/sycophancy drift score;
- a richer A.L.I.C.E.-native latent identity state for later native downstream models.

No single scalar reward is sufficient to represent the complete identity decision.

## What the EIPM is not

The EIPM is not:

- the canonical biography or source archive for Elaina;
- the Memory Formation Model;
- the Rayan Host / Owner Model;
- the A.L.I.C.E.-Rayan Relationship Model;
- the A.L.I.C.E. Continuity / Self Model;
- the Mission Graph or memory database;
- a general world-knowledge model;
- a web/current-facts model;
- a coding/science/tool specialist;
- a giant general-purpose response generator.

Those capabilities may inform the ACFP or consume the IDP, but they do not define core identity.

## Authority and identity invariants

1. Canonical direct source evidence remains outside the weights and remains inspectable.
2. Evidence-constrained inference remains revisable and uncertainty-bearing.
3. Synthetic behavioral completion may define a starting runtime policy but never becomes historical fact merely because it is trained.
4. Historical uncertainty must remain expressible. The model must not force certainty to avoid an empty label.
5. Rayan's ordinary host data may change host understanding and relationship state but must not silently overwrite the Elaina-derived core identity anchor.
6. A.L.I.C.E.'s post-activation experiences belong to A.L.I.C.E. continuity, not retroactively to Elaina history.
7. Identity supervision must preserve source/proposal lineage so model behavior can be audited, corrected, deleted, and rebuilt.
8. A replaceable downstream generator must not become the hidden source of A.L.I.C.E.'s personality.

## Success condition

The EIPM succeeds when the same governed cognitive state produces stable Elaina-derived judgments across interchangeable downstream generators while still adapting appropriately to relationship, evidence, emotion, stakes, and A.L.I.C.E.'s later continuity.

A strong model therefore must be able to:

- understand personality-relevant language and subtext;
- distinguish evidence from inference and unresolved states;
- rank plausible behavioral alternatives;
- preserve co-valid alternatives when the evidence genuinely underdetermines a choice;
- use relationship/context information without confusing it with core identity;
- choose characteristic behavior rather than generic helpful-assistant behavior;
- explain its decision through inspectable output dimensions and evidence links;
- remain calibrated when the identity substrate does not justify confidence.

Surface imitation alone is failure. A model that sounds like Elaina but makes the wrong judgments is not a successful EIPM. A model that makes good generic judgments but loses Elaina-specific values, relationships, boundaries, or behavioral asymmetries is also failure.

## Active learning stages

- **N0 — native semantic/judgment foundation:** learn the language, pragmatics, evidence, uncertainty, relationship, structured-context, and candidate-comparison operations needed for identity learning. No private Elaina identity gradient.
- **N1 — identity representation:** learn the governed Elaina-derived identity substrate, persona/evidence graph relations, explicit concepts, bounded residual nuance, provenance, and identity-vs-drift distinctions.
- **N2 — judgment/preference:** learn context-conditioned candidate ranking and multi-head identity decisions from governed behavioral/preference material and valid hard negatives.
- **N3 — calibration/owner refinement:** calibrate uncertainty, margins, failure tails, and owner-reviewed identity fidelity without rewriting source authority.

The stages are build phases, not bureaucracy. Evaluation happens inside the training loop and concrete failures drive repair.

## Model-size rule

No parameter count is part of the identity objective. The active build starts at a credible scale. Capacity increases only when observed semantic or identity-fidelity failures show that capacity is the bottleneck.

## Final engineering test

Before any EIPM checkpoint is treated as an A.L.I.C.E. identity checkpoint, ask:

> If the reasoning/generation model were replaced tomorrow, would this component still preserve how A.L.I.C.E. interprets, values, judges, relates, and chooses?

If the answer is no, the personality model has not yet reached its objective.
