# Full EIPM Workload Envelope v0.1

**Date:** 2026-09-20  
**Status:** architecture requirement, not a benchmark description  
**Applies to:** N0 architecture decisions that must remain extensible through N1/N2/N3 and runtime IDP production

## 1. Why this exists

The current QSRE work began from a concrete N0 relation/evidence failure. That failure is useful evidence, but it is not the definition of the personality model.

The architecture target is the full EIPM role:

> Given the current situation, relationship state, relevant evidence, uncertainty, and possible interpretations/actions/responses, preserve the Elaina-derived A.L.I.C.E. identity/judgment policy independently of whichever downstream reasoner, generator, speech renderer, tool system, or world model is attached later.

Therefore a current benchmark is only a causal probe. It may reveal a broken abstraction, but passing it is not sufficient architecture justification.

The correct design rule is:

> Choose an architecture family that is sufficient for the full expected workload envelope, while avoiding mechanisms not justified by either that envelope or observed failure evidence.

This is **full-envelope sufficiency**, not "minimum fix for the current bug."

---

## 2. Functional workload families

### W1 — Language, implication, and pragmatic interpretation

The model must work with:
- explicit requests;
- indirect requests;
- implication and subtext;
- sarcasm, teasing, humor, embarrassment;
- ambiguity;
- politeness and social framing;
- incomplete or elliptical language;
- literal versus intended meaning.

Architecture consequence:
- token-level semantic information must remain available;
- one pooled semantic vector cannot be the only interface.

### W2 — Evidence, provenance, and epistemic role

Inputs may contain:
- immutable historical evidence;
- revisable inference;
- synthetic behavioral completion;
- post-activation experience;
- current self-state;
- uncertain or conflicting sources;
- temporal updates and supersession.

The model must distinguish:
- evidence from inference;
- fact from proposal;
- historical source authority from runtime policy;
- old evidence from later updates;
- co-valid alternatives from true contradiction.

Architecture consequence:
- provenance/epistemic role is structured state, not text decoration;
- evidence support and truth authority remain distinct;
- one retrieval relevance score cannot encode all epistemic roles.

Recent long-term-agent research independently supports this separation: provenance-aware memory systems increasingly use typed or parallel memory representations rather than one flat retrieval surface. This is architectural precedent, not identity authority.

### W3 — Relational and causal reasoning

The system must support:
- source/target argument roles;
- causal direction;
- support/correction/supersession;
- temporal successor relations;
- derivation;
- arbitrary future relation types through vocabulary migration;
- one-hop and multi-hop composition;
- one or many relevant relations.

Architecture consequence:
- relation/argument role must be first-class;
- query conditioning must participate inside execution;
- relevant support must be able to define the execution domain.

### W4 — Temporal state and change

The model must reason about:
- before/after;
- current versus historical state;
- changing beliefs or preferences;
- supersession;
- relationship evolution;
- repeated events;
- duration and recency;
- state trajectories.

Architecture consequence:
- time is a typed variable;
- the architecture must not assume static snapshots;
- ordered relational/path state must remain representable.

### W5 — Relationship-sensitive interpretation

Later private stages require behavior conditional on:
- actor identity and role;
- closeness/trust;
- public/private setting;
- relationship history;
- current relationship state;
- obligations and boundaries;
- asymmetry between people and roles.

Architecture consequence:
- relational execution cannot be limited to factual KG relations;
- arbitrary actor/relationship state must enter through typed context and continuous operator state;
- current public relation vocabulary is not the final identity relation ontology.

### W6 — Values and trade-offs

N2/N3 judgment requires:
- several simultaneously active values;
- context-dependent priority changes;
- consequences and stakes;
- competing goals;
- privacy, trust, loyalty, risk, fairness, affection, autonomy, etc.;
- non-binary trade-offs.

Architecture consequence:
- no scalar personality reward can be the whole IDP;
- multiple active considerations and competing alternatives must remain available to later judgment layers.

### W7 — Candidate comparison and action/response ranking

The EIPM may receive:
- candidate interpretations;
- actions;
- response summaries;
- generated responses.

It must:
- rank;
- preserve ties/plurality;
- reject out-of-character options;
- distinguish "not chosen" from "wrong";
- expose why candidates differ.

Architecture consequence:
- candidate cardinality is variable;
- the architecture must support setwise comparison without permanently fixing candidate count;
- structural/semantic evidence must remain inspectable.

### W8 — Emotional interpretation and response posture

The EIPM must support:
- warmth;
- seriousness;
- affection;
- vulnerability;
- grief;
- stress;
- anger;
- playful teasing;
- embarrassment;
- repair/reconciliation;
- emotional intensity and uncertainty.

Architecture consequence:
- emotion is not one global trait scalar;
- emotional posture is context-conditioned state that may depend on evidence, relationship, and stakes.

### W9 — Voice-first expressive policy

Voice interaction may supply:
- transcript;
- ASR confidence;
- timing;
- interruptions;
- paralinguistic observations;
- emotional cues.

The EIPM must output expressive intent:
- pace;
- emphasis;
- confidence;
- hesitation;
- intensity;
- warmth;
- playfulness;
- seriousness;
- vulnerability.

It is not a waveform model.

Architecture consequence:
- delivery policy is another structured output family;
- voice observations are optional context views;
- absence of voice input must not break identity judgment.

### W10 — Uncertainty and plurality

The model must preserve:
- insufficient evidence;
- multiple plausible interpretations;
- multiple plausible behavioral priors;
- ambiguous relation/operator hypotheses;
- ties or near-ties;
- defer/question behavior.

Architecture consequence:
- no premature hard singleton commitment;
- support and operator state may remain plural;
- calibration must remain explicit.

### W11 — Generic-assistant and sycophancy drift detection

The model eventually must detect:
- generic helpfulness replacing identity;
- excessive agreement;
- caricature;
- context-insensitive tone;
- behavior inconsistent with evidence-supported identity.

Architecture consequence:
- downstream generator text cannot be the hidden identity policy;
- EIPM must have its own latent/structured judgment state independent of final prose.

### W12 — Continuity and later self-development

Post-activation A.L.I.C.E. may:
- gain experience;
- revise non-E0 priors;
- develop stable self-state;
- preserve provenance from prior states.

Architecture consequence:
- later A-EXP/A-SELF can enter as typed context/views;
- historical Elaina evidence remains separate;
- the architecture must allow successor vocabularies and views without replacement.

### W13 — Heterogeneous context fusion

The wider N0/N1 stack may provide:
- raw semantic text;
- structured fields;
- evidence graph;
- relation execution state;
- relationship state;
- memory-derived state;
- voice observations;
- identity concepts;
- candidate sets.

Architecture consequence:
- no fixed three-view permanent ceiling;
- new views must be migratable;
- QSRE is one composable view/capability, not the whole cognitive fabric.

### W14 — Long-context and sparse relevance

Runtime context can exceed current checkpoint windows or contain many fields/edges.

Architecture consequence:
- serving should retrieve/expand adaptively;
- field, edge, support, hop, candidate, context, or view counts are operating shapes, not capability ceilings;
- sparse execution is an efficiency tool and causal isolation mechanism, not a permanent fixed-k assumption.

### W15 — Fast path and deliberate path

Common turns should remain efficient.

Hard turns may require:
- broader evidence;
- more supports;
- additional hops;
- more candidate comparisons;
- deeper deliberation.

Architecture consequence:
- latency defaults must not become ability ceilings;
- support cardinality and compute can expand when fidelity requires it.

---

## 3. Workload stress axes

Every future architecture review must consider combinations across:

- **support cardinality:** 0 / 1 / many / ambiguous;
- **hop depth:** 0 / 1 / multi-hop / branching;
- **relation vocabulary:** known / newly migrated / unknown;
- **candidate count:** 1 / few / many;
- **actor count:** one / dyadic / group;
- **relationship role:** stranger / friend / intimate / authority / adversarial / mixed;
- **stakes:** trivial / meaningful / high;
- **social setting:** private / public / group;
- **emotional intensity:** low / moderate / high / mixed;
- **temporal structure:** static / updated / superseded / trajectory;
- **provenance:** direct / inferred / synthetic / experiential / current self;
- **evidence consistency:** aligned / incomplete / conflicting / stale;
- **operator certainty:** clear / plural / unresolved;
- **context modality:** text / structured / voice cues / later additional views;
- **context size:** compact / broad / very large;
- **runtime mode:** fast path / expanded reasoning.

No single current benchmark crosses all axes. Architecture must remain capable of doing so.

---

## 4. N0 versus later-stage boundary

N0 does **not** need to learn private Elaina identity now.

N0 must build operations later identity learning depends on:

- semantic understanding;
- pragmatic interpretation;
- typed provenance;
- relation/argument execution;
- temporal reasoning;
- uncertainty/plurality;
- candidate comparison;
- heterogeneous context fusion;
- evidence sufficiency;
- stable structured outputs.

N1 adds governed identity representation.

N2 adds context-conditioned preference/judgment learning.

N3 calibrates and owner-refines identity fidelity.

A good N0 architecture is therefore not "generic QA." It is a public, identity-neutral **cognitive operation substrate** for later personality policy.

---

## 5. QSRE-specific implication

QSRE must be judged as a reusable relational/evidence primitive against this envelope.

It must not assume:
- only factual relations;
- one correct edge;
- one-hop reasoning;
- one actor pair;
- one current truth;
- one candidate answer;
- one emotion;
- one relation vocabulary;
- one evidence source;
- one fixed support count;
- one fixed context size.

QSRE does not itself implement values, personality, voice rendering, or continuity learning.

It supplies a trustworthy relation/evidence execution state that those later modules can use.

---

## 6. Full-envelope architecture rule

A future QSRE or N0 design advances only if:

1. it repairs the current proven failure boundary;
2. it remains structurally compatible with W1-W15;
3. it preserves provenance and uncertainty;
4. it supports variable cardinality and future schema migration;
5. it does not turn current benchmark dimensions into permanent product limits;
6. it remains composable with the existing semantic/structured/fusion/latent fabric;
7. its failure modes can be localized rather than hidden in one monolith;
8. efficiency optimizations do not delete necessary behavioral information.

This is the architecture selection standard going forward.
