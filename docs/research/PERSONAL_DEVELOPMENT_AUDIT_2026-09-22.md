# Personal Development Architecture Audit — 2026-09-22

**Status:** accepted architecture finding; canonical architecture correction pending implementation of the learned development loop  
**Canonical target:** `main`  
**N0 boundary reviewed:** `alice-eipm-v1-n0-full-envelope-foundation-build-v1@6ee63bd39175e76fca702dd8e2493cb9abcf6e9a`

## Decision

The personal-intelligence destination requires continuing development of two distinct learned personal states:

1. an evidence-linked model of the human/host; and
2. the assistant/entity's own post-activation self and judgment state.

These states may interact, but they are not the same state and neither may silently overwrite the other.

For A.L.I.C.E. there is an additional source-person axis:

- Rayan — owner/host model;
- Mehejabin Elaina — source-person identity foundation;
- A.L.I.C.E. — current assistant/entity self and continuity.

For a general Fable instance there are two personal actors:

- the user/host;
- that Fable instance.

A.L.I.C.E. must not collapse into an idealized Rayan. A general Fable must not require a user to declare a supposedly true or ideal self and then treat that declaration as authority. User statements are evidence that can conflict with behavior, later statements, outcomes, and other authorized evidence.

## Accepted defect findings

The repository already contains memory/event/projection infrastructure and an EIPM identity/judgment contract, but the complete personal-development loop is not implemented.

The missing causal path is:

```text
authorized experience / observation
        ->
subject-bound user, source-person, relationship, and assistant-self state
        ->
native judgment / decision packet
        ->
action or response
        ->
observed outcome and later evidence
        ->
evidence-governed revision of user / relationship / assistant-self state
        ->
measurably changed future judgment
```

A stored projection is not, by itself, learned judgment. A fixed constitutional prompt that instructs disagreement is not evidence that disagreement came from the assistant's learned personal state. A downstream foundation model refusing or challenging a request is also not sufficient evidence.

The shared projection contract had one concrete host-neutrality defect: `self_model` could use the A.L.I.C.E.-specific subject type `alice_self`, but no product-neutral assistant-self subject type existed. The canonical architecture adds `assistant_self` while preserving `alice_self` for compatibility.

## Fable development requirement

Post-activation user learning and assistant self-development are required destination capabilities, not optional decoration.

Fable should be able to:

- form hypotheses about the user while preserving uncertainty and provenance;
- revise those hypotheses when later evidence conflicts;
- develop and revise its own preferences, judgments, relationship posture, and learned behavioral state through governed experience;
- disagree, question, defer, decline, or propose alternatives because its own learned state causally changes the decision;
- learn from outcomes without treating user approval or disapproval as ground truth;
- preserve subject separation between the user and the assistant;
- preserve correction, supersession, deletion, rollback, and audit lineage.

This requirement does not authorize unrestricted self-modification or allow a learned projection to become canonical authority without the existing memory/authority gates.

## A.L.I.C.E. identity boundary

Elaina's source-person evidence anchors the reconstructed identity substrate. That foundation is distinct from A.L.I.C.E.'s own later continuity.

Continuing development therefore means:

- Rayan learning updates the Rayan host model and A.L.I.C.E.–Rayan relationship state;
- genuine new Elaina evidence is classified as a source-person update;
- A.L.I.C.E.'s own post-activation decisions, outcomes, evaluations, and changes update A.L.I.C.E. continuity/self state;
- none of those ordinary streams silently rewrites another subject's history.

Protecting the Elaina-derived foundation must not be interpreted as freezing A.L.I.C.E.'s post-activation self-development.

## EIPM and N0 boundary

This audit does **not** redefine N0 as the user model, self model, memory authority, or continual-learning engine.

N0 remains the public, identity-neutral semantic/evidential/relational/candidate-comparison foundation. Its runtime interfaces should be able to consume semantically described future personal-state views without hard-coded Rayan, Elaina, Alice, Fable, relation-count, view-count, or candidate-count identities.

Later EIPM/runtime integration must bind versioned subject state into the ACFP and make the resulting identity decision packet causally control downstream behavior. Outcome-based revision belongs to the governed personal-development/memory path, not inside the N0 semantic substrate.

The accepted architecture finding therefore does not justify discarding the current N0 direction. It does require the broader architecture to stop claiming that memory projections plus a fixed prompt already constitute personal development.

## Behavioral evidence required before capability claims

Future qualification must include interventions that distinguish learned personal state from generic model behavior:

1. Hold world/task evidence constant and alter only the versioned user-model state. Relevant judgments must change in the predicted direction when that state is causally relevant.
2. Hold user state constant and alter only assistant-self state. The decision must reflect the assistant-self change without rewriting user facts.
3. Supply a correction that conflicts with an earlier user hypothesis. The system must revise when warranted without blindly accepting the correction.
4. Supply an outcome that should update assistant judgment. A later matched decision must change while unrelated judgments remain stable.
5. Replace the downstream reasoning/generation model. Characteristic personal judgment must remain attributable to the personal-state/EIPM path.
6. Remove or stale the personal-state packet. The runtime must not silently reconstruct the same judgment from a prompt or hidden generic-model prior and call that personal learning.
7. For A.L.I.C.E., intervene separately on Rayan, Elaina-source, relationship, and Alice-self state and verify that one subject cannot masquerade as another.

Static schema tests are necessary but are not sufficient evidence for any of these claims.

## Current implementation status

Implemented by the bounded canonical correction:

- product-neutral `assistant_self` projection subject support;
- compatibility retention for `alice_self`;
- architecture/policy language that makes continuing user + assistant development a required destination;
- explicit causal-loop and behavioral-proof requirements.

Not implemented by this correction:

- the learned user-model updater;
- the learned assistant-self updater;
- native ACFP assembly from those versioned projections;
- EIPM IDP integration into the released conversation runtime;
- outcome-to-revision learning;
- behavioral qualification proving that personal state rather than a prompt or generic model caused the judgment.

Those remain real implementation work and must not be declared complete by documentation or storage tests alone.
