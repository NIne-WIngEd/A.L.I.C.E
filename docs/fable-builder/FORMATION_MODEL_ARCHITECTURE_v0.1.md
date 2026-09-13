# Fable Builder Model Architecture v0.1

**Status:** working architecture; not yet owner-finalized  
**Date:** 2026-09-13

## 1. Problem

During A.L.I.C.E. construction, Sol, Astra, other assistants, scripts, and the owner collectively perform a repeatable set of cognitive operations: interpret source evidence, infer tentative personality structure, synthesize missing behavioral coverage, reject unsupported material, create teaching examples, evaluate fidelity, and repair gaps.

A consumer Fable installation cannot assume those external assistants are available. Those operations must therefore become a native Fable capability.

The working solution is a **Fable Builder Model (FBM)**: a host-neutral formation model that bootstraps a consumer-specific personality model from host-authorized data while preserving provenance, uncertainty, and identity boundaries.

## 2. What FBM is not

FBM is not the consumer's personality.

FBM is not historical authority.

FBM must not turn inference into evidence, synthetic completion into biography, or runtime experience into source-person history.

FBM must not need a fixed third-party teacher or judge at product runtime.

FBM should not contain Elaina-specific private content. A.L.I.C.E. supplies development lessons and training/evaluation methodology, not transferable private identity payloads.

## 3. Primary operating modes

### 3.1 Bootstrap formation mode

Used before personality activation.

Responsibilities:

1. ingest and normalize host-authorized source data;
2. extract source-grounded observations and behavioral evidence;
3. create structured evidence units with provenance;
4. classify epistemic status;
5. propose E-INF hypotheses only when support exists;
6. preserve UNKNOWN where support is inadequate;
7. generate A-SYN starting policies where historical truth is unknown but a runtime behavior is needed;
8. map behavioral/personality coverage;
9. generate competing hypotheses and counterfactual branches;
10. deduplicate and detect mechanically redundant variants;
11. detect contradictions, unsupported specificity, generic-assistant drift, and context errors;
12. compile a personality-teaching substrate;
13. generate semantic, pragmatic, social, emotional, judgment, calibration, and identity-specific curricula;
14. evaluate the candidate personality model;
15. drive targeted repair from observed failures.

### 3.2 Runtime formation mode

Used after activation for new ordinary experiences.

The shared formation backbone may continue as or initialize the **Memory Formation Model**. Runtime output is a structured memory proposal. Authority remains with deterministic memory/authority governance.

### 3.3 Governed identity-maintenance mode

Optional post-activation mode. It may examine accumulated evidence, feedback, and fidelity failures and propose personality-model maintenance material.

This mode must be separately gated from ordinary memory formation. It cannot silently rewrite the core personality model during normal conversation.

## 4. Capability decomposition

FBM should initially be designed as one product capability with separable modules/heads rather than prematurely forcing one monolithic objective.

### A. Source Interpreter

Converts heterogeneous user-authorized data into normalized evidence units. Maintains source IDs, timestamps, modality, speaker/author attribution, relationship context, and confidence in extraction.

### B. Provenance and Epistemic Classifier

Separates:

- direct source evidence / E0-equivalent material;
- evidence-constrained inference / E-INF;
- synthetic runtime prior / A-SYN;
- historical UNKNOWN;
- later lived experience / A-EXP-equivalent material;
- current self/continuity state where applicable.

The exact public Fable names may later differ, but the semantic distinctions must survive.

### C. Hypothesis Generator

Constructs conditional personality hypotheses from evidence. It should produce multiple branches when the data underdetermines behavior and explicitly surface counterevidence and boundary conditions.

### D. Behavioral Synthesizer

Produces non-historical synthetic starting policies for uncovered situations. Its purpose is behavioral completion, not rewriting biography.

### E. Coverage Planner

Represents meaningful combinations of values, relationships, social setting, emotional state, stakes, temporal state, public/private context, uncertainty, and other behavior-changing variables. It identifies weak or missing regions without creating a blind Cartesian explosion.

### F. Novelty and Consistency Critic

Detects near-paraphrases, mechanical transformations, contradictions, trait overconcentration, relationship-context errors, historical/runtime confusion, unsupported specificity, sycophancy, and generic-assistant contamination.

### G. Curriculum Composer

Turns public semantic material plus consumer-specific substrate into training exercises for the target personality model. It creates contrastive pairs, candidate sets, hard negatives, contextual variants, ACFP-like frames, graph-alignment tasks, uncertainty tasks, and targeted repair examples.

### H. Fidelity Evaluator

Tests the personality model against held-out evidence and scenario obligations. It reports failure categories and proposes targeted repair rather than merely producing one scalar score.

### I. Training/Update Planner

Chooses what data and objectives should be used for the next personality-model update. It proposes operations; deterministic product policy controls whether training actually occurs.

## 5. Authority separation

A critical design lesson from A.L.I.C.E. is that interpretation and authority must remain distinct.

```text
source evidence
      |
      v
FBM interpretation / proposal
      |
      v
deterministic provenance + authority rules
      |
      v
accepted evidence / hypothesis / synthetic policy
      |
      v
personality-model training substrate
```

FBM may assign scores and make recommendations. Product governance decides what may become canonical training material and what may update a live model.

## 6. Why FBM should not simply become the personality model

The builder must reason *about* a personality, compare alternatives, detect uncertainty, and generate educational material. The personality model must *instantiate* the personality.

Combining both roles into the same live weights risks self-confirming drift: the personality could generate its own interpretation, approve it, train on it, and gradually reinforce unsupported behavior.

Therefore the preferred design is shared representation where useful but distinct role boundaries.

## 7. Best post-activation descendant

The strongest current mapping is:

**FBM bootstrap backbone -> Memory Formation Model backbone + governed identity-maintenance capability.**

Reasons:

- both consume new evidence;
- both infer structured semantic meaning;
- both need entity, temporal, relationship, uncertainty, and contradiction understanding;
- both should propose rather than directly grant authority;
- both benefit from the same provenance-aware representations.

However, bootstrap-only heads such as broad synthetic personality expansion and curriculum generation should normally be dormant after activation. They can be re-enabled only inside explicit update/reconstruction workflows.

The personality model itself remains a separate learned component.

## 8. Training FBM from the A.L.I.C.E. build process

Every useful build action should be captured as a supervised or evaluable transformation where possible:

```text
input state + source references + constraints
                   |
                   v
              operation class
                   |
                   v
proposal / decision / repair / curriculum artifact
                   |
                   v
observable validation outcome + later owner feedback
```

This produces a future FBM training corpus without preserving private hidden reasoning.

Positive examples include successful evidence classification, useful E-INF proposals, good A-SYN completion, correct rejection decisions, strong hard-negative construction, and successful failure-driven repair.

Negative or contrastive examples include unsupported specificity, wrong provenance class, contradiction with source evidence, generic-assistant contamination, duplicate expansion, false historical certainty, and repairs that failed to improve fidelity.

## 9. Mainstream development doctrine

FBM itself must not become another MC10-style qualification project.

Development order:

1. capture the real A.L.I.C.E. build operations;
2. formalize recurring operations into schemas and modules;
3. automate deterministic pieces first;
4. train the smallest native formation model that can reproduce the learned operations;
5. evaluate it directly on held-out construction tasks;
6. repair concrete failures;
7. increase capability or scale only when observed failures justify it.

External models may generate alternate examples or solve difficult edge cases during development, but they are optional specialists rather than permanent dependencies.

## 10. Privacy and consumer boundary

Fable must operate only on host-authorized data. A consumer's raw evidence, derived personality substrate, embeddings, training examples, and learned personality weights are private to that instance unless the user explicitly authorizes another use.

Cross-user learning, if ever implemented, must use a separate consent and privacy design. The default architecture must not require sending one consumer's identity corpus to another user's model or to A.L.I.C.E.

## 11. Open research questions

These remain intentionally open until build evidence answers them:

- whether FBM should ultimately be one multi-head model or a small model family;
- how much of the formation backbone can be shared with Memory Formation;
- whether synthetic generation requires a decoder head or can be tool-assisted by a small local generator;
- the optimal parameter scale for offline consumer hardware;
- whether training occurs fully local, hybrid local/cloud, or via multiple hardware profiles;
- how to perform long-running background personality construction with resumable checkpoints;
- how much runtime identity maintenance should alter weights versus governed memory;
- what minimum evidence is required before a consumer personality can safely activate.

These should be answered through actual Fable/A.L.I.C.E. development rather than through a large pre-build qualification tournament.
