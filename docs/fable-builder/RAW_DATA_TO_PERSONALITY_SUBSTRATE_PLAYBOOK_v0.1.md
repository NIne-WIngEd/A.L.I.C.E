# Raw Data -> Personality Substrate Playbook v0.1

**Date:** 2026-09-13  
**Status:** retrospective process capture from A.L.I.C.E. construction

## Purpose

Capture the reusable method that Sol, Astra, scripts, and the owner used to turn raw personal/source data into a personality-model substrate. This is a process document for the future Fable Builder Model (FBM). It is not a copy of Elaina's private data and does not grant historical authority to synthetic material.

## 1. Ingest and normalize source material

Start from host-authorized raw material. Preserve the original source reference before interpretation.

For each item, retain where available:

- source object/file identifier;
- content hash;
- author/speaker;
- recipient/audience;
- timestamp or temporal range;
- modality;
- relationship context;
- conversation/thread context;
- extraction confidence;
- whether wording is exact, reconstructed, or summarized.

Do not destroy the raw source after extracting structured material. Structured outputs remain derived artifacts.

## 2. Decompose source into evidence units

Break source material into the smallest useful semantic/behavioral units without atomizing it so far that context is lost.

A useful unit should make it possible to answer:

- what was directly observed or stated;
- who it concerns;
- when it applied;
- the relevant relationship/social context;
- whether it reflects a fact, preference, value, judgment, emotion, behavior, boundary, habit, or interaction pattern;
- what original source supports it.

For A.L.I.C.E., direct owner-attested/source-supported units became the canonical direct-evidence layer. In a generic Fable product, the schema name may differ.

## 3. Separate direct evidence from interpretation

Never silently promote interpretation into source truth.

For every candidate statement, ask whether the source directly establishes it. If yes, preserve it as direct evidence with provenance. If the statement generalizes beyond the observed event, predicts behavior, or interprets motivation, treat it as a hypothesis.

A hypothesis should carry:

- supporting direct-evidence IDs;
- counterevidence/tensions;
- conditions under which it applies;
- confidence/support strength;
- reasons for uncertainty;
- disconfirmation conditions;
- whether it is about historical behavior or a runtime policy.

## 4. Preserve underdetermination

When evidence does not support a historical conclusion, explicitly preserve uncertainty rather than filling the gap with a convenient story.

The builder should be able to say, in effect:

> historical behavior is not determined by the available evidence.

This is a positive epistemic action, not a failure.

## 5. Generate evidence-constrained hypotheses

For behaviorally important areas that are not directly stated, derive multiple plausible conditional hypotheses from the smallest relevant evidence packet.

The successful A.L.I.C.E. method was:

1. select the relevant evidence units;
2. identify the personality dimension and context actually constrained by those units;
3. generate more than one plausible branch when evidence permits multiple interpretations;
4. attach support, counterevidence, uncertainty, boundary conditions, and disconfirmation conditions;
5. retain an explicit unresolved competitor when the historical answer remains unknown;
6. reject hypotheses that require unsupported specificity.

Do not convert model confidence into probability that a historical claim is true.

## 6. Build a behavioral coverage map

Model personality as context-dependent behavior rather than a flat trait list.

Coverage dimensions used during A.L.I.C.E. enrichment included combinations of:

- values and priorities;
- trust and loyalty;
- boundaries/privacy;
- affection and closeness;
- humor/teasing;
- criticism/disagreement;
- conflict/reconciliation;
- embarrassment/shame/vulnerability;
- ambition/risk/failure;
- injustice/authority;
- grief/fear/stress;
- jealousy/status;
- strangers/friends/close relationships;
- relationship-specific context;
- public vs private setting;
- stakes/intensity;
- temporal state;
- uncertainty state.

Do not create a blind Cartesian product. Add a cell only when the combination could plausibly change behavior.

Mark cells as covered, weak, missing, redundant, or unsupported.

## 7. Generate synthetic behavioral completion

A runtime personality cannot leave every historically unknown situation behaviorally blank.

For a missing but useful runtime behavior, create a synthetic starting policy that is compatible with known evidence while remaining explicitly non-historical.

A good synthetic behavioral proposal should include:

- scenario/context;
- behavioral response or policy;
- source-compatible rationale;
- relationship/social/emotional conditions;
- boundary conditions;
- uncertainty;
- competing branches when appropriate;
- explicit statement that the proposal is not historical fact and not lived memory.

Prefer conditional policies to absolute traits.

## 8. Generate alternatives, not just one answer

Where the evidence does not uniquely determine behavior, create competing branches.

Useful alternatives include:

- a likely evidence-consistent branch;
- another plausible branch with different tradeoffs;
- a conservative/uncertain branch;
- an explicit unresolved historical branch;
- a counterfactual branch that is plausible but inconsistent with evidence and may later serve as a hard negative.

Do not automatically treat every unselected alternative as a negative. Some are simply unordered competitors.

## 9. Novelty and duplicate control

Before keeping a generated proposal, compare it with the existing substrate and proposals from the same wave.

A new row should add at least one of:

- a new context;
- a new boundary condition;
- a new relationship condition;
- a new intensity/stakes regime;
- a genuine value tradeoff;
- a distinct behavioral branch.

Mechanical paraphrases, templated rewrites, and transformations of the same evidence should not be counted as independent evidence.

## 10. Curate against source authority

Generated material is proposal material until checked.

Curation checks used during A.L.I.C.E. included:

- support from direct evidence and already accepted inference;
- contradiction with direct evidence;
- unsupported specificity;
- historical-vs-runtime confusion;
- relationship-context mistakes;
- public/private or intensity mistakes;
- generic-assistant/sycophancy contamination;
- duplication;
- uncertainty calibration;
- whether the candidate genuinely expands useful behavior coverage.

Repair a promising candidate when the flaw is local. Reject it when the premise itself is unsupported.

## 11. Run global consistency after local curation

Individually plausible rows can still create a bad personality globally.

Check the accepted substrate for:

- contradictory tendencies that lack context explaining the difference;
- overrepresentation of one trait or style;
- missing behavioral tails;
- near-duplicate clusters;
- relationship-specific leakage;
- historical certainty inflation;
- generic helpful-assistant drift;
- missing negative/alternative cases;
- unsupported source dependencies.

## 12. Recompute coverage and regenerate only real gaps

After curation, rebuild the coverage map.

If curation removed or weakened a meaningful region, generate specifically for that region. Do not restart a giant blind expansion campaign.

The productive loop is:

```text
raw source
 -> evidence units
 -> hypotheses + unresolved states
 -> coverage map
 -> synthetic behavioral completion
 -> curation/repair
 -> global consistency
 -> recompute gaps
 -> targeted regeneration
 -> training substrate
```

## 13. Compile teaching material

Once provenance and behavioral material are stable enough, turn it into personality-model teaching tasks rather than merely a bag of text rows.

Examples:

- evidence -> correct provenance/epistemic category;
- scenario + context -> rank candidate behaviors;
- evidence + hypothesis -> support/contradiction judgment;
- scenario -> choose between characteristic, generic-assistant, sycophantic, exaggerated, or unsupported responses;
- relationship state -> behavior adjustment;
- source text -> structured personality frame;
- graph/evidence nodes -> relationship/value inference;
- ambiguous case -> calibrated uncertainty/abstention;
- failure case -> targeted repair examples.

## 14. Feedback closes the loop

After the personality model is trained, observed failures become new builder supervision.

For each meaningful failure:

1. classify the failure;
2. locate whether it comes from semantic understanding, evidence representation, personality judgment, relationship state, calibration, or capacity;
3. generate the smallest targeted teaching set that addresses it;
4. retrain/update;
5. re-run the failed case and regression set;
6. log whether the repair helped, harmed, or had no effect.

This `failure -> diagnosis -> teaching -> observed outcome` chain is especially valuable future FBM training data.

## 15. What the A.L.I.C.E. process taught us

The main reusable lessons are:

- provenance is part of the target, not bookkeeping;
- unresolved history must remain unresolved;
- synthetic behavior is allowed without pretending it happened historically;
- personality coverage should be conditional and relationship-aware;
- raw generated row count is not evidence count or coverage quality;
- generated alternatives should not be silently converted to negatives;
- curation can expose new gaps, so regeneration should be targeted;
- generic assistant behavior is a distinct contamination mode that must be tested;
- deterministic authority rules should remain outside learned proposal generation;
- building and validating should be one feedback loop rather than separate qualification projects.

## Consumer portability

The public Fable product does not need to expose A.L.I.C.E.'s exact labels. It needs this method.

A consumer-specific builder should reproduce the same epistemic distinctions and formation process using that consumer's own authorized evidence and personality target.
