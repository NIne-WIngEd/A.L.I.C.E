# Historical Synthetic and Teaching Pipeline v0.1

**Status:** retrospective process map  
**Date:** 2026-09-13  
**Scope:** reusable method only; no private identity payloads

## Purpose

This document reconstructs the major external operations that Sol, Astra, other assistants, scripts, and the owner have already performed while creating the A.L.I.C.E. EIPM substrate. It is the initial historical training map for the Fable Builder Model (FBM).

Source architecture/process records include:

- `docs/eipm/EIPM_EXPANSION_GENERATOR_HANDOFF_SPEC_v0.1.md`
- `docs/eipm/EIPM_ASTRA_EXPANSION_DOCTRINE.md`
- `docs/eipm/EIPM_ASTRA_SATURATION_EXPANSION_PROTOCOL.md`
- `docs/eipm/EIPM_PERSONALITY_COVERAGE_HARD_RULE.md`
- `docs/eipm/EIPM_UNKNOWN_BEHAVIOR_COMPLETION_AND_CONTINUITY_RULE.md`
- `docs/eipm/EIPM_POST_CURATION_TARGETED_REGENERATION_RULE.md`
- curated-frontier policies and receipts
- owner/Sol decisions recorded on `alice-context`

This is not a claim that every historical micro-step is fully reconstructable. It captures the stable operations we know were actually used and that should become native Fable capabilities.

## Historical pipeline

### 1. Protect canonical evidence

Direct source-person evidence was treated as immutable E0 authority. Synthetic generation was not permitted to modify E0 or silently relabel generated material as historical truth.

**Future FBM capability:** provenance and authority separation.

### 2. Decompose evidence into usable behavioral units

Source material was transformed into structured semantic/behavioral units so downstream systems could reason about values, preferences, relationships, communication patterns, situations, temporal context, and other identity-relevant structure without discarding source linkage.

**Future FBM capability:** source interpretation and evidence decomposition.

### 3. Route evidence by identity-training role

Not every source fragment was equally appropriate for direct identity loss. Material was routed into roles such as direct identity supervision, conditional supervision, episodic behavior, context-only conditioning, style/voice, or exclusion from identity loss.

**Future FBM capability:** training-role router.

### 4. Generate E-INF only from evidence-constrained support

When direct historical behavior was not explicitly recorded but the evidence supported a tendency, the development models generated revisable E-INF hypotheses. Strong practice included support IDs, uncertainty, boundary conditions, counterevidence/tension, and disconfirmation conditions.

The intended operation was not `guess what the person would do`. It was `derive the narrowest useful hypothesis that the evidence can support`.

**Future FBM capability:** evidence-constrained hypothesis generation.

### 5. Preserve historical UNKNOWN

When the evidence did not justify a historical claim, UNKNOWN was retained instead of inventing certainty. UNKNOWN represented an epistemic limit rather than a desired runtime refusal to have behavior.

**Future FBM capability:** calibrated uncertainty and abstention.

### 6. Generate A-SYN for behavioral completion

When historical truth was unknown but the future agent still needed a practical starting behavior, A-SYN proposals were created as explicitly synthetic behavioral priors compatible with known patterns.

A-SYN was kept separate from both historical E0 and later lived Alice experience.

**Future FBM capability:** safe synthetic personality completion.

### 7. Build behavioral coverage maps

Development models examined meaningful combinations of personality dimension, relationship context, social context, emotional state, stakes/intensity, public/private setting, temporal state, and uncertainty.

The method explicitly rejected a giant blind Cartesian product. A combination was useful only if it could plausibly change behavior.

**Future FBM capability:** adaptive personality-space coverage planner.

### 8. Select weak and missing regions

Coverage cells were classified as covered, weak, missing, redundant, or unsupported. Generation effort was directed toward weak/missing regions rather than fixed row quotas.

**Future FBM capability:** gap selection and active curriculum planning.

### 9. Generate competing branches

For ambiguous regions, generators produced multiple plausible branches rather than one falsely authoritative answer. Candidate roles included primary branches, alternatives, historical-UNKNOWN competitors, and counterfactual competitors.

**Future FBM capability:** multi-hypothesis generation.

### 10. Control novelty

Candidates were compared against existing material and the current generation wave. Near-paraphrases and mechanical duplicates were not supposed to count as new personality coverage. New material had to add a context, boundary condition, trade-off, intensity regime, relationship condition, or genuinely distinct behavioral branch.

**Future FBM capability:** semantic novelty and redundancy control.

### 11. Curate candidates

Sol/Astra/owner review looked for:

- direct contradiction with source evidence;
- unsupported specificity;
- generic-assistant contamination;
- sycophancy;
- duplication;
- relationship-context errors;
- public/private or intensity errors;
- historical-person versus current-agent confusion;
- wrong provenance class;
- poor uncertainty calibration;
- failure to add meaningful behavioral coverage.

Candidates could be accepted, rejected, repaired, or retained as useful alternatives/hard negatives depending on their role.

**Future FBM capability:** source-grounded critic and curator.

### 12. Run global consistency and coverage review

A candidate could look plausible individually while still harming the overall personality model. Global passes checked contradictions, overrepresented traits, underrepresented tails, near-duplicates, and missing contextual combinations.

**Future FBM capability:** corpus-level identity consistency analysis.

### 13. Recompute coverage after curation

Curation could remove unsupported or redundant rows and expose new holes. The correct loop became:

```text
raw expansion
    -> curation/repair
    -> recompute coverage
    -> targeted regeneration where needed
    -> curation
    -> freeze
```

This was explicitly preferred over restarting a giant blind expansion campaign.

**Future FBM capability:** failure-driven targeted regeneration.

### 14. Preserve alternative candidates without inventing preference labels

Plausible alternatives were useful for later judgment training, but an unordered alternative was not automatically a rejected response or negative preference label. The development process learned to preserve alternatives without silently manufacturing pairwise preference truth.

**Future FBM capability:** candidate-set preservation and safe preference construction.

### 15. Compile the training substrate with provenance intact

Accepted identity material, context-only material, alternatives, uncertainty sets, structured context frames, graph/concept representations, and provenance were compiled while preserving their different roles.

**Future FBM capability:** personality-substrate compiler.

### 16. Design teaching material around the target representation

The later EIPM architecture work moved beyond merely collecting behavioral rows. Teaching was mapped to the intended personality model capabilities: semantics, temporal/causal understanding, pragmatics, social/emotional reasoning, epistemic grounding, ranking/judgment, structured ACFP-like context, persona/evidence graph alignment, uncertainty, and drift detection.

**Future FBM capability:** curriculum composer tied to the target model architecture.

### 17. Shift from external qualification to direct build-and-repair

The owner identified a repeat of the MC10 failure pattern: too much effort was being spent qualifying datasets, teachers, models, and validators before developing Alice itself.

The corrected doctrine is:

```text
build -> evaluate directly -> classify concrete failures -> repair -> repeat
```

Sol becomes the primary development-time teacher for general reasoning and curriculum generation. Canonical evidence remains the identity authority. Rayan remains final owner authority. External models become optional specialists used only when a concrete capability gap justifies them.

**Future FBM capability:** integrated teacher/evaluator with failure-driven escalation.

## What FBM must learn from the failed process, not just the successful process

Historical inefficiencies are training data too.

### Excessive qualification chains

Multiple judges, teacher tournaments, and qualification layers can consume more time and compute than the target capability. FBM should validate inside the normal build loop and escalate only when a real ambiguity or failure appears.

### Generator/model authority confusion

A strong generator can still be wrong about the person's identity. Builder intelligence must never be treated as source authority.

### Raw row-count saturation

Large candidate volume can create an illusion of coverage. Saturation must be evaluated after deduplication and curation.

### Mechanical transformations as fake evidence

Transformed examples may improve training coverage but do not become independent source evidence.

### Technical failure versus semantic failure

Infrastructure errors must not be interpreted as evidence that a semantic candidate or model is bad. The builder should separate execution validity from identity validity.

### Generic-assistant attraction

Powerful assistant models naturally produce broadly helpful/polite answers. That is not automatically the target personality. The future builder must detect this bias explicitly.

## Initial FBM training interpretation

The historical process can be converted into several kinds of future supervision:

- **procedure imitation:** reproduce the correct operation from a structured input state;
- **classification:** choose evidence class, provenance class, training role, or failure category;
- **generation:** produce E-INF, A-SYN, alternatives, boundary conditions, or curriculum examples;
- **ranking:** select the candidate most consistent with source evidence and context;
- **critique:** identify why a candidate is unsafe, unsupported, duplicated, generic, or contradictory;
- **repair:** transform a flawed candidate into a better one without changing provenance truth;
- **planning:** choose which personality gap to work on next;
- **evaluation:** diagnose a personality-model failure and propose the smallest targeted repair.

These are more useful targets for a consumer builder than training it to reproduce a developer model's hidden reasoning style.
