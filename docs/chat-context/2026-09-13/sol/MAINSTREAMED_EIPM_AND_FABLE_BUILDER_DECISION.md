# Mainstreamed EIPM + Fable Builder Decision

**Date:** 2026-09-13

## Owner correction

The owner identified that the N0 path was beginning to repeat the MC10 failure pattern: too much time and resource spent on qualification, validation chains, multiple teacher selection, and external judging before actual A.L.I.C.E. capability was built.

The active doctrine is now:

> Build A.L.I.C.E. directly. Validate inside the build loop. Use external specialists only when a concrete failure or ambiguity justifies them.

Sol is the primary development-time teacher/curriculum generator. Canonical Elaina evidence remains the authority on identity. Rayan remains final owner authority.

The earlier staged N0 R1/R2/R3/R4/R5/R6/R7 sequence is no longer the required execution path. The streamlined sequence is N0A native foundation, N0B preprocessing/curriculum, N0C first native checkpoint, N0D direct evaluation/repair, then N1 private identity learning when authorized.

A new doctrine was committed on `alice-eipm-v1-build`:

- `docs/eipm/EIPM_MAINSTREAMED_BUILD_AND_TEACHING_DOCTRINE_2026-09-13.md`

## Fable product addition

The owner introduced a new product requirement for Fable Sleight: a consumer installation cannot depend on hosted Sol/Astra access to reconstruct the user's personality model. Fable therefore needs a built-in host-neutral model/capability that can perform the construction work currently done manually during A.L.I.C.E. development.

A dedicated branch was created:

- `fable-builder-model`

Working capability name:

- **Fable Builder Model (FBM)**

Initial committed artifacts:

- `docs/fable-builder/README.md`
- `docs/fable-builder/FORMATION_MODEL_ARCHITECTURE_v0.1.md`
- `docs/fable-builder/TRACE_SCHEMA_v0.1.md`
- `docs/fable-builder/history/HISTORICAL_SYNTHETIC_AND_TEACHING_PIPELINE_v0.1.md`
- `docs/fable-builder/traces/FBM_TRACE_20260913.jsonl`

## FBM responsibilities

FBM is intended to learn the reusable construction process rather than inherit a developer assistant's personality. Its capability map includes source interpretation, provenance/epistemic classification, E-INF hypothesis generation, UNKNOWN preservation, A-SYN behavioral completion, coverage planning, competing-branch generation, novelty/consistency criticism, curation, curriculum creation, personality-model evaluation, and failure-driven repair.

From now on, material Sol/Astra/other-model operations used to build or teach the personality model should be captured as lightweight process traces on `fable-builder-model` when technically possible. The trace stores observable inputs, operation, constraints, criteria, output, validation, lesson, and future FBM capability mapping. It does not require hidden chain-of-thought and must not leak private source payloads into the public repo.

## Post-activation evolution hypothesis

The closest current A.L.I.C.E. descendant is the **Memory Formation Model**, because both consume evidence and produce structured semantic proposals. The current working design is a shared formation backbone with separate modes/heads:

- pre-activation FBM bootstrap formation;
- post-activation Memory Formation runtime head;
- separately governed identity-maintenance/reflection head;
- bootstrap-only synthetic expansion/curriculum modes dormant during ordinary runtime.

The finished personality model remains a distinct learned component. This separation prevents a live personality from generating, approving, and training on its own unsupported identity changes.

## Privacy boundary

No raw Elaina/Rayan/private consumer payload belongs on the public FBM branch. Store reusable methods, schemas, hashes, safe references, aggregate receipts, and process traces only.
