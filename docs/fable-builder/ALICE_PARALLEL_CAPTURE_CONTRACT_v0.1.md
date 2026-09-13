# A.L.I.C.E. <-> Fable Builder Parallel Capture Contract v0.1

**Date:** 2026-09-13  
**Status:** active working contract

## Rule

From N0 onward, A.L.I.C.E. development and FBM process capture run in parallel.

A.L.I.C.E. remains the primary product being built. FBM capture is subordinate and lightweight.

For every material A.L.I.C.E. build step, capture only the reusable construction operation needed to later teach a native Fable Builder Model.

## Required pairing

For each material Alice step:

```text
A.L.I.C.E. artifact / training / evaluation action
                    |
                    +--> compact FBM trace
```

Examples:

- N0 corpus preprocessing -> source-normalization trace
- N0 semantic curriculum generation -> curriculum-composition trace
- N0 model failure -> failure-classification + targeted-repair trace
- N1 identity substrate compilation -> evidence/provenance transformation trace
- N1 candidate behavior generation -> hypothesis/synthetic-policy generation trace
- N1 fidelity review -> personality-fidelity evaluation trace
- later memory formation work -> runtime-formation trace

## What gets stored

Store:

- operation class;
- safe input references/hashes;
- reproducible procedure;
- constraints/invariants;
- decision criteria;
- safe output references/hashes;
- observed validation result;
- owner correction when applicable;
- whether the operation later proved useful or harmful;
- which future FBM capability the example teaches.

Do not store private source payloads on the public branch.

## Granularity rule

Do not trace every prompt, token, or micro-decision.

A single coherent operation should normally create one trace row. A long operation may create a small number of rows only when it contains genuinely distinct reusable transformations.

## No-blocking rule

FBM logging must not block or delay Alice training, evaluation, or debugging.

If time is limited, build Alice first and record the compact trace immediately after the material step.

## Stage continuity

The same rule applies across:

- N0 native semantic foundation;
- N1 identity representation/learning;
- N2 judgment and behavioral preference learning;
- N3 calibration/owner refinement;
- later runtime memory formation;
- later governed identity maintenance.

The purpose is to accumulate a real longitudinal corpus showing how a personality system is built, corrected, and maintained.

## Authority rule

The FBM branch records process. It does not authorize Alice model promotion, change source authority, or supersede the active Alice build branch.
