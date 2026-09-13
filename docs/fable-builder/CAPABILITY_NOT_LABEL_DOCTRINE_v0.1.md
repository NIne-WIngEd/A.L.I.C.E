# Fable Builder Model — Capability-Not-Label Doctrine v0.1

**Date:** 2026-09-13  
**Status:** owner-directed active working doctrine

## Core rule

The Fable Builder Model (FBM) does **not** need Elaina/A.L.I.C.E.-specific vocabulary such as `E0`, `E-INF`, `A-SYN`, or `UNKNOWN` baked into its identity or conceptual world model.

It must learn the **operations that produce those classes correctly**.

The labels are one project's schema. The capability is product-general.

A consumer Fable instance may expose different names, but the builder must still be able to distinguish:

- direct source-grounded evidence;
- evidence-constrained inference;
- unresolved/underdetermined historical truth;
- synthetic runtime behavior created to avoid a behavioral blank;
- later lived/post-activation experience;
- current self/continuity state.

For A.L.I.C.E., those product-general operations map onto the current provenance vocabulary:

| Product-general operation | A.L.I.C.E. schema output |
|---|---|
| preserve direct source evidence with exact provenance | E0 |
| derive a revisable hypothesis from supporting evidence | E-INF |
| preserve underdetermination instead of inventing history | UNKNOWN |
| create a non-historical starting behavior for an uncovered runtime situation | A-SYN |
| record later actual lived experience | A-EXP |
| maintain current self/continuity state | A-SELF / continuity structures |

## What FBM is actually learning

FBM should learn transformations such as:

```text
raw authorized data
  -> evidence extraction
  -> speaker/entity/temporal attribution
  -> direct-vs-inferred separation
  -> uncertainty representation
  -> hypothesis generation
  -> alternative generation
  -> synthetic behavioral completion where needed
  -> contradiction/novelty checks
  -> teaching substrate
  -> personality-model evaluation and repair
```

It should not memorize that a particular text string is called `E-INF`. It should learn why a statement is an inference rather than direct evidence and how to construct one without overstating support.

## Portability rule

All FBM training traces should therefore capture:

1. the input evidence state;
2. the requested transformation;
3. the externally expressible rules/constraints;
4. the generated proposal or decision;
5. validation/owner feedback;
6. the product-specific label only as an output mapping when useful.

This keeps the learned builder portable from A.L.I.C.E. to Fable consumers whose data, relationships, personality, and schemas are different.

## Privacy rule

Public FBM records store methods, schemas, hashes, counts, and sanitized examples only. Elaina-specific private payloads and future consumer payloads remain outside the public branch.

## Development implication

A.L.I.C.E. is the first full construction case for FBM.

We should not pause A.L.I.C.E. to build FBM separately. Each meaningful A.L.I.C.E. construction step should emit a compact reusable process trace. Over time those traces become the supervised/evaluation corpus for a native builder model.
