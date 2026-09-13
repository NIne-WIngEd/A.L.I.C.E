# Fable Personality Builder Model Workstream

**Branch:** `fable-builder-model`  
**Status:** active architecture and process-capture workstream  
**Created:** 2026-09-13

## Purpose

Fable Sleight cannot depend on a hosted Sol, Astra, or another manually selected frontier assistant to build each consumer personality. A consumer installation needs a built-in, product-controlled formation capability that can take host-authorized personal data and perform the same classes of work currently being done manually during A.L.I.C.E. construction.

This branch captures that capability from the process that is actually building A.L.I.C.E. rather than inventing it later from memory.

The working name is **Fable Builder Model (FBM)**. The name is provisional. FBM is the host-neutral model/capability that reconstructs and bootstraps a new personality from evidence. It is separate from the finished consumer personality model.

## Core doctrine

The FBM should learn the *construction process*, not copy any one developer model's personality.

Its responsibilities include:

- ingesting authorized source material;
- extracting evidence and structured behavioral observations;
- preserving provenance and epistemic class;
- proposing evidence-constrained E-INF hypotheses;
- preserving UNKNOWN when evidence is insufficient;
- generating A-SYN behavioral priors when runtime behavior needs initialization despite historical uncertainty;
- mapping personality-space coverage and finding meaningful gaps;
- generating competing branches rather than a single overconfident answer;
- detecting duplicate, mechanically-derived, contradictory, generic-assistant, sycophantic, or unsupported candidates;
- curating and repairing candidate substrate;
- generating teaching curricula and hard cases for the personality model;
- evaluating the resulting personality model against source evidence and owner/host feedback;
- iterating from observed failures instead of running large external qualification tournaments.

The consumer's source evidence remains the authority about that consumer-derived personality. FBM is a builder and interpreter, not the historical authority.

## Relationship to A.L.I.C.E.

A.L.I.C.E. is the first full development case. The Elaina-derived EIPM process provides the first high-fidelity trace of how a personality is reconstructed under explicit provenance classes, uncertainty, synthetic completion, coverage expansion, curation, and fidelity evaluation.

Private Elaina or Rayan payloads MUST NOT be copied into this public branch. This branch stores reusable methods, schemas, process traces, decision rules, failure lessons, capability mappings, and sanitized aggregate receipts only.

## Runtime evolution hypothesis

The closest existing A.L.I.C.E. component to the post-activation descendant of FBM is the **Memory Formation Model**, because both interpret new evidence and produce structured semantic proposals. However, FBM has broader bootstrap duties such as identity reconstruction, synthetic behavioral completion, curriculum generation, and personality-model evaluation.

Therefore the current architecture does **not** collapse FBM directly into the runtime Memory Formation Model. The working design is:

```text
pre-activation
source data -> FBM bootstrap mode -> personality substrate -> personality model

post-activation
FBM shared formation backbone
        |-> Memory Formation runtime head
        |-> optional identity-maintenance / reflection head
        |-> optional curriculum / repair mode, invoked only under governed update workflows
```

This allows the expensive formation capability to remain useful after activation without giving ordinary runtime memory ingestion unrestricted authority to rewrite core personality.

## Process capture rule

From this point forward, every material Sol/Astra/other-model step used to build or teach the personality model should produce a reusable process trace on this branch when technically possible.

A process trace records reproducible external behavior: inputs, operation, constraints, decision criteria, output type, validation, failure lessons, and the FBM capability it corresponds to. It does **not** require or preserve private hidden chain-of-thought. Stable operating recipes and observable decisions are the target artifact.

See:

- `FORMATION_MODEL_ARCHITECTURE_v0.1.md`
- `TRACE_SCHEMA_v0.1.md`
- `history/HISTORICAL_SYNTHETIC_AND_TEACHING_PIPELINE_v0.1.md`
- `traces/FBM_TRACE_20260913.jsonl`

## Mainstreaming rule

FBM development follows the corrected A.L.I.C.E. doctrine:

> Build the capability directly. Validate inside the build loop. Use external specialists only when a concrete failure or capability gap justifies them.

External models are optional tools, not standing authorities or mandatory committees.
