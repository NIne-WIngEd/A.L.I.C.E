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
        |-> required personal-development / reflection capability
        |      |-> evidence-linked user/host model revision
        |      |-> assistant-self and relationship-state revision
        |      |-> native judgment integration
        |      |-> outcome-based revision
        |-> optional curriculum / repair mode, invoked only under governed update workflows
```

This allows the expensive formation capability to remain useful after activation without giving ordinary runtime memory ingestion unrestricted authority to rewrite core personality.

Continuing personal development is a **required destination capability**, not optional personality decoration. A Fable instance must keep its model of the user separate from its own developing self. It may revise both from authorized experience and outcomes under provenance and authority controls. A user's declaration of an ideal or "true" self is evidence, not identity authority.

The capability is not complete merely because projections can be stored. The eventual runtime must demonstrate a causal loop from experience -> subject-bound user/self/relationship state -> native judgment -> outcome -> governed revision -> changed later judgment. A fixed prompt or generic downstream-model refusal does not count as evidence that Fable learned the judgment.

## Process capture rule

From this point forward, every material Sol/Astra/other-model step used to build or teach the personality model should produce a reusable process trace on this branch when technically possible.

A process trace records reproducible external behavior: inputs, operation, constraints, decision criteria, output type, validation, failure lessons, and the FBM capability it corresponds to. It does **not** require or preserve private hidden chain-of-thought. Stable operating recipes and observable decisions are the target artifact.

See:

- `FORMATION_MODEL_ARCHITECTURE_v0.1.md`
- `CONVERSATION_AND_SOURCE_SELECTION_v0.1.md` — first-release builder scope, selectable outside-data sources, and conversation handoff
- `TRACE_SCHEMA_v0.1.md`
- `history/HISTORICAL_SYNTHETIC_AND_TEACHING_PIPELINE_v0.1.md`
- `traces/FBM_TRACE_20260913.jsonl`

## Latest A.L.I.C.E. process capture — 2026-09-22

The active N0 full-envelope rebuild has continued to produce reusable builder lessons after earlier green static receipts. The latest captured boundary is A.L.I.C.E. branch `alice-eipm-v1-n0-full-envelope-foundation-build-v1@ac0defbc67f260d77d0d7cd664209e247f457867`.

The new traces record six high-value construction lessons:

- a candidate-answer path can silently become a second judgment policy unless candidate-specific scoring is forced to interact with the governed latent state;
- Binder support must represent query-relevant structural support rather than relation-key equality, while remaining neutral to reliability/recency/temporal/provenance arbitration that belongs downstream;
- shortcut auditing must be executable before gradient and should include matched context-swaps where the same candidate set changes target only because evidence changes;
- program completion is a joint probability event, not a product of marginal start/non-truncation quantities; incomplete branches must not contaminate completed relational state;
- runtime qualification must exercise every text-bearing semantic surface under the same long-context policy, and fixture sizes must remain operating points rather than hidden product ceilings;
- the successor FINAL package must be built, independently audited, hash-frozen, and bound to the registered full-envelope system before gradient while keeping candidate results unopened;
- architecture support for long context is not training coverage: every governed text surface needs explicit TRAIN/DEV examples that actually cross the native-window operating point while preserving the same causal targets;
- curriculum builders must return row-owned runtime schema objects. Shared aliases into canonical schema banks can let one downstream intervention silently mutate later TRAIN/DEV/FINAL materialization;
- stage ownership is semantic, not cosmetic: J1 long semantic supervision must come from rows that actually own semantic-operator program/factor/token-evidence targets rather than borrowing superficially similar full-fabric rows;
- staged optimization can keep one full topology instantiated while refusing meaningless losses through random downstream arbiters; activating a family and all of its causal owners together is different from shrinking the architecture;
- a public-mixture manifest should hash-bind every optimizer-facing lane separately, including supplemental lanes, so a declared capability cannot disappear between curriculum construction and the trainer;
- qualification fixtures must preserve real objective prerequisites. For example, a contrastive teacher objective needs enough groups to exercise the actual loss instead of weakening the loss for a smoke test;
- GPU memory qualification must cover structurally different expensive cases such as high runtime cardinality and long semantic context, rather than treating one convenient row as representative.

These are process seeds for FBM. They do not grant Fable/A.L.I.C.E. model authority and do not contain private identity payloads.

## Mainstreaming rule

FBM development follows the corrected A.L.I.C.E. doctrine:

> Build the capability directly. Validate inside the build loop. Use external specialists only when a concrete failure or capability gap justifies them.

External models are optional tools, not standing authorities or mandatory committees.
