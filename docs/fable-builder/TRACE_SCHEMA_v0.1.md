# Fable Builder Process Trace Schema v0.1

**Status:** active logging contract  
**Date:** 2026-09-13

## Purpose

This schema captures the external, reproducible work performed by Sol, Astra, other assistants, scripts, and the owner while building A.L.I.C.E. The goal is to turn the real construction process into future training/evaluation material for the Fable Builder Model (FBM).

The trace is an operational audit record. It is **not** a chain-of-thought archive. Hidden model reasoning is neither required nor treated as a product dependency. What matters is the observable transformation, the rules used, the evidence consulted, the output, and whether the output later proved useful.

## Required JSONL fields

Each material build step should be logged as one JSON object with these fields:

- `trace_id`: unique stable ID.
- `timestamp_utc`: ISO-8601 timestamp when known.
- `actor`: `owner`, `sol`, `astra`, `other_model`, `script`, `human_reviewer`, or another explicit actor label.
- `actor_instance`: optional model/version or tool identity when known.
- `project_stage`: e.g. `EIPM_EXPANSION`, `CURATION`, `N0_TEACHING`, `N1_IDENTITY`, `RUNTIME_FORMATION`.
- `operation_type`: normalized operation class.
- `input_classes`: logical input types used by the operation.
- `input_refs`: safe file/record/hash references where available. Do not embed private payloads in this public branch.
- `goal`: concise description of the desired transformation.
- `procedure`: short reproducible operating recipe.
- `constraints`: invariants the actor had to preserve.
- `decision_criteria`: externally expressible checks used to choose, reject, classify, or repair an output.
- `output_classes`: logical outputs created.
- `output_refs`: safe artifact references or hashes where available.
- `authority`: what this step was allowed to do and explicitly not allowed to do.
- `validation`: observable checks applied after the step.
- `result`: `success`, `partial`, `failed`, `superseded`, or `historical`.
- `failure_or_lesson`: reusable lesson, especially when a prior method was wasteful or wrong.
- `fbm_capabilities`: one or more future FBM modules this step teaches.
- `training_value`: `positive_example`, `negative_example`, `contrastive_pair`, `procedure_only`, `evaluation_case`, or combinations.
- `privacy`: classification such as `PUBLIC_METHOD_ONLY`, `PRIVATE_INPUT_REFERENCED`, or `SANITIZED_AGGREGATE`.
- `notes`: optional short context.

## Normalized operation types

Initial operation vocabulary:

- `SOURCE_INGEST`
- `EVIDENCE_DECOMPOSITION`
- `PROVENANCE_CLASSIFICATION`
- `E0_ROUTING`
- `EINF_HYPOTHESIS_GENERATION`
- `UNKNOWN_PRESERVATION`
- `ASYN_BEHAVIORAL_SYNTHESIS`
- `COVERAGE_MAPPING`
- `GAP_SELECTION`
- `COMPETING_BRANCH_GENERATION`
- `NOVELTY_DEDUP`
- `CONTRADICTION_CHECK`
- `GENERIC_ASSISTANT_CONTAMINATION_CHECK`
- `CURATION_ACCEPT_REJECT_REPAIR`
- `GLOBAL_CONSISTENCY_PASS`
- `TARGETED_REGENERATION`
- `TRAINING_SUBSTRATE_COMPILATION`
- `CURRICULUM_GENERATION`
- `HARD_NEGATIVE_GENERATION`
- `MODEL_TEACHING`
- `FIDELITY_EVALUATION`
- `FAILURE_CLASSIFICATION`
- `TARGETED_REPAIR`
- `OWNER_REVIEW`
- `RUNTIME_MEMORY_FORMATION`
- `IDENTITY_MAINTENANCE_PROPOSAL`

The vocabulary may grow when a genuinely new operation appears. Do not create a new label merely for wording differences.

## Procedure-writing rule

The `procedure` field should describe a process that another implementation could reproduce. Example:

> Select the smallest relevant source evidence packet. Identify whether it directly supports a behavioral tendency. Generate multiple conditional hypotheses. Attach support IDs, uncertainty, boundary conditions, and an explicit UNKNOWN branch when the evidence does not justify historical certainty.

Do not write vague labels such as `thought carefully` or attempt to reconstruct hidden private model reasoning.

## Feedback linkage

When a later step validates, rejects, repairs, or supersedes an earlier trace, add the earlier `trace_id` in `input_refs` or `notes`. This creates a causal training history:

```text
generation trace
      -> curation trace
      -> training trace
      -> evaluation trace
      -> repair trace
```

This lineage is more useful for FBM learning than a flat collection of final accepted rows.

## Private data rule

The public branch may record private artifact hashes, logical identifiers, counts, schemas, or sanitized aggregate outcomes. It must not contain raw private identity text, private source-person records, private consumer data, credentials, or secret model artifacts.

A future private Fable training store may link the process trace to authorized payloads by content-addressed identifiers.

## Minimal capture rule

Logging must not become another bureaucracy. A material step should normally require only one compact trace row. Deterministic scripts can emit these automatically. Human/model steps can be logged at the operation level rather than every token or micro-decision.

The trace system succeeds only if it captures development without slowing development materially.
