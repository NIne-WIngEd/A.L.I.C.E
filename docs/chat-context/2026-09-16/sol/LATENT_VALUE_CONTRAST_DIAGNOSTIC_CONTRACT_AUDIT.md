# Latent Value-Contrast Diagnostic Contract Audit

Date: 2026-09-16

## Trigger

Magnolia job 575683 failed before producing a diagnostic result with:

`current value 'dense' not found in target for N0V02-LATCHAL2-SUPERSESSION_HISTORICAL-01`

The owner explicitly warned against entering a bug-hotfix-wrapper loop and requested a bigger-picture diagnosis.

## Root cause

The v0.1 diagnostic constructed foils for all 160 frozen challenge rows and globally assumed every target contained the domain's current value. This is false for query-conditioned historical families. The GPU evaluator was incorrectly reconstructing semantic meaning from row IDs, family identity, and generator internals.

This is a diagnostic-contract design failure, not a latent-model failure and not a capacity signal. Job 575683 produced no diagnostic output and no new model evidence.

## Historical state preserved

- Frozen challenge v0.2 remains an immutable FAIL.
- It is not reclassified as pass.
- Step 360 remains unchanged and unratified.
- No challenge rows are used for training.
- No new gradient is authorized.
- Scaling remains unauthorized until a valid causal measurement chain exists.

## Corrective architecture

Build branch now uses a v0.2 diagnostic contract:

- CPU compiler materializes target/foil/query semantics only for the 32 explicitly counterfactual-required rows.
- Historical and uncertainty rows are not silently forced into a current-value contract.
- Regression tests verify exact 32-row coverage and historical-row exclusion.
- GPU evaluator consumes the sidecar and does not know generator-family semantics.
- It measures target/foil semantic separability, required-source margin, permutation-free latent-set margin, pooled margin, and changes after pre-fusion ablation.
- CPU preparation hash-locks the sidecar, frozen challenge/result, candidate, and diagnostic before P100 use.

Authoritative build state: `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.4.json`.
Audit: `docs/eipm/N0_LATENT_VALUE_CONTRAST_DIAGNOSTIC_CONTRACT_AUDIT_2026-09-16.md`.

## Reusable rule

Evaluation semantics must be explicit data. Evaluators may measure semantics but must not invent or reconstruct them from naming conventions, row IDs, or global assumptions. Temporal/current/historical/uncertainty semantics are query-conditioned.
