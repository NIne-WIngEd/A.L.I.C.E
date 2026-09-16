# N0 Latent Value-Contrast Diagnostic Contract Audit — 2026-09-16

## Scope

This audit records the failure of Magnolia job 575683 and the corrective design for the latent-pool value-contrast diagnostic. It does not change the frozen latent challenge v0.2 result, ratify the latent pool, authorize training, or authorize scaling.

## Historical facts

- Frozen latent challenge v0.2 remains `FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION`.
- Step 360 remains the unchanged candidate with SHA-256 `503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`.
- Job 575683 failed before producing a diagnostic result.
- Its error was `current value 'dense' not found in target for N0V02-LATCHAL2-SUPERSESSION_HISTORICAL-01`.
- No diagnostic conclusion about the latent model can be drawn from job 575683.

## Root cause

The v0.1 diagnostic reconstructed answer semantics inside the GPU evaluator. It iterated over all frozen challenge rows and assumed the target summary must contain the domain's `current` value. That assumption is false for query-conditioned historical families. For example, `supersession_historical` intentionally asks for the previous value.

The immediate exception was therefore only a symptom. The deeper contract error was that semantic meaning was implicit and reconstructed from generator internals instead of being materialized as first-class evaluation data before GPU execution.

## Why a hotfix is forbidden

The following would be unsafe repairs:

- skip only `supersession_historical` after seeing the error;
- swap current/previous based on a family-name suffix;
- catch the exception and continue;
- add family-specific string replacements inside the GPU evaluator;
- treat any successful rerun of that patched script as evidence about capacity.

Those approaches would keep the same architectural flaw: the evaluator would still own semantic interpretation and could silently mis-score future temporal, uncertainty, or authority-reversal cases.

## Corrected contract

The v0.2 diagnostic architecture separates semantic compilation from measurement.

### CPU semantic-contract compiler

`scripts/eipm/n0/compile_n0_v02_latent_pool_value_contrast_contract_v0_2.py`

The compiler:

- selects only rows explicitly marked `counterfactual_required=true`;
- requires exact agreement with the four eligible families and their required source views;
- materializes target text, foil text, target value, foil value, query-answer role, and foil grounding before GPU execution;
- does not apply a global current-value assumption to historical or uncertainty rows;
- binds the sidecar to the frozen v0.2 challenge hash;
- has no ratification or training effect.

### GPU diagnostic consumer

`scripts/eipm/n0/diagnose_n0_v02_latent_pool_counterfactual_value_contrast_v0_2.py`

The GPU evaluator consumes the compiled sidecar. It does not reconstruct family semantics or domain values. It reports a causal chain:

1. target-versus-foil semantic separability;
2. required-source target-versus-foil margin;
3. normal latent-set target-versus-foil margin;
4. counterfactual latent-set margin after the required source is removed before parent encoding and fusion;
5. pooled-state target-versus-foil margin and its counterfactual change.

The set-level contrast is permutation-free. No fixed personality-slot meaning is introduced.

## Interpretation rule

A capacity or model-repair conclusion requires a valid measurement chain. At minimum:

- the target and foil must be distinguishable in the semantic reference space;
- the required source must carry a positive target-over-foil distinction;
- the normal latent representation must preserve a useful distinction;
- pre-fusion removal of the required source must be evaluated against that same distinction.

If those prerequisites fail, the diagnostic is not sensitive enough to support a model conclusion.

If they hold and the latent representation still fails to preserve or causally use the distinction, that is real evidence for model repair. Only after the failure mode is localized should capacity scaling be considered.

## Governance

- Frozen challenge v0.2 remains an immutable FAIL.
- Job 575683 remains a failed diagnostic execution with no model result.
- No challenge row is authorized for training.
- No private identity data or private gradient is involved.
- No latent weights are changed by this audit.
- Scaling remains unauthorized until valid evidence isolates a capacity bottleneck.
