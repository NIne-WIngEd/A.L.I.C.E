# N0 Query-Semantics Layerwise Audit Handoff

**Date:** 2026-09-18  
**Status:** 575801 completed the first architecture audit; final token states are stronger than pooled views but causes/supports remain at chance; gradient work remains frozen pending one no-gradient layerwise audit

## Current frontier

`alice-eipm-v1-query-semantics-architecture-audit @ 900db6f09b38a5ae6328d897b520e59335201728`

Stable build remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

## 575801 architecture audit

Execution:

- job: `575801`
- state: `COMPLETED`
- exit: `0:0`
- node: `gpu001`
- elapsed: `00:00:49`

Result SHA-256:

`8bb4aa0f249b763342d78f5bf0f2f960a0bd73fafa67d18f8e0e7342f62f24b6`

No gradient was performed. No model parameter was mutated. No heldout row was used.

Representation role accuracy:

- raw mean pool: `0.5625`
- trained semantic projection: `0.5625`
- token late interaction: `0.6875`

Mean opposite-role pair distance:

- raw mean pool: `0.05102736999591192`
- trained semantic projection: `0.013605944812297821`
- token late interaction: `0.11760250230630238`

Token-level per-relation role accuracy:

- corrects: `0.875`
- supersedes: `0.75`
- temporal_successor: `0.75`
- derived_from: `0.75`
- causes: `0.5`
- supports: `0.5`

## Architecture interpretation

This is a mixed localization.

The final-layer token representation preserves materially more directional semantic signal than either pooled representation.

Therefore the graph's current one-vector query interface discards useful information for several relation families.

However token-level final-layer states still leave causes and supports at chance.

Therefore it is not yet valid to train a token-level graph adapter.

The trained semantic projection is also not a solution: it matches raw-pool accuracy and compresses source-target pair distance further.

## External research update

Layerwise relation probing literature provides a plausible next localization step:

- relation information can peak at intermediate layers rather than the final layer;
- directional relation concepts can have asymmetric representation profiles;
- selected earlier/intermediate activations can expose relational concepts more strongly than generic final representations.

No external source is treated as authority over ALICE. It motivates a diagnostic only.

## Next diagnostic

Research note:

`docs/research/eipm-n0-query-semantics-layerwise-audit-v0.1.md`

Script:

`scripts/eipm/n0/audit_n0_v02_query_semantic_layers_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_query_semantics_layerwise_audit_v0_1.sh`

Magnolia job:

`scripts/eipm/n0/magnolia_n0_v02_query_semantics_layerwise_audit_v0_1.sbatch`

The audit inspects every hidden-state output from the frozen 16-layer semantic model.

For each layer:

- token late-interaction SOURCE/TARGET role accuracy;
- mean-pool SOURCE/TARGET role accuracy;
- opposite-role pair distance;
- causes token accuracy;
- supports token accuracy.

No optimizer, gradient, checkpoint update, or model mutation is allowed.

## Decision after layerwise audit

If an earlier layer materially restores causes/supports:
- directional semantics exist upstream;
- design the future language↔relation interface to tap informative token layer(s);
- do not retrain the graph selector on final pooled queries.

If no layer restores causes/supports:
- stop graph repair;
- reopen public semantic representation training/objectives and directional relation coverage.

No training is authorized before interpretation of the layerwise audit.

## Authoritative state

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.15.json`

## CI

Workflow:

`N0 Query Semantics Layerwise Audit Contract Check`

Run:

`35391604417`

Result:

`SUCCESS`

## Anti-loop rules

- gradient work remains frozen;
- no QRR v0.4;
- no router/loss/step-count patch;
- no semantic-projection swap;
- no token-graph training before layerwise localization;
- no heldout opening;
- no frozen-challenge rerun;
- no threshold change;
- no scale authorization.
