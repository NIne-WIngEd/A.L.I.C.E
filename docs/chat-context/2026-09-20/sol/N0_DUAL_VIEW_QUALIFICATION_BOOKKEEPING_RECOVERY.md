# N0 Dual-View Qualification Harness Recovery — One CPU Requalification Ready

**Date:** 2026-09-20

## Attempt 1

Source revision:
`b736f55363254eb887664f33c35ac1a144a0b08e`

Execution:
- CPU / no gradient
- no optimizer
- no GPU
- no TEST
- no training authorization

Terminal failure:

`direction-invariance pair coverage drift: 144`

This is **not model evidence**.

## Root cause

The qualifier expected 72 unique direction-reversal comparisons:

- 6 relations
- 6 DEV quads per relation
- 2 query roles per quad

Total:

`6 × 6 × 2 = 72`

The implementation instead iterated all 144 DEV rows. For every row it looked up the same A/B pair for that row's `(quad_id, query_role)`. Therefore each unique reversal pair was counted twice:

- once when the loop visited direction A;
- once when the loop visited direction B.

The scientific expectation of 72 was correct.

The harness counting logic was wrong.

## Correction

The qualifier now first constructs the unique set:

`(quad_id, query_role)`

from DEV rows.

It requires exactly 72 unique keys.

It then performs one A-versus-B comparison per key.

The following are unchanged:

- architecture;
- audit lineage;
- DEV data;
- binding thresholds;
- permutation criterion;
- direction-invariance tolerance;
- parent exactness criterion;
- no-gradient boundary.

Do **not** change the expectation to 144.

## Preservation

The failed attempt-1 directory is preserved:

`query-edge-dual-view-late-interaction-v0.1/qualification/`

The corrected qualification writes to:

`query-edge-dual-view-late-interaction-v0.1/qualification-v0.2/`

Do not delete or overwrite attempt 1.

## Current state

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.34.json`

Current branch:

`alice-eipm-v1-dual-view-late-interaction-binding`

Exact recovery frontier:

`edcee7dd7a53bac158793b805a29cfaa0c1b11b0`

Static architecture contract after qualifier fix:

`35491405602 — SUCCESS`

Qualification-recovery contract:

`35491442003 — SUCCESS`

Recovery CI verifies:

- unique direction pairs are deduplicated;
- 72 remains the expected unique count;
- architecture/scientific criteria are unchanged;
- attempt 1 remains non-model evidence;
- attempt 1 is preserved;
- corrected output uses `qualification-v0.2`;
- no destructive cleanup;
- no GPU/training submission.

## Next action

Run exactly one corrected CPU/no-gradient requalification using:

`scripts/eipm/n0/run_n0_v02_qualify_dual_view_late_interaction_v0_2.sh`

If a true runtime invariant fails after this bookkeeping fix, stop and treat that invariant as new evidence. Do not automatically authorize another recovery or model change.
