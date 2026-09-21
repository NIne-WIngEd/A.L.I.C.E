# N0 Production P1 STOP-tail failure localized; causal recovery ready

**Date:** 2026-09-20
**Experiment branch:** `alice-eipm-v1-qsre-production-core-v1`
**Qualified recovery HEAD:** `dba3d4b100f081b0c09fe63ae002205244af5b44`

## Observed production run

Magnolia job `575955` executed the single governed Production N0 pipeline from revision `b32fabe49f206c2d71e17df5197a62b2a37c8c43`.

P0 completed correctly:
- exact 1,016-row production curriculum built;
- production and final dynamic schemas materialized;
- production semantic cache materialized;
- 320-row native final validation frozen before any Production P1 result;
- P0 real-artifact runtime qualification passed.

P1 then produced genuine model/runtime evidence rather than an infrastructure crash. Across every saved checkpoint through step 1000, almost all DEV families were already perfect while:
- ordered_path = 0.0;
- path_latest = 0.0;
- path_role = 0.5;
- three_hop = 1.0.

The invariant pattern did not move with training, so optimization was not the causal boundary.

## Root cause

The oracle P1 operator is represented at the curriculum maximum relation-program width. Shorter programs therefore have inactive/STOP tail slots with relation_step_mass=0.

The Production executor multiplied each active step by relation_step_mass, correctly yielding no next-frontier contribution for an inactive tail, but then replaced the current path frontier with that zero next frontier. Consequently:
- a 2-hop path stored in a 3-slot program was correctly advanced twice and then erased by the inactive third slot;
- a 3-hop path survived because it had no inactive tail;
- SOURCE-side path-role cases could partially survive through path_origin while TARGET-side cases depended on the erased reached frontier.

This exactly matches job 575955: ordered_path/path_latest 0, path_role 0.5, three_hop 1.0.

This is a production variable-program semantic defect, not a benchmark patch. It would also have broken P2 serving because P2 uses a larger runtime step budget than many natural relation programs.

## Causal architecture correction

`QSREProductionExecutor` now treats inactive/STOP relation steps as semantic no-ops for path state:

- active mass 1: advance normally;
- active mass 0: preserve the reached frontier exactly;
- fractional survival mass: combine the new active contribution with the unconsumed old frontier.

New mechanics test `test_path_stop_tail_is_semantically_idempotent_forward_and_reverse` proves that the same forward/reverse path yields identical frontier and relational output with or without an inactive STOP tail.

The Production Core machine contract now explicitly requires:
- `inactive_or_stopped_relation_step_preserves_frontier=true`;
- `stop_tail_is_semantically_idempotent=true`;
- path frontier persistence across inactive STOP tails;
- shorter-path semantic equivalence under larger runtime budgets.

The original production training plan bytes and all precommitted LR/step/width/threshold settings remain unchanged.

## Recovery policy

No blind P1 rerun is authorized.

First, zero-gradient requalification evaluates every checkpoint already produced by job 575955 under the corrected executor semantics. If an existing checkpoint satisfies the original P1 gate, that exact unchanged checkpoint is copied into the recovery lineage and P2 begins.

Only if no preserved checkpoint qualifies may the same single recovery job perform one corrected P1 training run, using the exact original data, optimizer, learning rate, max steps, batch size and eligibility thresholds. There is no LR/step/width search and no second automatic P1 attempt.

After P1 passes, the same job executes P2 -> P3 -> P4 -> native final N0 validation.

The final validation is NOT rebuilt. Recovery reuses the exact corpus, caches, manifest and freeze receipt created before the original P1 result. This preserves the untouched native holdout.

## Qualified implementation

- `scripts/eipm/n0/requalify_n0_v02_qsre_production_p1_stop_tail_v1.py`
- `scripts/eipm/n0/run_n0_v02_qsre_production_stop_tail_recovery_v1.sh`
- `scripts/eipm/n0/magnolia_p100_n0_v02_qsre_production_stop_tail_recovery_v1.sbatch`

Static recovery qualification:
- GitHub Actions `35548666874` — SUCCESS

Production mechanics after causal fix:
- GitHub Actions `35548858918` — SUCCESS
- 22 Production QSRE mechanics tests passed, including STOP-tail forward/reverse semantic idempotence.

## Next authorized action

Exactly one owner launch of the stop-tail causal recovery P100 workflow from HEAD `dba3d4b100f081b0c09fe63ae002205244af5b44`.

Preserve `$ALICE_N0_WORKDIR/qsre-production-core-v1` permanently; it is the failed source evidence. Recovery writes only to `qsre-production-core-v1-stop-tail-recovery-v1`.

N0 closes only on `PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE` with `n0_complete=true` and `n1_authorized=true`.
