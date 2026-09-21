# N0 final Production v3 qualified for one owner launch

**Date:** 2026-09-21
**Experiment branch:** `alice-eipm-v1-qsre-production-core-v1`
**Qualified exact HEAD:** `2204f61f3a2d49eccc10d834dda83921a9af5be1`
**Final qualification:** GitHub Actions `35553707475` — SUCCESS

## Why v3 is the final pre-launch architecture

Two prior Production results are preserved as causal evidence:

1. Magnolia job `575955` exposed a path-program STOP-tail semantic defect. The unchanged P1 step-50 checkpoint later passed completely under the corrected executor. Ratified P1 checkpoint SHA-256:
   `91f2c78dcc35967064189af4a7110cdee83e5652fb037d3d88f58500ac3aaab9`.

2. Magnolia job `575956` produced a genuine P2-v1 operator failure: training loss collapsed while DEV relation transfer, open-schema transfer, UNKNOWN termination and several operator factors remained far below the precommitted gate. P3 was never authorized.

A fresh architecture review before another GPU run found that sparsifying relation hypotheses inside the operator was itself premature and that exposing held-out schema relations as training negatives would weaken the zero-shot claim.

## Final architecture

### Operator v3

`src/alice_personality/n0/qsre_production_operator_v3.py`

- full 640-dimensional semantic width;
- full hidden-stack query access;
- shared query/schema semantic projection;
- symmetric token-level late interaction;
- recurrent ordered relation program with shared weights;
- continuous relation hypotheses through the operator;
- no exact-zero relation sparsity before structural binding;
- relation selection separate from CONTINUE / STOP / UNKNOWN;
- dedicated role, traversal, direction, modifier, control and applicability query slots;
- P1 schema relation state is executor interface only, not relation-matching authority;
- dense relation/event/factor logits for stable supervision;
- continuous residual operator state retained for executor compatibility.

### True zero-shot schema boundary

P2 gradient sees only the six core training relation descriptions. The four Production open-schema DEV relations do not appear as positive labels, negatives, schema-identity calibration targets or downstream query candidates during gradient.

They first enter the query-conditioned operator at DEV/runtime.

Final-only ENABLES/PREVENTS remain outside Production P1/P2/P3 entirely and first enter at the frozen native final validation.

### Binder

`src/alice_personality/n0/qsre_production_binder_v2.py`

This remains the final structural binder because its interface already satisfies the desired boundary:

- shared query/field metric;
- symmetric query/field late interaction;
- relation semantics owned by operator relation mass;
- no P1 relation-state matching authority;
- no operator continuous-state support authority;
- exact runtime schema type constraints;
- learned path focus without oracle;
- adaptive exact-zero sparse edge support;
- no fixed top-k, support-count, field-count or edge-count ceiling.

Thus the final causal order is:

`continuous operator -> exact sparse structural support -> relational execution -> structural readout -> N0 fusion/latent fabric`.

## Training / validation governance

Final plan:
`configs/eipm/n0/n0_v02_qsre_production_training_plan_v3.json`

Recovery contract:
`configs/eipm/n0/n0_v02_qsre_final_production_recovery_contract_v1.json`

No P1 retraining.
No P2-v1 retraining.
P2-v2 was never run on GPU.
Exactly one P2-v3 run is authorized.
P3-v3 runs only if P2 passes.
P4-v3 is zero-gradient and runs only if P3 passes.
The original frozen 320-row native final validation is reused unchanged.
No final corpus rebuild.
No threshold change.
No LR search.
No step search.
No width search.
No batch search.
No automatic rerun/hotfix.
No TEST opening.
No private identity gradient.

## Qualification

GitHub Actions `35553707475` passed on exact HEAD `2204f61f3a2d49eccc10d834dda83921a9af5be1`.

It proved:
- complete Python compilation and shell syntax;
- Production operator/binder/executor mechanics;
- continuous relation hypotheses and structural-only exact sparsity;
- core-only gradient boundary for open-schema relations;
- unchanged model width, optimizer, compute settings and eligibility gates versus the original precommitted Production plan;
- exact P1 checkpoint reuse;
- original frozen final-validation reuse;
- no oracle support/focus in P4;
- final-only relation isolation;
- complete Magnolia CLI/uDocker lineage.

## Next action

One owner-submitted Magnolia P100 job using:
`scripts/eipm/n0/magnolia_p100_n0_v02_qsre_final_production_v3.sbatch`

Output root:
`$ALICE_N0_WORKDIR/qsre-production-core-v1-final-v3`

N0 closes only if the native final result is:
`PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE`
with:
`n0_complete=true`
and
`n1_authorized=true`.

Otherwise preserve all evidence and stop for causal localization. No automatic architecture iteration is authorized.
