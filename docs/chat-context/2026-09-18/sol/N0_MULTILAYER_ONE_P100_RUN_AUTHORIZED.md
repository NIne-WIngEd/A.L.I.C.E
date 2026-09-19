# N0 Multi-Layer Interface — One P100 Run Authorized

**Date:** 2026-09-18

## Live experiment frontier

Branch:

`alice-eipm-v1-relation-conditioned-multilayer-interface`

Head:

`8a8531884bd9d59dfff5023579b0789dc7146ea7`

Contract workflow run:

`35420955972`

Conclusion:

`SUCCESS`

Stable build base remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

## Parent-only calibration is complete

Status:

`PASS_PARENT_ONLY_PRESERVATION_CALIBRATION`

Calibration source revision:

`4a223b0b35920ded5c81b05f53243024de919357`

Calibration receipt SHA-256:

`d485fce0e4304d9cbd8af1658ed7cfedfd1bd4d44b0f40cf56a2618d5ad2390b`

Preservation policy SHA-256:

`fb88b9580e38a648101277815f82c3ba60c19f3bfd16dbe3029ded40b04aea56`

Two canonical-parent repeats were numerically identical for every governed preservation metric.

Frozen parent baselines:

- ordinary macro target-support mass: 0.9810013264417649
- ordinary minimum target-support mass: 0.8611709574858347
- ordinary macro top-1 support accuracy: 1.0
- ordinary minimum top-1 support accuracy: 1.0
- endpoint pair accuracy: 0.9583333333333334
- endpoint minimum family pair accuracy: 0.6666666666666666
- endpoint row accuracy: 0.96875
- endpoint mean graph target margin: 0.8209280247489611

Every governed metric uses:

`atol = 1e-6`

`rtol = 0`

The candidate did not participate in calibration.

## Current state

Authoritative experiment state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.21.json`

Status:

`PARENT_ONLY_PRESERVATION_CALIBRATION_PASS;EXACT_CALIBRATION_BOUND;ONE_P100_CAUSAL_INTERFACE_RUN_AUTHORIZED`

Exactly one P100 training run is now authorized.

The canonical training runner hash-gates:

- preparation receipt;
- compiled layer map;
- causal curriculum;
- causal curriculum manifest;
- exact calibration receipt;
- exact preservation policy;
- calibration source revision.

The runner claims the training output directory before model work begins. A failure leaves the claim/evidence in place. Rerun requires explicit diagnosis rather than deletion or automatic retry.

## Candidate architecture

`frozen semantic intermediate token layers -> RelationConditionedMultiLayerQueryInterface -> edge_relation_query_state -> zero-initialized signed source/target residual -> frozen parent dual-endpoint graph`

Trainable:

- `query_interface.*`
- `interface_endpoint_read.*`

Frozen:

- semantic backbone
- structured-state encoder
- evidence-view adapter
- selected parent graph

Not present:

- QRR v0.4
- generic endpoint classifier
- raw pooled-query router
- final-layer-only shortcut
- hardcoded layer 12
- previous semantic-role residual reuse

## Gradient/data boundary

Gradient-bearing:

- fresh causal train quads only

Evaluation-only:

- causal dev
- ordinary preservation replay
- endpoint-role preservation replay

Unopened:

- causal test
- frozen latent challenge

No scale, heldout opening, semantic retraining, graph-parent retraining, private identity gradient, production promotion, or N0 completion is authorized.

## Failure doctrine

If this one model run fails:

1. stop;
2. do not start a local repair/hotfix chain;
3. re-evaluate N0's purpose and the failed component's role in the complete personality architecture;
4. localize the actual failure boundary;
5. review relevant frontier research and frontier model architectures for that boundary;
6. decide whether the interaction topology, representation, objective, curriculum, or component abstraction is wrong;
7. make one causal architectural change only after that larger diagnosis;
8. validate again.

Failure does not automatically justify:

- more steps;
- more width;
- another router;
- threshold relaxation;
- replay-gradient injection;
- reopening a previously failed repair family.

This rule is part of v0.21 experiment state.

## Next action

Submit exactly one Magnolia P100 job from the repository root using:

`scripts/eipm/n0/magnolia_p100_n0_v02_relation_conditioned_multilayer_interface_v0_1.sbatch`

Do not submit a second copy.

Do not delete `training-v0.1` if the job fails. Preserve the output and inspect the failure as evidence.
