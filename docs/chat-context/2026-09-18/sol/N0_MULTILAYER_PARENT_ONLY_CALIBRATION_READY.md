# N0 Multi-Layer Interface — Parent-Only Calibration Ready

**Date:** 2026-09-18  
**Continuity status:** causal-study preparation passed; one bounded multi-layer interface experiment is designed and CI-qualified; parent-only preservation calibration is the single next authorized action

## Live experiment frontier

Branch:

`alice-eipm-v1-relation-conditioned-multilayer-interface`

Head:

`4a223b0b35920ded5c81b05f53243024de919357`

Contract workflow:

`N0 Relation Conditioned MultiLayer Interface Contract Check`

Run:

`35419300501`

Conclusion:

`SUCCESS`

Stable build base remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

## Preserved Magnolia evidence

Runtime qualification:

- status: `PASS_EXACT_MAP_NO_GRADIENT_RUNTIME_CONTRACT`
- receipt SHA-256: `779b19459fa8ab67bb1c3ed56b3e9cdb1b89b8382f145e1b3d7c4d98312b2b57`
- compiled layer-map SHA-256: `ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304`
- source layerwise audit SHA-256: `ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab`

Causal-study preparation:

- status: `PASS_CAUSAL_STUDY_INPUTS_FROZEN_TRAINING_DECISION_STILL_REQUIRED`
- source revision: `02a05bf19545b5b281c3cfe7dc2f674b35534793`
- preparation receipt SHA-256: `54dc915f10ee58d994e387ca59ab1180f96bd0e3523780f3e75fb14d6ac30690`
- curriculum SHA-256: `706fb2944ae3f95f0bf93c93763c57d54a76be5068f665d9a9be1f2425dadaab`
- curriculum manifest SHA-256: `6ca37c884b9a75d3fed9ef0e1b8a19704c03d246bb2d32f33cc538546072ba72`
- preservation contract SHA-256: `65b3fa76487fef346ef227dc77ed57da504e2dcf891e32416ce60ef5a6df99dd`

Frozen runtime artifacts:

- semantic checkpoint: `6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43`
- structured checkpoint: `77e9793f50bf1ada2f5f22b59e5cd023668af9392921886102fbdc3b7fee186b`
- parent adapter: `50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df`
- selected parent graph: `3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`
- ordinary replay cache: `10cf39b6382e66c4eeef6a771d748e3eadedab54e6c0ffc6f900d8bfbdea5534`
- endpoint replay cache: `251c965e2113d66fb56122130bf59f9740d46f9fdf4672322daacdb576c7f02c`

## Scientific decision

One bounded interface-only experiment is justified, but gradient/GPU execution is conditional on a successful parent-only preservation calibration.

Decision file:

`configs/eipm/n0/n0_v02_relation_conditioned_multilayer_training_decision_v0_1.json`

Decision:

`AUTHORIZE_ONE_BOUNDED_INTERFACE_EXPERIMENT_AFTER_PARENT_ONLY_CALIBRATION`

Current authoritative experimental state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.20.json`

Current status:

`CAUSAL_STUDY_PREPARATION_PASS;ONE_BOUNDED_MULTILAYER_EXPERIMENT_DECIDED;PARENT_ONLY_CALIBRATION_REQUIRED_BEFORE_GRADIENT`

## Candidate architecture

Integration source:

`src/alice_personality/n0/evidence_graph_multilayer_interface.py`

Classes:

- `RelationConditionedMultiLayerQueryInterface`
- `RelationConditionedMultiLayerEvidenceGraphEncoder`

Path:

`frozen semantic intermediate token layers -> relation-conditioned token attention/layer mixing -> edge_relation_query_state -> zero-initialized signed source/target residual -> frozen parent graph pool logits`

The parent dual-endpoint graph remains the baseline expert.

The new signed edge readout is zero-initialized. Therefore the candidate must produce exact parent behavior before training.

Trainable scope after calibration:

- `query_interface.*`;
- `interface_endpoint_read.*`.

Frozen:

- semantic backbone;
- structured-state encoder;
- evidence-view adapter;
- selected parent evidence graph.

Explicitly absent:

- QRR v0.4;
- generic SOURCE/TARGET classifier;
- raw mean-pool router;
- final-layer-only router;
- hardcoded layer 12;
- previous semantic-role residual reuse;
- parent-graph rewrite.

## Parent-only preservation calibration

Script:

`scripts/eipm/n0/calibrate_n0_v02_multilayer_preservation_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_calibrate_multilayer_preservation_v0_1.sh`

This is CPU-only and candidate-free.

It evaluates the exact frozen parent twice on:

- ordinary graph replay;
- endpoint-role replay.

Frozen rule:

`atol = max(1e-6, 2 * max_pairwise_abs_repeat_delta)`

`rtol = 0`

Required output status:

`PASS_PARENT_ONLY_PRESERVATION_CALIBRATION`

The receipt must state:

- candidate supplied false;
- candidate result observed false;
- gradient performed false;
- optimizer created false;
- GPU required false;
- interface training gate satisfied true;
- scope exactly one bounded relation-conditioned multi-layer interface experiment;
- scale false;
- heldout opening false;
- frozen challenge rerun false;
- N0 complete false.

## Bounded training design already implemented but not executable yet

Trainer:

`scripts/eipm/n0/train_n0_v02_relation_conditioned_multilayer_interface_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_relation_conditioned_multilayer_interface_v0_1.sh`

P100 wrapper:

`scripts/eipm/n0/magnolia_p100_n0_v02_relation_conditioned_multilayer_interface_v0_1.sbatch`

The training runner refuses to run without the valid calibration receipt.

Operating study budget:

- one run maximum;
- one P100;
- 160 optimization steps;
- checkpoints every 40 steps.

Those values are causal-study operating points, not permanent product ceilings.

Only fresh causal train quads are gradient-bearing.

Ordinary replay, endpoint replay, causal dev, causal test, and frozen challenge are not gradient-bearing.

Causal test and frozen challenge are not evaluated in the training run.

## Causal dev readiness

A checkpoint is eligible only if:

1. every parent-only preservation metric passes its frozen policy;
2. no directed relation family dev row accuracy regresses versus exact parent;
3. `causes` strictly improves;
4. `supports` strictly improves;
5. overall causal-quad accuracy strictly improves;
6. worst-family quad accuracy does not regress;
7. mean target margin strictly improves.

Selection is dev-only.

If no checkpoint is eligible:

`FAIL_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`

Then stop and reopen the language↔graph boundary.

If one or more checkpoints are eligible:

`PASS_CAUSAL_DEV_AND_PRESERVATION_READY_FOR_SEPARATE_HELDOUT_DECISION`

Then stop and make one separate heldout-opening decision without changing the candidate.

## CI correction history

Several post-design workflow runs failed due static source-string assertions, not implementation/model failures.

The workflow originally expected JSON-like parameter-report syntax in Python source and later checked a different heldout log phrase than the runner emits.

These assertions were corrected without changing architecture, data, thresholds, trainable scope, or experiment policy.

Final head `4a223b0...` passes the complete workflow.

Do not classify the failed CI runs as model evidence.

## Single next authorized action

Run the parent-only CPU preservation calibration once through whole-stage Magnolia udocker.

Do not submit the P100 training job until the calibration receipt has been inspected and its SHA-256 recorded.

## Magnolia doctrine

- use whole-stage udocker;
- do not use host Python;
- CPU calibration sets `RAYAN_UDOCKER_NVIDIA=0`;
- do not use `git checkout` as a prerequisite on Magnolia;
- pull the named branch with `git pull --ff-only origin ...`;
- preserve existing outputs instead of deleting them;
- infrastructure failure is not model evidence.

## Capability doctrine

N0 remains the full-production personality-model foundation.

The current layer map, 720-row curriculum, interface width, one-run budget, and 160-step training budget are not permanent capability ceilings.

Efficiency may reduce redundant computation. It may not remove required representational capacity or identity-fidelity capability.
