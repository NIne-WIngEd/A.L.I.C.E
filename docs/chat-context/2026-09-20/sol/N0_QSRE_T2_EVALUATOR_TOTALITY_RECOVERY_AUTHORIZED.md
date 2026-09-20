# N0 QSRE T2 Evaluator-Totality Recovery Authorized

**Date:** 2026-09-20
**Status:** job 575933 classified as evaluator-harness failure, not a model-gate result; one fresh classified replacement run authorized after static qualification

## Failed governed run

- job: `575933`
- source revision: `c43df19e79dc63ed00f15a313d71f7873fa5b71f`
- scheduler state: FAILED
- exit: 1:0
- elapsed: 00:01:10
- last completed DEV evaluation: step 25
- result.json: not produced
- model-gate conclusion: **UNDETERMINED**

Observed partial diagnostics are retained but are not promoted to PASS or FAIL.

At step 25 the learned operator was improving, but the run later stopped while evaluating a provisional T2 prediction:

`ValueError: PATH_FOLLOW requires a non-empty oracle focus frontier`

## Root cause

The frozen T1 executor is correct to reject PATH_FOLLOW when the focus frontier is empty.

During T2 learning, however, an immature operation head can temporarily predict PATH_FOLLOW on rows whose oracle focus is empty. The T2 evaluator passed every provisional prediction directly into the strict T1 validator. Therefore an ordinary wrong intermediate prediction could raise an exception instead of being scored as wrong.

This is an evaluator-domain / totality defect. It is not evidence that the T2 model architecture, semantic backbone, T1 executor, optimizer, loss, or capacity failed.

## Bounded correction

Scientific correction revision:

`bff3b2e7c31255e8ca9354add5dafa1bde177b18`

Only the evaluator/trainer bridge was corrected.

For safe frozen-T1 execution:
- impossible provisional PATH_FOLLOW predictions are canonicalized to ROLE_SELECT at the execution boundary;
- if the predicted control is RELATIONAL, that row remains explicitly invalid and is forced to downstream failure;
- its downstream probability is zeroed so safe execution cannot give it accidental credit;
- if control is FALLBACK/DEFER, operation is semantically inactive, so canonicalization only satisfies the frozen executor input contract;
- new diagnostics report `invalid_relational_path_without_focus_count/rate`.

The frozen T1 executor was not changed. The T2 operator architecture was not changed. LR, optimizer, loss weights, curriculum, semantic checkpoint, T1 checkpoint, eligibility thresholds, and TRAIN/DEV rows were not changed.

## Qualification

Evaluator-totality qualification:

- revision: `feee4f2b8cc36dbcf1c3805f401f4347315be945`
- workflow run: `35533710899`
- conclusion: SUCCESS

Passed:
- syntax
- failure-classification governance
- frozen scientific boundary
- evaluator totality mechanics self-test
- correction-scope audit

## Recovery authorization

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.54.json`

Recovery contract:

`configs/eipm/n0/n0_v02_qsre_t2_training_recovery_contract_v0_2.json`

Qualified recovery scientific core:

`157153b124d665880de237299d8dd8c81403d45c`

Recovery authorization workflow:

- run: `35533910879`
- conclusion: SUCCESS

The CI explicitly proved that the recovery contract has the **same**:
- T2 architecture
- frozen semantic checkpoint
- frozen T1 checkpoint
- optimizer and learning rate
- loss weights / continuous objective
- seed
- batch size
- max steps
- eval cadence
- curriculum and preparation lineage
- eligibility thresholds
- selection rule

No scientific tuning was made in response to the exception.

## Preservation / anti-loop behavior

The original failed output root remains immutable:

`$HOME/rayan-compute/rayan-n0/n0-v02/qsre-t2-operator-v0.1/training-v0.1`

The recovery runner requires the failed root, hidden cache, and best-observed checkpoint to exist and refuses to delete or overwrite them.

The replacement run uses a fresh root:

`$HOME/rayan-compute/rayan-n0/n0-v02/qsre-t2-operator-v0.1/training-v0.2-evaluator-totality-recovery`

It restarts from the original seed. It does not reuse partial optimizer state or promote the step-25 checkpoint.

Exactly one classified replacement P100 run is authorized. This is not an automatic rerun and not an LR/step/model hotfix.

Still closed:
- T1 rerun
- semantic-backbone gradient
- T1-executor gradient
- T3 / learned support
- causal TEST
- frozen challenge
- private identity gradient
- production promotion

If the replacement reaches a genuine model FAIL, preserve the result and localize before any further change.
