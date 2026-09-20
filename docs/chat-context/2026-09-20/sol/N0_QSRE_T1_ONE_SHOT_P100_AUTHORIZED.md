# N0 QSRE T1 One-Shot P100 Authorized Frontier

Date: 2026-09-20

## Authoritative branch

alice-eipm-v1-qsre-t1-executor-training

HEAD:
f4a9edb1915e39d893224f4503746bfd61fc6424

State:
configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.48.json

Training contract:
configs/eipm/n0/n0_v02_qsre_t1_training_contract_v0_2.json

## Pretraining evidence

Corrected-pretraining CI at current authorization commit:
35502469015 — SUCCESS

One-shot authorization CI:
35502469056 — SUCCESS

Magnolia real-cache preparation:
PASS_QSRE_T1_REAL_CACHE_PREPARATION

Pinned hashes:
- source cache: 5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823
- curriculum: 155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063
- curriculum receipt: 55f7bc40a5d5f1ff5c134e1f4ad00c1788748f317d7c0b118114c111e794b60d
- prepared cache: 03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564
- preparation receipt: b92ddc17749f03230f3feff1d70942ec2fa5d893638574d8c81ee2957608e9db

Prepared rows:
- TRAIN 360
- DEV 144

Prepared field-state width:
640

Verified:
- paired causal representation identity true
- TRAIN/DEV representation pools isolated
- direct field support all zero
- edge support exact membership
- no TEST
- no private identity
- preparation optimizer/gradient/GPU false

## T1 causal question

Can the corrected QSRE executor and support-local structural readout learn public relation/role, plurality, reliability, temporal, and causal-pair behavior when:
- operator is oracle;
- structural support is oracle;
- field representations are frozen production-real N0 semantic states?

T1 does NOT test:
- learned natural-language operator extraction;
- learned support discovery;
- semantic backbone learning;
- parent graph learning;
- private identity.

## Run governance

Exactly one P100 run is authorized.

Runner:
scripts/eipm/n0/run_n0_v02_qsre_t1_executor_training_v0_1.sh

Slurm wrapper:
scripts/eipm/n0/magnolia_p100_n0_v02_qsre_t1_executor_training_v0_1.sbatch

Training output:
$ALICE_N0_WORKDIR/qsre-t1-executor-v0.1/training-v0.1

The runner refuses:
- wrong prepared-cache bytes;
- wrong preparation receipt;
- wrong curriculum bytes;
- pre-existing training output;
- tracked repository drift;
- TEST/challenge/private-identity opening.

The trainer stops at the first fully eligible DEV checkpoint. If no checkpoint is eligible by max_steps, it writes a genuine T1 executor-failure result and stops.

No automatic rerun or hotfix is authorized.

## After the run

If PASS:
- interpret the selected checkpoint;
- decide whether T2 learned operator is scientifically justified;
- do not auto-start T2.

If FAIL:
- analyze family and causal-pair metrics;
- no LR tuning, longer run, rerun, or local architecture repair before first-principles localization.
