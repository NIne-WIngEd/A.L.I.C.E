# N0 QSRE T2 One-Shot Operator Training Authorized

**Date:** 2026-09-20
**Status:** tokenized preparation passed; one governed T2 P100 run is statically qualified and authorized

## Active experiment frontier

- branch: `alice-eipm-v1-qsre-t2-operator-learning`
- head: `6f20c807d0281fb00d6cd6def05a8ee8eef01518`
- authoritative state: `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.52.json`
- training contract: `configs/eipm/n0/n0_v02_qsre_t2_training_contract_v0_1.json`
- static authorization CI: workflow run `35532580204` — SUCCESS
- scientific core binding: `9924227e7e7d7f3b26c2e294b5a3dde455b4cb12`

## Resolved preparation boundary

The earlier tokenized-preparation stop was an infrastructure import-boundary failure, not model evidence. Recovery preserved the already valid curriculum and resumed only the missing tokenization step.

Observed Magnolia preparation PASS:

- status: `PASS_QSRE_T2_TOKENIZED_PREPARATION`
- T2 curriculum SHA-256: `40e2d608eaa59ad13e2f9b37a2f3da5d6c20101a28ab76434c11be3d36cb5590`
- curriculum receipt SHA-256: `26719ccff1ad8b08140a495a9cfefa8696270b70e3efd5d0a9b32e46393a2165`
- prepared cache SHA-256: `774722d57ccaacc7dba8e53ff3aec66bc4b4b5c6efd20db6907ac5a70654e781`
- preparation receipt SHA-256: `52e1e0599ecda2992dd2476305c377a4645a7ad079fc6e7ea660684fa35a5967`
- TRAIN rows: 720
- DEV rows: 288
- TRAIN query width: 46
- DEV query width: 44
- semantic hidden states materialized: false
- optimizer/gradient/GPU: false
- TEST/private identity: false

## T2 one-shot scientific question

Can a learned natural-language operator encoder over the frozen full N0 semantic hidden-state stack recover the ordered relation sequence, role, operation, and control required by the exact frozen T1 executor while structural support and focus binding remain oracle?

## Frozen and trainable boundaries

Frozen:
- ratified N0 v0.2 semantic backbone, step80
- exact selected T1 executor, step50
- structural support
- focus binding
- all other parent N0 machinery

Trainable:
- `QSRET2OperatorEncoder` only

Still closed:
- semantic-backbone gradient
- T1-executor gradient
- learned support / T3
- causal TEST
- frozen challenge
- private identity gradient
- production promotion

## Training mechanics

The governed run materializes all 17 frozen semantic token-state layers once. Special tokens are excluded from the T2 attention mask. The cache is stored fp16 for efficiency; T2 consumes batches in fp32.

The operator model keeps:
- multi-layer token access
- learned layer embeddings
- latent cross-attention
- ordered relation slots
- first-class role, operation, and control latents
- learned schema-anchor readout
- retained continuous operator state
- no oracle relation/role/support input

The continuous operator state is now given a supervised contrastive operator-signature objective. This prevents the retained continuous projection from being an unused/random side output while leaving the frozen T1 interface unchanged.

The continuous state is still not injected into T1 because T1 was trained with zero operator context. Doing so now would create a new executor input distribution and confound T2.

## Selection contract

The run stops at the first DEV checkpoint satisfying all exact operator and downstream contracts.

Operator gates include:
- exact relation-sequence accuracy = 1.0
- relational role accuracy = 1.0
- relational operation accuracy = 1.0
- control accuracy = 1.0
- full operator exact accuracy = 1.0
- query-view pair consistency = 1.0
- minimum family operator success = 1.0

Downstream frozen-T1 gates retain the strict T1 causal requirements, including:
- control accuracy = 1.0
- single-target top1 = 1.0
- causal pair completion = 1.0
- minimum family success = 1.0
- plural L1 <= 0.1
- outside-support mass = 0
- outside-support invariance max delta <= 1e-7

No TEST/challenge is opened by this run.

## Anti-loop policy

Exactly one P100 training run is authorized. Automatic rerun and automatic hotfix chains are forbidden.

If the model gate fails:
- preserve all hidden cache, metrics, result, and checkpoint evidence;
- do not tune LR or steps and rerun;
- first localize the failure to operator component, semantic representation, or downstream executor interaction.

Infrastructure failures remain separate from model evidence.

## Next action

Submit:

`sbatch scripts/eipm/n0/magnolia_p100_n0_v02_qsre_t2_operator_training_v0_1.sbatch`

from the A.L.I.C.E repo root after fast-forwarding to the exact current branch head.
