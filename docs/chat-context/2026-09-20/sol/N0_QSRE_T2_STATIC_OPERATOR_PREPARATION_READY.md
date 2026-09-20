# N0 QSRE T2 Static Operator / CPU Preparation Ready

**Date:** 2026-09-20
**Status:** durable handoff after T1 checkpoint binding and T2 static contract pass

## Authoritative experimental frontier

- branch: `alice-eipm-v1-qsre-t2-operator-learning`
- head: `b4eb70634f68ac3c6a34f431d016083c2c9002ce`
- authoritative state: `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.50.json`
- T2 design: `configs/eipm/n0/n0_v02_qsre_t2_operator_learning_design_v0_2.json`
- T2 curriculum contract: `configs/eipm/n0/n0_v02_qsre_t2_curriculum_contract_v0_1.json`
- static CI run: `35531513467` — SUCCESS

## Frozen T1 evidence

Magnolia post-run binding passed:

- T1 result SHA-256: `ba6b82c10fcb13f403569ad24dbd77cf0b3657dfc3eaf9cb9d10b6690b2d32be`
- selected checkpoint step: `50`
- selected T1 checkpoint SHA-256: `483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb`
- post-run receipt SHA-256: `fe07035348e2f5e2d523ec94ae786dbe535a2a45e6fddd727254f3765b927b8f`
- causal TEST stayed closed
- frozen challenge stayed closed
- T2 gradient remains closed

T1 is not to be rerun or hotfixed.

## T2 causal boundary

T2 asks only whether natural-language semantic hidden states can recover the QSRE operator needed by the frozen proven executor while structural support and focus binding remain oracle.

Trainable when later authorized:
- T2 query-operator encoder only.

Frozen:
- semantic backbone
- selected T1 executor/readout
- graph parent and other N0 parent modules
- oracle structural support
- oracle focus binding in the first T2 study

The operator encoder receives no oracle relation/role labels and no graph/support input.

## T2 architecture now staged

`src/alice_personality/n0/qsre_t2_operator.py` implements CPU-qualified mechanics:

- input: frozen full semantic hidden-state stack `[B,L,T,640]`
- projection to 512
- learned layer embeddings
- latent cross-attention over all layer/token states
- ordered relation-slot latents
- first-class role latent
- operation latent
- control/applicability latent
- continuous residual latent
- learned migratable schema anchor banks
- token/layer attention diagnostics
- no hard relation-specific layer mask
- no parent global field logits
- no graph support input
- no oracle relation or role input

For the initial T2-to-T1 bridge, learned continuous context is intentionally not injected into the frozen T1 executor because T1 was trained with zero operator-context vectors. Injecting a new varying context would change the executor input distribution and confound the T2 operator-extraction question. The continuous latent remains available for later integrated QSRE stages.

## Public T2 curriculum

The initial T2 curriculum is deterministically derived from the exact T1 v0.2 structural rows while adding natural-language operator queries.

Expected rows:
- TRAIN: 720
- DEV: 288
- total: 1008

It covers all nine T1 families with two natural-language query views per row. TRAIN and DEV template banks are disjoint. Relation wording includes literal and relation-name-free paraphrases. No TEST or private identity data is present.

The first CI run `35531380426` correctly caught an outside-support query-invariance bug before training: the counterfactual LOW/HIGH rows differed only because phrase selection still included the row variant. Operator mechanics had already passed. This was a curriculum/data-contract failure, not model evidence.

Only the query-key construction was corrected. No model architecture, training threshold, permission, optimizer, or causal boundary changed.

The corrected run `35531513467` passed syntax, governance, all 5 operator-mechanics tests, deterministic T1/T2 curriculum construction/evaluation, and preparation import.

## Next action

Run exactly one CPU/no-gradient tokenized T2 preparation on Magnolia using the existing whole-stage uDocker route.

Expected terminal status:

`PASS_QSRE_T2_TOKENIZED_PREPARATION`

This preparation:
- hash-binds the exact T1 curriculum, prepared cache, and selected step-50 T1 checkpoint;
- reuses the existing T1 structural tensors;
- tokenizes the new T2 natural-language queries with the N0 v0.2.1 tokenizer;
- produces 720 TRAIN / 288 DEV rows;
- does not materialize semantic hidden states yet;
- creates no optimizer;
- performs no gradient;
- requires no GPU;
- opens no TEST/challenge/private identity data;
- does not authorize T2 training.

## Anti-loop and capability doctrine

- no T1 rerun
- no T1 hotfix
- no automatic T2 rerun
- no LR/step tuning after a future genuine model failure without first-principles localization
- infrastructure/data-contract failures are not model evidence
- engineering may be parallelized while scientific causality stays sequential
- full EIPM workload envelope remains the architecture target
- current relation slots, relation vocabulary, role vocabulary, hidden width, field/edge/support/hop count, context length, parameter count, and compute route are not permanent product ceilings
- Graphify is navigation-only; consequential claims resolve to original branch/path sources
