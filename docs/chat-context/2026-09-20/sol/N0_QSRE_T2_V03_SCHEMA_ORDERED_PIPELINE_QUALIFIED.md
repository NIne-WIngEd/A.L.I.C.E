# N0 QSRE T2 v0.3 schema-grounded ordered pipeline qualified

**Date:** 2026-09-20  
**Experiment branch:** `alice-eipm-v1-qsre-t2-operator-learning`  
**Qualified experiment HEAD:** `e9b602457404ed6adeb301a9cf37dcbdf994ab57`  
**Authoritative state:** `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.58.json`

## T2 v0.2 result

Magnolia preparation job 575953 completed 0:0 and bound the realistic public T2 v0.2 curriculum, exact prepared cache, frozen semantic checkpoint, frozen T1 executor, and prelocked 60-row operator challenge.

Magnolia P100 job 575954 completed 0:0 and is a valid model failure:

`FAIL_QSRE_T2_OPERATOR_DEV_CONTRACT`

Result SHA:
`6e4fa39775a985b3265e4e1a84851ebc344310d7dd699e1649e807ab9f8f6025`

Final summary SHA:
`03f9a639b47de06f4b1faa00465dffb20489c6e328b12eedae225f491df97dfb`

Best observed step:
`1175`

Best observed checkpoint:
`5ff543b8c87543681286d55a3f982ecf31af93588d65b979056992a4ff041574`

The locked operator challenge remained closed exactly as intended.

## Failure localization

The failure is not treated as global semantic-backbone insufficiency.

The frozen token stack allowed the existing T2 interface to learn:

- operation accuracy = 1.0;
- control accuracy = 1.0 at some checkpoints;
- relational role accuracy up to 0.96484375.

The weak axis was relation identity/order:

- relation-sequence exact max = 0.5381944179534912;
- full operator exact max = 0.4618055522441864;
- query-view pair consistency max = 0.5416666666666666;
- ordered-path operator success remained especially weak.

This matches the actual T2 v0.2 source: relation labels were opaque randomly initialized learned anchors, and the two relation slots were not explicitly semantically conditioned in sequence.

No T2 v0.2 rerun, LR search, step extension, width search, or hotfix was authorized.

## T2 v0.3 causal architecture change

Only relation extraction changes.

Kept exactly:

- frozen semantic checkpoint;
- all 17 token-level hidden outputs;
- frozen T1 executor;
- exact realistic v0.2 prepared TRAIN/DEV cache;
- oracle support/focus;
- role head family;
- operation head family;
- control head family;
- continuous operator objective;
- optimizer, LR, seed, batch size, max steps, eval cadence, eligibility, and first-perfect-DEV selection.

New relation path:

`frozen semantic relation schema -> token/layer late interaction -> ordered relation-slot conditioning -> structural STOP`

Relation schema:
`configs/eipm/n0/n0_v02_qsre_t2_relation_schema_v0_1.json`

SHA:
`bae45945fcb1a7f33b1177bc65da02c90335d443a0718aa7c796e744ace2e2c9`

The schema contains public natural-language directional definitions for SUPPORTS, CORRECTS, SUPERSEDES, DERIVED_FROM, CAUSES, and TEMPORAL_SUCCESSOR. NONE is a structural STOP, not a seventh semantic relation.

The same frozen semantic backbone encodes those schema glosses into all 17 hidden layers. They are stored as a frozen operator buffer, not relation-specific learned class parameters.

The ordered relation decoder:
- re-attends full multi-layer query token memory for every current relation step;
- computes token-level semantic similarity against the relation schema at matching hidden layers;
- feeds the soft prior relation decision into the next relation slot;
- does not hard-select a transformer layer;
- preserves fail-closed sequence termination.

The current six relation IDs and two-step T1 bridge are current causal-fixture shapes, not product ceilings.

## Research rationale

The design is consistent with recent schema-anchored latent reasoning work: semantic schema elements participate directly in latent inference rather than existing only as arbitrary labels.

Recent multi-hop layer-order work also argues against assuming hop 1 and hop 2 live at fixed successive transformer depths. Therefore T2 v0.3 keeps all token layers available and makes relation order a decoder property.

## Static qualification

Workflow:
`n0-qsre-t2-v03-schema-ordered-contract`

Run:
`35539728367`

Conclusion:
`SUCCESS`

Qualified HEAD:
`e9b602457404ed6adeb301a9cf37dcbdf994ab57`

The workflow passed:

1. Python/shell syntax;
2. six schema-ordered operator mechanics tests;
3. trainer gradient/bridge self-test;
4. proof that only the relation architecture changed relative to v0.2;
5. exact relation-schema and locked-challenge hashes;
6. behavioral Slurm `afterok` dependency test;
7. no-rerun/no-hotfix/no-T3/no-private-identity governance.

## One owner-side Magnolia launch

Launcher:

`scripts/eipm/n0/launch_n0_v02_qsre_t2_schema_ordered_pipeline_v0_3.sh`

It submits exactly:

1. one CPU node job that loads **real Magnolia artifacts**, the exact frozen semantic checkpoint, relation schema, realistic prepared cache, and frozen T1 executor; materializes real schema/query hidden states and executes T2->T1 with no gradient;
2. one P100 job dependent on `afterok:<CPU qualification job>`.

If the CPU runtime qualification fails, the P100 job will not start.

If T2 v0.3 DEV fails, the locked challenge remains closed and the pipeline stops.

If DEV reaches the precommitted perfect contract, the already-locked 60-row operator-only challenge opens once in the same P100 job.

No T3, causal TEST, frozen challenge, semantic gradient, private identity gradient, rerun, hotfix chain, LR search, step search, width search, or production promotion is authorized.
