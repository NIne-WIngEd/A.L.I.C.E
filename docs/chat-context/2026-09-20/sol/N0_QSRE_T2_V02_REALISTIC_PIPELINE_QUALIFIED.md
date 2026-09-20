# N0 QSRE T2 v0.2 realistic pipeline qualified

**Date:** 2026-09-20  
**Experiment branch:** `alice-eipm-v1-qsre-t2-operator-learning`  
**Qualified experiment revision:** `9676c4cfd13d18dc185d06b893c9a151c0ca5cff`

## Reality-audit result and correction

Magnolia job 575950 completed successfully and established:

- behavioral Bash/uDocker parity PASS;
- T2->T1 totality fuzz PASS over 480 provisional operator combinations;
- synthetic T2 v0.1 language is easier for shallow lexical models than the realistic audit distribution;
- simple mean-pooled hidden-state centroid probes transfer weakly on several operator heads.

The automatic v0.56 decision `REOPEN_FROZEN_SEMANTIC_REPRESENTATION_BEFORE_T2` is **not ratified as a semantic-backbone failure** because the semantic probe used a mismatched representation class:

- it mean-pooled each hidden layer and used nearest-centroid classification;
- T2 actually consumes all token states from all 17 hidden-state outputs using learned token/layer cross-attention.

Prior exact N0 evidence already showed this distinction:
- job 575801: final token late interaction role accuracy 0.6875 versus 0.5625 pooled;
- job 575804: causes/supports directional information appeared in intermediate token layers and disappeared in some later/pool views;
- job 575933, although invalid as a final model gate, showed T2 step-25 learning from the frozen token stack: control 0.8854, relational role 0.7773, relational operation 0.6406, relation sequence 0.2014.

Therefore job 575950 supports “simple pooled readouts are insufficient,” not “the token-level frozen semantic stack lacks the needed information.”

Research decision:
`docs/research/eipm-n0-qsre-post-reality-audit-interpretation-v0.57.md`

Authoritative state remains:
`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.57.json`

## T2 v0.2 data redesign

A new realistic public-only T2 curriculum was built while preserving the exact T1 structural distribution and the same T2 scientific core.

- curriculum SHA: `5aa8f0aa3210466964f38b14081c71c1d3c58f4405247067c76a7cdfe115963c`
- 720 TRAIN / 288 DEV
- 168 cue-collision rows
- disjoint TRAIN/DEV names and phrase banks
- entity-rich natural-language scenarios
- no relation-name requirement
- no explicit fallback/defer/path/role label leakage
- no private identity data
- TEST absent

Static curriculum workflow:
- run `35537157415`: SUCCESS
- locked challenge workflow run `35537274555`: SUCCESS

Query-only unigram diagnostic on T2 v0.2:
- DEV full signature: 0.39236
- realistic audit full signature: 0.25
- DEV control: 0.74306
- DEV operation: 0.88542
- DEV relation: 0.30903
- DEV role: 0.63542

These head-level scores remain diagnostic, not pass/fail thresholds. Operation language is naturally more lexically recoverable; final operator behavior and the locked challenge remain the causal evidence.

## Locked operator-only challenge

A fresh 60-row public challenge is fixed before training.

- expected SHA: `197cebe5ed4c59455839db5b2822582a03021c92a94670e1aede568278c1f205`
- never training data
- only opens if the T2 DEV contract reaches a perfect governed PASS
- no T1 downstream scoring in this challenge; it isolates operator generalization

## Scientific core preservation

T2 v0.2 keeps exactly the T2 v0.1:
- operator architecture;
- frozen semantic checkpoint;
- frozen T1 executor;
- AdamW optimizer;
- learning rate;
- loss weights;
- continuous operator objective;
- seed;
- batch size;
- max steps;
- eval cadence;
- eligibility thresholds;
- first-perfect-DEV selection rule.

Only the query curriculum and its preparation lineage changed.

Pipeline contract workflow:
- first run `35538203657` failed because the new contract had a wording-only metadata difference in the continuous-objective `role` field; no model evidence.
- corrected exact objective wording.
- run `35538256393`: SUCCESS.

The workflow additionally executes:
- trainer gradient/bridge self-test;
- challenge hash rebuild;
- behavioral fake-Slurm dependency test proving GPU submission has `afterok:<prep-job>`;
- no-tuning/no-hotfix governance.

## One owner-side launch, no micro-gate

The pipeline is now intentionally two Slurm jobs with one owner-side launch:

1. CPU preparation;
2. P100 training automatically released only if CPU preparation exits 0.

Launcher:
`scripts/eipm/n0/launch_n0_v02_qsre_t2_realistic_pipeline_v0_2.sh`

The GPU job:
- validates the exact preparation authority;
- dynamically binds only the prepared-cache SHA into a resolved copy of the frozen training contract;
- performs exactly one T2 v0.2 P100 run;
- if DEV does not perfectly pass, challenge remains closed and the workflow stops;
- if DEV perfectly passes, opens the already-locked 60-row operator-only challenge in the same job;
- never opens T3, causal TEST, frozen challenge, private identity gradient, or promotion.

No automatic rerun, LR search, step search, size search, or hotfix chain is authorized.

## Next action

Pull exact experiment revision `9676c4cfd13d18dc185d06b893c9a151c0ca5cff` and run the single launcher. Return both Slurm job IDs and, after completion, the prep/training stdout and stderr. Do not submit any separate recovery or additional GPU job.
