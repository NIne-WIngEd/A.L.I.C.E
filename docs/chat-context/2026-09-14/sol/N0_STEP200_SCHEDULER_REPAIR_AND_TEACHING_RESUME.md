# N0 Step-200 Scheduler Repair and Teaching Resume — 2026-09-14

## Authoritative real-learning state

Job `575546` is the first real durable N0 MLM learning run.

- state: COMPLETED 0:0
- elapsed: 00:21:57
- node: gpu001
- 2× Tesla P100-PCIE-12GB
- canonical endpoint: `$HOME/rayan-compute/rayan-n0/n0-v01/checkpoints/step-00000200`
- tokens seen total: 3,276,800
- parameters: 352,184,960
- sequence length: 512
- fp16, world size 2, gradient checkpointing true
- step-200 model SHA-256: `120eb602cb118eb2e45057362208be2b0835646812c557e8a5c044502f327cbf`
- private identity gradient: false

Training loss moved from 9.6375 at step 10 to 6.5328 at step 200.

Job 575547 was a duplicate submission attempt and is not a second learning segment. Durable checkpoint state outranks launcher history.

## Scheduler defect discovered from model evidence

The logged LR reached zero at global step 100 and then rose again:

- step 10: 3.0e-4
- step 100: 0.0
- step 190: 3.0e-4
- step 200: 2.9095e-4

Root cause: Accelerate `AcceleratedScheduler`, with `split_batches=false`, advances the wrapped scheduler once per process for every synchronized optimizer update. N0 had passed a 200-step global horizon directly to the scheduler under 2-process DDP, so the underlying cosine schedule was consumed twice as fast as intended and then continued into the next cosine half-cycle.

The weights are retained. This is a training-control defect, not a data-lineage or architecture failure.

## Build repair now tracked

On `alice-eipm-v1-build` the code now:

- converts global warmup/horizon to Accelerate-internal scheduler steps;
- separates segment `max_steps` from `scheduler_total_steps`;
- records scheduler global/internal values in MLM receipt schema v0.3;
- marks continuation from the old v0.2 checkpoint as `legacy_scheduler_rebase=true`;
- adds deterministic held-out MLM dev evaluation;
- adds a before/after continuation path that resumes step 200 and refuses duplicate target checkpoints.

Prepared continuation path:

- `scripts/eipm/n0/continue_mlm_p100x2_segment.sh`
- `scripts/eipm/n0/magnolia_p100x2_mlm_continue.sbatch`

Default next endpoint is step 1000 with a provisional 10,000-global-step scheduler horizon. The 10,000 value is an LR horizon, not a mandatory final training length. The next decision is made from held-out and training evidence.

## Sol teaching resumed

Do not bulk-generate synthetic teaching rows just to increase count.

Original immutable teacher seed remains:

- `training/eipm/n0/sol_curriculum_seed_v0.1.jsonl`
- 60 rows
- owner-authorized Sol teacher
- no private identity data

A targeted additive coverage shard now exists:

- `training/eipm/n0/sol_curriculum_coverage_v0.2.jsonl`
- 26 rows
- fills only train/dev gaps in the 43-competency N0 registry
- seed + coverage now gives every competency at least one train and one dev row
- no private identity data or private identity gradient authorization

Curriculum trainer/evaluator now accepts multiple governed shards and records each shard separately in receipts. Future teaching expansion must be driven by observed per-competency failures.

## Operating rule

Remote compute is for actual weight learning/evaluation only. Git/local work remains the main development path. Do not reopen Magnolia/Kaggle infrastructure qualification unless a new concrete failure requires it.

The next model path is:

1. corrected N0 continuation with held-out before/after MLM evidence;
2. train/evaluate Sol semantic/judgment ranker on complete train/dev competency coverage;
3. add teaching examples only against observed failures;
4. then move into N1 identity representation and N2 context-conditioned judgment using the governed private substrate when owner authorization for private gradient is explicitly given at that boundary.
