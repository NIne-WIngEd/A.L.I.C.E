# N0 Teaching Continuation — 2026-09-14

## Purpose

Return the EIPM build from infrastructure work to actual semantic/judgment teaching while preserving the real step-200 checkpoint from job 575546.

This is a model-development loop, not another route qualification loop.

## What step 200 taught us

The first real N0 run produced durable model signal:

- 352,184,960 trainable parameters;
- 3,276,800 public-corpus tokens seen;
- training loss moved from about 9.64 at step 10 to about 6.53 at step 200;
- the checkpoint is bound to the verified public corpus/tokenizer lineage;
- no private Elaina identity gradient occurred.

The run also exposed a concrete learning-control defect. On 2-process Accelerate DDP with `split_batches=false`, the wrapped scheduler advanced twice per global optimizer update. The old 200-step cosine schedule therefore hit zero near global step 100 and rose again afterward.

The weights remain useful. The next segment resumes them with corrected scheduler semantics rather than discarding or repeating them.

## Scheduler repair

N0 now treats these as different quantities:

- `max_steps`: the end of the current durable learning segment;
- `scheduler_total_steps`: the provisional global optimizer-update horizon of the LR schedule.

Global scheduler warmup/horizon values are converted to Accelerate-internal steps using process count when batches are sharded across processes. Checkpoint receipt schema v0.3 records both global and internal values.

The first repaired continuation uses a provisional 10,000-global-step LR horizon. This is not a claim that N0 must train exactly 10,000 steps. It prevents a short checkpointing segment from accidentally defining the entire cosine schedule. Capability evidence can change later stop or schedule decisions.

## Evaluation added

`scripts/eipm/n0/evaluate_mlm.py` evaluates a checkpoint against the deterministic held-out `dev` document split using fixed dynamic masking. It records masked-token NLL and perplexity plus corpus/tokenizer/checkpoint lineage.

The next remote learning segment evaluates step 200 before training and the new endpoint afterward in the same allocated runtime. This makes the compute decision evidence-bearing instead of relying only on training loss.

## Sol teaching data

The original Sol curriculum remains immutable:

`training/eipm/n0/sol_curriculum_seed_v0.1.jsonl`

It contains 60 owner-authorized, non-private candidate-ranking examples across the N0 competency registry.

A new additive shard is active:

`training/eipm/n0/sol_curriculum_coverage_v0.2.jsonl`

It adds 26 carefully targeted rows only where the original seed lacked either a train or dev example. The union now gives every registered N0 competency at least one train and one dev example without replacing the original seed.

Training/evaluation code now accepts additive governed curriculum shards and records each shard/manifest separately in receipts. Future Sol teaching should be failure-driven: add examples because model evidence shows a weakness, not merely to inflate row count.

## Next real learning segment

Prepared route:

`scripts/eipm/n0/magnolia_p100x2_mlm_continue.sbatch`

Default behavior:

1. require canonical parent `step-00000200`;
2. refuse if target `step-00001000` already exists;
3. evaluate step 200 on held-out MLM dev data;
4. resume exact model/optimizer state;
5. rebase the legacy scheduler state onto corrected distributed semantics;
6. continue to global step 1000;
7. checkpoint every 200 updates;
8. evaluate step 1000 on the same deterministic held-out dev mask;
9. emit v0.3 receipt lineage.

This gives roughly 800 additional global optimizer updates before the next decision, rather than committing scarce compute to an arbitrary long run.

## Responsibility rule

A completed job is not automatically a good teaching run. Inspect what the model actually experienced: data lineage, loss, LR behavior, held-out metrics, checkpoint state, and competency failures. Preserve useful learned weights. Repair the smallest real defect. Then continue from the durable state.

No private E0/E-INF/A-SYN gradient is authorized at this stage.
