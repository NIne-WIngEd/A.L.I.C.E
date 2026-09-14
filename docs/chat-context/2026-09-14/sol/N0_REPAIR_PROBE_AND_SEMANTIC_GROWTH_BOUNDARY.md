# N0 Repair Probe and Semantic-Growth Boundary — 2026-09-14

## Current durable evidence

Job 575571 completed the failure-driven frozen-backbone repair probe on `step-00001000`.

The three public Sol curriculum shards now define the active N0 semantic/judgment development benchmark:

- seed v0.1: 60 rows
- coverage v0.2: 26 rows
- repair v0.3: 42 rows

Combined active split used by the repair probe:

- train: 71 rows
- dev: 57 rows
- competencies represented in each split: 43

No private identity data or private identity gradient was involved.

## Job 575571 result

The previously trained frozen head scored 0.6491228070 top-1 on the expanded 57-row dev set.

A fresh head trained against the expanded curriculum and selected by best dev performance reached:

- train top-1: 0.8591549296
- dev top-1: 0.7017543860
- dev separation: 0.7017543860
- dev margin: 0.3980430840
- failed dev examples: 17

Remaining failure competencies:

`ALIGN-02, ALIGN-03, CAUS-02, EPI-04, PRAG-04, PRAG-05, RANK-04, RANK-05, SEM-03, SEM-04, SEM-06, SOC-02, SOC-05, TEMP-01`

The result is useful but not strong enough to justify another tiny repair-data loop. Some targeted dimensions improved, while several previously passing dimensions became errors under the newly fit head. The representation is partially useful but still fragile.

## Decision

Do not author repair v0.4 yet.

Freeze seed v0.1 + coverage v0.2 + repair v0.3 unchanged while the N0 backbone receives more public MLM learning. This gives the next run a stable measurement target and separates representation growth from curriculum growth.

Next bounded segment:

- parent: exact `step-00001000`
- target: `step-00002500`
- added global optimizer steps: 1,500
- sequence length: 512
- scheduler horizon: 10,000 global steps
- save every: 500
- 2x P100
- no private identity gradient

After the target checkpoint is written, run a fresh frozen-backbone semantic-readiness head against the unchanged three-shard curriculum and an 8-mask held-out MLM evaluation. Compare the new dev top-1 and MLM NLL against step 1000.

Tracked runner:

- `scripts/eipm/n0/grow_n0_semantic_p100x2_segment.sh`
- `scripts/eipm/n0/run_n0_semantic_readiness_probe.sh`
- `scripts/eipm/n0/magnolia_p100x2_n0_semantic_growth.sbatch`

The next decision is evidence-driven:

- if both language-model quality and fixed semantic-readiness improve materially, continue public N0 representation learning;
- if MLM improves but fixed semantic readiness stalls, generic MLM alone is no longer sufficient and the next intervention should train semantic/judgment representations directly;
- if both stall, inspect optimization/capacity/data before spending further accelerator time.

Do not move to N1 private identity learning while N0 remains this fragile.
