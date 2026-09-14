# Magnolia P100 N0 Failure-Driven Repair Probe — Job 575571

Date: 2026-09-14
Status: completed payload
Purpose: measure whether targeted Sol repair examples improve generic N0 semantic/judgment separability without changing the N0 backbone

## Runtime

- node: `gpu001`
- accelerator: 1x Tesla P100-PCIE-12GB
- checkpoint: `step-00001000`
- backbone frozen: true
- trainable parameters: 404,097 ranking-head parameters only
- private identity gradient: false
- repair epochs requested: 12
- repair batch size: 4
- repair learning rate: 2e-4
- combined train rows: 71
- combined dev rows: 57
- train competencies: 43
- dev competencies: 43

The job payload reached its completion marker at 2026-09-14T12:06:29-05:00.

## Controlled comparison

The prior step-1000 frozen ranker was first evaluated on the expanded 57-row dev set that includes the new repair examples:

- top-1: 0.6491228070
- supported-set separation: 0.6491228070
- failed examples: 20

A fresh frozen-backbone ranker was then trained over the full 71-row train set. `train_curriculum_ranker.py` retained the best dev checkpoint rather than the last epoch. The retained head reached:

- train top-1: 0.8591549296
- dev top-1: 0.7017543860
- dev supported-set separation: 0.7017543860
- dev mean separation margin: 0.3980430840
- failed dev examples: 17

This is an absolute +0.0526315790 increase in expanded-dev top-1 versus the pre-repair head.

## Failure structure

The retained repaired head still has failures in 14 competencies:

- ALIGN-02
- ALIGN-03
- CAUS-02
- EPI-04
- PRAG-04
- PRAG-05
- RANK-04
- RANK-05
- SEM-03
- SEM-04
- SEM-06
- SOC-02
- SOC-05
- TEMP-01

The repair data clearly helped some originally weak dimensions (for example ALIGN-05, EPI-02, PRAG-01, and RANK-01), but performance is still fragile. Several previously passing dimensions also became errors under the newly fit linear head. That pattern is evidence that the step-1000 representation is only partially linearly separable for the intended semantic/judgment operations.

## Decision

Do not create another targeted repair shard yet. Repeatedly adding tiny supervised repairs would increasingly optimize the development set and would not establish that the 352M-parameter N0 backbone has learned the underlying operations.

Freeze the current three public Sol curriculum shards as the next readiness benchmark:

- `sol_curriculum_seed_v0.1.jsonl`
- `sol_curriculum_coverage_v0.2.jsonl`
- `sol_curriculum_repair_v0.3.jsonl`

The next learning segment should therefore improve the public N0 backbone itself through additional MLM exposure, then rerun a fresh frozen-head readiness probe against the same unchanged curriculum. This separates representation growth from curriculum expansion.

Next target: resume exact `step-00001000` to `step-00002500` at sequence length 512 under the corrected global 10,000-step scheduler horizon. This is a bounded 1,500-step semantic-growth segment, not a new infrastructure qualification.

No private Elaina identity gradient is authorized by this result.
