# N0 teacher probe 575565 and failure-driven repair — 2026-09-14

## Durable result

Magnolia job `575565` completed the single-P100 step-1000 N0 teacher diagnostic. This was not additional backbone training.

Step-1000 backbone state remains:

- `alice-n0-semantic-v0.1`
- 352,184,960 parameters
- 16,384,000 public MLM tokens seen
- private identity gradient: false

Robust true-held-out MLM evaluation reused the same pre-existing dev documents across eight deterministic masking patterns:

- masked-token exposures: 3,960
- visible-token exposures: 26,528
- mean NLL: 7.886467143622312
- perplexity: 2661.026271237024
- repeat NLL stddev: 0.21838322864697152

The frozen-backbone Sol probe trained only the 404,097-parameter generic ranking head. Original governed curriculum coverage was 43 train rows and 43 dev rows across all 43 N0 competencies.

Best saved dev result:

- top1: 29/43 = 0.6744186046511628
- supported-set separation: 0.6744186046511628
- mean separation margin: 0.18771825817435286
- failures: 14

Failed competencies:

`ALIGN-02`, `ALIGN-03`, `ALIGN-05`, `EPI-02`, `EPI-04`, `PRAG-01`, `PRAG-04`, `RANK-01`, `RANK-04`, `SEM-03`, `SEM-04`, `SOC-02`, `SOC-05`, `TEMP-01`.

The frozen head later fit all 43 train examples while dev performance degraded, establishing sparse-curriculum overfit. Do not unfreeze the 352M backbone on this tiny curriculum. Do not respond by blindly extending MLM either; first test whether targeted public teaching repairs transfer with the backbone still frozen.

## Teaching response

Sol authored `training/eipm/n0/sol_curriculum_repair_v0.3.jsonl` under existing owner authorization.

- 42 generic/public rows
- 28 train
- 14 new dev
- exactly two train and one new dev example for each of the 14 failed competencies
- prior 60-row seed and 26-row coverage shard remain immutable
- no private Elaina identity data
- no private identity gradient authorization

The repair experiment is intentionally comparative:

1. evaluate the old job-575565 frozen ranker on the expanded 57-row dev set;
2. train a fresh frozen head on the expanded 71-row train set;
3. evaluate the repaired head on the same expanded dev set;
4. inspect remaining failure competencies;
5. only then decide between another targeted lesson batch and additional public MLM learning.

Entry points:

- `scripts/eipm/n0/run_n0_repair_probe.sh`
- `scripts/eipm/n0/magnolia_p100_n0_repair_probe.sbatch`

This uses one P100 and leaves the step-1000 backbone unchanged.

Infrastructure remains closed unless a concrete runtime failure occurs. Magnolia is only the accelerator for actual learning/measurement.
