# Magnolia 2×P100 Real N0 Training — Job 575546

**Date:** 2026-09-14  
**Status:** completed successfully; durable weights retained; LR scheduler defect identified before continuation  
**Role:** first durable real N0 MLM learning segment on the imported public corpus/tokenizer lineage

## Scheduler/job result

- job: `575546`
- name: `rayan-n0-p100-pilot`
- state: `COMPLETED`
- exit code: `0:0`
- elapsed: `00:21:57`
- node: `gpu001`
- observed route: 2× Tesla P100-PCIE-12GB

This run is real N0 learning state, not runtime/DDP qualification. Runtime and distributed-mechanics gates were already closed before this job.

## Step-200 durable checkpoint

Canonical checkpoint:

`$HOME/rayan-compute/rayan-n0/n0-v01/checkpoints/step-00000200`

Receipt facts:

- schema: `alice.eipm.n0.mlm-checkpoint-receipt.v0.2`
- model id: `alice-n0-semantic-v0.1`
- step: `200`
- sequence length: `512`
- world size: `2`
- mixed precision: `fp16`
- gradient checkpointing: `true`
- parameters total/trainable: `352,184,960`
- tokens seen this launch/total: `3,276,800`
- corpus receipt SHA-256: `6cf717095e0eea4e790d0c852014a8047fd5ffc9f30550b2445c8a992a0bc293`
- corpus verified bytes: `422,476,606`
- corpus verified shards: `5`
- tokenizer SHA-256: `9ce759f063d92acfdfd65b24ee8f4cad180d7dd8a37fef1a5da8a67df0a8b88d`
- tokenizer receipt SHA-256: `821520db2ee5005fbbad800e0bcc4b0747610b68564526ed312296724fdbd844`
- tokenizer origin git revision: `cb23ac483d0ec5bcc9107f895ada0a6a9c8c34ba`
- training code git revision: `79d4b2d0b4c3bff8932dbc5c5a68f79256d59b9e`
- step-200 `model.safetensors` SHA-256: `120eb602cb118eb2e45057362208be2b0835646812c557e8a5c044502f327cbf`
- private identity gradient: `false`
- resume parent: `null`

The step-100 and step-200 checkpoints are both retained. Job `575547` was a duplicate submission attempt and must not be treated as a second learning segment.

## Observed learning curve

Training loss moved from `9.6375` at step 10 to `6.5328` at step 200. Selected points:

| Global step | loss | logged LR |
| ---: | ---: | ---: |
| 10 | 9.6375 | 3.000e-4 |
| 20 | 7.6938 | 2.910e-4 |
| 30 | 7.2067 | 2.649e-4 |
| 70 | 6.8518 | 7.500e-5 |
| 100 | 6.9379 | 0.0 |
| 150 | 6.8595 | 1.760e-4 |
| 180 | 6.5640 | 2.910e-4 |
| 200 | 6.5328 | 2.910e-4 |

The weights learned useful signal and are retained. However, the LR trace exposed a concrete scheduler configuration defect: the cosine schedule reached zero at global step 100 and then rose again, even though the run target was 200 optimizer updates.

## Root cause and repair

N0 passed the unscaled global `max_steps=200` and `warmup_steps=20` directly into the wrapped scheduler. With Accelerate and `split_batches=False`, `AcceleratedScheduler` advances the underlying scheduler once per process for each synchronized optimizer update. On 2-process DDP, the underlying scheduler therefore consumed approximately two scheduler steps per global optimizer update. The 200-step cosine horizon was exhausted at about global step 100, after which the cosine lambda continued into its next half-cycle and increased the LR again.

This is a training-control bug, not a model-architecture or data-lineage failure.

The build now repairs it by:

1. expressing scheduler horizon/warmup in global optimizer updates;
2. converting those values to Accelerate-internal scheduler steps using process count when `split_batches=false`;
3. separating the LR schedule horizon from the current segment stop step;
4. recording global/internal scheduler horizons in checkpoint receipt schema v0.3;
5. marking continuation from this v0.2 checkpoint as a legacy scheduler rebase;
6. adding deterministic held-out MLM evaluation before longer continuation decisions.

The step-200 optimizer/model state remains usable. The next run must resume this exact checkpoint with the repaired schedule. It must not restart from random initialization or repeat steps 0–200.

## Warnings classified

- Magnolia host kernel `3.10.0` is below Accelerate's recommended Linux kernel. This is a known route risk; job 575546 still completed normally.
- tied `decoder.weight` save warning is expected for tied embeddings. Reload behavior was already proven during prior DDP resume mechanics. No evidence of checkpoint corruption appeared.

## Next learning boundary

Do not spend another remote run merely to prove infrastructure. Continue model teaching/building in Git first:

- repair scheduler semantics (done in tracked code);
- add deterministic held-out MLM evaluation (done in tracked code);
- expand Sol-authored semantic/judgment teaching data;
- preserve step-200 as the parent for the next N0 learning segment;
- choose the next segment length from held-out and training evidence rather than a repeated pilot ritual.

No private E0/E-INF/A-SYN gradient is authorized by this result.
