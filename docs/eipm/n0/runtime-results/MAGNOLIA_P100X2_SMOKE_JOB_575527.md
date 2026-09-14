# Magnolia 2×P100 N0 Runtime Smoke — Job 575527

**Date:** 2026-09-13  
**Branch:** `alice-eipm-v1-build`  
**Git HEAD:** `22a129784af5a5a456d6e5f5f49f9b3cc48ac386`  
**State:** COMPLETED  
**Exit code:** `0:0`  
**Elapsed:** `00:06:59`  
**Node:** `gpu001`

## Result

The tracked Magnolia command

```bash
sbatch scripts/eipm/n0/magnolia_p100x2_runtime_smoke.sbatch
```

completed successfully as SLURM job `575527`.

The runtime smoke reported:

- 2 × Tesla P100 GPUs visible;
- CUDA available;
- N0 dependency/runtime checks passed;
- public smoke corpus materialization and receipt verification passed;
- 48,000-token Alice-native tokenizer training and verification passed;
- native N0 model construction passed;
- actual parameter count: **352,184,960**;
- CUDA forward pass passed;
- CUDA backward pass passed;
- `runtime_smoke_receipt.json` produced with `status=PASS`;
- stderr was empty;
- no private identity data was used;
- no private identity gradient occurred.

## Interpretation

This closes the basic single-process/runtime mechanics question for the proven Magnolia P100 route. It does **not** yet prove distributed training, optimizer/checkpoint state saving, or checkpoint resume across a 2-GPU training process.

The next cheapest evidence-producing step is therefore a tiny 2×P100 distributed mechanics run that intentionally uses the already-created smoke corpus/tokenizer. That run is non-promotable and exists only to verify:

1. 2-process `accelerate`/DDP startup;
2. fp16 training on both P100 devices;
3. gradient accumulation and optimizer stepping;
4. distributed checkpoint/state save;
5. exact resume from the saved state;
6. monotonic cumulative token accounting after resume.

Only after those mechanics pass should the real bounded N0 corpus/tokenizer lineage be built and the first meaningful public N0 checkpoint trained.

## Evidence status

This repository record is based on the owner-returned tracked SLURM result and requested output summary. Raw SLURM stdout/stderr and the private runtime receipt remain outside public Git.
