# N0 DDP PASS and Real Lineage Start — 2026-09-13

**Status:** active continuation pointer after Magnolia job 575537  
**Supersedes:** the earlier immediate pointer that still asked for the DDP mechanics job

## DDP mechanics gate

Magnolia job `575537` completed successfully on `gpu001` with exit code `0:0` in `00:03:23`.

The two-leg non-promotable mechanics run proved:

- 2 × Tesla P100-PCIE-12GB;
- udocker P2 stage-level execution using `rayan-n0-base`;
- 2-process distributed N0 training;
- 352,184,960 trainable parameters;
- fp16 trainer state;
- step 0 -> 4 PASS;
- 8,192 cumulative tokens at step 4;
- exact resume from the step-4 accelerator/model state;
- step 4 -> 8 PASS;
- 16,384 cumulative tokens at step 8;
- `world_size=2`;
- `ddp_mechanics_receipt.json status=PASS`;
- no private identity data or gradient;
- checkpoint explicitly non-promotable.

Receipt hashes:

- step 4: `ba7490c9b24260159bc78d5f6830ac74f1652a36d2e8ba0df91ca34fc4facfc0`
- step 8: `654e7d7a21a92132eddf8a022790d0e37ec4d7f9554bf4b3415e3611beb1168a`

Public result record:

`docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_DDP_JOB_575537.md`

The DDP/checkpoint-resume mechanics gate is closed. Do not repeat it merely to reconfirm the same behavior.

## Warning disposition

The completed job emitted launcher-default warnings, a legacy-kernel warning, a tied-weight save warning, and a process-group shutdown warning.

Disposition:

- launcher defaults: hardened in tracked launchers by explicitly setting one machine, fp16, and no Dynamo backend;
- process-group shutdown: `train_mlm.py` now calls `accelerator.end_training()` on normal and already-complete exits;
- kernel 3.10 warning: unavoidable host-kernel risk, not a failed qualification; retain bounded/checkpointed runs;
- tied `decoder.weight` warning: empirically non-blocking because leg 2 reloaded the saved state and completed successfully.

## Real N0 lineage begins now

The real N0 corpus/tokenizer lineage has not yet been created and must remain separate from the smoke lineage.

Canonical root:

```text
$HOME/rayan-compute/rayan-n0/n0-v01
```

Smoke artifacts under `p100x2-runtime-smoke-575527` remain qualification-only and must not be copied into `n0-v01`.

The next sequence is:

```text
Magnolia CPU corpus-bootstrap
  -> corpus receipt verification
  -> Magnolia CPU tokenizer
  -> tokenizer receipt
  -> 2xP100 200-step meaningful MLM pilot
  -> inspect throughput/loss/checkpoint behavior
  -> continue/rescale only from evidence
```

## Magnolia CPU preprocessing route

The shared udocker wrapper now supports both CPU and GPU stages. CPU jobs set `RAYAN_UDOCKER_NVIDIA=0` and still use the same validated Debian 12/Python 3.11 user-space container.

Tracked jobs:

- `scripts/eipm/n0/magnolia_cpu_corpus_bootstrap.sbatch`
- `scripts/eipm/n0/magnolia_cpu_tokenizer.sbatch`

Defaults:

- partition `node`;
- 8 CPUs;
- 32 GB RAM;
- real workdir `$HOME/rayan-compute/rayan-n0/n0-v01`;
- HF cache `$HOME/rayan-compute/rayan-n0/hf-cache`;
- bounded corpus target: 100,000,000 accepted characters per configured source;
- 128 MB shards;
- tokenizer: Alice-native 48k byte-complete BPE.

## Current EIPM branch

`alice-eipm-v1-build` after recording the DDP pass, cleaning distributed shutdown, pinning launcher settings, and adding CPU preprocessing jobs:

`0e5469916b5800ad622d459431f2524a223b06f9`

(Branch may advance again as this continuation is committed.)

## Owner-help boundary

The next owner action is to sync `alice-eipm-v1-build` on Magnolia and submit the CPU corpus bootstrap. No GPU allocation and no private-gradient authorization are needed for this step.
