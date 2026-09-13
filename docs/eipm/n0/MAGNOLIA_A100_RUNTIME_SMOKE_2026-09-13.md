# Magnolia A100 N0 Runtime Smoke — 2026-09-13

**Status:** ready for owner execution  
**Purpose:** prove the real Magnolia GPU/runtime path before spending a long N0 allocation  
**Private identity data/gradient:** none

## Known schedulable route

Earlier A.L.I.C.E. Magnolia route qualification established a working A100 request with:

- partition: `suliaoma`
- QOS: `normal`
- resource: `gpu:a100:1`
- previously schedulable A100 node: `gpu003`

The repository therefore includes:

`scripts/eipm/n0/magnolia_a100_runtime_smoke.sbatch`

It intentionally does not add an account directive or a site-specific module command that has not already been established.

## Before submission

Use the A.L.I.C.E. checkout that contains the current `alice-eipm-v1-build` branch. Activate the Python/Conda environment that should be used for N0 before calling `sbatch`; SLURM inherits the submitted environment.

The smoke script checks the required N0 packages itself:

- torch
- transformers
- tokenizers
- accelerate
- safetensors
- datasets
- huggingface_hub

It also refuses to treat a CPU-only runtime as a passing Magnolia GPU smoke.

## Submit

From the A.L.I.C.E. repository root on Magnolia:

```bash
git switch alice-eipm-v1-build
git pull --ff-only
sbatch scripts/eipm/n0/magnolia_a100_runtime_smoke.sbatch
```

The job derives the repository root from `SLURM_SUBMIT_DIR`, so it must be submitted from the repository root.

By default each job receives a fresh private work directory:

```text
$HOME/rayan-compute/alice-n0/runtime-smoke-$SLURM_JOB_ID
```

The job is limited to one hour. It is a smoke/preflight allocation, not the long semantic-foundation run.

## What the job executes

The job calls `runtime_smoke.sh`, which performs:

1. environment/Git/disk/CUDA capture;
2. dependency import/version check;
3. public `corpus-smoke` materialization;
4. exact corpus-receipt/shard verification;
5. separate smoke tokenizer training plus tokenizer receipt;
6. native N0 model construction;
7. CUDA forward/backward preflight;
8. final hashed `runtime_smoke_receipt.json`.

No private curated E0/E-INF/A-SYN package is read or required.

## Observe

Useful scheduler commands:

```bash
squeue -u "$USER"
```

The SLURM stdout/stderr files are written in the repository submit directory as:

```text
alice-n0-smoke-<jobid>.out
alice-n0-smoke-<jobid>.err
```

The durable smoke receipt and detailed logs remain under the job-specific private work directory printed in stdout.

## After PASS

A successful smoke is permission to continue the **public N0** execution path, not permission for a private Elaina gradient.

The next build sequence is:

```bash
bash scripts/eipm/n0/run_stage.sh corpus-bootstrap
bash scripts/eipm/n0/run_stage.sh verify-corpus
bash scripts/eipm/n0/run_stage.sh tokenizer
N0_PREFLIGHT_BACKWARD=1 bash scripts/eipm/n0/run_stage.sh preflight
bash scripts/eipm/n0/run_stage.sh train-mlm
```

Long-run resource sizing can then be chosen from the observed A100 runtime/VRAM behavior instead of guessed in advance.
