# Magnolia 2×P100 N0 Runtime Smoke — 2026-09-13

**Status:** ready for owner execution  
**Purpose:** prove the real Magnolia N0 runtime on the route the owner actually has  
**Private identity data/gradient:** none

## Qualified route

The previously proven Magnolia route is:

- partition: `gpu`
- QOS: `normal`
- GRES: `gpu:p100:2`
- observed node: `gpu001`
- observed devices: 2 × Tesla P100-PCIE-12GB

The earlier A100 route is not part of the active N0 plan because the owner does not have usable A100 access from the actual working directory/account path. The active script is therefore:

`scripts/eipm/n0/magnolia_p100x2_runtime_smoke.sbatch`

## Submit

From the A.L.I.C.E. repository root on Magnolia, with the intended N0 Python/Conda environment active:

```bash
git switch alice-eipm-v1-build
git pull --ff-only
sbatch scripts/eipm/n0/magnolia_p100x2_runtime_smoke.sbatch
```

The job requests the same 2×P100 resource shape that was physically qualified earlier. It requires two CUDA devices and verifies that the first two visible GPU names contain `P100` before running the N0 smoke.

## What it checks

The generic N0 smoke then verifies:

1. Python/package imports;
2. CUDA availability and at least two visible CUDA devices;
3. public corpus smoke materialization;
4. corpus receipt and shard hashes;
5. smoke tokenizer construction and tokenizer receipt;
6. native N0 model construction;
7. CUDA forward/backward preflight;
8. final hashed runtime smoke receipt.

No private Elaina payload is read. No E0/E-INF/A-SYN gradient occurs.

## Output

SLURM output files:

```text
alice-n0-p100-smoke-<jobid>.out
alice-n0-p100-smoke-<jobid>.err
```

Private work directory:

```text
$HOME/rayan-compute/alice-n0/p100x2-runtime-smoke-<jobid>
```

Final receipt:

```text
runtime_smoke_receipt.json
```

## After PASS

P100 is the primary free N0 execution route. Kaggle remains the fallback/overflow route when Magnolia is blocked, too slow for a particular stage, or cannot support the required runtime. Kaggle quota should not be consumed first when Magnolia can perform the same work.

After the smoke passes, the next step is a short 2×P100 public MLM pilot. Its purpose is to measure real memory use, throughput, stability, and checkpoint/resume behavior before committing to a longer N0 run.
