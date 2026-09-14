# Magnolia 2×P100 N0 Runtime Smoke — Job 575527

**Date:** 2026-09-13  
**Execution branch:** `alice-eipm-v1-build`  
**Execution Git HEAD:** `22a129784af5a5a456d6e5f5f49f9b3cc48ac386`  
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

## Magnolia runtime discoveries

The successful run closed a sequence of environment-specific failures that must not be rediscovered by future operators.

### Host runtime is not a valid N0 Python environment

Magnolia's login/compute host stack is CentOS 7 with glibc 2.17. The default host `python` is Python 2.7.5. A module Python 3.11 exists, but the required modern PyTorch CUDA wheels target a newer glibc baseline. Native pip/Conda-style attempts therefore do not provide the validated N0 runtime.

Do not rebuild PyTorch from source for this path and do not spend GPU allocations retrying the native host environment.

### Validated user-space runtime

The working route is udocker in P2 mode with a Debian 12 / Python 3.11 container. The validated container name on Magnolia is `rayan-n0-base`. The successful runtime used:

- Python 3.11.16 inside the container;
- glibc 2.36 inside the container;
- PyTorch `2.7.1+cu118`;
- `transformers 5.17.0`;
- `tokenizers 0.23.2`;
- `accelerate 1.15.0`;
- `datasets 4.8.5`;
- `huggingface-hub 1.31.0`;
- `safetensors 0.8.0`;
- 2 × Tesla P100-PCIE-12GB.

The NVIDIA binding must be refreshed on the allocated compute node with `udocker setup --nvidia --force` before the container workload starts.

### Failed compatibility attempts and lesson

Two failed tracked runs are retained only as engineering evidence:

- job `575524` failed because a fake `python` shim was put first in `PATH`. udocker itself resolved that shim through `/usr/bin/env python`, recursively re-entered udocker, and exhausted process creation;
- job `575525` proved the separated Python shim could reach Python 3.11 and both P100s, but starting a fresh udocker process for each Python invocation failed during `corpus-smoke` with an Intel oneMKL / `libtorch_cpu.so` load failure.

The robust boundary is therefore: **launch the whole N0 stage inside one udocker session**. Do not proxy individual Python calls through udocker.

Job `575523` had already demonstrated that whole-stage container boundary successfully through an external wrapper. Job `575527` then reproduced the same architecture while invoking the tracked runtime-smoke sbatch path.

## Repository-side fix after qualification

After the successful smoke, `alice-eipm-v1-build` was updated so the Magnolia wrappers encode the discovered runtime boundary directly instead of depending on the temporary shell hook used during qualification.

The branch now contains:

- `scripts/eipm/n0/magnolia_udocker_exec.sh` — shared one-container execution boundary;
- an udocker-routed `magnolia_p100x2_runtime_smoke.sbatch`;
- an udocker-routed `magnolia_p100x2_ddp_mechanics.sbatch`;
- an udocker-routed `magnolia_p100x2_mlm_pilot.sbatch`.

Magnolia-visible SLURM job names and stdout/stderr names now use the `rayan-n0-*` namespace. Internal software interfaces such as `ALICE_N0_WORKDIR` remain unchanged for compatibility.

## Private Magnolia namespace

Owner-side Magnolia artifacts were consolidated under the private tree:

```text
$HOME/rayan-compute/
├── rayan-eipm-main/
├── rayan-n0/
├── udocker-store/
├── udocker-tmp/
└── tools/
```

The repository clone is `$HOME/rayan-compute/rayan-eipm-main`. The runtime tree is `$HOME/rayan-compute/rayan-n0`. The validated container is `rayan-n0-base`.

The owner verified the top-level compute directory, repository, runtime directory, udocker store, and udocker tmp as owner-only (`drwx------`). A home-directory search found no remaining `A.L.I.C.E-main`, `ALICE-main`, or `alice-main` clone.

This naming/privacy rule applies to Magnolia-facing directories, containers, SLURM names, and output filenames. It does not rename tracked A.L.I.C.E. project identifiers or provenance interfaces.

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

This repository record is based on owner-returned SLURM output and runtime receipts. Raw SLURM stdout/stderr and private runtime artifacts remain outside public Git.

The runtime-smoke gate is closed. Do not rerun it merely to reconfirm the environment. Future compute should target the next unresolved mechanics or meaningful training stage.
