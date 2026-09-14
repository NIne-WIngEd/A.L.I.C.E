# N0 Build Continuation and Owner Authorization — 2026-09-13

**Status:** active continuation pointer  
**Purpose:** supersede stale “research-only / teacher qualification” next-action language without deleting historical context.

## Owner correction retained

The owner explicitly states that the project has permission to use Sol/ChatGPT-produced teaching material for A.L.I.C.E. and Fable model development. The Sol teaching seeds must remain part of the governed training lineage rather than being removed because of a generic hosted-model-origin assumption.

A.L.I.C.E. and the OpenAI assistant are not treated as the same system. The EIPM remains an A.L.I.C.E.-native identity/judgment component with its own architecture, weights, evidence authority, memory boundaries, and runtime role.

Sol is the primary development-time teacher for general semantic, pragmatic, social, emotional, epistemic, ranking, and structured-alignment competencies. Canonical Elaina evidence remains the authority for Elaina-specific identity targets. Rayan remains final authority on fidelity.

## Mainstreamed build rule

Do not recreate MC10 as a teacher/judge qualification bureaucracy. Build A.L.I.C.E. directly. Keep integrity, provenance, held-out evaluation, regression, and source-authority checks inside the actual build loop. Escalate only when a concrete failure or ambiguity warrants it.

The active loop is:

```text
public semantic corpus
  -> Alice-native tokenizer
  -> Alice-native random-init N0 semantic model
  -> owner-authorized Sol curriculum
  -> candidate-ranking / judgment training
  -> per-competency + row-level failure evaluation
  -> targeted Sol repair curriculum
  -> regression
  -> repeat only while failures justify more teaching or capacity
  -> N1 governed Elaina identity learning after explicit private-gradient authorization
```

## Parameter scale correction

The approximately 350M N0 configuration is an initial engineering reference, not a global model-size requirement or ceiling. The first real instantiation has now been measured at **352,184,960 parameters** on the actual P100 runtime. Scale changes remain evidence-driven.

## Successful Magnolia P100 smoke

Tracked command used for qualification:

```bash
sbatch scripts/eipm/n0/magnolia_p100x2_runtime_smoke.sbatch
```

Authoritative successful result returned by owner:

- job: `575527`
- state: `COMPLETED`
- exit: `0:0`
- elapsed: `00:06:59`
- node: `gpu001`
- 2 × Tesla P100 visible
- CUDA available
- 48k tokenizer trained and verified
- native N0 construction PASS
- forward/backward preflight PASS
- actual parameter count: 352,184,960
- `runtime_smoke_receipt.json`: `status=PASS`
- stderr empty
- execution branch/head: `alice-eipm-v1-build@22a129784af5a5a456d6e5f5f49f9b3cc48ac386`

Public summary:

`docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_SMOKE_JOB_575527.md`

The runtime-smoke gate is closed. Do not rerun it merely to reconfirm the environment.

## Magnolia runtime discoveries retained

The working account has a legacy CentOS 7 / glibc 2.17 host environment. Default host `python` is Python 2.7.5. Although a Python 3.11 module exists, the modern PyTorch CUDA runtime required by N0 is not a valid native fit for that glibc baseline.

Native pip/Conda retries and building PyTorch from source are not the active path.

Validated user-space route:

- udocker 1.3.17;
- execution mode P2;
- Debian 12 container;
- Python 3.11.16 inside the container;
- glibc 2.36 inside the container;
- PyTorch `2.7.1+cu118`;
- two P100 devices;
- container name after owner namespace cleanup: `rayan-n0-base`.

NVIDIA bindings must be refreshed on the allocated compute node with `udocker setup --nvidia --force` before the stage starts.

Two failed tracked attempts are useful negative evidence:

- `575524` failed because placing a fake Python shim first in `PATH` caused udocker to recursively resolve the same shim through `/usr/bin/env python`, eventually exhausting process creation;
- `575525` separated the shim successfully and reached Python 3.11 plus both P100s, but wrapping each Python invocation in a fresh udocker process failed during `corpus-smoke` with an Intel oneMKL / `libtorch_cpu.so` load failure.

The stable rule is therefore:

> Run the whole N0 stage inside one udocker session. Do not proxy individual Python calls through udocker.

External wrapper job `575523` first proved that boundary. Tracked job `575527` reproduced it successfully.

## Owner-private Magnolia namespace

The owner requested that Magnolia-visible work identify Rayan rather than exposing `alice` names to peers, while all work remains inside the owner's directory.

Canonical Magnolia layout:

```text
$HOME/rayan-compute/
├── rayan-eipm-main/
├── rayan-n0/
├── udocker-store/
├── udocker-tmp/
└── tools/
```

Canonical paths/names:

- repository clone: `$HOME/rayan-compute/rayan-eipm-main`
- runtime tree: `$HOME/rayan-compute/rayan-n0`
- udocker container: `rayan-n0-base`
- future scheduler/output namespace: `rayan-n0-*`

The owner verified `rayan-compute`, `rayan-eipm-main`, `rayan-n0`, `udocker-store`, and `udocker-tmp` as owner-only (`drwx------`). A search under `$HOME` found no remaining `A.L.I.C.E-main`, `ALICE-main`, or `alice-main` clone.

Important distinction: external Magnolia directory/container/job/output names use `rayan`; tracked internal project identifiers and interfaces such as `ALICE_N0_WORKDIR`, the repository identity, provenance labels, and A.L.I.C.E. source names remain unchanged.

## Current build state

`alice-eipm-v1-build` current tip after this Magnolia hardening pass:

- `fbee8a5b88a2be21cb1967a19c81abde0716af6d`

The branch contains:

- `train_mlm_p100x2_ddp_mechanics.sh` — deliberately non-promotable two-leg DDP mechanics test;
- `magnolia_p100x2_ddp_mechanics.sbatch` — exact 2×P100 DDP mechanics wrapper;
- `magnolia_udocker_exec.sh` — shared whole-stage udocker execution boundary;
- udocker-routed runtime smoke, DDP mechanics, and bounded MLM pilot sbatch wrappers;
- Magnolia-visible `rayan-n0-*` SLURM/output naming;
- `$HOME/rayan-compute/rayan-n0/...` default runtime paths;
- runtime-result documentation containing the failure chronology and validated environment.

DDP mechanics design retained:

- leg 1 trains only to step 4 and saves distributed accelerator/model/tokenizer state;
- leg 2 resumes from that exact step-4 state and continues to step 8;
- PASS requires world size 2, fp16, correct resume ancestry, and increasing cumulative token count;
- final `ddp_mechanics_receipt.json` marks the checkpoint non-promotable and private-data-free.

The meaningful 200-step public N0 pilot remains prepared, but it should not run until the cheaper DDP/resume uncertainty is closed and the real bounded corpus/tokenizer lineage has been prepared.

## Active compute route

Do not assume Magnolia A100 access from this working account unless access is explicitly revalidated later.

Active proven route:

- partition `gpu`
- QOS `normal`
- `--gres=gpu:p100:2`
- observed node `gpu001`
- two Tesla P100-PCIE-12GB devices

Kaggle remains fallback/overflow. Preserve Kaggle quota while Magnolia P100 can perform the same stage.

## FBM state

`fable-builder-model` current tip after recording this chat's infrastructure lessons:

- `73b6575b5ceb5edc4689c70a45d4d338dc03b430`

FBM now retains three linked builder lessons around this route: adapt to the real owner-available hardware; advance compute in the cheapest evidence-producing increments; and place containerization at the validated stage boundary rather than repeatedly wrapping interpreter calls.

## Immediate operational pointer

The local Magnolia clone was intentionally renamed and will be behind GitHub until pulled. Do not recreate the old `A.L.I.C.E-main` directory.

From Magnolia:

```bash
cd "$HOME/rayan-compute/rayan-eipm-main"
git checkout alice-eipm-v1-build
git pull --ff-only
```

The successful smoke lineage now lives under the renamed private runtime tree:

```bash
export N0_SMOKE_WORKDIR="$HOME/rayan-compute/rayan-n0/p100x2-runtime-smoke-575527"
```

After pulling the hardened build branch, the tracked DDP wrapper itself handles the validated udocker boundary. The temporary `BASH_ENV` compatibility hook used to qualify job 575527 is historical evidence and is no longer the preferred launch mechanism.

Submit the next tiny mechanics test with:

```bash
sbatch --export=ALL,N0_SMOKE_WORKDIR="$N0_SMOKE_WORKDIR" \
  scripts/eipm/n0/magnolia_p100x2_ddp_mechanics.sbatch
```

This next job exists only to prove two-process P100 training plus state save/resume. It does not create a promotable N0 checkpoint and does not touch private identity data.

If PASS, build the real bounded public corpus/tokenizer lineage next and then run the 200-step meaningful N0 MLM pilot.

No private E0/E-INF/A-SYN gradient is authorized by this continuation note. That transition remains an explicit owner action. Private curated payloads remain outside public Git.

## Current owner-help boundary

Sol can continue repository design, curriculum construction, result interpretation, failure repair, and branch maintenance independently.

Rayan's next needed contribution is only to sync the hardened branch to Magnolia and submit the tiny 2×P100 DDP mechanics job when ready. No private-gradient decision is needed yet.
