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

Tracked command:

```bash
sbatch scripts/eipm/n0/magnolia_p100x2_runtime_smoke.sbatch
```

Result returned by owner:

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
- branch/head at execution: `alice-eipm-v1-build@22a129784af5a5a456d6e5f5f49f9b3cc48ac386`

Public summary is recorded at:

`docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_SMOKE_JOB_575527.md`

This closes basic target-runtime construction/backprop mechanics. It does not by itself close distributed training/checkpoint-resume mechanics.

## Current build state

`alice-eipm-v1-build` current tip:

- `277745d2cdcc1a7928936d1c5244009450992520`

The branch now additionally contains:

- `train_mlm_p100x2_ddp_mechanics.sh` — a deliberately non-promotable two-leg DDP mechanics test;
- `magnolia_p100x2_ddp_mechanics.sbatch` — the exact 2×P100 job wrapper;
- leg 1 trains only to step 4 and saves distributed accelerator/model/tokenizer state;
- leg 2 resumes from that exact step-4 state and continues to step 8;
- PASS requires world size 2, fp16, correct resume ancestry, and increasing cumulative token count;
- the final `ddp_mechanics_receipt.json` explicitly marks the checkpoint non-promotable and private-data-free.

The meaningful 200-step public N0 pilot remains prepared, but it should not run until the cheaper DDP/resume uncertainty is closed and the real bounded corpus/tokenizer lineage has been prepared.

## Active compute route

The owner does **not** have usable Magnolia A100 access from the actual working account/directory path. Do not plan N0 around A100 unless that access changes later.

Active Magnolia route:

- partition `gpu`
- QOS `normal`
- `--gres=gpu:p100:2`
- observed node `gpu001`
- two Tesla P100-PCIE-12GB devices

Kaggle remains fallback/overflow. Preserve Kaggle quota while Magnolia P100 can perform the same stage.

## FBM state

`fable-builder-model` current tip:

- `17663e5e04f5a2c16e2dc672b20c48e2e9ae167a`

FBM now captures both the compute-route correction and the successful runtime-smoke/next-cheapest-uncertainty pattern. The reusable lesson is: each build-time compute test should eliminate one real uncertainty and immediately target only the next unresolved one.

## Immediate operational pointer

Use the already-passed smoke lineage. Do not rebuild it.

From the A.L.I.C.E. repo root on Magnolia:

```bash
git switch alice-eipm-v1-build
git pull --ff-only
export N0_SMOKE_WORKDIR="$HOME/rayan-compute/alice-n0/p100x2-runtime-smoke-575527"
sbatch --export=ALL,N0_SMOKE_WORKDIR="$N0_SMOKE_WORKDIR" scripts/eipm/n0/magnolia_p100x2_ddp_mechanics.sbatch
```

This next job is intentionally tiny. It exists only to prove 2-process P100 training plus state save/resume. It does not create a promotable N0 checkpoint and does not touch private identity data.

If PASS, build the real bounded public corpus/tokenizer lineage next and then run the 200-step meaningful N0 MLM pilot.

No private E0/E-INF/A-SYN gradient is authorized by this continuation note. That transition remains an explicit owner action. Private curated payloads remain outside public Git.

## Current owner-help boundary

Sol can continue repository design, curriculum construction, result interpretation, failure repair, and branch maintenance independently.

Rayan's next needed contribution is only to submit the tiny 2×P100 DDP mechanics job and return its stdout/stderr plus `ddp_mechanics_receipt.json`. No private-gradient decision is needed yet.
