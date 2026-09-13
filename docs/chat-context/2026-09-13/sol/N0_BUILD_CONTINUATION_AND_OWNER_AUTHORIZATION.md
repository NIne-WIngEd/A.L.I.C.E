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

The approximately 350M N0 configuration is an initial engineering reference, not a global model-size requirement or ceiling. Scale changes are driven by observed capability or efficiency evidence.

## Current build state

`alice-eipm-v1-build`:

- current tip after the P100 correction/build pass: `22a129784af5a5a456d6e5f5f49f9b3cc48ac386`

The branch contains a directly executable N0 pipeline:

- the 60-row owner-authorized Sol curriculum is active and provenance-bound;
- curriculum rows receive structural validation before training;
- native tokenizer loading is explicit from `tokenizer.json`;
- candidate evaluation distinguishes top-1 correctness from supported-set separation;
- development metrics are emitted per competency;
- `evaluate-curriculum` emits concrete failed rows for the next Sol repair batch;
- public corpus receipts bind active source config, exact resolved source revisions, shard sizes, and SHA256 values;
- tokenizer receipts bind the tokenizer to the exact corpus lineage;
- MLM checkpoints save Accelerator state and are resumable across allocation boundaries;
- checkpoint receipts bind config/tokenizer/corpus/code/runtime lineage and cumulative token counts;
- smoke corpus/tokenizer artifacts are separated from real training lineage;
- the first real corpus path is bounded by default (`corpus-bootstrap`);
- runtime smoke now supports a required minimum CUDA-device count;
- the stale Magnolia A100 execution artifacts were removed after the owner corrected the access assumption;
- the active Magnolia route is the previously proven 2×P100 route;
- an explicit 2×P100 distributed MLM launcher and bounded 200-step pilot job are ready.

## Active compute route

The owner does **not** have usable Magnolia A100 access from the actual working account/directory path. Do not plan N0 around A100 unless that access changes later.

The previously proven usable Magnolia route is:

- partition `gpu`
- QOS `normal`
- `--gres=gpu:p100:2`
- observed node `gpu001`
- two Tesla P100-PCIE-12GB devices

The P100 smoke verifies both the device count and device family before N0 preflight.

Kaggle remains the fallback/overflow route. Preserve Kaggle quota when Magnolia P100 can perform the same stage. Use Kaggle if Magnolia is blocked, materially too slow, or incompatible with a required runtime.

## FBM state

`fable-builder-model`:

- current tip after the compute-route correction trace: `53ccb5bfac576b07b974f79c2bd7367ec5ac4d2a`

FBM retains:

- the exact owner-authorized Sol bootstrap seed;
- the failure-driven teaching-loop builder event;
- bounded-compute/lineage lessons;
- the infrastructure-correction lesson: owner/device reality outranks an idealized accelerator assumption, and a future builder must adapt the training plan to the actually available machine.

## Immediate operational pointer

Repository-side preparation has reached the real target-runtime boundary.

From the A.L.I.C.E. repo root on Magnolia, with the intended N0 Python/Conda environment active:

```bash
git switch alice-eipm-v1-build
git pull --ff-only
sbatch scripts/eipm/n0/magnolia_p100x2_runtime_smoke.sbatch
```

The smoke performs no private identity training. It verifies the 2×P100 route, public corpus smoke, tokenizer, native model construction, and CUDA backward pass.

After a clean smoke, prepare the bounded public corpus/tokenizer in the persistent N0 workdir and submit:

```bash
sbatch scripts/eipm/n0/magnolia_p100x2_mlm_pilot.sbatch
```

The default pilot is intentionally bounded to 200 optimizer steps, sequence length 512, micro-batch 1 per process, gradient accumulation 16, two P100 processes, fp16, and checkpoints at steps 100 and 200. Those defaults can be changed from observed evidence instead of guesswork.

No private E0/E-INF/A-SYN gradient is authorized by this continuation note. That transition remains an explicit owner action. Private curated payloads remain outside public Git.

## Current owner-help boundary

Sol can continue repository design, teaching-curriculum construction, failure analysis, and branch maintenance independently.

The current piece Sol cannot execute from chat is the real Magnolia GPU allocation. Rayan's next needed contribution is only to submit the 2×P100 smoke job and return its stdout/stderr/receipt. No private-gradient decision is needed yet.
