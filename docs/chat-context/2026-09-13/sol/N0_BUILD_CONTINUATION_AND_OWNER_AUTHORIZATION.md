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

The approximately 350M N0 configuration is an initial engineering reference, not a global model-size requirement or ceiling. The obsolete 300M-to-400M preflight/test gate is removed. Scale changes are driven by observed capability or efficiency evidence.

## Current build state

`alice-eipm-v1-build`:

- tip: `507fbe1d266a3e74455c5b668d05188fb236b25e`

The branch now contains a directly executable N0 pipeline rather than only a research plan:

- the 60-row owner-authorized Sol curriculum is active and provenance-bound;
- curriculum rows receive structural validation before training;
- native tokenizer loading is explicit from `tokenizer.json`;
- candidate evaluation distinguishes top-1 correctness from supported-set separation;
- development metrics are emitted per competency;
- `evaluate-curriculum` emits concrete failed rows for the next Sol repair batch;
- the approximately 350M size is a starting reference rather than a hard preflight envelope;
- public corpus receipts bind active source config, exact resolved source revisions, every shard size, and every shard SHA256;
- tokenizer receipts bind the tokenizer to the exact corpus shard hashes used to train it;
- MLM checkpoints save Accelerator state and are resumable across allocation boundaries;
- checkpoint receipts bind config/tokenizer/corpus/code/runtime lineage and cumulative token counts;
- smoke corpus/tokenizer artifacts are physically separated from the real training lineage;
- the first real corpus path is bounded by default (`corpus-bootstrap`) instead of silently materializing an unbounded corpus;
- a one-command target-runtime smoke performs dependency/CUDA/data/tokenizer/forward/backward checks and emits a hashed PASS receipt;
- a Magnolia A100 SLURM smoke job and route document are ready.

The known Magnolia A100 scheduler route encoded in the smoke job is:

- partition `suliaoma`
- QOS `normal`
- `--gres=gpu:a100:1`

The job is intentionally submitted from the repo root and derives the checkout from `SLURM_SUBMIT_DIR`; no unverified account or module directive is invented.

## FBM state

`fable-builder-model`:

- tip: `5ac6a343ead63131abc2f9fa25e5808e9f7b8b85`

FBM now retains:

- the exact owner-authorized Sol bootstrap seed;
- the failure-driven teaching-loop builder event;
- the bounded-compute/lineage builder event covering smoke-vs-real separation, bounded first corpus, corpus/tokenizer/checkpoint ancestry, resumability, and target-runtime fail-fast behavior.

This preserves the construction knowledge, not just the final Alice artifacts.

## Immediate operational pointer

Repository-side preparation has reached the first actual target-runtime execution boundary.

The cheapest next action is the Magnolia A100 smoke. From the A.L.I.C.E. repo root on Magnolia, with the intended N0 Python/Conda environment active:

```bash
git switch alice-eipm-v1-build
git pull --ff-only
sbatch scripts/eipm/n0/magnolia_a100_runtime_smoke.sbatch
```

The smoke job performs no private identity training. It downloads/materializes only the public smoke corpus, verifies it, builds the smoke tokenizer, constructs N0, and performs a CUDA backward preflight.

After PASS, continue the first bounded public N0 execution path:

```text
corpus-bootstrap
  -> verify-corpus
  -> tokenizer
  -> preflight
  -> train-mlm
  -> train-curriculum
  -> evaluate-curriculum
  -> targeted Sol repair
```

No private E0/E-INF/A-SYN gradient is authorized by this continuation note. That transition remains an explicit owner action. Private curated payloads remain outside public Git.

## Current owner-help boundary

Sol can continue repository design, teaching-curriculum construction, failure analysis, and branch maintenance independently.

The current piece Sol cannot execute from chat is the real Magnolia GPU job. Rayan's next needed contribution is therefore only to submit the A100 smoke job (or provide its stdout/stderr/receipt after it runs). No private-gradient decision is needed yet.
