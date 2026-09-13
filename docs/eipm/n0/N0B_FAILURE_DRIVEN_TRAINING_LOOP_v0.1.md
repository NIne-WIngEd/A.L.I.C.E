# N0B Failure-Driven Training Loop v0.1

**Date:** 2026-09-13  
**Status:** active implementation  
**Private identity gradient:** not authorized by this document

## Purpose

N0B turns the selected Alice-native semantic foundation into a direct build/teach/evaluate/repair loop. Evaluation is part of the build. It is not a separate teacher qualification project.

The build now separates three concerns that were previously too easy to blur together:

1. **target-runtime smoke** — prove the intended machine/runtime can materialize the public sample, build the native tokenizer, instantiate the model, and complete a backward pass;
2. **bounded first real N0 foundation run** — build a provenance-bound public corpus and resumable semantic checkpoint without accidentally consuming unbounded storage/compute;
3. **teacher loop** — apply the owner-authorized Sol curriculum, evaluate concrete failures, and generate only the next teaching material the failures justify.

## Current execution path

```text
runtime_smoke.sh
  -> corpus-smoke
  -> tokenizer-smoke
  -> preflight-smoke

then, after target-runtime smoke passes:

corpus-bootstrap
  -> verify-corpus
  -> tokenizer
  -> preflight
  -> train-mlm (resumable)
  -> train-curriculum
  -> evaluate-curriculum
  -> inspect concrete failure rows
  -> Sol targeted repair batch
  -> regression
  -> repeat only while failures justify more teaching or capacity
```

All stages use `scripts/eipm/n0/run_stage.sh` and default to a private runtime directory under `.alice-private/` so generated corpora, checkpoints, predictions, logs, and receipts do not enter public Git accidentally.

## One-command target-runtime smoke

`scripts/eipm/n0/runtime_smoke.sh` is the preferred first command on the intended GPU runtime. It:

- records host, Git head/branch, disk state, Python/package versions, and CUDA devices;
- requires CUDA by default so a large backward preflight is not accidentally run on a login CPU;
- materializes a separate smoke corpus lineage;
- verifies every smoke corpus shard against its receipt;
- trains a separate smoke tokenizer and records its exact input shard hashes;
- constructs the native N0 model and performs the backward preflight;
- writes a hashed `runtime_smoke_receipt.json` on success.

A second smoke run must use a fresh `ALICE_N0_WORKDIR`. This prevents accidental appends or silent reuse of an earlier smoke lineage.

## Public corpus lineage

The public materializer resolves each configured Hugging Face dataset revision to an exact commit SHA. Each accepted row retains source ID, revision, row ID, explicit license, provenance, and source URL metadata where present.

The new corpus verifier requires:

- the active source-config SHA to match the corpus receipt;
- configured source IDs/order to match the receipt;
- every source to have an exact resolved revision and usable accepted text;
- every manifested shard to exist;
- exact shard byte counts and SHA256 hashes to match the receipt;
- `private_identity_data=false`.

Tokenizer training is then bound to those exact shards through `tokenizer_receipt.json`.

## Bounded first real corpus

The first real execution path is `corpus-bootstrap`, not the unbounded `corpus` mode.

By default it accepts at most **100,000,000 normalized characters per configured source**. With the current five-source manifest, this bounds the first accepted-text payload to roughly 500 million characters before JSON/provenance overhead. The value is a bootstrap engineering budget, not a permanent corpus ceiling.

The unbounded `corpus` stage refuses to run unless `N0_ALLOW_UNBOUNDED_CORPUS=1` is explicitly set. This is a storage/compute guardrail rather than a capability restriction.

A corpus target that already contains shards or a receipt is not appended to. Rebuilds use a fresh runtime lineage so receipts remain unambiguous.

## Active curriculum

The bootstrap teacher artifact is:

- `training/eipm/n0/sol_curriculum_seed_v0.1.jsonl`
- 60 rows
- owner-authorized Sol teacher origin
- semantics, temporal/causal reasoning, pragmatics, social/emotional interpretation, epistemics, ranking, and ACFP/graph alignment coverage

Its origin manifest remains mandatory. Authorization removes an origin veto; it does not remove integrity or semantic checks.

Before training, the curriculum loader verifies:

- valid JSONL;
- unique stable IDs;
- explicit competency IDs;
- train/dev split presence;
- supported task type;
- non-empty prompt/source/candidates;
- at least two candidates;
- valid and non-duplicated preferred indices.

## Multi-valid-answer evaluation

Some EIPM-relevant situations legitimately permit more than one answer. Exact numerical score equality is not a useful requirement.

N0 therefore tracks two distinct signals:

1. **top-1 accuracy** — whether the highest-scoring candidate is one of the supported candidates;
2. **supported-set separation** — whether every supported candidate scores above every unsupported candidate.

The second signal prevents a row with two valid alternatives from looking solved merely because one valid option happened to rank first while another valid option was incorrectly pushed below a bad option.

The evaluator also records the minimum supported-vs-unsupported separation margin. Negative margins become direct repair signals.

## Per-competency diagnosis

Development metrics are emitted separately for each competency represented in the active dev split. This lets Sol repair the actual weakness instead of generating another broad curriculum round.

`evaluate-curriculum` writes:

- `dev_metrics.json` (or the requested split);
- `dev_predictions.jsonl`;
- `dev_failures.jsonl`;
- `dev_evaluation_receipt.json`.

A failure row is emitted when top-1 is unsupported or the complete supported set is not separated from unsupported candidates.

## Resumable MLM checkpoint lineage

Every saved N0 MLM step now stores Accelerator state plus a receipt binding the checkpoint to:

- model ID and config SHA256;
- exact tokenizer SHA256 and tokenizer-receipt SHA256;
- exact public corpus-receipt and source-config SHA256;
- verified corpus shard count/bytes and the exact training shard set;
- seed, sequence length, precision, world size, CUDA device names, and code Git revision;
- total/trainable parameter counts;
- cumulative tokens seen;
- parent checkpoint when resumed;
- SHA256 maps for saved model and tokenizer artifacts;
- `private_identity_gradient=false`.

`N0_RESUME_FROM=/path/to/step-XXXXXXXX` resumes the optimizer/scheduler/model runtime state instead of restarting the first public foundation run after an allocation boundary.

This is intentionally lightweight: enough information to reproduce and audit the gradient lineage without constructing an MC10-style independent validation project.

## Model scale rule

The current approximately 350M configuration is the first build reference. It is not a 300M-to-400M gate and not a permanent ceiling.

Preflight records actual and planned parameter counts but does **not** reject a future configuration merely because it is smaller or larger. Scale changes follow observed semantic/identity failures and efficiency evidence.

## Public corpus source verification

On 2026-09-13 the configured Common Pile filtered source names were rechecked against the live Hugging Face dataset pages. The configured Project Gutenberg, pre-1929 books, USGPO, regulations, and Python Enhancement Proposal datasets exist. Sampled filtered records expose the `text`, `id`, and `metadata` structure expected by the materializer and show `metadata.license = "Public Domain"` where the current configuration requires it.

Runtime remains the authority for the actual training artifact: the materializer resolves an exact source revision and rejects rows whose explicit license field does not match the allowlist.

## Next compute event

Run the cheap target-runtime gate first, inside the actual GPU environment:

```bash
bash scripts/eipm/n0/runtime_smoke.sh
```

If it passes, the first real bounded path is:

```bash
bash scripts/eipm/n0/run_stage.sh corpus-bootstrap
bash scripts/eipm/n0/run_stage.sh verify-corpus
bash scripts/eipm/n0/run_stage.sh tokenizer
N0_PREFLIGHT_BACKWARD=1 bash scripts/eipm/n0/run_stage.sh preflight
bash scripts/eipm/n0/run_stage.sh train-mlm
```

After the first usable MLM checkpoint:

```bash
bash scripts/eipm/n0/run_stage.sh train-curriculum
bash scripts/eipm/n0/run_stage.sh evaluate-curriculum
```

The resulting failure JSONL is the input to the next Sol teaching pass.

No private E0/E-INF/A-SYN gradient is authorized by this document or by N0 runtime success. That remains a later explicit owner transition.
