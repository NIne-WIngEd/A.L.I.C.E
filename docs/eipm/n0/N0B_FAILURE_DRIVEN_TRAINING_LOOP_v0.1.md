# N0B Failure-Driven Training Loop v0.1

**Date:** 2026-09-13  
**Status:** active implementation  
**Private identity gradient:** not authorized by this document

## Purpose

N0B turns the selected Alice-native semantic foundation into a direct build/teach/evaluate/repair loop. Evaluation is part of the build. It is not a separate teacher qualification project.

## Current execution path

```text
corpus-smoke
  -> tokenizer
  -> preflight
  -> corpus
  -> train-mlm
  -> train-curriculum
  -> evaluate-curriculum
  -> inspect concrete failure rows
  -> Sol targeted repair batch
  -> repeat only where failures justify it
```

All stages use `scripts/eipm/n0/run_stage.sh` and default to the private runtime directory `.alice-private/n0` so generated corpora, checkpoints, predictions, and receipts do not enter public Git accidentally.

## Active curriculum

The bootstrap teacher artifact is:

- `training/eipm/n0/sol_curriculum_seed_v0.1.jsonl`
- 60 rows
- owner-authorized Sol teacher origin
- semantics, temporal/causal reasoning, pragmatics, social/emotional interpretation, epistemics, ranking, and ACFP/graph alignment coverage

Its origin manifest remains mandatory. Authorization removes an origin veto; it does not remove integrity or semantic checks.

Before training, the curriculum loader now verifies:

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

## Model scale rule

The current approximately 350M configuration is the first build reference. It is not a 300M-to-400M gate and not a permanent ceiling.

Preflight records:

- actual total parameter count;
- trainable parameter count;
- configured approximate reference count;
- fractional difference from that reference.

It does **not** fail because a future configuration is smaller or larger. Scale changes should follow observed capacity/efficiency evidence.

## Public corpus source smoke verification

On 2026-09-13 the configured Common Pile filtered source names were rechecked against the live Hugging Face dataset pages. The configured Project Gutenberg, pre-1929 books, USGPO, regulations, and Python Enhancement Proposal datasets exist. The sampled dataset viewers/cards expose `text`, `id`, and `metadata` structures consistent with the materializer design; sampled filtered records show `metadata.license = "Public Domain"` where the current configuration requires it.

The runtime still resolves each dataset revision to an exact Hugging Face commit SHA and rejects rows whose explicit license field does not match the allowlist.

## Next compute event

Before a long run, execute the cheap path first:

```bash
bash scripts/eipm/n0/run_stage.sh corpus-smoke
bash scripts/eipm/n0/run_stage.sh tokenizer
N0_PREFLIGHT_BACKWARD=1 bash scripts/eipm/n0/run_stage.sh preflight
```

If those pass on the target runtime, materialize the intended corpus and start a resumable N0 MLM run. Do not spend a long GPU allocation debugging packaging or tokenizer/runtime incompatibilities.

After the first MLM checkpoint:

```bash
bash scripts/eipm/n0/run_stage.sh train-curriculum
bash scripts/eipm/n0/run_stage.sh evaluate-curriculum
```

The resulting failure JSONL is the input to the next Sol teaching pass.
