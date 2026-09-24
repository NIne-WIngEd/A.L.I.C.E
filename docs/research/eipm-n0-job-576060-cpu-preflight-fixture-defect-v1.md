# N0 job 576060 — CPU preflight fixture defect, not model evidence

**Date:** 2026-09-24  
**Source revision:** `ab59c02d9a625b2b376d14ad0d9efd0250b305f3`  
**Magnolia job:** `576060`  
**Observed Slurm state:** `FAILED 2:0`  
**Failed stage:** P40C semantic-operator long-token alignment audit  
**Classification:** qualification-fixture defect; not Magnolia failure; not learned-model failure

## Evidence

Magnolia reached node051 and executed the CPU-only runtime path. The static proof suite completed with zero pytest failures, errors, or skips. The tokenizer stress audit and ordinary semantic-operator evidence-token audit also passed.

The failure occurred before any optimizer, gradient, GPU training, DEV checkpoint selection, or FINAL opening. The P40C receipt reported:

- `gradient=false`
- `optimizer=false`
- `gpu_training_authorized=false`
- `final_opening_authorized=false`
- `final_results_observed=false`

The only errors were:

- `solc_train_relation_schema_0002: decisive evidence did not remain beyond native window: 5 <= 4096`
- `solc_dev_relation_schema_0002: decisive evidence did not remain beyond native window: 5 <= 4096`

stderr was empty.

## Root cause

The semantic-operator long-context row uses base example 2, which is a multi-step ordered relation program. It therefore owns more than one supervised relation-schema evidence span.

The long-context builder selected:

`target_index = spans[0]["candidate_index"]`

and longified only that first supervised relation-schema candidate.

The second supervised relation candidate stayed in its original short form. The exact token-evidence compiler correctly emitted positive evidence for both program steps. The tokenizer audit then correctly saw the unshifted second-step evidence near token 5 and failed.

This is a curriculum/materialization bug. It says nothing about learned model capability because no model-learning step ran.

## Repair

The active full-envelope branch now:

1. longifies every unique relation-schema candidate referenced by a supervised relation-schema evidence span;
2. shifts and revalidates every corresponding span;
3. records all supervised relation candidate indices in the long-context locator;
4. makes the static curriculum audit require that every supervised relation candidate reaches the declared long operating point;
5. makes the real-token audit require locator coverage of every supervised relation candidate and conservatively uses the shortest supervised candidate length;
6. requires step-conditioned factor evidence to remain beyond the native window as well;
7. adds a static regression test covering TRAIN and DEV query, multi-step relation-schema, factor-schema, and step-factor evidence placement;
8. adds that regression test file to the workflow trigger paths so a future change cannot silently bypass CI.

## Anti-loop consequence

Job 576060 does not authorize architecture revision, optimizer tuning, more training steps, model resizing, threshold changes, or a GPU run.

The only justified next execution is one exact-head CPU qualification after the repaired static contract is green. A failure at a later gate must be classified from its own evidence; it must not automatically produce another patch ladder.
