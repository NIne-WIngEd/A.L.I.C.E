# N0 job 575962 semantic-v2 closure — qualified for one owner launch

**Date:** 2026-09-21  
**Status:** valid P2S model failure preserved; semantic-v2 repair implemented; exact-head static/CPU qualification passed; exact-source Graphify verification passed; Fable trace updated; one owner Magnolia launch may proceed  
**Semantic-v2 branch:** `alice-eipm-v1-n0-closure-semantic-v2`  
**Qualified exact HEAD:** `1c94065cd30abb9764260e239b2227b27d098040`  
**Qualification:** GitHub Actions `35566747300` — SUCCESS  
**Graphify exact-source verification:** workflow `35566838101`, attempt 2 — SUCCESS  
**Fable branch:** `fable-builder-model@032654de6cdbde2f7a533b14f3f052bf47851de6`

## Valid source failure

Magnolia job `575962` ran the previously qualified closure package on exact revision:

`fefc2e4ec54ddde113165b1a3103a1add9be34e8`.

Infrastructure and lineage passed:
- P100/CUDA available;
- exact source revision;
- ratified P1 step-50 checkpoint;
- job-575958 failed-P2 lineage;
- original frozen final validation reused;
- no private identity gradient;
- no post-result threshold/search changes.

The P2S matcher then consumed the full precommitted 1000-step budget and returned:

`FAIL_QSRE_CLOSURE_SCHEMA_MATCHER`

with:
- `selected=null`;
- `p2_authorized=false`;
- no Production P2/P3/P4/final run.

Best P2S observation was step 850:
- auxiliary unseen-holdout relation top-1: `0.5416666666666666`;
- auxiliary seen relation top-1: `0.5625`;
- Production core single-relation top-1: `0.7090909090909091`;
- held-out factor macro accuracy: `0.6666666666666667`.

The run is valid model evidence and remains preserved at:

`$ALICE_N0_WORKDIR/qsre-n0-closure-pass-v1`.

## Why this failure is different from job 575958

Job 575958 showed that the Production-v3 operator could fit TRAIN while failing relation identity/order/open-schema transfer.

P2S was introduced to test semantic-schema generalization before another Production P2.

Job 575962 failed that earlier gate.

The important diagnostic is that seen and unseen auxiliary relations were similarly weak. This does not look like a simple recurrence of the six-relation ontology problem. The matcher was broadly unable to reach the required semantic matching fidelity.

The trajectory plateaued far below the precommitted gates. More steps, another learning rate, a lower threshold or an identical rerun are not authorized.

## Source-level diagnosis

Inspection of the failed P2S implementation localized one semantic-matcher capability/alignment boundary.

### Shallow semantic interaction

The v1 matcher removed relation-ID parameters, but its learned semantic surface was still primarily:
- one shared projection;
- cosine token similarity;
- tokenwise max similarity;
- weighted late interaction.

That is insufficient proof of compositional runtime-schema matching for direction, argument roles, dependency language and closely related relation meanings.

### Encoding-schema mismatch

The v1 auxiliary schema text was encoded as an opaque key plus a short meaning.

Actual Production runtime schemas are encoded through `relation_schema_text(...)` with:
- relation meaning;
- source argument;
- target argument;
- directional/symmetric statement.

Thus the P2S generalization gate and the actual Production runtime schema were not using the same semantic encoding grammar.

### Factor train/evaluation mismatch

The v1 meta-schema already had explicit factor training phrases.

The v1 trainer did not use those phrase banks in gradient. It trained factors from full Production queries, then evaluated on standalone semantic factor instructions.

Semantic-v2 makes the train phrase bank gradient-bearing while retaining Production factor-query supervision.

### Reproducibility defect

The v1 trainer seeded after matcher construction.

Semantic-v2 applies the precommitted seed before matcher construction.

## Semantic-v2 matcher

`src/alice_personality/n0/qsre_schema_matcher.py`

The matcher remains relation/factor identity neutral.

It has:
- zero relation-identity parameters;
- zero factor-identity parameters;
- zero candidate-count parameter axis;
- zero reasoning-step parameter axis;
- no fixed runtime schema cardinality ceiling.

The v2 architecture adds:

### Fixed semantic anchor

A fixed relation-independent projection preserves the frozen N0 semantic geometry.

The learned semantic projection is residual to the anchor, so P2S cannot freely rotate the entire semantic space around its training labels.

### Shared layer mixture

All frozen semantic hidden states are combined through one globally shared layer mixture.

The mixture has no relation, factor, candidate or hop index.

### Bidirectional candidate-conditioned pair refinement

Every runtime candidate performs shared query-to-schema and schema-to-query soft alignment.

Pair refinement composes:
- left semantic state;
- right semantic state;
- elementwise interaction;
- absolute difference.

This refines candidate-specific token evidence and a bounded pair score while retaining the direct semantic anchor.

The same weights apply to:
- auxiliary relations;
- Production relations;
- operator factors;
- unseen runtime schemas;
- every recurrent operator step.

The historical `[B,C,L,T]` query-evidence interface is preserved, so ordered coverage remains intact downstream.

## Semantic-v2 P2S objective

New files:
- `configs/eipm/n0/n0_v02_qsre_closure_schema_meta_v2.json`
- `configs/eipm/n0/n0_v02_qsre_closure_matcher_plan_v2.json`
- `scripts/eipm/n0/train_n0_v02_qsre_closure_matcher_v2.py`

Auxiliary relation schemas now call the exact Production runtime `relation_schema_text(...)` formatter.

Relation keys are metadata only and are not semantic tokens.

Factor schema keys are metadata only.

Factor train phrases are now gradient-bearing.

Held-out factor phrases remain evaluation-only.

Auxiliary seen relation paired-view consistency and Production core paired-view consistency are both gradient-bearing.

Actual Production open-schema meanings remain forbidden from P2S TRAIN:
- `CONFLICTS_WITH`
- `EXEMPLIFIES`
- `PREREQUISITE_FOR`
- `PART_OF`

Final-only meanings remain forbidden:
- `ENABLES`
- `PREVENTS`

The v2 plan keeps the exact v1 P2S:
- seed;
- batch size;
- max steps;
- evaluation cadence;
- learning rate;
- weight decay;
- gradient clip;
- eligibility thresholds.

This is an architecture/objective repair, not a hyperparameter search.

## Downstream architecture remains unchanged

No job-575962 evidence invalidated:
- corrected P1;
- ratified P1 step-50 checkpoint;
- ordered query-evidence coverage;
- continuous relation hypotheses;
- semantic factor schema interface;
- Binder v2 exact sparse structural support;
- Production P2/P3/P4 optimizer and eligibility plan;
- original frozen native final validation.

If semantic-v2 P2S passes, its matcher is loaded into Production P2 and frozen.

Production P2 cannot rewrite the schema matcher.

Binder v2 remains the first exact sparse boundary.

## Governed runner

New runner:

`scripts/eipm/n0/run_n0_v02_qsre_n0_closure_semantic_v2.sh`

Magnolia wrapper:

`scripts/eipm/n0/magnolia_p100_n0_v02_qsre_n0_closure_semantic_v2.sbatch`

Before creating new evidence, the runner requires the preserved v1 closure root and proves its P2S result is the exact valid 575962 failure:
- schema v1;
- `FAIL_QSRE_CLOSURE_SCHEMA_MATCHER`;
- `selected=null`;
- `p2_authorized=false`;
- full 1000 steps;
- exact step-850 best metrics;
- no Production-open or auxiliary-holdout description leakage.

New immutable output root:

`$ALICE_N0_WORKDIR/qsre-n0-closure-semantic-v2`

The stage order remains:

`P2S-v2 -> Production P2 -> P3 -> zero-gradient P4 -> original frozen native N0 validation`.

Every stage fail-closes the next.

## Exact-head qualification

GitHub Actions `35566747300` passed on exact source:

`1c94065cd30abb9764260e239b2227b27d098040`.

It verified:
- Python and shell compilation;
- 52 matcher/operator/binder/core mechanics tests;
- valid 575962 causal authority;
- unchanged v1 P2S training settings and eligibility thresholds;
- Production runtime schema-text grammar alignment;
- factor train/heldout phrase separation;
- Production/final holdout isolation;
- identity-neutral matcher parameter axes;
- fixed semantic anchor;
- residual learned projection;
- shared layer mixture;
- bidirectional pair refinement;
- strict v2 checkpoint/result loading in Production P2;
- matcher freeze before Production P2;
- unchanged downstream Production plan;
- one new evidence root and original frozen final reuse.

## Graphify verification

Graphify v0.9.63 rebuilt from the exact semantic-v2 source worktree.

Published graph metadata:
- source branch: `alice-eipm-v1-n0-closure-semantic-v2`;
- source commit: `1c94065cd30abb9764260e239b2227b27d098040`;
- extraction: `exact-source-worktree`;
- workflow: `35566838101`;
- authority: navigation-only.

The first workflow attempt assembled the code graph successfully but publication fail-closed because `fable-builder-model` moved while the catalog was running.

Attempt 2 refreshed the catalog and completed successfully.

The maximal experiment frontier after refresh is:
1. `alice-eipm-v1-n0-closure-semantic-v2@1c94065...`
2. `alice-eipm-v1-qsre-production-p2-ordered-evidence-v3@342bdbb...`
3. `alice-eipm-v1-qsre-production-p2-failure-localization@c9b1d35...`

Graphify navigation points directly to:
- `QSRESchemaMatcher`;
- `auxiliary_relation_schema_text()`;
- `factor_instruction_losses()`;
- `QSREProductionBinderV2`;
- the preserved ordered-evidence and Production interfaces.

Consequential claims were checked in the original exact-head source, not promoted from Graphify alone.

No source-level bypass of the v2 matcher, semantic relation-key injection, relation/factor identity parameter axis, holdout-description gradient path, loss of ordered coverage, early exact relation sparsity, factor-cache lineage break or frozen-final replacement was found in the qualified path.

The current Magnolia job 575962 output is also the host-specific private runtime evidence used for the failure classification.

## Fable continuity

`fable-builder-model@032654de6cdbde2f7a533b14f3f052bf47851de6`

adds:

`docs/fable-builder/traces/FBM_TRACE_20260921_N0_JOB575962_SEMANTIC_V2.jsonl`

with three material traces:
- valid 575962 P2S failure;
- semantic-boundary diagnosis;
- qualified semantic-v2 architecture.

Fable remains builder-process evidence. It does not claim model success.

## One authorized next action

Exactly one owner-submitted Magnolia P100 semantic-v2 closure run.

No automatic rerun.
No LR search.
No step search.
No width search.
No batch search.
No threshold reduction.
No TEST opening.
No private identity gradient.

If P2S v2 fails, stop before Production P2 and preserve the evidence.

If P2/P3/P4/final fails, preserve the first valid failing stage and stop for causal localization.

N0 is still incomplete.

N0 closes only on the unchanged native final result:

`PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE`

with:

`n0_complete=true`

and:

`n1_authorized=true`.
