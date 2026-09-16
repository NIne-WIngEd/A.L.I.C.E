# Fusion v0.2 Validator Overconstraint + Corrected Constraint Gate v0.3 — 2026-09-16

## Observed v0.2 result

Job 575622 completed cleanly with no gradient. The preselected source-anchored repair-step240 candidate remained hash `4d51494beb788f74ddc03590da05ea00f0e36574294649c2cd5d438f9472e577`.

Challenge v0.2 failed its frozen gate:
- macro exact-routing similarity: 0.9373158878 vs 0.94
- worst-family exact-routing similarity: 0.8328953013 vs 0.90
- decisive top-view accuracy: 1.0
- fused semantic cosine: 0.9412938865
- disagreement geometry MAE: 0.0363150912
- missing-view max weight: 0.0
- source summary anchor error: 0.0
- source token anchor error: 0.0

Worst exact-routing families were `consensus_one_stale` and `lexical_distractor_low_authority`.

v0.2 remains immutable and failed. It is retired for ratification and its rows/templates/entities remain forbidden for training.

## Post-result validator diagnosis

Inspection found that v0.2 judged routing by L1 similarity to hand-authored exact soft percentages. This was over-specific for a latent routing mechanism.

Examples:
- consensus case required approximately `0.34/0.33/0.33` although all views support the same current conclusion;
- lexical-distractor case required `0.08/0.48/0.44` although either reliable structured/evidence view may legitimately dominate while stale prose remains suppressed.

Therefore v0.2 mixed genuine capability evaluation with arbitrary internal-percentage matching. This violates the owner doctrine that validation must not become an artificial limitation.

No model gradient is authorized from this finding. The correct order is validator correction first, then evaluate the unchanged model.

## New authoritative routing doctrine

Build branch document:
`docs/eipm/EIPM_FUSION_ROUTING_VALIDATION_DOCTRINE_2026-09-16.md`

Exact routing proportions are not ground truth unless a separately grounded probabilistic source model exists.

Valid routing constraints include:
- allowed top-view set;
- minimum/maximum per-view mass;
- pairwise dominance margin;
- minimum/maximum set mass;
- unavailable-view zero mass;
- query-conditioned reversals such as current vs historical state.

Executable implementation:
`src/alice_personality/n0/cross_context_fusion_routing_constraints.py`

Tests:
`tests/eipm/test_n0_cross_context_fusion_routing_constraints.py`

If future fusion training is necessary, use constraint/set-valued routing supervision instead of arbitrary exact-distribution cross entropy unless calibrated probabilities exist.

## Frozen challenge v0.3

The same repair-step240 model is the only candidate. Its weights have not changed since v0.2.

Config:
`configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.3.json`

Builder:
`scripts/eipm/n0/build_n0_v02_cross_context_fusion_frozen_challenge_v0_3.py`

Evaluator:
`scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge_v0_3.py`

Preparation runner:
`scripts/eipm/n0/run_n0_v02_cross_context_fusion_frozen_challenge_v0_3_prepare.sh`

Evaluation runner:
`scripts/eipm/n0/run_n0_v02_cross_context_fusion_frozen_challenge_v0_3_eval.sh`

Magnolia launcher:
`scripts/eipm/n0/magnolia_p100_n0_v02_cross_context_fusion_frozen_challenge_v0_3.sbatch`

Coverage: 160 rows, 20 families, 8 rows/family. New Atlas/Beacon/Crux/Delta/Ember/Fable/Glyph/Helix entities and new templates. Historical challenge/training row IDs are checked for overlap during freeze.

Frozen v0.3 numeric gates before any result:
- overall routing constraint pass >= 0.95
- worst-family routing constraint pass >= 0.875 (7/8)
- singleton required-top accuracy >= 0.95
- overall fused semantic cosine >= 0.90
- worst-family fused semantic cosine >= 0.86
- exact source summary anchor error == 0
- exact source token anchor error == 0
- disagreement geometry MAE <= 0.12
- missing-view max weight <= 1e-6
- mean routing-constraint violation <= 0.01

The compatibility `target_view_distribution` field exists only because the frozen parent-cache builder expects it. v0.3 explicitly marks it `parent_cache_compatibility_only_not_used_for_ratification`.

## Decision rule

If the unchanged candidate passes v0.3, write the fusion ratification manifest and advance directly to adaptive multi-view latent pooling.

If it fails real behavioral constraints, retire v0.3 and perform failure-driven constraint-based model/data work. Do not use v0.3 rows for training and do not shrink the architecture merely for efficiency.

Private identity gradient remains CLOSED.
