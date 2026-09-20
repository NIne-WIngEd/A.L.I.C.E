# N0 Dual-View Specialist Failure — Counterfactual Localization Ready

**Date:** 2026-09-20

## Source experiment

Magnolia job:
`575912`

Source revision:
`57b085fd815cc784ae4090b8db7b619adec57040`

Result SHA-256:
`1a38c37a41de68bea0bb9bc897b86d62cbc457b8b93189c2d16731110c1694ec`

Status:
`FAIL_DUAL_VIEW_SPECIALIST_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`

No eligible checkpoint.
No selected candidate.
TEST unopened.
Frozen challenge unopened.
Binding scale frozen.
No edge-identity training loss.
Parent graph unchanged.

## What the run proved

### Pair/edge identity stayed solved

Across every checkpoint:

- conditional all-four edge top1 = 1.0
- conditional same-relation top1 = 1.0
- mean target conditional probability = 0.4847296124531163

So the trained failure is downstream of edge identity.

### Old-task preservation is not the dominant problem

From step 80 onward, the canonical preservation policy passes.

At step 200:
- ordinary no-op top1 = 1.0
- endpoint no-op top1 = 1.0
- parent preservation passes
- parent graph parameters unchanged

### Specialist activation is unstable on causal DEV

- initial: 0.0 top1 at exactly p=0.5
- step 40: 0.5347
- step 80: 0.8889
- step 120: 0.7778
- step 160: 0.5278
- step 200: 0.5556

This happens while TRAIN causal specialist BCE falls near zero.

Therefore the explicit causal-vs-preservation activation classifier generalizes unstably.

### Directional residual reasoning is independently weak

At step 80, causal DEV specialist activation is already 0.8889 and mean specialist probability is 0.8955, yet:
- row accuracy = 0.0833
- quad accuracy = 0.0

Thus activation failure alone cannot explain the field-selection failure.

By step 200:
- row accuracy = 0.1667
- quad accuracy = 0.0278
- mean margin improves from -0.6462 to -0.4664
- only derived_from has non-zero quad accuracy (0.1667)

The residual path is learning something, but it is not learning endpoint-role reasoning robustly enough.

## Current hypothesis space

Three failures may coexist:

1. **Activation calibration / utility mismatch**
   - a binary causal=on, preservation=off BCE may learn dataset identity rather than expert utility;
   - causal and preservation scores might still be rank-separable even when p=0.5 top1 looks poor.

2. **Diffuse edge execution**
   - edge top1 is perfect but target conditional probability averages only 0.4847;
   - all active edges still receive residual mass;
   - irrelevant edge residuals may dilute a correct binding.

3. **Endpoint-role residual failure**
   - even with the correct pair, source vs target must still be selected;
   - the residual proposal may not encode this role robustly.

## Counterfactual audit

Branch:
`alice-eipm-v1-dual-view-specialist-failure-localization`

Exact frontier:
`9befcf4ae3574d671da211f994ff7529979403de`

Authoritative state:
`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.36.json`

CI:
`35492586940 — SUCCESS`

Audit:
`scripts/eipm/n0/audit_n0_v02_dual_view_specialist_failure_localization_v0_1.py`

Runner:
`scripts/eipm/n0/run_n0_v02_dual_view_specialist_failure_localization_v0_1.sh`

For trained checkpoints 40/80/120/160/200 it measures:

### Gate separation
- causal specialist probability distribution
- ordinary preservation distribution
- endpoint preservation distribution
- causal-vs-preservation AUC
- strict rank separation
- p=0.5 accuracies

This distinguishes poor calibration from actual overlap.

### Four execution counterfactuals

1. actual activation + soft binding
2. forced specialist + soft binding
3. actual activation + predicted hard-top1 binding
4. forced specialist + predicted hard-top1 binding

Hard top1 is **model predicted**, not target-label oracle. This is valid because the already-qualified DEV edge top1 is 1.0.

### Endpoint-role diagnostics

For the query-relevant edge:
- source/target residual proposal role accuracy
- mean proposal role margin
- family role accuracy
- within-relevant-pair endpoint accuracy for every execution counterfactual

### Interpretation

- forcing specialist helps -> activation bottleneck evidence
- hard top1 helps -> diffuse mixture bottleneck evidence
- forced specialist + hard top1 still weak and proposal role accuracy weak -> directional residual bottleneck evidence

Multiple failures may coexist.

## Closed

No:
- optimizer
- gradient
- GPU
- retraining
- threshold tuning
- binding-scale tuning
- new router
- LR tuning
- step tuning
- TEST
- challenge
- scale
- semantic retraining
- parent retraining
- private identity gradient
- promotion

The next architecture decision must follow the counterfactual localization result.
