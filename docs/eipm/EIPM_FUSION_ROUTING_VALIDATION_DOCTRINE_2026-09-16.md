# EIPM Fusion Routing Validation Doctrine — 2026-09-16

**Status:** current N0 interpretation authority  
**Trigger:** confirmatory fusion challenge v0.2 exposed over-specific routing supervision  
**Scope:** cross-context fusion routing supervision, evaluation, and future repair work

## Core rule

Fusion routing weights are an internal latent mechanism. They are not themselves historical truth, personality truth, or calibrated probabilities unless a separate grounded probabilistic model justifies that interpretation.

Therefore, A.L.I.C.E. must not be trained or rejected merely because its internal routing vector differs from a hand-authored exact percentage split when multiple splits are behaviorally equivalent.

Validation must constrain what matters for capability and fidelity, not force arbitrary internal geometry.

## What may be supervised or gated

Routing supervision/evaluation may require behaviorally necessary properties such as:

- one view must be the top view;
- the top view may be any member of a justified set of views;
- a stale, contradicted, low-authority, or unavailable view must not dominate;
- a required view must receive at least a minimum amount of routing mass;
- a distractor view must receive no more than a maximum amount of mass;
- one view must exceed another by a grounded minimum margin;
- a set of mutually supportive reliable views must collectively receive sufficient mass;
- unavailable views must receive effectively zero mass;
- changing the query from current-state to historical-state may require a different dominance relation;
- unresolved conflict must remain unresolved rather than being collapsed to a single false authority.

These requirements should be represented as set-valued or inequality constraints.

## What must not be treated as ground truth without evidence

The following are forbidden as ratification requirements unless separately justified by a grounded probabilistic source model:

- exact hand-authored routing percentages such as `0.34/0.33/0.33`;
- exact ratios between two agreeing authoritative views;
- exact equality among consensus views;
- arbitrary preference for structured versus evidence routing when both support the same correct conclusion;
- a requirement that an internal routing vector reproduce the training author's intuition more closely than is needed for the correct judgment.

A soft target distribution may remain in historical training artifacts as a heuristic. It must not be reinterpreted as calibrated truth.

## v0.2 challenge finding

Confirmatory challenge v0.2 remains an immutable historical failed evaluation. Its thresholds are not changed retroactively and the run is not reclassified as a pass.

However, post-result analysis found that its decisive routing metric used L1 similarity against exact hand-authored `target_view_distribution` values. Some of the worst-scoring families admitted multiple behaviorally equivalent internal routing distributions. For example, a consensus case required a near-uniform split despite all available views supporting the same answer, and a distractor case imposed an exact structured/evidence split despite both reliable views supporting the same conclusion.

That means v0.2 mixed two questions:

1. Did the model choose and combine evidence correctly?
2. Did its internal routing percentages match the author's arbitrary soft target?

Only the first question is a valid capability gate absent probabilistic grounding.

## Corrective procedure

Because no model gradient has occurred after challenge v0.2, the correct next action is **validator correction before model correction**.

1. Keep the v0.2 result immutable and retire it for ratification.
2. Do not train on v0.2 challenge rows, entities, or templates.
3. Do not train another repair merely to match v0.2's exact routing percentages.
4. Build a new independent frozen challenge with new entities/templates.
5. Evaluate the unchanged preselected repair-step240 checkpoint.
6. Gate routing with behaviorally justified constraints rather than exact distributions.
7. Continue to hard-gate semantic conclusion quality, exact source anchors, missing-view safety, disagreement preservation, provenance boundaries, and no-gradient lineage.
8. If the unchanged model fails the corrected constraints, treat that as real evidence for failure-driven model/data work.
9. Any future routing repair should use constraint/set-valued objectives rather than exact-percentage cross entropy unless calibrated probabilities exist.

## Validation versus limitation

This doctrine follows the owner-ratified rule that validation and limitation are different.

A validator is valid when it tests a necessary capability or provenance boundary. A validator becomes a limitation when it forces one arbitrary internal implementation despite multiple equally correct representations.

The corrected routing contract constrains outcome-relevant behavior while leaving the model free to discover the internal routing geometry that best supports A.L.I.C.E.'s ultimate fidelity objective.

## Capacity policy

This doctrine changes no model parameter ceiling, view ceiling, graph ceiling, context ceiling, or training ceiling. It does not authorize shrinking the model. It does not authorize private identity gradient.

If corrected capability tests later show a real architecture bottleneck, the architecture may expand or change without a preset parameter limit.
