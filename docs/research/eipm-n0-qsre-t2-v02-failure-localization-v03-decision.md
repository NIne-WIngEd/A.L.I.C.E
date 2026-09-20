# N0 QSRE T2 v0.2 failure localization and T2 v0.3 architecture decision

**Date:** 2026-09-20  
**Evidence jobs:** 575953 preparation, 575954 realistic one-shot T2 v0.2  
**Decision:** retain frozen semantic/T1 parents; replace only the relation extractor with a schema-grounded ordered decoder.

## Result status

Preparation job 575953 completed 0:0 and bound:

- realistic T2 v0.2 curriculum SHA `5aa8f0aa3210466964f38b14081c71c1d3c58f4405247067c76a7cdfe115963c`;
- prepared cache SHA `f7d93a1ad0559ae2ad29d909062a88fbf2d1eea4a05246a5fb5a95e01d6b69ac`;
- locked challenge SHA `197cebe5ed4c59455839db5b2822582a03021c92a94670e1aede568278c1f205`;
- semantic checkpoint SHA `6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43`;
- T1 executor checkpoint SHA `483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb`.

Training job 575954 also completed 0:0. It reached the precommitted 1200-step boundary without an eligible DEV checkpoint.

Training result SHA:

`6e4fa39775a985b3265e4e1a84851ebc344310d7dd699e1649e807ab9f8f6025`

Final summary SHA:

`03f9a639b47de06f4b1faa00465dffb20489c6e328b12eedae225f491df97dfb`

Status:

`FAIL_QSRE_T2_OPERATOR_DEV_CONTRACT`

The locked operator challenge correctly remained unopened.

This is a valid model failure, not an infrastructure failure.

## Failure localization

The run does **not** look like a globally inadequate frozen semantic representation.

Across the governed DEV trajectory:

- relational operation accuracy reached **1.0** by step 50;
- operator control accuracy reached **1.0** by step 75;
- relational role accuracy reached a maximum of **0.96484375**;
- relation-sequence exact accuracy reached only **0.5381944**;
- full operator exact accuracy peaked at **0.4618056**;
- query-view pair consistency never exceeded **0.5416667** and was **0.375** at the best-observed late checkpoint;
- ordered-path operator success was only **0.15625** at step 1175.

Downstream behavior tracked the relation failure:

- best row success: **0.6284722**;
- best single-target top-1: **0.6041667**;
- best causal-pair completion: **0.4305556**;
- best plural L1: **0.3957110**;
- no DEV checkpoint approached the exact causal contract.

The architecture therefore learned control, operation, and endpoint role from the frozen token stack while failing to robustly recover relation identity/order across paraphrases. This narrows the bottleneck to the relation extraction interface.

## Why T2 v0.2 relation extraction is the wrong abstraction

T2 v0.2 represents each relation with a randomly initialized learned anchor. Two learned relation-slot queries attend the whole frozen query memory and then classify independently against those learned prototypes.

That creates three problems for the full A.L.I.C.E. workload:

1. **No semantic schema grounding.**  
   The six relation labels are learned as opaque prototype IDs. Natural-language definitions of support, correction, supersession, provenance, causation, and chronology never participate in scoring.

2. **Weak ordered composition.**  
   The two relation slots are different learned queries, but relation 2 is not explicitly conditioned on the semantic decision for relation 1. This is especially poor for ordered paths containing two relation descriptions.

3. **Poor migration/systematicity.**  
   A new relation requires learning a new opaque class anchor instead of supplying a schema meaning and learning how to align language to that meaning.

These are architecture issues. They are not reasons to tune LR, extend steps, increase width, or reopen support learning.

## Research connection

Recent work strengthens the same direction without dictating A.L.I.C.E.'s design.

**SALR — Schema-Anchored Latent Reasoning (2026)** grounds continuous reasoning states in an explicit schema codebook and delays discrete commitment. Its relevant lesson here is that schema elements should participate semantically in latent reasoning instead of existing only as arbitrary output IDs.

**Layer-Order Inversion (2026)** reports that multi-hop information does not reliably emerge as a simple one-hop-per-layer chain. That argues against hardwiring relation slot n to a particular transformer depth. T2 v0.3 therefore keeps all frozen token layers available and makes ordering a decoder property rather than a layer-order assumption.

Recent constrained semantic-parsing work also supports separating semantic schema selection from structural grammar/termination. T2 v0.3 treats NONE as a structural STOP decision rather than inventing a seventh semantic relation.

## One causal architecture change

T2 v0.3 changes **only relation extraction**.

Kept unchanged:

- frozen N0 semantic checkpoint;
- all 17 query hidden-state outputs;
- T1 executor checkpoint;
- oracle support/focus;
- realistic T2 v0.2 TRAIN/DEV data and exact prepared cache;
- role head;
- operation head;
- control head;
- continuous operator state;
- AdamW settings;
- seed;
- batch size;
- 1200-step boundary;
- eval cadence;
- eligibility thresholds;
- first-perfect-DEV selection rule.

Replaced:

`random learned relation anchors + independent relation-slot classification`

with:

`frozen semantic relation schema + token/layer late interaction + sequential relation-slot conditioning + structural STOP`

### Relation schema grounding

The public relation codebook is:

`configs/eipm/n0/n0_v02_qsre_t2_relation_schema_v0_1.json`

SHA-256:

`bae45945fcb1a7f33b1177bc65da02c90335d443a0718aa7c796e744ace2e2c9`

Each current relation contains natural-language meaning and directional semantics. The schema text is encoded by the same frozen semantic backbone. No private identity data is used.

### Ordered relation decoder

For each current relation slot:

1. initialize from the existing T2 relation latent;
2. re-attend the complete multi-layer token memory;
3. score each relation by token-level semantic similarity against the frozen relation schema at matching layers;
4. represent termination with a separate learned STOP decision;
5. feed the **soft previous schema decision** into the next relation slot.

No hard transformer layer is selected.

No hard relation span is selected.

No relation-specific learned class anchor exists.

Schema row permutation must permute semantic relation logits correspondingly; CI tests this directly.

The current two-step bridge exists because the frozen T1 causal fixture contains up to two hops. It is explicitly not a product ceiling.

## Failure interpretation for T2 v0.3

### If v0.3 passes realistic DEV

Open the already-locked operator-only public challenge exactly once in the same governed run.

Do not open T3 automatically.

### If v0.3 fails mainly on relation identity/order

The schema-grounded relation interface has failed under a frozen semantic parent.

That is the point at which reopening the semantic relation objective becomes justified.

Do not tune v0.3.

### If relation accuracy passes but downstream remains weak

The bottleneck has moved to the T2->T1 operator/execution bridge.

Do not retrain semantics merely because downstream execution fails.

## Anti-hotfix rule

T2 v0.2 will not be rerun.

T2 v0.3 gets one governed GPU run only after:

- static architecture/unit qualification;
- exact CPU runtime qualification with real Magnolia artifacts.

Those stages are chained under one owner-side launcher. There is no intermediate permission gate.
