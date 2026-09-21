# N0 closure-pass architecture after Magnolia job 575958

**Date:** 2026-09-21  
**Status:** implementation and static qualification package; GPU remains closed until exact-head qualification passes  
**Source failure:** Magnolia job `575958`  
**Source branch:** `alice-eipm-v1-qsre-production-core-v1@2204f61f3a2d49eccc10d834dda83921a9af5be1`  
**Closure branch:** `alice-eipm-v1-n0-closure-pass`

## 1. Why this is not another local P2 hotfix

Job 575958 is a valid learned-model failure. CUDA, uDocker, exact source revision, preserved P1 lineage, and the original frozen final-validation lineage all passed before P2 began. P2 then completed the full 1600-step governed budget and returned `FAIL_QSRE_PRODUCTION_P2_OPERATOR_V3`.

The failure is not "train longer" evidence.

Across the run:
- TRAIN relation/factor losses became very small;
- DEV relation-program exactness remained roughly in the 0.35–0.45 range;
- open-schema relation exactness peaked early and then collapsed to near zero;
- traversal DEV accuracy stayed pinned near 0.789 despite near-zero TRAIN traversal loss;
- P3 remained correctly closed.

That pattern says the final-v3 *package abstraction* still behaved like a small closed classifier even though its source code no longer contained an explicit learned relation-ID table.

The response is therefore not P2-v4. The response is to repair the semantic induction boundary once, before another Production operator run.

## 2. Architecture mistakes exposed by 575958

### 2.1 An implicit six-relation ontology remained

Production v3 removed an explicit `[R,D]` learned relation table, but its shared semantic projection, normalization, and layer geometry were still optimized directly on six core relation labels.

The four Production open-schema descriptions were intentionally excluded from gradient. That preserved the benchmark boundary, but it also meant the trainable metric itself could rotate around the six observed meanings. Job 575958 showed exactly that behavior: core TRAIN fit improved while runtime schema expansion degraded.

A dynamic candidate axis is not enough. The *matching operation itself* must be trained to generalize across relation meanings and then protected from a six-class Production objective.

### 2.2 The independently localized ordered-evidence repair was lost

Job 575957 had already shown that relation activity/survival was mostly correct while identity/order collapsed sharply with path length. The ordered-evidence experiment therefore added a token-position coverage state so later recurrent steps operate on remaining relation evidence.

During the later Production-v3 recomposition, the useful insight "keep relation hypotheses continuous until structural binding" was preserved, but the explicit remaining-evidence state was dropped. Final v3 re-attended the complete query at every recurrent relation step.

Both requirements are compatible and required:
- continuous semantic hypotheses before binding;
- explicit differentiable ordered evidence consumption across recurrent steps.

Closure pass keeps both.

### 2.3 Factor semantics were still opaque learned class heads

Role, traversal, direction, control and modifiers were described as stable factors, but the implementation still used learned categorical heads such as `role_head`, `traversal_head` and `control_head`.

The stable ~0.789 traversal DEV accuracy while TRAIN traversal loss approached zero is direct evidence that this surface was also overfitting its small vocabulary rather than learning a reusable semantic operation.

Closure pass represents factor meanings as runtime semantic descriptions and applies the same relation-identity-independent matcher to them. The current grammar cardinalities remain checkpoint/runtime interfaces, not semantic parameter axes.

### 2.4 Static qualification proved mechanics, not the missing generalization property

The previous qualification proved useful invariants:
- candidate permutation equivariance;
- continuous relation distributions;
- dynamic relation count;
- no fixed top-k before the binder;
- final holdout isolation.

Those tests were necessary but insufficient. A toy runtime candidate can win under a hand-constructed embedding while a trainable six-class semantic metric still destroys real unseen-schema transfer after optimization.

The missing proof is an **unseen semantic schema generalization gate before Production P2**.

## 3. Closure architecture

The closure package adds one upstream semantic induction stage and then freezes it.

### P2S — public schema-generalization matcher

`QSRESchemaMatcher` learns a relation-identity-independent operation:

> given query evidence and a supplied semantic schema description, estimate whether that description is expressed by the query.

Properties:
- no relation-key parameter;
- no candidate-count parameter axis;
- no hop-specific parameter;
- shared query/schema semantic projection;
- candidate-conditioned query-token evidence;
- symmetric query-to-schema and schema-to-query late interaction;
- public identity-neutral data only.

P2S is trained on:
- a broad auxiliary set of public relation meanings unrelated to the Production holdouts;
- the six Production core meanings;
- semantic schemas for role/traversal/direction/control/modifier factors;
- paired paraphrase consistency.

P2S is evaluated on a separate auxiliary relation set whose descriptions and queries never enter gradient.

The real Production open-schema relations and final-only relations are explicitly forbidden from both P2S TRAIN and its auxiliary holdout:
- CONFLICTS_WITH
- EXEMPLIFIES
- PREREQUISITE_FOR
- PART_OF
- ENABLES
- PREVENTS

Thus passing P2S demonstrates a reusable schema-matching operation without contaminating the actual N0 open/final holdouts.

### Production P2 — ordered program learning around a frozen matcher

After P2S passes, its matcher is frozen.

Production P2 may learn:
- recurrent program ordering state;
- CONTINUE / STOP / UNKNOWN event state;
- applicability;
- continuous executor context;
- downstream execution compatibility.

Production P2 may **not** rotate the semantic schema metric around the six core relation identities.

The recurrent program also restores differentiable query-token coverage:
- coverage is over token positions;
- coverage is shared across semantic layers;
- no left-to-right monotonic parser is assumed;
- selected relation evidence is softly consumed;
- a small nonzero recovery floor preserves the ability to revise an earlier soft hypothesis.

Relation hypotheses remain continuous softmax distributions. Exact zeros still begin only in Binder v2 structural support.

### Semantic factor schemas

The fixed learned categorical heads for role/traversal/direction/control/modifiers are removed.

The same frozen matcher evaluates supplied factor descriptions. This turns the factor surfaces into semantic interfaces instead of tiny lexical classifiers while retaining the executor's current stable factor indices as interface order.

## 4. What remains proven and unchanged

No evidence from 575958 rejects:
- the corrected P1 executor;
- the STOP-tail semantic correction;
- the selected unchanged P1 step-50 checkpoint;
- Binder v2's structural sparsity boundary;
- the existing P4/frozen-native validation policy;
- the original 320-row frozen native holdout;
- full 640-dimensional semantic width;
- runtime-variable relation schema;
- continuous relation hypotheses before structural binding.

Those components are reused rather than rewritten.

## 5. Compute and anti-loop rule

This package is designed and qualified as one closure package before another GPU launch.

The sequence is:

`P2S unseen-schema matcher gate -> Production P2 -> P3 Binder v2 -> zero-gradient P4 -> original frozen native N0 gate`

Failure stops the sequence and preserves all artifacts.

There is:
- no automatic rerun;
- no LR search;
- no step search;
- no width search;
- no threshold change after results;
- no TEST opening;
- no private identity gradient;
- no deletion or overwrite of jobs 575955, 575956, 575957, or 575958 evidence.

The next GPU result is allowed to falsify the closure architecture. It is not permission for an automatic version ladder.

## 6. N0 close condition

N0 still closes only on the already-frozen native objective:

`PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE`

with:

`n0_complete=true`

and:

`n1_authorized=true`.

Anything else is preserved as model evidence and stops the launch.
