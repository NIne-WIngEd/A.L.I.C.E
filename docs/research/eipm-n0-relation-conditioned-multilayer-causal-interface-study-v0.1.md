# N0 Relation-Conditioned Multi-Layer Causal Interface Study v0.1

**Date:** 2026-09-18  
**Status:** fresh causal curriculum and preservation contract defined; CPU-only preparation authorized; interface training remains unauthorized

## Trigger

The exact-map runtime qualification completed successfully on Magnolia.

Runtime qualification receipt SHA-256:

`779b19459fa8ab67bb1c3ed56b3e9cdb1b89b8382f145e1b3d7c4d98312b2b57`

The qualification established:

- exact map binding to `ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304`;
- exact 575804 source-audit binding to `ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab`;
- parameter state exactly unchanged;
- no gradient;
- no optimizer;
- no GPU;
- no heldout or frozen challenge use;
- no training or scaling authorization.

Therefore the architecture's pre-gradient step 3 is complete.

The required next work is to define the fresh causal curriculum and preservation contract before deciding whether one bounded interface-training experiment is justified.

## Causal question

The experiment must answer one question:

> Can relation-conditioned access to intermediate semantic token layers recover query-role-dependent evidence selection while leaving the frozen semantic backbone and already-proven evidence graph intact?

It must **not** answer:

- whether a wider router can memorize endpoint labels;
- whether another residual can overpower the parent;
- whether more training steps can brute-force a failed objective;
- whether heldout thresholds can be tuned after observing a candidate.

## Fresh factorial curriculum

Builder:

`scripts/eipm/n0/build_n0_v02_relation_conditioned_multilayer_interface_curriculum_v0_1.py`

The curriculum uses six directed relation families:

- `corrects`;
- `supersedes`;
- `derived_from`;
- `causes`;
- `supports`;
- `temporal_successor`.

Every semantic scenario is a four-row causal quad.

Two independent factors are varied:

1. query role: source-seeking vs target-seeking;
2. stored edge direction: A vs reversed B.

Within one quad:

- field text is identical;
- relation type is identical;
- entity/attribute content is identical;
- only query role and edge direction change;
- source-seeking versus target-seeking must flip the target;
- reversing the edge direction must also flip the target.

This prevents a successful candidate from solving the dataset with one global SOURCE or TARGET preference.

The operating curriculum budget is:

- 18 train quads per relation;
- 6 dev quads per relation;
- 6 unopened test quads per relation;
- 30 quads per relation;
- 180 quads overall;
- 720 rows overall.

These counts are a first causal-study operating budget, not a permanent capability limit.

Train/dev/test query templates, subject pools, and attribute pools are split-disjoint.

No prior QRR, relation-semantic, residual, endpoint-heldout, missing-evidence-localization, or frozen-challenge row is reused.

### Important authorization boundary

The curriculum is *defined* before the training decision.

The rows marked as intended future train rows still contain:

`training_authorized=false`

No optimizer may consume the curriculum until a later explicit decision changes that state.

## Preservation contract

Contract:

`configs/eipm/n0/n0_v02_relation_conditioned_multilayer_preservation_contract_v0_1.json`

The candidate is required to preserve:

1. ordinary graph replay behavior;
2. the selected endpoint-role repair capability;
3. symmetric `conflicts_with` behavior;
4. PAD/unmapped fail-closed behavior.

The selected parent graph remains:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

The following parent components remain frozen:

- semantic backbone;
- structured-state encoder;
- evidence-view adapter;
- selected evidence graph.

A future interface experiment may not rewrite those parents merely to make the causal curriculum pass.

## Numerical preservation policy

No preservation threshold is invented from a future candidate.

Before any candidate training:

1. repeatedly evaluate the frozen canonical parent on each numerical preservation lane;
2. measure actual numerical repeatability;
3. freeze tolerances from that parent-only repeatability evidence;
4. only then expose a trained candidate to the preservation gate.

Candidate outputs are forbidden inputs to tolerance calibration.

This is the same causal discipline used by the prior downstream arbitration work, but kept compact: one calibration required for interpreting one candidate, not a new validator program.

## CPU-only preparation

Runner:

`scripts/eipm/n0/run_n0_v02_prepare_relation_conditioned_multilayer_causal_study_v0_1.sh`

The preparation step:

- verifies the exact compiled map hash;
- verifies the exact runtime-qualification receipt hash;
- verifies the selected parent graph hash;
- compiles the fresh curriculum;
- validates all 2x2 causal quads;
- binds semantic, structured, adapter, graph, and replay artifacts by SHA-256;
- validates that the preservation contract was fixed before candidate results;
- writes one immutable preparation receipt.

The preparation step does not create:

- an optimizer;
- a gradient;
- a GPU allocation;
- a candidate checkpoint;
- a heldout result;
- a training authorization.

## Training-decision rule

After the CPU preparation receipt exists, make one explicit architecture-training decision.

A bounded interface-only training study is justified only if all of the following remain true:

1. the runtime interface qualification passed;
2. the causal curriculum remains fresh and factorial;
3. parent artifacts are exact and frozen;
4. preservation calibration can be performed without candidate knowledge;
5. the trainable path tests the actual multi-layer token-interface hypothesis rather than recreating QRR/residual routing;
6. the experiment has a clear failure interpretation that returns to the language-graph boundary rather than spawning hotfixes.

If those conditions hold, implement one bounded candidate path.

If they do not, do not create an optimizer.

## Future candidate architecture constraint

If interface training is later authorized, the trainable component must use the real:

`RelationConditionedMultiLayerQueryInterface`

and its per-edge `edge_relation_query_state`.

The experiment must not substitute:

- raw pooled query semantics;
- final-layer-only token state;
- hardcoded layer 12;
- QRR v0.4;
- the previous semantic-role residual;
- a new generic endpoint classifier.

The parent graph must remain an immutable expert.

Any composition with the parent should initialize at exact parent behavior and expose the new interface contribution explicitly enough to measure whether the improvement came from relation-conditioned multi-layer semantic access.

## Failure interpretation

### Causal dev fails, preservation passes

The richer semantic interface did not solve the directional evidence-selection problem.

Do not widen or hotfix it automatically.

Reopen whether the relation distinction belongs:

- in a different language-graph interaction mechanism;
- in the semantic training objective;
- or at a higher reasoning layer instead of evidence selection.

### Causal dev passes, preservation fails

The interface learned the new behavior by damaging proven behavior.

Reject the candidate.

Do not open test.

### Both pass

A separate explicit decision may authorize one heldout opening or further bounded study.

Neither action is automatic.

## Capability doctrine

This study is part of the full production N0 build.

The curriculum size, four candidate layers per relation, current semantic depth, current graph width, current parent graph, and any future interface width are not permanent A.L.I.C.E. capability ceilings.

Efficiency should come from selective layer access, sparse relation-conditioned computation, caching, batching, and architecture quality rather than deleting needed representational capacity.
