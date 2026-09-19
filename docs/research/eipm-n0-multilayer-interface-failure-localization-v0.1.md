# N0 Multi-Layer Interface Failure Localization v0.1

**Date:** 2026-09-19

## Classification

Magnolia job `575825` is a valid model experiment, not an infrastructure failure.

- scheduler state: `COMPLETED`
- exit code: `0:0`
- source revision: `8a8531884bd9d59dfff5023579b0789dc7146ea7`
- result SHA-256: `6f2f3fe6ddab764d947a7c63fe94ef24d5cd77e3cc68821c8678fdb8f2ea36bd`
- result status: `FAIL_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`
- eligible checkpoints: none
- heldout opened: false
- frozen challenge evaluated: false
- parent graph parameters changed: false
- preservation rows used for gradient: false

The single authorized experiment is consumed. No rerun or local hotfix is authorized.

## What the experiment actually showed

The new interface did learn some of the intended query-role signal.

Baseline causal dev:

- row accuracy: 0.5555555555555556
- quad accuracy: 0.08333333333333333
- mean target margin: 0.08108323460651769
- causes row accuracy: 0.5
- supports row accuracy: 0.5

At step 40:

- quad accuracy increased to 0.2222222222222222
- mean target margin increased to 0.08724810348616706
- causes row accuracy increased to 0.5416666666666666
- supports row accuracy increased to 0.5833333333333334

But endpoint preservation collapsed immediately:

Parent-only endpoint baseline:

- pair accuracy: 0.9583333333333334
- family-min pair accuracy: 0.6666666666666666
- row accuracy: 0.96875
- mean target margin: 0.8209420529504617

Step 40 endpoint preservation:

- pair accuracy: 0.6041666666666666
- family-min pair accuracy: 0.0
- row accuracy: 0.7083333333333334
- mean target margin: 0.24894556403160095

Later checkpoints became worse on ordinary preservation as well.

This is not evidence that the interface needed more steps, more width, a different learning rate, relaxed tolerances, or another checkpoint-selection rule.

## Structural localization

### The parent read is endpoint-aware

`DualEndpointEvidenceGraphEncoder._query_conditioned_relation_bias` builds directed relation features from:

- parent query state;
- relation embedding;
- source graph state;
- target graph state.

It then computes **independent source and target biases**.

The parent therefore decides endpoint relevance using the semantic content of the actual graph endpoints.

### The failed residual is not endpoint-aware

`RelationConditionedMultiLayerQueryInterface.forward` receives:

- frozen semantic hidden states;
- query attention mask;
- edge relation type;
- edge validity.

It does **not** receive:

- source graph state;
- target graph state;
- source field semantics;
- target field semantics;
- edge identity beyond relation type.

For a fixed query and relation type, the resulting `edge_relation_query_state` is therefore identical for every same-type edge in that example.

`RelationConditionedMultiLayerEvidenceGraphEncoder._interface_relation_bias` then reads one scalar `delta` from that query-only edge state and applies:

- `+delta` to the source;
- `-delta` to the target.

The new path therefore implements a learned **query-conditioned endpoint polarity**, not query-to-edge semantic binding.

It is also artificially zero-sum: any source increase forces an equal target decrease. The proven parent does not have this restriction; its source and target heads are independent.

## Why the causal curriculum could reward the wrong abstraction

Every causal row in the v0.1 curriculum contains:

- exactly two fields;
- exactly one directed relation edge;
- one relation family;
- a target that is either the edge source or edge target.

That factorial design successfully prevents a constant SOURCE or TARGET shortcut.

It does **not** require the interface to identify which graph edge or endpoint content is semantically relevant because there is only one edge and two endpoints.

A query-conditioned polarity signal can therefore improve the causal quads while still being fundamentally incapable of safe behavior on mixed or multi-edge evidence graphs.

That is exactly the observed pattern: targeted causal improvement appears while the already-proven endpoint read collapses.

## Larger architectural role

The language↔graph boundary should not be a second endpoint classifier.

Its role is to bind the user's semantic intent to **specific structured evidence** while preserving the graph model's relation and endpoint competence.

The boundary must answer:

> Given this query, this relation, this source state, this target state, and the surrounding graph context, what information from the language representation should modify this specific edge decision?

The failed architecture instead answered:

> For this query and relation type, should source generally beat target?

Those are different capabilities.

## Frontier design evidence

The next design is informed by architectural patterns rather than copied wholesale.

### Cross-attention bridges

Flamingo (Alayrac et al., 2022, arXiv:2204.14198) uses gated cross-attention to bridge frozen pretrained streams instead of collapsing one modality into a detached global scalar.

TransNAR (arXiv:2406.09308) keeps a pretrained graph reasoner frozen and uses cross-attention between text and graph/node/edge representations. Its motivation is closely aligned with preserving structured reasoning while allowing language-conditioned access.

### Shared parent plus routed specialist

DeepSeekMoE (Dai et al., 2024, arXiv:2401.06066) separates shared experts from routed specialists so common capability does not have to be relearned by every specialist.

Remembering Transformer (Sun et al., 2024, arXiv:2404.07518) uses routed adapters plus knowledge distillation to reduce interference during continual capability acquisition.

The transferable lesson is not to turn N0 into a generic MoE. It is to keep the proven graph read as the shared expert and make any new language bridge a selectively activated specialist.

### Zero-init is necessary but insufficient

LLaMA-Adapter (Zhang et al., 2023, arXiv:2303.16199) uses zero-initialized attention/gating to preserve pretrained behavior at initialization.

Our experiment already validated that exact-parent initialization works.

The failure demonstrates that zero-init alone says nothing about preservation **after the specialist is trained**.

### Stability-plasticity constraints

Continual-learning and multi-task methods such as PCGrad (Yu et al., 2020, arXiv:2001.06782), Gradient Projection Memory (Saha et al., 2021, arXiv:2103.09762), and later routed-adapter approaches all address the same general issue: a new objective can improve one capability while destructively interfering with an old one.

This does not authorize importing a generic continual-learning algorithm blindly. It establishes that explicit routing or preservation-aware optimization is a first-class design concern when the new module acts on an existing decision surface.

## Rejected architecture family

The following family is closed by this result:

`query-only relation-conditioned scalar -> +source / -target`

Do not create:

- QRR v0.4;
- a smaller-learning-rate copy;
- a wider copy;
- a longer-trained copy;
- a new signed source/target scalar with another name;
- a version that only changes preservation thresholds;
- a version that adds replay loss to the same query-only scalar.

The failure is architectural.

## New hypothesis: Query–Edge Cross-Attention Bridge

The next boundary should be built around a real edge-specific interaction.

For each active directed edge, the bridge should jointly consume:

- relation-conditioned intermediate query token states;
- source graph state;
- target graph state;
- relation embedding;
- optionally the parent query state and parent edge score context.

The query attention itself should be conditioned by the specific edge, rather than producing one relation-global query summary first.

The bridge should emit **independent source and target residuals**. No forced anti-symmetry.

A learned edge-specific gate should control whether the specialist contributes at all.

The parent dual-endpoint graph remains frozen and authoritative at initialization.

The specialist must be exact-zero at initialization.

## Training study must also change

A new edge-binding architecture cannot be evaluated on the old one-edge/two-field curriculum alone.

The next fresh causal curriculum must include:

- multiple fields;
- multiple directed edges;
- multiple edges sharing the same relation type;
- distractor relations;
- content-matched and content-mismatched endpoints;
- source-role and target-role queries;
- reversed edge directions;
- train/dev/test lexical and entity isolation.

The correct answer must require both:

1. understanding the query role;
2. binding that role to the correct edge content.

A relation-global endpoint polarity shortcut must score near chance.

## Preservation training without contaminating preservation evaluation

The v0.1 study intentionally made all preservation rows evaluation-only.

That was a valid test of whether the architecture could preserve old behavior structurally. It failed.

For the next routed specialist, preservation **training anchors** may be drawn only from pre-existing TRAIN splits of the ordinary and endpoint source curricula.

Frozen preservation DEV remains evaluation-only and must not be used for gradient, hyperparameter selection, routing thresholds, or loss-weight tuning.

Causal TEST and frozen challenge remain unopened.

This changes the scientific question from:

> can an unconstrained specialist avoid interference by luck?

to:

> can an edge-routed specialist learn a new boundary capability while its training objective explicitly preserves the proven shared expert on independent old-task training data?

## Next authorized work

Authorized now:

1. record this failed result and close the old one-shot experiment;
2. implement the Query–Edge Cross-Attention Bridge;
3. build a fresh multi-edge causal curriculum;
4. build a preservation-anchor compiler using TRAIN splits only;
5. perform static and no-gradient exact-parent qualification.

Not authorized now:

- optimizer creation;
- gradient training;
- P100 training;
- causal test opening;
- frozen challenge rerun;
- scale;
- parent graph retraining;
- semantic backbone retraining;
- private identity gradient;
- promotion;
- N0 completion.

Training requires a separate explicit decision after the new architecture and data contracts pass qualification.

## Standing failure doctrine

If the next qualified architecture later fails a genuine model gate, stop again.

Do not patch symptoms.

Re-evaluate N0's purpose, the component's role, the failure boundary, and relevant frontier research before another model change.
