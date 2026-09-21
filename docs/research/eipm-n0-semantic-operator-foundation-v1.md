# N0 Semantic-Operator Foundation v1 — architecture decision after job 575990

**Date:** 2026-09-21  
**Status:** architecture decision; no gradient or GPU authorization  
**Branch:** `alice-eipm-v1-n0-semantic-operator-foundation-v1`  
**Parent diagnostic:** `alice-eipm-v1-n0-p2a-semantic-localization-v1@8180ed500fcf2181d3a70f5a01481d00ebbbb24a`  
**Source evidence:** Magnolia job `575990`, completed `0:0`, empty stderr

## Decision

Reopen the **public N0 semantic representation objective and semantic/operator boundary**.

Do **not** build another P2S/P2A matcher, fusion repair, prompt wrapper, or fixed-class semantic head.

The next N0 architecture is a **schema-conditioned semantic-operator foundation** trained end-to-end on public identity-neutral data so that dynamic runtime schema semantics, relation/argument roles, operator factors, uncertainty, and token-level evidence are part of the representation learning objective itself.

The existing corrected QSRE structural executor, ordered evidence coverage, adaptive structural support boundary, Binder v2, selected repaired evidence graph, structured-state encoder, fusion system, latent pool, and original final N0 validation remain preserved unless later evidence directly falsifies them.

## Why job 575990 closes the old semantic-authority path

The audit reconstructed job 575986 before interpretation.

Current frozen authority:

- Production core single relation: `0.4068181813`;
- auxiliary seen relation: `0.2708333433`;
- auxiliary held-out relation: `0.2604166567`;
- held-out factor macro: `0.3666666731`.

Using the trained teacher heads in teacher-native prompt/candidate geometry did not rescue the result:

- Production core: `0.3840909004`;
- auxiliary seen: `0.2291666716`;
- auxiliary held-out: `0.2604166567`;
- held-out factors: `0.4166666731`.

Even a post-hoc diagnostic upper bound that replaces the deterministic layer mixture with the best hidden-layer token view per task remained weak:

- Production core: `0.4090909064`;
- auxiliary seen: `0.3333333433`;
- auxiliary held-out: `0.3541666567`;
- held-out factors: `0.4166666731`.

Therefore:

1. prompt geometry is not the primary failure;
2. equal-weight P2A fusion is not the primary failure;
3. choosing a better frozen hidden layer is not enough;
4. the current frozen representation does not expose the required broad open-schema semantics through any tested zero-gradient view.

Job 575966 remains important counter-evidence against the stronger claim that the backbone contains no useful information. Its learned P2S-v2 matcher achieved much stronger metrics, including Production core `0.9477272727`, auxiliary seen `0.71875`, auxiliary held-out `0.6666666667`, and held-out factors `0.7166666667`, with later Production-core fit reaching `1.0`.

The correct interpretation is:

> useful semantic information exists, but the ratified semantic representation was not trained to make dynamic open-schema relation/factor semantics a native, broadly generalizing geometry.

That is an objective/representation-boundary problem, not a reason for another readout patch.

## Why the old semantic base was insufficient for this role

The ratified `alice-n0-semantic-v0.2` checkpoint is a 16-layer 136,594,435-parameter ModernBERT-family bidirectional encoder.

Its public semantic training emphasized:

- span MLM;
- candidate preference;
- principle/rationale alignment;
- semantic contrastive learning.

The registered teacher bank contains 1,020 rows over 51 public competencies. The selected targeted repair added only 20 repair-train rows and updated the top four backbone layers plus semantic heads.

That training was useful for the original semantic readiness task. It was never evidence that the model had learned:

- arbitrary runtime relation descriptions;
- unseen relation-family discrimination;
- explicit source/target argument semantics;
- dynamic traversal/control/modifier schema;
- recurrent ordered relation programs;
- open-world out-of-schema uncertainty.

Semantic-base ratification therefore remains valid **for the evidence on which it was ratified**, but it is no longer sufficient authority for the broader QSRE/N0 workload now exposed.

## Governance correction: do not freeze the foundation before the full-envelope contract

The main process error was not that earlier validations were useless. It was that a locally ratified semantic component was allowed to become a **frozen architectural boundary before the broader N0 workload had been closed**.

The semantic base was correctly validated for its original scope. Later QSRE work introduced requirements that were not present in that ratification contract: dynamic relation schemas, explicit argument-role semantics, operator factors, open-schema uncertainty, and ordered compositional execution.

Downstream work then tried to recover those missing capabilities with adapters, matchers, routers, and frozen-authority readouts around a semantic representation that could no longer co-adapt.

New rule:

> A foundation representation may be preserved as a baseline or temporarily frozen for causal experiments, but it must not become permanently frozen authority until the full downstream workload that depends on it has passed an interface-sufficiency review.

For N0 specifically:
- local component PASS does not imply permanent architectural freeze;
- causal isolation may freeze a parent temporarily;
- if a later full-envelope requirement reaches below that boundary, the boundary may reopen once evidence localizes the deficit;
- re-opening does not erase the historical validity of the earlier ratification;
- once the joint semantic/operator foundation passes full-envelope proof obligations, downstream freezes may become durable again.

This policy is intended to prevent another sequence where increasingly capable downstream heads are forced to compensate for a representation objective that was frozen too early.

## Architecture family

### 1. Shared semantic token foundation

Retain a bidirectional transformer semantic backbone as the base language representation family.

The current ratified checkpoint may be used as an initialization baseline. It is not frozen authority.

All semantic layers may receive public gradient when the eventual training plan is authorized.

No current hidden size, depth, sequence length, or parameter count is a permanent capability ceiling.

### 2. Runtime schema is first-class input

Each runtime relation or operator factor is represented by natural-language schema content:

- relation description;
- source argument meaning;
- target argument meaning;
- symmetry/direction;
- domain/range/type constraints;
- optional provenance/temporal semantics.

Relation keys remain metadata only.

No relation ID, factor ID, candidate count, or hop index is allowed to become a learned semantic-identity parameter axis.

### 3. Shared schema-conditioned token interaction

Query tokens, schema tokens, and relevant typed-context tokens interact through shared trainable attention/late-interaction layers.

The semantic model is trained with the same kind of query/schema interaction that runtime will use.

This removes the current pattern:

`generic semantic model -> freeze -> invent a new semantic metric afterward`.

The model instead learns:

`language representation <-> runtime schema interpretation <-> operator state`

jointly.

### 4. Continuous operator latent

A shared continuous operator state receives:

- query token information;
- runtime relation-schema states;
- role/direction/traversal/control/modifier schema states;
- typed context;
- uncertainty/plurality state.

It exposes continuous distributions and evidence, not fixed class-head authority.

### 5. Shared recurrent relation-step transition

Ordered multi-hop reasoning uses one shared recurrent transition.

Each step consumes:

- previous operator latent;
- remaining query-evidence coverage;
- current runtime schema candidate states;
- structural/context state.

It emits:

- continuous relation hypotheses;
- applicability;
- structural stop/continue/defer state;
- updated operator latent;
- evidence coverage.

There is no learned slot per hop and no fixed hop-count parameter axis.

### 6. Semantic identity remains continuous until structural binding

Relation/factor hypotheses remain continuous through semantic/operator inference.

Exact zero support remains a structural decision owned by Binder v2 or its evidence-supported successor.

Do not reintroduce early hard top-1 relation selection.

### 7. Existing QSRE structural execution remains downstream

The semantic-operator foundation does **not** replace the proven structural half.

Its operator output feeds the existing design family:

`semantic/operator state -> query-conditioned relational execution -> adaptive support -> Binder v2 -> N0 fusion/latent fabric`.

The old global-softmax/residual family remains closed.

## Training objective family

The eventual full public training objective must include all of the following in one governed curriculum.

### Broad semantic replay

Preserve language/pragmatic competence with public MLM/denoising and governed semantic replay.

### Dynamic schema discrimination

Given a natural query and runtime relation descriptions, rank or distribute mass over the correct semantics.

Training and DEV must hold out entire relation families, not just paraphrases.

### Argument-role intervention

For the same evidence and relation:

- SOURCE versus TARGET requests must change the operator state;
- relation reversal must change state where semantically required;
- symmetric relations must preserve plurality.

### Relation intervention

Hold endpoints fixed and change relation semantics.

The operator must change when the relation meaning changes.

### Factor semantics

Role, direction, traversal, control, reliability, recency, temporal constraint, provenance constraint, and future migrated factors are supplied as semantic schemas rather than fixed opaque class labels.

### Ordered composition

Train noncommutative multi-step relation sequences using the shared recurrent transition.

Order interventions must be behaviorally visible.

### Token-level evidence grounding

The semantic/operator state must expose which query tokens and schema tokens support the current hypothesis.

This is both a learnability signal and a later ordered-coverage interface.

### Open-schema and unknown handling

Include:
- unseen relation families;
- near-miss descriptions;
- irrelevant schemas;
- insufficient schema;
- ambiguous co-valid schemas;
- truly out-of-schema requests.

The model must learn defer/unknown rather than being forced to select a supplied relation.

### Hard semantic negatives

Negatives must include:
- lexical overlap but wrong direction;
- same endpoints but wrong relation;
- same relation but wrong requested role;
- same operation cue but wrong semantics;
- temporal/provenance conflicts;
- paraphrases with changed operator truth.

### Existing semantic/judgment objectives

Candidate preference, rationale alignment, paraphrase semantics, pragmatics, ambiguity, and public judgment competence remain useful. They become part of one broader representation objective rather than being treated as universal post-hoc semantic-authority heads.

## Data contract

The new curriculum is not a small matcher dataset.

It must be broad enough that runtime-schema generalization is a representation-learning problem rather than memorization of a few relation labels.

Required split isolation:

- relation-family split;
- entity split;
- lexical/paraphrase split;
- template split;
- domain split;
- multi-hop composition split;
- factor-combination split.

Synthetic public data is permitted when provenance is explicit, but generated examples must be diversified and adversarially audited for shortcuts.

Natural public language must remain substantial so the model cannot learn a private miniature instruction language.

Private Elaina/A.L.I.C.E identity data remains forbidden in N0 gradient.

## What is preserved

Job 575990 does not falsify:

- the semantic tokenizer;
- the usefulness of the existing semantic checkpoint as initialization;
- structured-state encoding;
- selected repaired evidence graph;
- evidence-view adapter;
- cross-context fusion;
- adaptive multi-view latent pool;
- corrected P1 executor mechanics;
- ordered query-evidence coverage;
- continuous relation hypotheses;
- query-conditioned structural execution;
- Binder v2's structural sparsity role;
- original frozen final N0 validation;
- full-scale/no-ceiling doctrine.

## What is closed

Do not create:

- P2S-v3;
- P2A-v4;
- another frozen-head ensemble;
- prompt-template search;
- learned fusion weights over the same failed frozen views;
- relation-ID embeddings;
- fixed factor-class semantic authority;
- fixed top-k relation commitment;
- LR/step/width/batch search as a substitute for the objective redesign.

## Scale policy

This is a **full-capability architecture**, not a reduced pilot.

Current model dimensions are operating points.

There is:

- no hard parameter ceiling;
- no hard width ceiling;
- no hard depth ceiling;
- no runtime relation-count ceiling;
- no runtime factor-count ceiling;
- no runtime hop ceiling;
- no context-view ceiling.

Parameter growth is allowed only when the architecture/data/objective is correct and measured capacity evidence shows that scaling is the remaining bottleneck.

The job-575990 evidence does **not** by itself prove that 136M parameters are too small. It proves the current learned representation objective is insufficient for the required open-schema semantics.

## Next authorized work

Authorized now:

1. build the public full-envelope semantic/operator curriculum contract;
2. define the joint schema-conditioned semantic/operator module interfaces;
3. build shortcut and leakage audits;
4. build static/no-gradient mechanics tests;
5. define one precommitted training/evaluation plan;
6. preserve the current semantic checkpoint as a comparison baseline.

Not authorized yet:

- optimizer;
- gradient;
- GPU launch;
- private identity training;
- TEST/final opening;
- N0 completion.

The next GPU run must train the **joint semantic-operator foundation**, not another repair head.
