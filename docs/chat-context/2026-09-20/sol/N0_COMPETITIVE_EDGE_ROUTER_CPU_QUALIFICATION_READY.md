# N0 Competitive Edge Router — CPU Qualification Ready

**Date:** 2026-09-20

## Source failure

Magnolia job `575888` completed normally and produced valid model evidence.

- scheduler: `COMPLETED`
- exit: `0:0`
- source revision: `cf15e2b13a5ca4d22a7070e54c95460959206c33`
- result SHA-256: `99fc21394bdfbb8e3500bf0efa7c3ce836a2e071b8b11a1c733f73aa99b6fea2`
- result status: `FAIL_QUERY_EDGE_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`
- eligible checkpoints: none
- test opened: false
- frozen challenge opened: false
- parent graph parameters changed: false

The one authorized routed-specialist P100 run is consumed.

Do not rerun it.

## Failure localization

The edge-conditioned cross-attention representation remains supported.

What failed was the routing/control plane.

The training objective supervised:

`dot(edge_query_language_state, edge_binding_query)`

as a routing proxy.

Runtime specialist contribution was instead controlled by:

`tanh(gate_mlp(...))`

These were different quantities.

The proxy routing loss stayed near the random three-way baseline for most of training, while the actual independent edge gates became large on irrelevant edges.

By step 160–200, causal field behavior had improved and parent preservation was much closer than the previous query-only architecture, but routing remained below the required threshold and worst-family routing remained poor.

Therefore the rejected pattern is:

`proxy routing score + independent per-edge tanh contribution gates`

Do not repair that pattern by changing loss weights, learning rate, training length, width, thresholds, or adding gradient surgery.

## New branch

`alice-eipm-v1-query-edge-competitive-router`

Current exact head:

`93901b4ea0e3a9773e59047181414e507d3ed904`

Final static workflow:

`35486018658`

Conclusion:

`SUCCESS`

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.27.json`

## One causal architecture change

Preserve the qualified query-edge cross-attention representation.

Replace only the control plane with:

**Competitive Edge Router with Parent/No-Op Route**

For each active directed edge:

- edge-specific language/graph fusion is computed as before;
- one route logit is emitted;
- an explicit parent/no-op route is added;
- all active directed edges plus no-op compete in one softmax;
- edge route probability directly multiplies that edge's source and target residual proposals.

Therefore the quantity that future routing supervision would optimize is the exact quantity that controls runtime specialist contribution.

There is no proxy dot-product router.

There is no independent per-edge tanh gate.

## Exact-parent initialization

Source and target residual readout heads are exactly zero-initialized.

The route head is also zero-initialized.

At initialization:

- route probabilities are live and normalized;
- residual proposals are exactly zero;
- specialist contribution is exactly zero;
- parent graph output is therefore exact.

This avoids a dead first step:

- route supervision can update route logits immediately;
- field-selection loss can update residual readout weights immediately.

## Route space

Future causal route space:

`parent/no-op + all active directed edges`

The route is not restricted to same-relation candidates.

This makes relation selection and edge selection one production-real decision.

The existing multi-edge causal curriculum remains unchanged for causal attribution:

- three same-relation edges;
- one different-relation distractor;
- known relevant edge;
- TEST still unopened.

## Static contracts passed

Workflow `35486018658` verified:

- source failure lineage;
- CPU-only current state;
- competitive route is actual runtime control;
- explicit no-op route;
- no old proxy router;
- no independent tanh edge gates;
- zero-initialized route head;
- zero-initialized source and target residual heads;
- frozen parent integration;
- no-gradient qualifier contains an explicit non-persisted route-control probe;
- qualification runner is exact-hash bound and contains no training.

## Next action

Only one CPU/no-gradient runtime qualification is authorized.

Qualifier:

`scripts/eipm/n0/qualify_n0_v02_query_edge_competitive_router_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_qualify_query_edge_competitive_router_v0_1.sh`

It must prove:

- exact parent field weights;
- exact parent pooled state;
- parent tensors unchanged;
- route probabilities sum to one;
- explicit no-op has live probability;
- conflict/PAD do not receive directional route probability;
- same-relation edge states remain distinct;
- edge content still changes query-token attention;
- route probability is the actual source contribution multiplier;
- route probability is the actual target contribution multiplier.

The last two are tested with a temporary non-persisted no-gradient probe that makes residual proposals non-zero and checks exact contribution scaling.

## Still closed

No optimizer.

No gradient.

No GPU.

No P100.

No TEST.

No frozen challenge.

No scale.

No parent or semantic retraining.

No private identity gradient.

No production promotion.

N0 remains incomplete.

A CPU PASS still requires a separate training-objective decision.
