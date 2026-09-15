# N0 v0.2 graph interference and specialist design — 2026-09-15

## First relation pilot
Magnolia job 575604 completed successfully. The first evidence-graph comparison is diagnostic evidence, not a ratification event.

Key held-out results at step 240:
- compact graph-only: macro evidence mass 0.6949287444, worst family 0.4632357657, shared structured top1 preserved at 0.7960784314.
- compact joint: macro 0.8850235204, worst family 0.6445056001, but shared structured top1 fell to 0.7137254902.
- expanded joint 512x2: macro 0.8848500450, worst family 0.6723137995, but shared structured top1 fell to 0.6862745098.

Interpretation:
- relation-specialized adaptation is materially useful;
- directly rewriting the generic structured-state checkpoint causes representation interference;
- graph-only preserves generic capability but leaves too much evidence reasoning capability unused;
- more graph capacity improved the weakest family, though not the macro average.

## Architecture defect found
The v0.1 graph had fixed pooling priors such as supersedes/corrects target = -1.0. This is semantically wrong for historical queries. A superseded memory may be irrelevant for a current-state question but exactly correct for a historical-state question.

The corrected graph removes fixed relation-status priors and learns relation pooling from:
- semantic query;
- relation type;
- source node state;
- target node state.

Conflict pooling remains explicitly symmetric under endpoint reversal.

## Multi-view specialist design
The ratified structured-state step80 remains the immutable generic typed-state view.
A new trainable EvidenceViewAdapter creates a query-conditioned specialist evidence view for graph reasoning. The graph and specialist adapter can scale without overwriting the generic structured representation.

This is capability architecture, not a parameter minimization device. There is no hard parameter ceiling.

Second comparison variants:
1. specialized_compact_384x2_graph256x1
2. specialized_expanded_640x3_graph512x2

Selection priority:
- worst-family evidence fidelity;
- macro evidence fidelity;
- supersession-historical fidelity;
- semantic summary fidelity;
- relation counterfactual dependence;
- distribution quality;
- earlier checkpoint only when capability is tied.

The semantic core remains frozen and the graph semantic cache from job 575604 is reused.
Private identity data/gradients remain closed.

## Owner capacity rule
Parameter counts describe checkpoints/configurations. They are not personality-model ceilings. Use as much capacity as full personality fidelity requires; optimize latency, memory and parameter count only after capability requirements are satisfied.
