# N0 v0.2 relation-essential repair checkpoint

Date: 2026-09-16

## Frozen relation-essential result

Magnolia job 575616 completed the frozen relation-essential evaluation over 40 paired cases / 80 examples. The challenge keeps text and query identical inside each pair and changes only relation topology. Training on this challenge is forbidden.

Best pre-repair checkpoint:
- variant: specialized_expanded_640x3_graph512x2
- checkpoint: step-00000080
- example top-1: 0.80
- pair-flip accuracy: 0.65
- worst-family pair-flip: 0.0
- causal_chain: 0.0
- causes_direction: 0.0
- corrects_direction: 0.25
- temporal_direction: 0.50

Compact specialist was materially worse on this strict relation-only task (best pair-flip 0.375), so expanded capacity is retained as the repair parent even though ordinary graph-dev metrics had previously shown little size benefit.

## Diagnosis

This is not evidence for generic model scaling. The v0.2 graph read learned directed-relation pooling mainly on the relation target endpoint. Direction-sensitive questions also need an explicit learned source-endpoint role. The frozen challenge exposed this especially for causal/root, correction direction, and temporal direction.

## Repair architecture

New module:
- src/alice_personality/n0/evidence_graph_dual_endpoint.py

It subclasses the v0.2 graph encoder and adds a query-conditioned source_relation_pool_mlp while retaining the existing target relation MLP. Conflicts remain symmetric. The source head final layer is zero-initialized, so the expanded step80 parent loads compatibly and initially preserves old behavior.

The evidence-view adapter stays frozen. Structured step80 and the 136.6M semantic core remain frozen. Only relation-read parameters are trainable.

## Independent repair curriculum

- 320 rows / 160 pairs
- 240 train rows / 120 train pairs
- 80 dev rows / 40 dev pairs
- 10 relation families
- identical text/query inside each pair
- graph topology flips the correct evidence target
- independent public analogues; frozen relation-essential challenge rows are not used
- no private identity content
- no hard parameter ceiling

Training mixes paired relation repair with replay from the original graph curriculum. The final decision still requires evaluation on the untouched frozen relation-essential challenge.

## Execution

Build launcher:
- scripts/eipm/n0/magnolia_p100_n0_v02_relation_repair.sbatch

One P100, max 160 steps, save 40/80/120/160. This is a bounded repair experiment, not an architecture-size ceiling or open-ended tuning program.
