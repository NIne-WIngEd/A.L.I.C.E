# N0 v0.2 Evidence Graph Capability Pilot

Date: 2026-09-15

## Owner rule

There is no rigid parameter-count ceiling for the A.L.I.C.E. personality model. Exact parameter counts describe concrete checkpoints only. Future semantic, structured, graph, fusion, and identity architectures may grow whenever controlled capability/fidelity evidence shows that more capacity is needed. Efficiency is secondary to capability and personality fidelity.

Finite step counts, job walltimes, and checkpoint intervals are experiment-safety controls. They are not model-capacity ceilings.

## Current ratified parents

- Semantic base: targeted-repair-v0.1 step-00000080.
- Structured-state base: structured-state-pilot-v0.1 step-00000080.
- Structured step080 remains the current initialization/base; its 1,656,064 parameters are an observed checkpoint fact, not a future architecture constraint.

## Evidence graph preparation

CPU preparation passed on Magnolia:
- tests: 23 passed
- curriculum rows: 240
- train/dev: 180/60
- families: 10
- soft evidence targets: true
- relation counterfactual training: true
- historical query coverage: true
- unresolved conflict coverage: true
- multi-hop coverage: true
- private identity content: false
- hard parameter ceiling: none

Families cover supersession current/historical, correction, unresolved/resolved conflict, support aggregation, causal explanation, temporal latest/previous, and mixed update/support.

## Capability pilot

Build head at authorization: f0a26c843854fb56b6bf251dfe8a0fa05fd5d26d

One P100 comparison, semantic core frozen/cached once. Three variants:
1. compact_graph_only: graph 256 x 1, structured frozen
2. compact_joint: graph 256 x 1, structured trainable from step080
3. expanded_joint_512x2: graph 512 x 2, structured trainable from step080

Each variant is bounded to 240 optimizer steps with saves at 80/160/240. This is an experiment bound only.

Selection priority:
1. preserve ratified structured capability
2. worst-family evidence fidelity
3. macro evidence fidelity
4. semantic conclusion quality
5. relation-counterfactual dependence
6. efficiency only after capability

If expanded joint materially improves held-out capability, retain the larger architecture. If the best variant still has weak worst-family evidence or weak relation dependence, scale again rather than protecting parameter count.

Private identity data/gradient remain false. Production promotion remains false.
