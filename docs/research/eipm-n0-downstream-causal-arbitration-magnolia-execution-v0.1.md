# N0 downstream causal arbitration — Magnolia execution contract

## Scope

This is the real frozen downstream arbitration stage that follows canonical-only metric calibration. It does not train, repair, promote, scale, or mark N0 complete.

Execution is intentionally split into two Magnolia stages:

1. `magnolia_downstream_causal_arbitration_metric_calibration_v0_1.sbatch` runs the canonical graph twice through the complete production N0 stack and materializes the metric tolerance policy without receiving the candidate graph.
2. `magnolia_downstream_causal_arbitration_v0_1.sbatch` verifies that calibration belongs to the same canonical graph, binds the selected endpoint-repair candidate, prepares and preflights the frozen two-arm manifest, executes each arm once, then finalizes through the deterministic post-arbitration gate.

## Frozen graph arms

Canonical graph:

`$ALICE_N0_WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors`

Candidate graph:

`$ALICE_N0_WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors`

Expected candidate SHA-256:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

The arbitration runner refuses execution if the selected candidate does not match this hash.

## Calibration binding

The second stage refuses to proceed unless:

- the calibration receipt is complete;
- calibration declares that no candidate graph was supplied or observed;
- the calibration canonical graph hash equals the arbitration canonical graph;
- the metric policy says it was selected before arm results;
- the metric policy still says `results_observed=false`.

## Source freeze

Preparation binds the exact checked-out Git revision. Frozen execution then requires that same Git HEAD and a clean tracked worktree. Untracked run outputs are allowed. Pulling, switching branches, or editing tracked source between preflight and execution is not allowed.

## Decision boundary

Finalization recomputes the frozen comparisons and invokes the post-arbitration gate.

- `HARM` -> retain canonical.
- `EQUIVALENT` -> retain canonical.
- `IMPROVEMENT` with no frozen-metric degradation -> authorize at most one final frozen challenge.

The runner stops there. It never launches that challenge automatically. Candidate promotion, new training, scaling, and N0 completion remain false.
