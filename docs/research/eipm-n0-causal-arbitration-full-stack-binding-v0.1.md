# N0 downstream causal arbitration: full-stack graph binding v0.1

## Purpose

This binding closes the graph-to-downstream causal boundary without introducing a proxy model. Each arbitration arm is a real `DualEndpointEvidenceGraphEncoder` checkpoint. The evaluator runs that checkpoint through the production N0 path:

`semantic model -> structured state -> evidence adapter -> dual-endpoint evidence graph -> ratified source-anchored fusion -> adaptive multi-view latent pool -> frozen downstream metrics`

Only the evidence-graph checkpoint changes between arms. The latent checkpoint, evaluation set, semantic parent, structured parent, evidence adapter, fusion checkpoint/config/ratification, tokenizer, and all model configs remain common and frozen.

## Two-layer freeze

The outer `n0_downstream_causal_arbitration_v0.2` manifest hashes four regular files: the evaluator, latent checkpoint, evaluation set, and stack-binding manifest. It separately hashes the canonical and candidate graph checkpoints.

The stack-binding manifest then hashes the rest of the production stack. Regular files use raw SHA-256. Directory artifacts use a deterministic directory-tree SHA-256 over a canonical JSON list of every regular file's sorted relative path, file SHA-256, and byte size. Symlinks and empty directory artifacts are rejected.

This keeps the outer arbitration protocol simple while still freezing the directory-based checkpoints and tokenizer used by the production evaluator.

## Source binding and fail-closed behavior

The stack manifest must name an immutable 40-hex Git revision. At execution time the evaluator requires the checked-out repository `HEAD` to match it exactly. Any file hash, directory-tree hash, external input hash, artifact type, required artifact set, or governance invariant mismatch fails before CUDA/model loading.

The common stack manifest cannot contain a graph/evidence-graph artifact. That prevents the arm-varying checkpoint from being smuggled into common state. The evaluator loads the arm graph with the existing production `load_public_parents` path, which constructs `DualEndpointEvidenceGraphEncoder(expanded_graph_config())` and uses strict state-dict loading.

## Metrics

The evaluator emits the existing latent-pool `corrected_evaluate` metrics plus the existing counterfactual target-drop metrics. The outer arbitration manifest declares comparison direction and tolerances. Tolerances are intentionally not selected by this implementation. They must be frozen before preflight from the already-authorized comparison policy/evidence.

## Governance

This evaluator is diagnostic only. It has no optimizer, training, promotion, repair-ratification, or scaling path. A downstream `IMPROVEMENT` classification still does not ratify the endpoint repair; it only feeds the separate post-arbitration decision gate, which can authorize at most one final frozen challenge.
