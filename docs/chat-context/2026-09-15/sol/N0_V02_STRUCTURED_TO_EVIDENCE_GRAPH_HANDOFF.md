# N0 v0.2 Structured-State -> Evidence Graph Handoff

Date: 2026-09-15

## Ratified boundary preserved

The existing public structured-state pilot remains unchanged:

- implementation: `src/alice_personality/n0/structured_state.py`
- exact branch parameters: 1,656,064
- semantic parent: `targeted-repair-v0.1/step-00000080`
- semantic parent trainable parameters: 0
- private identity data/gradient: false
- bounded pilot: 240 steps, checkpoints 80/160/240

During graph work, an initial attempt placed relation-aware parameters inside `StructuredStateEncoder`. Review found that `train_n0_v02_structured_state_pilot.py` hard-locks `EXPECTED_BRANCH_PARAMETERS = 1_656_064`. The graph parameters were therefore separated rather than silently changing an already-authorized experiment.

## Evidence graph mechanics added on build branch

Build branch: `alice-eipm-v1-build`

New model sidecar:
- `src/alice_personality/n0/evidence_graph.py`
- `EvidenceGraphEncoder`
- exact default sidecar parameters: 937,728
- semantic core parameter growth: 0
- structured-state parameter growth: 0
- no positional embeddings
- no private identity parameters
- no graph-library dependency

The sidecar consumes contextualized structured field states rather than raw/private text. It implements sparse typed directed message passing plus query-conditioned evidence pooling.

Stable public relation vocabulary currently includes:
- supports
- corrects
- supersedes
- conflicts_with
- derived_from
- causes
- temporal_successor

Memory Core direction is preserved. `corrects` and `supersedes` point replacement -> historical record. `conflicts_with` is treated symmetrically even though Memory Core stores one canonical relation edge.

Lifecycle pooling priors are deterministic public mechanics:
- supports target: +0.20
- corrects historical target: -1.00
- supersedes historical target: -1.00
- conflicts both endpoints: -0.35

New Memory Core compiler:
- `src/alice_personality/n0/evidence_graph_data.py`
- consumes `alice_memory.temporal.MemoryRelation`
- preserves `from_memory_id -> to_memory_id`
- deterministic relation-id ordering
- ignores edges outside the selected graph slice
- explicit relation confidence
- fails closed on edge overflow; never silently truncates evidence

New tests:
- `tests/eipm/test_n0_evidence_graph.py`
- `tests/eipm/test_n0_evidence_graph_data.py`

Registered config:
- `configs/eipm/n0/n0_v02_evidence_graph_v0.1.json`
- status: `MECHANICS_READY_NO_GRAPH_GRADIENT_AUTHORIZED`

## Architectural properties tested in code

- consistent node permutation must preserve graph read
- supersession updates/penalizes the historical target, not the replacement source
- conflict behavior is symmetric independent of canonical storage orientation
- directed relations do not mutate unrelated/disconnected nodes
- query changes evidence pooling without requiring structured-field re-encoding
- active invalid edges fail closed
- sidecar stays below 1M parameters and does not grow the semantic core

## Next actual model action

Do not start graph training yet.

Run the already-authorized structured-state pilot first:

```bash
cd "$HOME/rayan-compute/rayan-eipm-main"
git pull --ff-only
sbatch scripts/eipm/n0/magnolia_p100_n0_v02_structured_state_pilot.sbatch
```

The launcher writes:
- `rayan-n0-struct-<jobid>.out`
- `rayan-n0-struct-<jobid>.err`
- `$HOME/rayan-compute/rayan-n0/n0-v02/structured-state-pilot-v0.1/structured_state_pilot_comparison.json`

After the run, select or reject the smallest useful structured checkpoint. Only then may a bounded public identity-neutral graph/evidence gradient be considered. Private N1 identity gradients remain closed.

## Anti-MC10D rule

Do not create a separate graph qualification program. Evaluation exists only to select or reject a concrete model capability update. Infrastructure must serve the model, not become the project.
