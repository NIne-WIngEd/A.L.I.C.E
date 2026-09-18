# Graphify Context Experiments

This file records empirical retrieval behavior from the external Graphify context substrate. Results are research evidence, not A.L.I.C.E. authority.

## Experiment 001 — Broad N0 topology query

- Date: 2026-09-17/18
- Graph mode: code-only
- Graph size: 10,832 nodes
- Request: identify code structurally connected to the EIPM personality-model N0 build
- Traversal: BFS depth 2
- Explicit relation context: none
- Result neighborhood: 3,785 nodes
- Returned before token truncation: 65 nodes
- Useful early hits:
  - `scripts/eipm/n0/run_n0_v02_first_tranche.sh`
  - `scripts/eipm/n0/train_n0_v02_cross_context_fusion_full_scale.py`
  - `src/alice_personality/n0/__init__.py`
  - `tests/eipm/test_n0_evidence_graph.py`
  - `src/alice_personality/n0/v02_training.py`
- Failure:
  - broad lexical seed selection admitted generic symbols and unrelated high-connectivity nodes;
  - two-hop BFS then expanded those seeds into thousands of candidates;
  - increasing the output budget would not fix the retrieval-quality problem.

### Lesson

Graph discovery is useful, but broad graph traversal cannot be the final context compiler. Discovery and precision retrieval should be separate stages.

## Experiment 002 — Relation-scoped N0 query

- Date: 2026-09-17/18
- Request: trace the N0 execution path around the first-tranche runner and full-scale cross-context-fusion trainer
- Traversal: BFS depth 2
- Explicit contexts: `call`, `import`
- Token budget: 3,500
- Result neighborhood: 555 nodes
- Returned before token truncation: 103 nodes
- Candidate-neighborhood reduction versus Experiment 001: about 85.3%
- High-value hits included:
  - `scripts/eipm/n0/train_n0_v02_cross_context_fusion_full_scale.py`
  - `scripts/eipm/n0/train_n0_v02_cross_context_fusion_repair_full_scale.py`
  - `scripts/eipm/n0/run_n0_v02_first_tranche.sh`
  - `scripts/eipm/n0/run_n0_v02_cross_context_fusion_full_scale.sh`
  - `src/alice_personality/n0/cross_context_fusion.py`
  - `CrossContextFusion`
  - `src/alice_personality/n0/v02_training.py`
  - `src/alice_personality/n0/config.py`
  - `src/alice_personality/n0/curriculum_data.py`
  - `src/alice_personality/n0/evidence_graph.py`
  - `src/alice_personality/n0/structured_state.py`
  - `src/alice_personality/n0/evidence_view_adapter.py`
  - `src/alice_personality/n0/cross_context_fusion_anchored.py`
  - `src/alice_personality/n0/cross_context_fusion_repair_objectives.py`
  - relevant N0 evaluation and diagnosis scripts

### Remaining failure

The result still contains generic or irrelevant seeds such as common helper names and unrelated test symbols. The query CLI uses a fixed two-hop traversal, so precise continuation work will need a second-stage operator such as exact-node explain/path/affected traversal or an A.L.I.C.E.-specific context compiler on top of the graph.

### Lesson

Relation scoping produces a large improvement without simply spending more tokens. Future A.L.I.C.E. context retrieval should favor:
- exact identifiers when known;
- typed relation constraints;
- authority and temporal filters;
- mission-local neighborhoods;
- a precision pass after broad discovery.

## Experiment 003 — Exact-looking symbol query still broadens lexically

- Date: 2026-09-17/18
- Request: `CrossContextFusion in the N0 personality-model path`
- Traversal: BFS depth 2
- Explicit context: `call`
- Token budget: 2,500
- Result neighborhood: 77 nodes
- Returned before token truncation: 71 nodes
- Correct primary hit:
  - `CrossContextFusion` at `src/alice_personality/n0/cross_context_fusion.py:L214`
- Incorrect or low-value seed broadening included:
  - `Path`
  - `_model()`
  - `n0/model.py`
  - unrelated Phase 2 memory-deletion tests and semantic-retrieval nodes

### Lesson

A natural-language query that contains an exact symbol name is still not equivalent to exact-node lookup. Lexical/semantic seed expansion can admit generic symbols before graph traversal begins.

For continuation work:
- `graphify query` is a discovery operator;
- exact-node explain/get-node/path/affected operators should be used for precision follow-up;
- generic symbol names should be down-weighted or excluded in an A.L.I.C.E.-native context compiler;
- source path and mission scope should be first-class retrieval constraints, not merely words in the query.

## Experiment 004 — Concurrent research update during graph publication

- Date: 2026-09-17/18
- Extraction itself succeeded:
  - 10,832 nodes
  - 37,624 edges
  - 322 communities
- Failure:
  - while the Action was extracting, a research-protocol commit advanced the branch;
  - the generated graph commit was rejected as a non-fast-forward push.
- Safety impact:
  - no corrupted graph was published;
  - however, a naive bot publication step is vulnerable to harmless research edits racing a long extraction.

### Fix

The graph publisher now:
1. stages generated artifacts outside the working tree;
2. fetches the newest research head and live A.L.I.C.E. source head;
3. aborts if the real source branch moved during extraction;
4. resets to the newest research head;
5. verifies that the current source SHA is still contained;
6. reapplies generated artifacts;
7. commits and pushes on top of the newest research state.

The query-result publisher uses the same pattern and also refuses to publish if a newer query request replaced the request it actually executed.

### Lesson

Long-lived context compilation must be transaction-like. Retrieval/index generation cannot assume the surrounding project state remains frozen while compilation is running.

## Operational lesson — Graphify v8 query CLI

Verified supported controls for `graphify query`:
- `--graph <path>`
- `--budget <tokens>`
- repeated `--context <relation-family>`
- `--dfs`

The query command does **not** use `--top` or `--json`. The first bridge revision passed those unsupported flags and Graphify silently ignored them. The bridge was corrected on 2026-09-17/18.

This is itself a continuity lesson: tool-interface assumptions must be captured after verification so a later session does not rediscover the same mistake.
