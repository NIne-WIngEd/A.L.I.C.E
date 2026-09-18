# Graphify Context Substrate Research

This branch is an external research and continuity substrate for work on A.L.I.C.E.

It does **not** make Graphify part of A.L.I.C.E.'s architecture, runtime, authority chain, memory system, or trusted computing base.

## Purpose

We are using Graphify for two linked goals:

1. Reduce context reconstruction cost for long-running A.L.I.C.E. development sessions.
2. Study which graph/context techniques are worth carrying forward into A.L.I.C.E.'s own continuous-memory and context architecture.

Graphify output is a routing aid. The authoritative source remains the original repository file, receipt, branch, commit, evaluation artifact, or owner-ratified record.

## Hard boundary

- No Graphify-derived claim becomes A.L.I.C.E. truth merely because it appears in a graph.
- Inferred graph edges are never treated as source authority.
- Material architectural, identity, memory, training, deployment, or governance decisions must be verified against original sources.
- This branch is research-only and must not silently merge into A.L.I.C.E. main.
- Private local material must not be added to this public branch.
- Graphify is replaceable. The research findings are the durable asset.

## Working loop

1. Build or update the Graphify graph against the current working tree.
2. Ask a narrow project-state question.
3. Use the returned subgraph to identify the smallest set of original sources that need reading.
4. Verify material claims in those sources.
5. Do the actual work.
6. Record retrieval misses, stale-state failures, bad inferences, and useful techniques in the seed ledger.
7. Update the graph after meaningful repository changes.

## Initial benchmark questions

The first benchmark set should include:

- Where exactly is EIPM personality-model N0 now?
- What is the next valid N0 action?
- Which Magnolia execution lessons are still active?
- Which historical errors must not be repeated?
- Which files and receipts justify the current personality-model architecture?
- Which constraints are intentional and which were temporary implementation gates?
- Which prior decision superseded an older one?

The objective is not merely token reduction. The context substrate must preserve decision lineage, temporal state, provenance, failure lessons, and source authority.

## Branch origin

Created from `alice-eipm-v1-build` on 2026-09-17 for the continuity failure observed across several long A.L.I.C.E. development chats.
