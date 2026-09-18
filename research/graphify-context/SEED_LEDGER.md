# Graphify Seed Ledger

This ledger captures transferable ideas and failure evidence from Graphify without importing Graphify as an A.L.I.C.E. dependency.

## Seed 001 — Persistent graph instead of full re-ingestion

Graphify persists a graph that can be queried later rather than rebuilding understanding from every raw file each session.

Potential A.L.I.C.E. lesson:
- session continuity should depend on durable state, not repeated prompt reconstruction;
- unchanged knowledge should not require full cognitive re-ingestion.

Status: **promising / requires A.L.I.C.E.-native design**

## Seed 002 — Incremental invalidation

Graphify fingerprints inputs and can update only changed material.

Potential A.L.I.C.E. lesson:
- memory/context compilation should be change-aware;
- unchanged evidence, claims, and relations should retain stable identities;
- invalidation and supersession should be explicit rather than handled by wholesale rebuild.

Status: **promising**

## Seed 003 — Separate extracted from inferred relations

Graphify marks graph relationships by confidence/origin rather than flattening all edges into equal truth.

Potential A.L.I.C.E. lesson:
- derived associations must remain distinguishable from source-grounded relations;
- inference should preserve provenance and be revisable;
- retrieval convenience must never silently promote inference to canonical fact.

Status: **strong fit with existing A.L.I.C.E. doctrine**

## Seed 004 — Community-aware navigation

Graphify clusters the graph into connected subsystems and exposes high-connectivity nodes.

Potential A.L.I.C.E. lesson:
- large context spaces need structural navigation before raw retrieval;
- mission, memory, code, identity, evaluation, and operational subgraphs may need different retrieval policies;
- high-connectivity does not equal high-authority, so centrality must not become importance by default.

Status: **useful with caution**

## Seed 005 — Query first, raw read second

Graphify's agent integration can make the graph the first navigation step before broad source reads.

Potential A.L.I.C.E. lesson:
- a context compiler should locate the relevant evidence neighborhood before expensive source loading;
- the system must still open original authority for consequential decisions.

Status: **immediate Sol workflow candidate**

## Seed 006 — Query/path/explain primitives

Graphify exposes distinct operations for scoped retrieval, connection tracing, and node explanation.

Potential A.L.I.C.E. lesson:
- one generic semantic search endpoint is probably insufficient;
- different cognitive questions require different retrieval operators;
- causal/lineage questions should be handled differently from topical recall.

Status: **promising**

## Seed 007 — Graph diff

Graphify includes graph-difference analysis between old and new states.

Potential A.L.I.C.E. lesson:
- continuity needs explicit change detection;
- context compilation should be able to answer not only "what is true?" but "what changed since the last trusted state?"

Status: **high value**

## Seed 008 — Rationale and documentation links as first-class context

Graphify can connect code to rationale-style documentation.

Potential A.L.I.C.E. lesson:
- implementation state without decision rationale is insufficient;
- "why" records should survive refactors and remain retrievable beside "what" records.

Status: **high value**

## Seed 009 — Local structural extraction where possible

Graphify parses code structurally without requiring an LLM for that portion.

Potential A.L.I.C.E. lesson:
- deterministic extraction should be preferred where the source domain permits it;
- models should be reserved for ambiguity, interpretation, synthesis, and cross-domain reasoning.

Status: **strong**

## Seed 010 — Graph alone is not continuous memory

Observed limitation for A.L.I.C.E.-level requirements:

A graph of concepts and relations does not by itself provide:
- canonical adjudicated truth;
- bitemporal validity;
- correction and supersession semantics;
- experience-to-learning conversion;
- identity continuity;
- mission state;
- selective retention and forgetting;
- owner/host/self separation;
- confidence-aware belief revision;
- procedural learning;
- long-horizon causal memory;
- deletion propagation across all derived state;
- context compilation optimized for a particular active goal.

Status: **hard design boundary**

---

## Experiment record template

For every substantial retrieval test, append:

### Test
- Date:
- Task:
- Repository ref:
- Graph version/hash:
- Query:
- Raw-context baseline:
- Graph-selected sources:
- Sources actually required:
- Important misses:
- False positives:
- Stale/superseded items:
- Unsupported inferences:
- Token/read reduction:
- Did the task complete correctly:
- Design lesson:
- Candidate A.L.I.C.E. requirement:
