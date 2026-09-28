# FBM transfer requirement — full Fable v1 cognitive fabric

**Date:** 2026-09-27  
**Status:** required builder-direction correction  
**Upstream architecture authority:** A.L.I.C.E. main `e6d0490cdf29ee56edbcb0e3d65d5783adfdf1a7`  
**Fable product authority:** Fable main `e57a0ab1639076db79406e92d4d675d8e9f6c4d6`

## Correction

FBM must not learn a procedure that produces a reduced "desktop profile."

The builder's target is the **complete transferable personal cognitive architecture**. Hardware adaptation changes placement and execution strategy. It does not delete cognitive capabilities.

The earlier interpretation "capability parity without daemon parity" was too easy to misuse as "collapse physical systems until a capability quietly disappears." That interpretation is superseded.

## What FBM must be able to assemble

FBM must eventually construct and wire:

- encrypted raw evidence/object storage;
- replayable Experience/Event fabric;
- bitemporal Claim Authority;
- episodes/autobiographical memory;
- Cognitive Multi-Graph;
- associative graph computation;
- vector/multimodal retrieval;
- source-native/live retrieval;
- protected perceptual personal memory;
- host / relationship / assistant-self / world / social / causal / mission / preference / trust / skill state;
- parametric personal memory with lineage;
- working/activation memory;
- Memory Resource Manager;
- Retrieval Orchestrator / Cognitive Recollection;
- Lifecycle Curator;
- cross-layer correction/deletion/unlearning;
- durable workflows;
- model/data lineage;
- multi-device/federation semantics;
- the personal-model set required by the user's actual evidence.

No fixed model count is part of the product definition.

## Current selected physical substrate

FBM should learn adapters/planners for the current implementation direction:

- NATS JetStream for Experience/Event;
- XTDB v2 for Claim Authority;
- content-addressed object storage;
- LadybugDB local graph placement;
- JanusGraph scale-out graph placement;
- Qdrant vector/multimodal;
- Temporal workflows;
- Valkey shared ephemeral state;
- PyTorch + Accelerate training.

These engine names are current implementation knowledge, not the builder's semantic target. FBM learns the **logical role first**, then selects/instantiates a qualified physical adapter.

## Hardware-adaptive planning

FBM may alter:

- process placement;
- host count;
- storage tier;
- sharding/partitioning;
- replication;
- batch size;
- precision;
- offload;
- worker count;
- index placement;
- cache placement;
- training schedule;
- owner-authorized remote compute.

FBM may not silently remove:

- graph memory;
- vector/multimodal memory;
- episodes;
- source-native retrieval;
- personal cognitive models;
- procedural memory;
- parametric learning;
- activation/working memory;
- deletion/unlearning;
- durable workflow semantics;
- multi-device continuity.

If the full qualified build does not fit current hardware, the builder must report the requirement, propose owner-authorized compute, or defer completion. It must not claim that a smaller system is the same Fable.

## Builder-training implication

Future FBM traces/cases should explicitly distinguish:

1. **capability decision** — what cognitive role is required;
2. **representation decision** — external/episodic/graph/vector/procedural/parametric/activation form;
3. **physical placement decision** — which engine/device/cluster implements it;
4. **resource decision** — how to schedule it on available hardware.

A weak machine may change (3) and (4). It must not change (1) merely for convenience.

## Deletion/unlearning implication

FBM must build lineage deep enough to trace influence through:

source -> Event -> Claim -> Episode/Graph/Vector/Summary -> Context/Plan -> Dataset/Replay -> Personal Model -> later regenerated memory.

This is required so user deletion can remove or quarantine downstream influence rather than only removing the visible source row.

## Release-boundary implication

The builder must not learn a staged product ladder where a partial personal architecture becomes a smaller consumer Fable.

F4 through F11 are internal qualification stages only:

- ingestion/runtime;
- selective memory/learning;
- personal intelligence;
- missions/proactive agency;
- actions/skills/self-evolution;
- expert feature integration;
- host-specific adaptation;
- persistent environment/multi-device continuity.

FBM may produce intermediate artifacts while constructing or testing these stages. Those artifacts are not consumer release targets.

The consumer release predicate is:

`full_personal_cognitive_foundation_after_f11`

General feature engines may remain API-backed. The builder may not satisfy the release predicate by outsourcing or omitting the personal cognitive core.

## Qualification implication

An FBM-built fresh instance does not pass because:

- the five initial models exist;
- chat works;
- memory search returns something;
- the API draft sounds personalized.

It passes only when the connected personal architecture works end to end and can scale its physical placement without losing cognitive planes.
