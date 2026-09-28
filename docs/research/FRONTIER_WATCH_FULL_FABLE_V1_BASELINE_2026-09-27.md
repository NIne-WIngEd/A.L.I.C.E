# Frontier Watch baseline — full Fable v1 cognitive architecture

**Date:** 2026-09-27  
**Status:** mandatory comparison baseline for future frontier-watch intake  
**A.L.I.C.E. architecture authority:** `main@e6d0490cdf29ee56edbcb0e3d65d5783adfdf1a7` or newer  
**Fable product authority:** `NIne-WIngEd/Fable_Sleight main@e57a0ab1639076db79406e92d4d675d8e9f6c4d6` or newer

## Why this note exists

Frontier Watch must not compare new research against the superseded September 27 interpretation that used a PostgreSQL-centered, minimum-daemon, progressively shippable consumer profile.

Before judging a paper, inspect current A.L.I.C.E. main and current Fable main. This branch is a research record, not the architecture source of truth.

## Current destination

Fable v1 is the **full transferable personal cognitive foundation**.

The personal architecture includes:

- Raw Evidence/Object;
- Experience/Event Fabric;
- bitemporal Claim Authority;
- Episodes/Autobiographical memory;
- Cognitive Multi-Graph;
- associative graph compute;
- vector/multimodal retrieval;
- exact/source-native/live retrieval;
- perceptual personal memory;
- host/source-person/relationship/self/world/social/causal/mission/skill/preference/trust state;
- parametric personal memory;
- working/activation memory;
- Memory Resource Manager;
- Retrieval Orchestrator / Cognitive Recollection;
- Lifecycle Curator;
- cross-layer correction/deletion/unlearning;
- durable workflows;
- model/dataset lineage;
- multi-device/federation semantics;
- Mission Graph;
- Cognitive Workspace;
- procedural learning;
- governed personal-model evolution and rollback.

General feature models may remain replaceable GPT/Claude-class services in v1. Those services do not own the personal cognitive foundation.

## Current selected physical architecture

Current implementation direction:

- Experience/Event -> NATS JetStream behind first-party event/evidence contracts;
- Claim Authority -> XTDB v2;
- raw evidence/artifacts -> encrypted content-addressed storage behind an S3-compatible abstraction;
- host-local Cognitive Multi-Graph -> LadybugDB;
- scale-out graph placement -> JanusGraph + qualified distributed storage;
- existing graph reference -> Neo4j;
- associative graph compute -> engine-independent first-party layer;
- vector/multimodal -> Qdrant Edge/server/cluster;
- source-native -> local/live source reads;
- durable workflows -> Temporal;
- shared ephemeral workspace -> process-local L1 + Valkey;
- model/data lineage -> content-addressed signed/hashed registry;
- personal training -> PyTorch + Accelerate with hardware-adaptive distribution.

A paper may justify changing one of these implementations when it reveals a concrete capability/correctness/scale/privacy/deletion/distribution/licensing/cost defect. The mere existence of another backend is not a reason to start a tournament.

## Hardware rule

Hardware adapts **placement**, not cognition.

Allowed adaptation includes:

- sharding;
- replication;
- process/device placement;
- storage tiers;
- precision;
- offload;
- batch size;
- worker count;
- index placement;
- private/owner-authorized remote compute.

Not allowed:

- removing graph memory because the laptop is small;
- removing vector/multimodal memory to simplify installation;
- replacing durable workflows with non-durable behavior and calling it equivalent;
- dropping personal-model evolution;
- dropping cross-layer deletion/unlearning;
- creating a smaller consumer intelligence tier.

If the full build cannot fit current hardware, the system must report that requirement or use an authorized larger placement.

## Release-boundary rule

PR #98 established the consumer release doctrine now on main.

F4 through F11 are **internal qualification milestones**, not progressively shippable cognitive tiers.

- F4 — ingestion/runtime;
- F5 — selective memory/learning;
- F6 — personal intelligence;
- F7 — missions/proactive agency;
- F8 — actions/skills/self-evolution;
- F9 — expert feature integration;
- F10 — host-specific model adaptation;
- F11 — persistent environment/multi-device.

The first consumer release predicate is:

`full_personal_cognitive_foundation_after_f11`

Historical `closed_alpha`, `alpha`, and `beta` names remain distribution/test-channel vocabulary only.

F12 is post-v1 platform/ecosystem work.

## How to classify new research

For every candidate paper:

1. inspect current main first;
2. identify the paper's actual mechanism and evidence;
3. compare against the **full** architecture above;
4. classify the delta as:
   - already covered/corroboration;
   - future challenger;
   - direct current capability improvement;
   - architecture-invalidating evidence;
5. never recommend a simplification merely because the paper's benchmark has a narrower target than Fable;
6. never let a paper's fixed ontology/model count/store count become a Fable ceiling;
7. preserve Experience/Claim authority and provenance unless strong evidence truly invalidates them;
8. if current work changes, identify exact affected contracts/tests/migrations;
9. if only future work changes, record it here without disrupting current execution;
10. do not reintroduce MC10-style infrastructure validation unless the result can change a real decision.

## Research areas that now matter especially

Prioritize mechanisms that can improve:

- lifelong event/episode formation;
- temporal/causal/social/multi-graph memory;
- associative recollection;
- multimodal/perceptual personal memory;
- memory scheduling across external/parametric/activation forms;
- procedural/environment experience;
- source-native retrieval;
- proposition-level memory influence;
- correction/deletion/unlearning across execution and model state;
- multi-device memory reconciliation;
- personal-model continual learning without identity collapse;
- host/relationship/self development;
- scalable private deployment of the full fabric;
- evaluation of the complete personal loop.

A paper that only improves a generic RAG benchmark without a concrete implication for this destination should not be surfaced.
