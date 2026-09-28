# A.L.I.C.E. / Fable continuation anchor — 2026-09-27

**Purpose:** prevent future continuation chats from reverting to the superseded "lighter Fable v1" interpretation or skipping required execution lanes.

## Canonical correction now on main

A.L.I.C.E. PR #97 merged to main at:

`e6d0490cdf29ee56edbcb0e3d65d5783adfdf1a7`

Fable PR #1 merged to main at:

`191b0dbf8c1c33114b56e8e5e2c4be52076cd4ce`

The correction is architectural, not cosmetic:

- Fable v1 is the **full transferable personal/cognitive architecture**.
- PR #98 later closed the remaining release-governance loophole: F4–F11 are internal engineering/qualification milestones, not progressively shippable alpha/beta cognitive tiers.
- The first consumer release uses one gate: `full_personal_cognitive_foundation_after_f11`.
- Historical `closed_alpha` / `alpha` / `beta` labels are distribution/test channels only and cannot override the v1 capability gate.
- There is no small/light/local-lite intelligence tier.
- Hardware can change placement, sharding, scheduling, precision, and service topology.
- Hardware may not remove a logical cognitive plane.
- Only replaceable general feature-model work is API-backed in v1.
- Personal memory, identity, host understanding, relationship state, Fable self, native judgment, learning, mission state, procedural experience, and continuity remain first-party.

## Selected full memory architecture

The current physical implementation direction is:

- Raw evidence / large objects -> encrypted content-addressed storage behind an S3-compatible abstraction;
- Experience/Event Fabric -> NATS JetStream behind project-owned event/evidence contracts;
- Claim Authority -> XTDB v2 bitemporal authority;
- Episodes -> governed derived autobiographical/episode plane;
- Cognitive Multi-Graph -> LadybugDB host-local, JanusGraph for scale-out private-cluster placement, Neo4j retained as existing A.L.I.C.E. reference;
- Associative graph compute -> engine-independent PPR / spreading activation / temporal decay / inhibition / path computation;
- Vector/Multimodal -> Qdrant Edge/server/cluster;
- Exact/source-native -> files, SQL/FTS, code/source search, structured APIs and live systems of record;
- Perceptual personal memory -> grounded specialist representations linked to exact source evidence;
- Personal cognitive projections -> host/source-person/relationship/self/world/social/causal/mission/skill/preference/trust state;
- Parametric personal memory -> versioned personal models/adapters/rankers/routers with influence lineage;
- Working/activation memory -> Cognitive Workspace plus execution-state provenance;
- Memory Resource Manager -> required first-party control plane;
- Retrieval Orchestrator/Cognitive Recollection -> required first-party control plane;
- Lifecycle Curator -> required first-party control plane;
- Deletion/unlearning -> cross-layer influence graph spanning durable, execution and parametric state;
- Durable workflows -> Temporal;
- Shared ephemeral state -> process-local L1 + Valkey;
- Model/data lineage -> content-addressed signed/hashed registry;
- Training -> PyTorch + Accelerate with hardware-adaptive distributed strategy.

This does **not** reintroduce MC10-style backend tournaments. The architecture is selected. Challengers are admitted only when a concrete observed blocker or strong frontier evidence can change a real decision.

## Consumer release-boundary correction

Canonical release-governance merge:

`e6d0490cdf29ee56edbcb0e3d65d5783adfdf1a7` (PR #98)

Interpretation:

- F4 = ingestion/runtime qualification;
- F5 = selective memory/learning qualification;
- F6 = personal intelligence qualification;
- F7 = mission/proactive-agency qualification;
- F8 = action/skill/self-evolution qualification;
- F9 = expert feature integration qualification;
- F10 = host-specific model-adaptation qualification;
- F11 = persistent environment/multi-device qualification;
- **none is a standalone consumer Fable release**;
- Fable v1 becomes release-eligible only after the complete F4–F11 personal foundation is qualified as one entity;
- F12 is post-v1 SDK/platform/ecosystem expansion.

Do not interpret any legacy profile named `friday.learning_alpha` as a reduced product tier. Its ID is retained for compatibility and now explicitly means internal qualification only.

## Non-skippable execution lanes

These are not one giant serial chain. They may overlap where dependencies permit, but no lane may be silently skipped.

### Lane A — N0 public identity-neutral foundation

Current active branch:

`alice-eipm-v1-n0-full-envelope-joint-ddp-v2@171e0cc21ef420a5bf93d01cd2631de8e789a2be`

Required order:

1. exact-source whole-step branch closure;
2. full two-rank CUDA joint backward;
3. actual optimizer-step peak-memory qualification;
4. exact checkpoint/resume qualification;
5. bounded DEV lifecycle;
6. actual public N0 training;
7. frozen full-envelope closure on unseen relation/schema/cardinality/composition/causal-flip/uncertainty/long-context/judgment/final suites.

Do not shrink the model or objective to fit old hardware.

### Lane B — N1 / N2 / N3 EIPM

Only after N0 readiness:

- N1 private identity representation;
- N2 native context-conditioned judgment/preferences and full IDP behavior;
- N3 calibration, failure tails, owner fidelity and provider-swap stability.

N0 may be progressively unfrozen if fidelity requires it. N0 is a foundation, not a permanent frozen ceiling.

### Lane C — MFM

Current branch:

`research/mfm-foundation-20260923@00583fb25fbe07c1452519992de2bc8055dcebd5`

Can progress in parallel where it does not require final EIPM weights:

- Formation Context Planner;
- gold semantic decomposition;
- learned proposal model;
- dynamic episodes/scenes/domains;
- fast/slow formation;
- MFM -> Gate -> projection -> retrieval qualification.

MFM proposes. It does not grant truth authority.

### Lane D — full Memory v4.x successor

Implement the full planes listed above.

Do not collapse back to "PostgreSQL plus optional graph/vector" or "minimal local services."

### Lane E — host / relationship / assistant-self development

Build and qualify:

- host/user model;
- relationship model;
- A.L.I.C.E./Fable self;
- outcome-driven governed revision;
- causal intervention tests showing each state changes only relevant judgment.

### Lane F — Mission Graph / Workspace / procedural learning / self-improvement

These are part of the personal foundation.

Do not postpone them merely because external APIs supply feature generation.

### Lane G — integrated Stage G

Qualify the **selected full cognitive-memory fabric**, not backend permutations.

Must include:

- Event -> MFM -> Gate -> Claim;
- episodes/graph/vector/personal/procedural projections;
- associative + vector + source-native + episodic + mission retrieval;
- Context Planner -> EIPM judgment;
- outcome -> Experience -> governed revision;
- deletion across durable/execution/parametric influence;
- multi-device reconciliation;
- rebuild/restore/rollback;
- realistic scale with no certification ceiling.

### Lane H — Phase 2 replacement

Non-skippable sequence:

- Stage H: bounded successor canary authority;
- Stage I: canonical cutover;
- Stage J: compatibility/fallback acceptance and final Phase 2 replacement/retirement.

Phase 2 is **not** fully replaced at H or I.

### Lane I — FBM transfer

Current branch:

`fable-builder-model@d8b477d33631f6c7ed67dff1e63450b7e19effd1`

FBM must learn the real construction process:

- source intake;
- evidence semantics;
- coverage planning;
- personal-model construction;
- full memory-fabric assembly;
- failure localization and repair;
- hardware-adaptive placement without capability deletion;
- cross-user qualification.

### Lane J — Fable v1

Fable main after PR #1:

`191b0dbf8c1c33114b56e8e5e2c4be52076cd4ce`

Release requires a fresh user corpus to produce one connected personal entity with:

- full memory fabric;
- host/relationship/self development;
- MFM + deterministic authority;
- native personal judgment;
- controlled API feature generation;
- outcome learning;
- mission/workspace/procedural memory;
- deletion/unlearning;
- model/provider/device continuity;
- physical scale-up/scale-out without a smaller intelligence tier.

## Frontier research lane

Current frontier branch:

`research/frontier-watch@7b250b98e34068ef6cd87c648a58f05fc7aa70ec`

Continue research intake, but compare every paper against the **full** destination above. A paper is not permission to narrow the architecture or start a benchmark tournament.

## Resume rule

When a future chat resumes A.L.I.C.E./Fable work:

1. read current repo/branch heads first;
2. use this file only as a continuation map;
3. treat main architecture docs as authority;
4. preserve solved failure evidence;
5. never infer that deployment simplification permits capability removal;
6. continue the active lane from its exact current blocker rather than restarting architecture selection.
