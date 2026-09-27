# A.L.I.C.E. execution path to Phase 2 replacement and Fable v1

**Date:** 2026-09-27  
**Status:** owner-directed execution plan  
**Purpose:** convert the existing research-heavy program into a capability-first build sequence with a selected default infrastructure and bounded validation.

## 1. North star

A.L.I.C.E. is not an infrastructure benchmark. The destination remains the companion described by the README and *The Second Mind*: a persistent personal intelligence that understands the whole life context, remembers selectively, develops judgment, can disagree, preserves source/host/self distinctions, learns from outcomes, and survives replacement of general-purpose models.

Infrastructure exists to support that behavior. It is not itself the product.

The execution rule is therefore:

> **Build the strongest coherent A.L.I.C.E. we can justify from the architecture we already understand. Validate enough to protect correctness and scarce compute. Do not run technology tournaments unless a concrete capability or operational problem creates a decision that cannot be made from current evidence.**

This rule supersedes the earlier interpretation of Stage G that required exhaustive all-candidate/all-pairs infrastructure qualification.

## 2. What is already done

The program already has substantial irreversible-value work behind it:

- Constitution, owner authority, provenance, correction, deletion, rollback, product separation, and capability-unblocking doctrine;
- Phase 1 private-evidence and retrieval foundation;
- Phase 2 released authoritative memory core;
- Phase 3 conversation/model abstraction;
- Phase 4 governed public-information path;
- Phase 5/M2 host-neutral memory contracts and reversible prototypes;
- migration Stage A+B inventory/adapters, C+E destination/shadow reads, D deterministic backfill machinery, F controlled mirroring, and G persistent reference projection receipts;
- Claim/Experience separation, bitemporal authority design, cognitive-graph/vector/object/workflow logical planes;
- EIPM private substrate, ACFP/IDP interfaces, N0 architecture, and the current full-envelope N0 training lineage;
- MFM formation contracts/evaluation program;
- FBM builder-data/seed program;
- Fable first-release product/conversation/privacy architecture;
- frontier-research additions that materially improve the design without replacing its core: citation lock, consolidation path-dependence checks, memory-use calibration, adaptive retrieval, source-native retrieval, failure-localized learning, and versioned context/harness evolution.

These are foundations to build on, not invitations to restart architecture selection.

## 3. Current critical path

### A. Finish N0 as actual model work

Current branch: `alice-eipm-v1-n0-full-envelope-joint-ddp-v2`.

The immediate job is not another infrastructure survey. The whole-step DDP repair must prove the real joint gradient/optimizer/resume path on the production topology. Once actual full-step memory is measured, choose hardware that fits it and train.

Do not:

- shrink registered capability to fit old P100s;
- rerun known-failed Magnolia/Kaggle capacity routes without new evidence;
- substitute AirLLM frozen-base LoRA for native N0 training;
- spend additional weeks qualifying runtimes that are already known well enough to proceed.

### B. In parallel, build MFM and FBM data, not backend tournaments

MFM should progress on formation gold, context planning, proposal quality, abstention, temporal/correction/deletion handling, and end-to-end formation -> gate -> projection behavior.

FBM should progress from procedure traces into reusable host-neutral input/target/outcome cases that teach source intake, coverage, model construction, failure localization, evaluation, repair, and assembly of the connected personal foundation.

### C. Move the memory program from prototypes to one selected integrated stack

Stop treating every backend in prior research inventories as a mandatory Stage G target. Implement the selected default stack below, run the integrated cognitive-memory path, and only open a challenger when a concrete trigger justifies it.

## 4. Selected infrastructure

The logical Memory v4.1 architecture remains unchanged. The physical implementation is simplified.

| Role | Selected default | Why |
| --- | --- | --- |
| Canonical Claim Fabric + Experience/Event ledger + outbox/reconciliation metadata | **PostgreSQL**, with separate logical schemas/contracts | Mature transactions, append-only/event patterns, bitemporal queries, local ownership, replication/partitioning path, permissive license, and fewer cross-authority services. |
| Raw evidence, datasets, checkpoints, exports, backups | **Content-addressed local filesystem/object store**, with optional owner-authorized S3-compatible remote backup | Raw bytes do not need another database. Hash-addressed objects make lineage, dedup, export, and rebuild straightforward. |
| Cognitive graph accelerator | **Neo4j** for A.L.I.C.E. full/research profile; PostgreSQL relation projection remains the rebuildable source/fallback | A.L.I.C.E. benefits from graph-native traversal and explanation, while authority remains outside the graph. Fable product profiles need capability parity, not mandatory Neo4j deployment. |
| Semantic/multimodal accelerator | **Qdrant** | Strong local vector/multimodal retrieval, filters, snapshots, and existing project experience. It is optional per query, never authority. |
| Exact/source-native retrieval | **Direct filesystem/SQL/API search**: names, fields, FTS, grep-style reads, live system-of-record queries | Stronger models should be allowed to search current sources directly when that is better than precomputed similarity. |
| Retrieval decision | **Context Planner / Retrieval Orchestrator** | Route among claim, exact/source-native, graph, temporal, vector, episode, and live-source paths based on the evidence problem. |
| Durable long-running workflows | **Temporal** for distributed/long-lived A.L.I.C.E. operations; local durable runner may implement the same contract in a single-host Fable profile | Use a real durable engine where missions, deletion, repair, migration, and training need it. Do not force the service into every desktop action. |
| Ephemeral workspace/cache | **In-process first; Valkey when shared/distributed state is actually needed** | Avoid making a cache another authority or mandatory local daemon. |
| Training | **PyTorch + Accelerate; DDP for the current route; FSDP/offload only when measured memory requires it; Slurm for cluster scheduling** | This is already the active engineering path. Do not run Ray/DeepSpeed/Kubernetes tournaments without a concrete blocker. |
| Personal-model serving | **PyTorch native first; compiled/ONNX-class export after behavioral equivalence is proven** | Avoid packaging work before the model behavior is stable. |
| Large open-model serving | **vLLM when the model/hardware profile fits it** | Mature high-throughput serving. It is not required for Fable v1's external GPT/Claude feature path. |
| Model/dataset registry | **Content-addressed artifacts + signed/hashed manifests first; MLflow only if experiment volume makes it useful** | Lineage is required; another always-on service is not. |

### Why not keep KurrentDB as a required event authority?

KurrentDB remains a legitimate future challenger, but A.L.I.C.E./Fable do not currently need a second canonical persistence system merely to gain event-stream semantics that PostgreSQL plus append-only tables/outbox/subscriptions can already provide at personal-AI scale. Open it again only if measured replay/subscription/throughput or operational behavior becomes a real bottleneck.

### Why keep Neo4j and Qdrant?

They serve genuinely different cognitive workloads. The current frontier evidence does not show that source-native search replaces semantic similarity, multimodal retrieval, graph traversal, or temporal/relationship reasoning. They remain rebuildable accelerators behind authority and are invoked only when useful.

## 5. Validation budget rule

Validation protects capability; it must not replace capability construction.

Infrastructure validation is now **decision-triggered**.

A challenger is opened only when at least one of these is true:

- the selected stack cannot satisfy a required semantic/authority contract;
- a real workload exposes a correctness, scale, latency, reliability, deletion, privacy, or packaging blocker;
- a frontier result shows a directly relevant capability delta that the selected path cannot express;
- licensing/distribution makes the selected component unsuitable for the target profile;
- measured cost is material enough to justify migration.

Otherwise:

1. use the selected stack;
2. run targeted integration/zero-tolerance tests;
3. move on to model, memory, judgment, and product work.

No all-pairs backend tournament. No Cartesian product of technologies. No benchmark built only to decide a question that the architecture and actual workload already answer.

## 6. Execution timeline

### Wave 0 — completed foundations

P0–P4, M2 contracts/prototypes, migration A–G reference work, EIPM/MFM/FBM architecture, Fable first-release design.

### Wave 1 — now: make N0 and the builder foundations real

- prove N0 whole-step CUDA gradient + optimizer + exact resume;
- buy/use suitable GPU time only after measured full-step requirement is known;
- train N0 until the semantic/judgment readiness suite shows it can support private identity learning;
- expand FBM into real builder-operation cases;
- build MFM context planner/gold/formation candidates in parallel.

### Wave 2 — private personal foundation

- N1: Elaina-derived identity representation under private provenance;
- N2: context-conditioned judgment/preference and full IDP behavior;
- N3: calibration, failure tails, owner fidelity review;
- train/qualify MFM;
- instantiate Rayan host state, A.L.I.C.E.–Rayan relationship state, and A.L.I.C.E. self/continuity state;
- prove isolated interventions in each state produce only relevant judgment changes.

### Wave 3 — integrated Stage G on the selected stack

Implement the full loop on real selected infrastructure:

```text
experience
  -> PostgreSQL evidence/event ledger
  -> MFM
  -> deterministic Memory Gate
  -> PostgreSQL Claim Fabric
  -> graph/vector/episode/host/relationship/self/mission projections
  -> Context Planner
  -> EIPM native judgment
  -> action/response
  -> outcome
  -> governed revision
```

Required integrated evidence is about A.L.I.C.E. behavior:

- provenance/authority correctness;
- identity separation;
- temporal correction;
- deletion/revocation;
- citation lock;
- memory-use calibration;
- consolidation schedule sensitivity;
- exact vs vector vs graph vs source-native retrieval routing;
- failure/recovery and rebuild;
- long-horizon continuity;
- realistic scale/latency on the selected profile.

This is not a backend tournament.

### Wave 4 — replace Phase 2

- Stage H: bounded successor canary authority;
- Stage I: explicit cutover after reconciliation/rollback rehearsal;
- Stage J: Phase 2 read-only compatibility/fallback, then owner-accepted retirement/replacement.

Phase 2 replacement is complete at Stage J acceptance, not before.

### Wave 5 — generalize A.L.I.C.E. into FBM

Turn the successful A.L.I.C.E. construction process into a host-neutral builder:

- source intake and subject attribution;
- outside-source-pack adequacy;
- personal foundation construction;
- MFM/personality/host/relationship/self assembly;
- memory/context wiring;
- training and evaluation;
- failure localization and repair;
- portability, deletion, rollback, and model replacement.

A.L.I.C.E. private identity/data never enters the shared builder.

### Wave 6 — Fable v1 productization

Fable v1 ships only when the complete personal loop is transferable to a fresh user.

The product profile is local-first and may collapse physical services compared with A.L.I.C.E.'s research/full profile while preserving the same logical contracts. Capability parity matters; backend parity does not.

First release contains:

- FBM;
- local evidence/Experience/Claim infrastructure;
- personality/identity, MFM, host, relationship, and Fable-self capabilities;
- native personal judgment and versioned response verdict;
- local conversation control, privacy/egress gateway, provider-independent draft checking;
- GPT/Claude feature APIs for general language/coding/research/vision/simulation where authorized;
- outcome-driven learning;
- inspection, correction, deletion, export, restore, rollback, and provider replacement.

The release gate is end-to-end personal behavior, not whether every research backend was benchmarked.

## 7. Frontier-watch integration

The research watch remains active, but new papers do not automatically create architecture work.

Classify every finding as:

- **already covered** — record only;
- **future challenger** — preserve in research notes;
- **direct current capability improvement** — modify the relevant model/contract now;
- **architecture-invalidating evidence** — reopen a decision only when evidence is strong enough to justify disruption.

Current frontier findings retained in the execution path:

- citation-locked evidence consumption;
- consolidation path-dependence checks;
- proposition-level memory-use calibration;
- dynamic retrieval routing including no-vector source-native search;
- retrieval/outcome learning as a derived projection;
- failure-localized skill/context-harness evolution;
- personalized counterfactual judgment and downstream behavior checks;
- history-aware privacy/egress evaluation for Fable conversation;
- independent audit evidence for later autonomous self-evolution.

Research-only mechanisms such as recurrent parametric memory, latent query-conditioned topology, hidden-state recall controllers, AirLLM-style streaming, and other frontier challengers do not block current construction.

## 8. Stop conditions against infrastructure drift

Stop infrastructure work and return to A.L.I.C.E. capability work when:

- the selected component meets the current required contract;
- remaining differences are speculative rather than tied to an observed A.L.I.C.E. failure;
- the next experiment would compare tools rather than advance memory, judgment, identity, learning, or product capability;
- a successful result would not change the next implementation decision.

This is the lesson from MC10: validation must answer a real decision. If it no longer changes the decision, stop validating.
