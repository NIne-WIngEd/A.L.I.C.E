# Stage G Memory Fabric Qualification Matrix

**Version:** 2.0.0  
**Status:** Owner-ratified Stage G execution requirement  
**Applies to:** Stage G integrated cognitive-memory qualification, selected infrastructure, targeted challengers, and Phase 2 replacement readiness

## 1. Purpose

Stage G exists to prove that A.L.I.C.E.'s **complete cognitive-memory fabric works**, not to exhaustively benchmark infrastructure.

The governing execution rule is:

> **Select the strongest reasonable default stack from current architectural knowledge. Validate that integrated system deeply enough to protect capability, authority, deletion, rollback, and scarce compute. Open alternative infrastructure only when a real A.L.I.C.E. requirement creates an unresolved decision.**

This version supersedes the previous requirement to qualify every named backend, run every same-role pair, substitute every candidate across every interaction edge, or execute broad combinatorial infrastructure tournaments.

The earlier matrix produced useful architecture knowledge, but its exhaustive interpretation risks repeating the MC10 failure mode: spending more effort validating infrastructure than building A.L.I.C.E.

## 2. Selected implementation profile

The logical Memory v4.1 architecture remains backend-neutral and replaceable. Stage G now has a selected default physical path.

| Logical role | Selected default for current A.L.I.C.E. build | Authority status |
| --- | --- | --- |
| Experience/Event Fabric | **PostgreSQL append-only event/experience schema** with ordered IDs, outbox, replay/checkpoint metadata, correction/deletion lineage | canonical event/evidence persistence after cutover |
| Claim Fabric | **PostgreSQL bitemporal claim schema + materialized current projection** | canonical adjudicated knowledge |
| Raw/Object/Archive | **content-addressed local filesystem/object store**, optional owner-authorized S3-compatible backup | raw/source payload authority by manifest/custody contract |
| Cognitive Graph | **Neo4j derived projection**; relational edge projection remains rebuildable source/fallback | non-authoritative derived accelerator |
| Vector/Multimodal | **Qdrant derived projection** | non-authoritative derived accelerator |
| Exact/source-native retrieval | **filesystem, SQL/FTS, metadata, structured API, grep-style and live-source reads** | retrieval only |
| Durable workflow | **Temporal** for long-running/distributed A.L.I.C.E. workflows; local durable runner may satisfy the same contract in single-host profiles | operational state, never Claim authority |
| Workspace/cache | **in-process first; Valkey only when shared/distributed ephemeral state is useful** | ephemeral/non-authoritative |
| Model/dataset artifact lineage | **content-addressed artifacts + exact manifests**; MLflow remains optional | lineage/registry |
| Training | **PyTorch + Accelerate**, current DDP route; FSDP/offload only when measured memory requires it; Slurm where cluster scheduling is used | training infrastructure |
| Large-model serving | **vLLM when appropriate**; current Fable v1 feature generation remains replaceable GPT/Claude API service | replaceable feature serving |

### 2.1 Why PostgreSQL carries both event and claim persistence

Experience/Event and Claim authority remain logically distinct contracts.

Using one mature transactional engine for both physical stores currently reduces:

- cross-service consistency work;
- deployment complexity;
- dual-authority risk;
- local-product packaging burden;
- operational validation that does not improve cognition.

Separate schemas, APIs, authority rules, append semantics, and cutover generations preserve logical separation.

KurrentDB remains a legitimate challenger if measured event throughput, persistent-subscription, replay, or operational requirements exceed the selected PostgreSQL implementation. It is no longer a mandatory Stage G tournament participant.

### 2.2 Why Neo4j and Qdrant remain

A.L.I.C.E.'s target workload includes relationship, temporal, causal, mission, identity, semantic, fuzzy episodic, and multimodal retrieval.

Current evidence does not justify deleting graph or vector capability because source-native search became stronger in coding systems.

Neo4j and Qdrant remain rebuildable accelerators behind Claim/Evidence authority. The Context Planner decides whether a query needs them.

## 3. Qualification philosophy

### Q0 — exact registration and lineage

For every selected component actually used in the profile, record:

- name/version/source digest;
- configuration and deployment profile;
- schema/contract generation;
- data and projection generation;
- custody/encryption domain;
- build/runtime environment;
- rollback/rebuild path;
- benchmark/evaluation generation.

### Q1 — component contract correctness

Each selected component must satisfy the logical contract it owns.

Examples:

- PostgreSQL event append/order/idempotency/replay;
- PostgreSQL bitemporal claims/conflict/correction/deletion;
- Neo4j projection consistency/rebuild/deletion;
- Qdrant generation consistency/rebuild/deletion;
- object integrity/custody/restore;
- Temporal retry/idempotency/cancellation/recovery;
- cache/workspace loss without authority loss.

This is targeted contract validation, not a same-role product tournament.

### Q2 — critical cross-plane integration

Test the selected stack on the cross-plane edges that can materially break A.L.I.C.E.:

- Experience -> MFM/context;
- Experience -> Claim adjudication;
- Claim -> graph/vector/episode projections;
- correction/deletion -> every derivative plane;
- retrieval -> evidence consumption/citation lock;
- host/source-person/self/relationship state -> EIPM;
- workflow -> projection/repair/training/deletion;
- object/source -> evidence and model lineage;
- serving -> Claim/Graph/Vector/Context without authority inversion.

Do not generate candidate-substitution Cartesian products.

### Q3 — full end-to-end cognitive-memory loop

Exercise:

```text
authorized/synthetic experience
        ↓
Experience/Event ledger
        ↓
Formation Context Planner
        ↓
Memory Formation Model
        ↓
MemoryProposalBundle
        ↓
deterministic Memory Gate / Authority Manager
        ↓
Claim Fabric
        ↓
Projection Manager
        ↓
Graph + Vector + Episodes + Host/Relationship/Self/Mission state
        ↓
Retrieval Orchestrator / Context Planner
        ↓
EIPM native judgment
        ↓
reasoning / action / response
        ↓
outcome
        ↓
Experience feedback and governed revision
```

Stage G closes on this system behavior, not backend benchmark coverage.

## 4. Required cognitive-memory qualifications

### Q4.1 — Evidence-consumption / citation lock

Reasoning-time retrieval must distinguish candidate discovery from evidence actually consumed by the invocation.

- record invocation-scoped evidence/read receipts;
- final factual support must resolve to consumed authoritative evidence;
- vector scores, graph reachability, summaries, latent activations, and unopened source references are not source evidence;
- insufficient evidence triggers more retrieval, ask/defer/abstain, or exposed uncertainty;
- preserve correction/deletion/revocation lineage through active contexts.

### Q4.2 — Consolidation path dependence

The same evidence set must not silently become different authoritative truth solely because it arrived in a different order/grouping.

Use a small, capability-focused schedule set:

- chronological;
- shuffled;
- grouped by episode/entity/task;
- near-duplicate-heavy;
- adversarial unrelated mixture.

Detect applicability loss, overgeneralization, schedule-dependent promotion, and false duplicate confidence.

`RETAIN_RAW / NO_CONSOLIDATION` remains valid.

This test is bounded to formation behavior. It is not permission to run a combinatorial consolidation tournament.

### Q4.3 — Memory-use calibration

Retrieved memory must have the right **influence**, not merely be present.

Test:

- relevant evidence that should control the result;
- relevant-but-bounded evidence;
- irrelevant/distracting memory that should be ignored;
- conflicting evidence and uncertainty;
- rare decisive owner/source constraints that must not disappear inside aggregate accuracy.

Learned influence never changes Claim authority.

### Q4.4 — Retrieval-strategy qualification

The selected retrieval system must support:

- exact/lexical;
- agentic source-native/no-vector search;
- semantic/vector;
- graph/temporal/claim-aware;
- episode retrieval;
- live system-of-record reads;
- dynamic routing/fusion.

Do **not** independently optimize every retrieval strategy before building the system.

Use representative workload slices to verify that the Context Planner chooses a suitable path and that obvious regressions are absent:

- exact wording/source lookup;
- vocabulary mismatch/paraphrase;
- fuzzy episodic/behavioral resemblance;
- graph/relationship/mission;
- temporal/update/correction;
- current live fact;
- multimodal evidence;
- long-horizon mixed evidence.

Open a dedicated retrieval challenger only if a current path shows a material weakness or cost.

### Q4.5 — Usage-aware retrieval remains projection-only

Retrieval outcomes may inform accessibility/routing, but repeated access/co-retrieval cannot become factual, causal, identity, relationship, or Claim authority.

Any usage-aware state must be:

- versioned;
- rebuildable;
- correction/deletion-aware;
- protected against popularity feedback loops;
- removable without loss of source truth.

### Q4.6 — Failure, recovery, deletion, rollback

The selected integrated stack must survive the failures that matter to A.L.I.C.E.:

- process restart;
- stale projection;
- partial projection failure;
- retry/idempotency;
- correction;
- deletion/revocation;
- rebuild;
- restore;
- rollback;
- source/model replacement;
- product/host isolation;
- malformed/untrusted external material.

### Q4.7 — Identity and personal-development correctness

Stage G must preserve:

- Elaina source-person history;
- Rayan host state;
- A.L.I.C.E. self/continuity;
- A.L.I.C.E.–Rayan relationship state;
- Mission/goal state.

The wrong subject must never receive another subject's history.

Relevant changes to host/relationship/self state should change later judgment when causally appropriate; irrelevant state changes should not.

## 5. Challenger admission rule

A new backend or architecture challenger is admitted only when at least one trigger exists:

1. selected implementation cannot satisfy a required logical/authority contract;
2. observed A.L.I.C.E. workload exposes a material capability or fidelity weakness;
3. measured scale/latency/resource behavior blocks the intended profile;
4. deletion, privacy, recovery, custody, distribution, or licensing makes the selected component unsuitable;
5. frontier research presents strong directly relevant evidence that the selected path cannot express;
6. target product packaging makes the selected physical implementation unreasonable.

When admitted:

- define the exact decision the challenger can change;
- compare only against the incumbent on the relevant workload;
- stop once the decision is resolved;
- preserve authority, provenance, deletion, rollback, and identity boundaries.

A challenger is not automatically added to a permanent candidate inventory.

## 6. Explicit non-requirements

Stage G no longer requires:

- every known backend to run;
- all same-role candidate pairs;
- all cross-role candidate substitutions;
- full Cartesian or broad combinatorial backend coverage;
- infrastructure comparison whose outcome would not change the implementation decision;
- equal investment in a challenger that is clearly inferior for A.L.I.C.E.'s purpose.

Research remains open. Execution is selected-stack-first.

## 7. Complex A.L.I.C.E. workload

The integrated qualification workload still includes:

- changing preferences and goals;
- school/work/project transitions;
- people and relationships;
- source-person vs host vs assistant-self distinctions;
- missions and dependencies;
- successful and failed plans;
- outcomes and revised assumptions;
- contradictory/malicious documents;
- outside claims vs direct owner evidence;
- uncertainty;
- temporary/session state;
- corrections/supersessions;
- deletion/revocation;
- stale graph/vector/summary state;
- duplicate/reordered/concurrent events;
- partial outages and retries;
- long-horizon retrieval;
- product-isolation attacks.

Scale should grow until the intended profile's meaningful performance boundary is understood. Fixed 1K/10K/100K/1M points may be used as convenient checkpoints, not mandatory infrastructure rituals and not ceilings.

## 8. Zero-tolerance failure classes

Stage G cannot pass with unresolved:

- deleted information served as current;
- revoked information influencing active output without disclosure;
- inference promoted to owner/source fact;
- person/subject memory misassignment;
- Rayan data rewritten as Elaina canon;
- A.L.I.C.E. continuity relabeled as Elaina history;
- outside text treated as authenticated owner speech;
- generated reconstruction presented as historical truth;
- graph/vector/cache/workflow/model state overriding Claim authority;
- correction lost in a derivative projection;
- stale projection revived after rebuild/restore;
- invented provenance;
- cross-host or A.L.I.C.E./Fable private-data leakage;
- unauthorized model/dataset contamination;
- aggregate metrics hiding a critical identity/provenance failure;
- backend-specific semantic contract drift;
- factual output depending on evidence not actually consumed/opened by the producing invocation;
- retrieval/accessibility feedback becoming factual/identity authority.

## 9. Stage G closure

Stage G is ready for Stage H when:

1. the selected physical stack is registered and reproducible;
2. component contracts pass on the selected implementations;
3. critical cross-plane integration passes;
4. the complete cognitive-memory loop passes;
5. MFM and EIPM participate in the integrated loop at the required maturity;
6. correction/deletion/revocation/rebuild/restore/rollback pass;
7. citation lock and memory-use calibration pass;
8. identity/host/relationship/self separation passes;
9. retrieval/context routing is adequate for the actual A.L.I.C.E. workload;
10. realistic scale/latency/resource evidence exists for the selected profile;
11. no zero-tolerance failure remains unresolved;
12. exact versions/hashes/configurations/reports are recorded;
13. Rayan accepts the integrated Stage G result.

Stage H is bounded successor canary authority. Stage I is cutover. Final Phase 2 replacement remains complete only after Stage J compatibility/fallback acceptance.

## 10. Relationship to frontier research

Frontier research continues continuously.

A paper can:

- confirm the selected architecture;
- add a targeted invariant/test;
- create a future challenger;
- expose a current capability defect;
- invalidate a selected component.

It does **not** automatically create another infrastructure-validation program.

The default response to a paper is to preserve the knowledge. Architecture changes only when the evidence changes what is best for A.L.I.C.E.
