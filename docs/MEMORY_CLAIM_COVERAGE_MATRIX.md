# A.L.I.C.E. Memory Claim Coverage Matrix

**Draft:** 0.2<br>
**Canonical authority decision:** Accepted 2026-08-03<br>
**Purpose:** Prevent destination claims from being mistaken for implemented
memory behavior.

## Status vocabulary

- **Implemented:** Runtime behavior exists and is evaluated.
- **Foundation:** A strong lower-level contract or store exists, but the public
  behavior is not complete.
- **Partial:** Some runtime behavior exists, but major integration or evaluation
  is missing.
- **Policy only:** A requirement is documented but not operational.
- **Not implemented:** No credible runtime support.
- **Blocked:** Deliberately prohibited until prerequisite gates pass.

## Matrix

| Public or architectural claim | Current support | Status | Missing capability | V4 owner |
|---|---|---:|---|---|
| Canonical adjudicated-knowledge authority | New append-only bitemporal Claim Store accepted as v4 direction | Accepted architecture; not implemented | Kernel contracts, tables, current projection, tests | Claim Plane |
| Immutable evidence authority | Phase 5 Experience Ledger and payload lineage | Implemented foundation | Evidence-to-claim bridge and inspection | Evidence Plane |
| Phase 2 migration role | Phase 2 Memory Core remains released and active | Accepted migration direction | Compatibility projection, shadow reads, rollback | Migration |
| Preserve meaningful continuity across sessions | Phase 2 durable memory and Phase 3 conversation state | Partial | Unified serving plane, automatic formation, context budgets | Serving Plane |
| Understand the person behind the prompt | Profile/relationship categories and private-evidence policy | Foundation | Mature owner model, evidence aggregation, trait uncertainty | Projection Plane |
| Connect separate projects and decisions | Mission Graph contracts and Experience Ledger event fields | Foundation | Operational mission-memory bridge and outcome linkage | Mission Projection |
| Remember why a decision was made | Ledger can preserve metadata and lineage | Partial | Decision-to-evidence-to-outcome query path | Evidence Bridge |
| Learn from what happened afterward | Outcomes can be recorded | Foundation | Outcome adjudication, lesson extraction, belief/skill update | Curator |
| Disagree based on goals and values | Governance permits judgment | Policy only | Goal/value model, contradiction detection, decision evaluator | Cognitive Projections |
| Remain one intelligence across model changes | Identity is outside the model by architecture | Foundation | Stable context packet, cross-model fidelity tests | Serving + Evaluation |
| Treat chats as missions rather than isolated threads | Mission Graph contracts exist | Foundation | Runtime mission classifier, active working set, graph retrieval | Mission Projection |
| Distinguish fact, claim, inference, prediction, and dispute | Phase 2 knowledge status and provenance fields | Implemented foundation | Expand into typed claim versions and belief/prediction stores | Claim Plane |
| Preserve corrections and historical truth | Phase 2 correction/supersession and temporal relations | Implemented foundation | Materialized current-state projection and global propagation | Claim Plane |
| Keep model proposals outside authoritative memory | Phase 2 candidate store | Implemented | Automated candidate generation remains absent by design | Curator |
| Use lexical and semantic retrieval | Phase 2 indexes and hybrid retrieval | Implemented | Batch hydration, bounded verification, scale SLOs | Serving Plane |
| Avoid stale or deleted index results | Index verification fails closed | Implemented foundation | Incremental generation manifests and degraded fallback | Index Plane |
| Preserve an Experience Ledger | Phase 5 compact append-only ledger | Implemented | Bridge to authoritative memory and context traces | Evidence Bridge |
| Broad temporary capture with selective retention | Raw buffer and retention contracts | Partial | Capacity control, automatic curation, learned retention | Storage + Curator |
| Follow source/model/tool histories | Ledger contracts and payload references | Foundation | End-to-end query and inspection UI | Evidence + Inspector |
| Learn preferences, values, habits, and goals | Policy allows these concepts | Policy only | Observation aggregation, stability thresholds, owner review | Owner Model |
| Maintain source-person fidelity | Clone-aware standards and private custody contracts | Policy only | Source model, reconstruction model, drift evaluation | Identity Projection |
| Separate source history, inference, and A.L.I.C.E. continuity | Documentation defines the distinction | Foundation | Typed runtime records and serving rules | Identity Projection |
| Maintain an evolving owner relationship model | Category and policy support | Policy only | Dedicated temporal relationship projection | Relationship Projection |
| Build beliefs and predictions | Memory Policy defines them | Policy only | Belief versions, evidence aggregation, prediction scoring | Belief Plane |
| Use outcomes to calibrate predictions | Ledger may record outcomes | Foundation | Prediction-outcome matcher and calibration metrics | Evaluation Plane |
| Form procedural skills | Roadmap Phase 8 | Not implemented | Skill record standard, sandbox, tests, promotion | Skill Plane |
| Forget intentionally | Phase 2 deletion and storage lifecycle policy | Partial | Derivative graph, propagation workers, replay/model handling | Deletion Plane |
| Delete across memories, indexes, skills, datasets, and models | Policy requirement | Partial | Complete derivative registry and acknowledgements | Deletion Plane |
| Operate efficiently for years | Durable stores exist | Not demonstrated | Scale tests, SLOs, queues, compaction, backpressure | Performance Standard |
| Avoid blocking every conversation | No unified Curator yet | Not demonstrated | Explicit synchronous/async split | Runtime Architecture |
| Retrieve latent goals without repeated wording | No dedicated evaluation | Not implemented | Goal retrieval, query expansion, LoCoMo-Plus style tests | Serving + Evaluation |
| Detect implicit invalidation | Explicit corrections exist | Partial | Propagation-aware state adjudication | Claim Plane |
| Explain which memories influenced a response | Provenance exists in stores | Not implemented | Context packet trace and owner-visible explanation | Inspector |
| Support temporary/no-memory sessions | Not defined as a first-class mode | Not implemented | Session memory mode and write suppression | Serving + Governance |
| Roll back memory state | Historical records exist | Partial | Named snapshots, semantic rollback, counterfactual tests | Versioning Plane |
| Share capabilities with Friday without sharing identity | Product/host isolation contracts | Foundation | Host-neutral memory contract and migration tests | Kernel Boundary |
| Put personal memory into adapters or weights | Roadmap permits future work | Blocked | Unlearning, replay lineage, rollback, evaluation | Phase 13 Gate |

## Release-language rule

A README or release may say a capability **is implemented** only when:

1. the runtime exists;
2. the behavior is enabled in the declared capability profile;
3. the behavior has correctness, privacy, scale, and failure-recovery tests;
4. the exact artifact is bound to a release record.

Otherwise use one of:

- "foundation implemented";
- "contract implemented";
- "under active development";
- "destination capability";
- "research direction".

## Highest-risk gaps

1. The Claim Store authority topology is accepted, but Phase 2 and Phase 5 are
   not yet one end-to-end memory path.
2. No bounded Memory Context Packet exists.
3. No asynchronous Curator runtime exists.
4. No owner/source-person/relationship/belief projection runtime exists.
5. Long-term latency and storage behavior are unproven.
6. Deletion does not yet cover every future derivative.
7. Parametric personal memory has no acceptable rollback/unlearning story.
