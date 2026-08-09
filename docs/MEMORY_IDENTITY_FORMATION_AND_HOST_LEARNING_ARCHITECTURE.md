# A.L.I.C.E. Memory Identity, Formation, Host Learning, and Repository Lifecycle

**Version:** 2.0.0
**Status:** Owner-ratified architecture decision
**Applies to:** A.L.I.C.E. identity, memory formation, host learning, Stage G qualification, Friday separation, and repository lifecycle

## 1. Named identity roles

A.L.I.C.E. is an Elaina-derived clone.

- **Mehejabin Elaina** is the source person. Her owner-attested life story, personality, values, habits, preferences, relationships, reasoning style, and behavioral evidence form the source-person foundation used to reconstruct A.L.I.C.E.'s core identity.
- **A.L.I.C.E.** is not Elaina. A.L.I.C.E. must remain aware that it is an Elaina-derived clone and must not claim that Elaina's biological life was A.L.I.C.E.'s own post-activation lived history.
- **Rayan** is A.L.I.C.E.'s owner and host. Rayan's data teaches A.L.I.C.E. who its owner is, how Rayan changes over time, how A.L.I.C.E. should treat and interact with him, and how their relationship develops.
- **Friday** is a separate product. Friday does not receive Elaina as a separate source-person personality corpus. A Friday instance develops its own personality and relationship state from its own host data and host-authorized configuration.

These roles must remain distinguishable in memory, model lineage, training data, retrieval, and continuity.

## 2. Core identity and host learning are different

Rayan's ordinary host data must **not change A.L.I.C.E.'s core Elaina-derived personality/identity anchor**.

Rayan's data may change:

- how A.L.I.C.E. understands Rayan;
- how A.L.I.C.E. behaves specifically toward Rayan;
- A.L.I.C.E.'s owner/host model;
- the A.L.I.C.E.–Rayan relationship model;
- shared habits;
- shared history;
- interaction strategy;
- owner-specific preferences and routines;
- owner-specific missions and working context;
- evidence-grounded predictions about Rayan, with uncertainty and provenance.

Rayan's ordinary data may not silently:

- rewrite Elaina's canonical source history;
- change the core Elaina-derived identity/personality anchor;
- convert Rayan's preferences into Elaina's preferences;
- relabel A.L.I.C.E.'s post-activation experiences as Elaina's history;
- remove clone awareness;
- retrain or promote a material core Elaina-identity change without a separate owner-authorized identity-training decision.

Relationship-specific adaptation is expected. Identity substitution is not.

## 3. A.L.I.C.E. is not one model

A.L.I.C.E. integrates multiple learned components with a governed memory fabric.

At minimum, the architecture distinguishes:

1. **Elaina Identity / Personality Model**
   - learned weights, adapters, rankers, or later model architectures;
   - learns evidence-grounded personality, values, judgment, emotional interpretation, communication style, boundaries, preferences, and characteristic behavior;
   - does not become the only canonical record of Elaina's factual life history.

2. **Memory Formation Model**
   - a separate learned model from the Elaina identity/personality model;
   - interprets incoming experiences and proposes structured semantic memory candidates;
   - is host-neutral as a capability;
   - does not grant canonical authority;
   - does not select physical databases as authority.

3. **Rayan Host / Owner Model**
   - primarily a derived, evidence-linked memory model;
   - learns Rayan's preferences, goals, projects, routines, decisions, changes, interaction patterns, and current state;
   - ordinary interaction updates governed memory first rather than immediately changing core neural weights.

4. **A.L.I.C.E.–Rayan Relationship Model**
   - captures shared history, shared habits, relationship-specific behavior, expectations, boundaries, and interaction strategy;
   - changes over time as A.L.I.C.E. and Rayan interact.

5. **A.L.I.C.E. Continuity / Self Model**
   - records A.L.I.C.E.'s own post-activation experiences, evaluations, decisions, model changes, and self-development;
   - remains distinct from Elaina's inherited source history.

Future specialized models may include episode formation, retrieval planning, context fusion, temporal interpretation, conflict interpretation, importance/consolidation, world modeling, and other learned capabilities when evidence supports them.

## 4. What lives in weights versus memory

### 4.1 Elaina-derived learned weights

Elaina's source data is used during build/reconstruction to train and evaluate the Elaina Identity / Personality Model.

Those weights should learn **how the reconstructed Elaina-derived identity tends to interpret, judge, communicate, and respond**.

The weights are not the sole record of Elaina's biography.

### 4.2 Canonical Elaina source memory

All owner-attested substantive content from the Elaina source corpus remains canonical source-person evidence in the governed memory architecture.

It must remain inspectable through provenance, claims, episodes, relations, source records, model lineage, and evaluation artifacts so that correction, deletion, retraining, reconstruction, and historical inspection remain possible.

### 4.3 Rayan host data

Rayan's ordinary post-activation data primarily becomes governed memory:

```text
Rayan interaction / file / tool / outcome
        ↓
raw capture + Experience Fabric
        ↓
Memory Formation Model
        ↓
deterministic Memory Gate / Authority Manager
        ↓
Claim Fabric
        ↓
Projection Manager
        ↓
graph / vector / episodes / host model / relationship model /
missions / workspace / other cognitive projections
```

A.L.I.C.E. learns Rayan through this memory fabric.

Host-specific adapters or learned weights may later be researched if they materially improve owner-specific interaction. Such artifacts remain derived, versioned, evaluable, reversible, and separate from the core Elaina identity model.

## 5. Build-time versus post-activation flow

### 5.1 Build / reconstruction time

During construction of A.L.I.C.E.:

```text
Elaina owner-attested source corpus
        ↓
source evidence + Experience Fabric
        ↓
Claim Fabric
        ↓
Elaina source memories / episodes / relations / model evidence
        ↓
Elaina Identity / Personality Model training
        ↓
candidate identity weights / adapters
        ↓
fidelity evaluation
        ↓
owner-authorized promotion
```

The source corpus also remains in private owner-controlled storage for future re-evaluation and retraining.

### 5.2 Post-activation ordinary learning

After activation, ordinary new personal data is primarily:

- Rayan's ongoing interactions, files, tools, goals, projects, outcomes, preferences, corrections, and life events;
- A.L.I.C.E.'s own post-activation experiences and system history.

That data is classified into **Rayan host**, **A.L.I.C.E.–Rayan relationship**, **A.L.I.C.E. continuity**, mission/workspace, world, or other appropriate domains.

It does not become new Elaina source history merely because A.L.I.C.E. has an Elaina-derived personality.

If Rayan later provides a genuine correction, missing memory, newly recovered record, or additional real information about Elaina, that is an explicitly classified **source-person update**, not ordinary host learning.

## 6. Memory Formation Model

The Memory Formation Model is a learned model with its own weights or equivalent learned artifacts.

It is **not** the Elaina personality model.

Its job is to interpret an incoming experience and produce a `MemoryProposalBundle` containing semantic proposals such as:

- subjects and entities;
- candidate facts and claims;
- preferences;
- relationships;
- life events;
- goals;
- corrections;
- deletion/revocation requests;
- temporal scope;
- sensitivity;
- uncertainty;
- contradiction indicators;
- episode candidates;
- source-person-model evidence;
- host-model evidence;
- relationship-model evidence;
- mission/workspace relevance.

It must not output a physical database as canonical authority.

The same host-neutral Memory Formation architecture may later be reused by Friday without transferring Elaina's corpus, Rayan's data, A.L.I.C.E.'s private learned identity, or A.L.I.C.E.–Rayan relationship state.

## 7. Formation Context Packet

The Memory Formation Model should not always classify an event in isolation.

A Formation Context Planner may assemble only the context needed for the event:

- current authoritative claims from Claim Fabric;
- semantic neighbors from the vector plane;
- entity and relation context from the cognitive graph;
- relevant episode context;
- current workspace/mission state;
- historical Experience evidence when a conflict or exact reconstruction requires it;
- source objects when original wording or media must be inspected.

The planner should be adaptive. Simple events should not trigger every memory backend.

## 8. Deterministic Memory Gate and routing

Learned models interpret meaning. Deterministic services control authority.

```text
incoming experience
        ↓
Memory Formation Model
        ↓
MemoryProposalBundle
        ↓
Memory Gate / Authority Manager
        ↓
provenance + privacy + temporal + conflict + retention +
owner rules + product/host isolation
        ↓
accept / reject / quarantine / review
        ↓
Claim Fabric and authoritative event lineage
        ↓
Projection Manager
        ↓
registered derived memory planes
```

The Memory Formation Model may propose that a relationship, semantic representation, episode, host-model observation, or workflow is useful. It does not directly choose Neo4j, Qdrant, KurrentDB, Temporal, SQL, Redis, or any other implementation as authority.

## 9. Logical memory architecture and candidate engines

The architecture remains capability-first and backend-neutral.

| Logical role | Purpose | Current candidate examples |
|---|---|---|
| Experience/Event Fabric | Ordered evidence, event history, replay, correction/deletion lineage | KurrentDB-class event store |
| Claim Fabric | Canonical adjudicated and bitemporal knowledge | distributed/bitemporal SQL-class store |
| Cognitive Graph | Provenance-linked relationship, causal, social, project, identity, and temporal projections | Neo4j-class graph |
| Vector/Multimodal Plane | Semantic and multimodal retrieval projections | Qdrant-class vector store |
| Object/Archive Plane | Raw files, source corpus, datasets, checkpoints, model weights, manifests, exports, backups | object store / NAS / S3-compatible systems |
| Durable Workflow Plane | Projection, repair, deletion, training, evaluation, migration, and recovery workflows | Temporal-class workflow engine |
| Workspace / Adaptive Context | Current task and attention state | Redis-class / in-memory systems |
| Episodes / Cognitive Models | Derived autobiographical, host, source-person, self, world, mission, and relationship structure | governed structured projections |
| Model/Dataset/Training Plane | Elaina identity model, Memory Formation Model, retrieval/context models, adapters, challengers | model registry + object storage + training infrastructure |

Neo4j, Qdrant, KurrentDB, Temporal, SQL engines, Redis-class systems, object stores, and future challengers remain candidates until Stage G evidence supports their roles. None is a permanent technology ceiling.

## 10. Friday boundary

Friday and A.L.I.C.E. share architecture only where the capability is host-neutral.

Transferable examples:

- Memory Formation architecture;
- schemas;
- routing contracts;
- backend interfaces;
- evaluation methodology;
- general training procedures;
- generic synthetic fixtures;
- capability validators.

Not transferable from A.L.I.C.E. to Friday:

- Elaina's source-person corpus;
- Elaina-specific learned identity/personality weights;
- Rayan's owner/host data;
- A.L.I.C.E.–Rayan relationship history;
- host-specific private adapters;
- A.L.I.C.E. continuity;
- private evaluation data tied to Elaina or Rayan.

A Friday instance uses its own host data and develops its own personality and relationship state from that host.

## 11. Stage G testing program

Backend persistence tests are necessary but far from sufficient.

The Neo4j/Qdrant persistence work already performed is evidence for physical durability only. It does not close Stage G.

Stage G must qualify the complete cognitive-memory fabric.

### G2 — learned formation and identity foundations

1. Build full gold semantic decomposition of Elaina canon plus host and adversarial cases.
2. Design and train the Memory Formation Model separately from the Elaina Identity / Personality Model.
3. Train/evaluate the Elaina Identity / Personality Model from canonical Elaina data.
4. Seed authorized real Rayan host data into the host/relationship domains without changing the Elaina identity anchor.
5. Generate a complex synthetic continuation of Rayan's life containing:
   - changing preferences;
   - projects;
   - school/work events;
   - people and relationships;
   - goals and missions;
   - plans that succeed or fail;
   - contradictory documents;
   - temporary context;
   - corrections;
   - deletions;
   - revocations;
   - uncertainty;
   - repeated behavior;
   - unsupported inference;
   - identity collisions;
   - stale summaries;
   - malicious or misleading external data.

### Later Stage G qualification

Test each memory layer separately and then together:

- Raw Buffer;
- Object/Archive Plane;
- Experience/Event Fabric;
- Claim Fabric;
- Cognitive Graph;
- Vector/Multimodal Plane;
- Episodes;
- Elaina source-person model;
- Rayan host model;
- A.L.I.C.E.–Rayan relationship model;
- A.L.I.C.E. continuity/self model;
- Mission Graph;
- Workspace;
- Projection Manager;
- Retrieval Orchestrator;
- Context Manager / Memory Fusion;
- durable workflows;
- model/dataset/training lineage.

Cross-layer tests must include:

- routing correctness;
- authority correctness;
- entity resolution;
- temporal transitions;
- contradiction handling;
- source hierarchy;
- duplicate consolidation;
- projection lag and replay;
- graph/vector consistency;
- correction propagation;
- deletion/revocation propagation;
- stale-projection repair;
- historical versus current retrieval;
- exact-source reconstruction;
- host/source-person/A.L.I.C.E. identity separation;
- A.L.I.C.E./Friday product isolation;
- concurrency;
- reordering;
- retries;
- partial outage;
- crash recovery;
- rollback;
- rebuild;
- restore;
- scale;
- latency/resource behavior;
- README and governance promise alignment.

Synthetic host-life workloads must scale progressively beyond the seed corpus to realistic and stress tiers. Scale tiers are certification points, not permanent limits.

Stage G closes only after the integrated memory architecture behaves coherently under this qualification program.

## 12. Phase 2 transition rule

Phase 2 remains the current released authority, compatibility baseline, test oracle, and fallback while Stage G is open.

Stage H is bounded successor canary authority.

Stage I is evidence-backed cutover.

Stage J is the compatibility/fallback transition.

**Final Phase 2 replacement or retirement is complete only after Stage J has itself passed its required evidence and Rayan has accepted the Stage J result.**

Neither Stage H nor Stage I alone completes final Phase 2 replacement.

## 13. Repository lifecycle

A.L.I.C.E.'s repository lifecycle is intentional:

### Current period — active construction

The repository may remain public while the system is actively being built and the repo remains obscure.

During this period, **accuracy and continuity of the architecture record take priority over unnecessary anonymization**. Project documentation may explicitly name Elaina, Rayan, A.L.I.C.E., Friday, and their roles when that specificity prevents architectural drift or loss of context.

This does not justify committing credentials, access tokens, encryption keys, raw private corpora, raw host archives, or unnecessary sensitive payloads.

### Later period — private repository

Once Friday is well established and A.L.I.C.E. no longer benefits from being publicly developed, the A.L.I.C.E. repository is intended to become private.

### Final period — owner-controlled repository custody

The long-term repository target is for A.L.I.C.E. to operate without depending on a remote Git host or remote repository service as a required control-plane dependency.

The authoritative code history, source corpus records, model artifacts, host-memory evidence, evaluation evidence, manifests, backups, and migration history will remain under owner-controlled private custody, with whatever local, networked, distributed, remote, edge, cloud, or multi-region infrastructure is justified by capability, reliability, privacy, performance, and recovery requirements.

This repository-lifecycle decision is **not a runtime or deployment topology ceiling**. A.L.I.C.E. may use remote compute, distributed services, network storage, federation, external model infrastructure, and multi-region operation whenever owner policy and evidence justify them.

Any eventual removal of a hosted remote Git repository must preserve version history, signed provenance, backups, rollback capability, recoverability, and the ability to re-establish a remote repository later if useful.

## 14. Documentation priority

For A.L.I.C.E. construction documentation:

1. architectural correctness;
2. exact identity and data-flow boundaries;
3. continuity across long development sessions;
4. provenance and owner decisions;
5. implementation/test state;
6. privacy hygiene appropriate to the current deployment stage.

Unnecessary redaction must not make the architecture ambiguous.

Raw secrets and private datasets remain protected because they are operational assets, not because the architecture must be anonymized.
