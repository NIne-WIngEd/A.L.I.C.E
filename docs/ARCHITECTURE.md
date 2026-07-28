# A.L.I.C.E. and Friday Permanent Cognitive Architecture

**Version:** 1.3.0
**Authority:** A.L.I.C.E. Constitution and ratified product-family policy

## 1. System family

The project consists of three architectural identities:

1. **Personal Cognitive Kernel** — host-neutral libraries, schemas, learning machinery, evaluators, and capability contracts.
2. **A.L.I.C.E.** — Rayan's owner-specific flagship and unrestricted long-term research implementation.
3. **Friday** — a separately distributed local-first product that creates a new host-specific intelligence for each installation.

A.L.I.C.E. and Friday may share kernel code. They do not share personal state, authority, learned identity, or product policy.

## 2. Model-independent identity

Base models are replaceable engines. Persistent identity lives in governed memory, beliefs, skills, adapters, evaluation history, constitutional configuration, and the Identity Capsule.

## 3. Major planes

### Identity and authority kernel

Contains owner/host identity, Constitution, mission mandates, autonomy classes, cryptographic authority, audit roots, and stop/rollback controls.

### Cognitive executive

Maintains goals, attention, priorities, task state, budgets, interruption, delegation, and model/tool routing.

### Experience and learning plane

Captures trajectories, outcomes, corrections, evaluations, and learning events. Hosts curation, reflection, consolidation, skill extraction, dataset formation, local training, replay selection, and evolution loops.

### Memory and knowledge plane

Contains episodic, semantic, temporal, causal, belief, user-model, self-model, world-model, and procedural stores.

### Storage lifecycle plane

Maintains the permanent compact event ledger, content-addressed encrypted blob store, retention classes, hot/warm/cold/quarantine tiers, storage budgets, backup manifests, restore verification, deletion lineage, and replay-buffer manifests. Capture is aggressive at the temporary boundary; durable retention is utility-weighted and evidence-linked. Deduplication is limited to the same host and encryption domain unless a separate privacy proof authorizes a broader scope.

### Reasoning and planning plane

Provides decomposition, search, simulation, counterfactual reasoning, critics, verifiers, formal tools, and specialist-agent coordination.

### Capability fabric

Exposes models, APIs, files, services, computers, scientific tools, multimodal perception, devices, robots, and other agents through versioned manifests.

### Evolution laboratory

Runs code modification, agent variants, training jobs, champion/challenger evaluation, canaries, promotion, rollback, and research experiments.

### Production execution plane

Executes mission-authorized actions, records state transitions, verifies outcomes, and exposes recovery.

### Product distribution plane

For Friday, distributes signed installers, common model packs, schema migrations, and public evaluations without receiving raw host state.

## 4. Core data flow

Observation → compact Experience Ledger + policy-bounded raw buffer → Curator → retain/compress/archive/delete decision → memory/belief/skill/training candidates → representative replay or model adaptation → executive → plan/simulation → action → verification → outcome → learning.

## 5. Product and host scoping

Every persistent event and artifact must identify:

- product identity;
- host instance;
- schema version;
- provenance;
- encryption domain;
- deletion lineage;
- content digest;
- retention class;
- active storage tier.

Shared kernel components may process multiple synthetic or separately authorized hosts, but production personal state is isolated by construction.

## 6. Model fabric

A router selects local or external models by competence, privacy, cost, latency, modality, context, hardware, and measured reliability. Multiple specialist models may collaborate behind one product identity.

Friday defaults to local execution. External compute is an optional declared capability.

## 7. Self-evolution

Every evolvable module declares interfaces, benchmarks, consequence class, rollback method, and promotion authority. Candidate versions may coexist. A.L.I.C.E. may pursue frontier self-evolution; Friday product promotion follows product release and host-authority policies.

## 8. Shared-kernel extraction

New Phase 5+ reusable work begins in host-neutral namespaces. Phase 1–4 modules may be migrated whenever they contain owner-specific coupling or product barriers. `docs/SHARED_KERNEL_EXTRACTION_STANDARD.md` is controlling for extraction.

## 9. Governance principle

Containment enables experimentation; evidence enables deployment. Permanent capability bans are not used as a substitute for engineering. Completed phases and tests are historical evidence, not veto authorities.

## 10. Extension rule

New capabilities enter through manifests and modules and connect to experience capture, evaluation, authority, and outcome verification. Product-specific behavior remains outside the shared kernel.
