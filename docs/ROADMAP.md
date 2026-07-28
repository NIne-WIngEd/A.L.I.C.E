# A.L.I.C.E. Roadmap 2.1

**Version:** 2.2.0
**Owner:** MK Rayan
**Status:** Final top-level capability domains with a mapped Friday product track
**Principle:** Capability, learning, and governance co-evolve. Earlier implementations remain changeable when necessary to serve the final architecture.

## Phase 0 — Identity and Authority Kernel

**Status:** Released baseline; evolvable.

Constitution, owner sovereignty, authority model, memory principles, evaluation, resilience, rollback, and repository governance.

## Phase 1 — Private Evidence Foundation

**Status:** Released baseline; evolvable.

Private vault, ingestion, classification, provenance, retrieval indexes, reproducibility, and grounded evidence.

This phase may be refactored to support host-neutral storage, product identity, local encryption, Identity Capsules, or multiple host instances.

## Phase 2 — Authoritative Memory Core

**Status:** Released baseline; evolvable.

Inspectable memory, temporal conflict, authorization-aware retrieval, sensitive storage, candidate promotion, correction, deletion, and rebuild guarantees.

This phase may be extended or replaced where necessary for automated curation, multi-host isolation, personal adapters, or Friday productization.

## Phase 3 — Conversational A.L.I.C.E.

**Status:** Released baseline; evolvable.

Model abstraction, private conversation state, grounded responses, orchestration, repair, local inference, and evaluation.

Released Phase 3 behavior is a compatibility profile, not a permanent limit on tools, retries, fallback, live retrieval, memory, or learning.

## Phase 4 — Public Web and Information Intelligence

**Status:** Active; current work may resume at P4.5 after architecture migration.

Provider-neutral search and fetch, temporal metadata, citation binding, source conflict, injection-resistant evidence handling, and learning-ready observation envelopes.

Phase 4's read-only release scope is a maturity boundary. Phase 4 documents, code, tests, and schemas may change when required to support the final learning architecture, product family, or migration path.

## Phase 5 — Experience Ledger, Evaluation Substrate, and Kernel Extraction

Trajectory capture, outcomes, corrections, decision lineage, source/model/tool histories, benchmark registry, resource accounting, candidate-learning extraction, and the storage lifecycle substrate. Phase 5 establishes an immutable compact event ledger, aggressive temporary capture, content-addressed blobs, host-scoped deduplication, retention classes, hot/warm/cold/quarantine tiers, storage-pressure controls, encrypted backup manifests, and restore verification.

**Friday action:** Begin code-level separation from A.L.I.C.E. at Phase 5.0 (P5.0). New reusable contracts are built in host-neutral `cognitive_kernel` namespaces with product and host scope. Storage lifecycle contracts must be host-isolated and portable before the Phase 6.5 split.

### Storage doctrine

A.L.I.C.E. captures broadly enough to preserve future learning opportunity, but permanent full-payload retention is not the default. The governing pattern is: permanent compact ledger + policy-bounded raw buffer + utility-weighted durable memory + representative replay + encrypted archive + verified deletion. `docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md` defines the controlling lifecycle.

## Phase 6 — Cognitive Control Plane, Inspector, UI, and Voice

Interfaces for memory, beliefs, models, skills, missions, autonomy, overrides, evaluations, rollback, voice, constitutional management, host enrollment, and local model management.

**Friday action:** Build generic host identity and product-neutral control interfaces.

### Phase 6.5 — Product Separation Gate

This is a release gate, not a new top-level phase. Create separate Friday and shared-kernel repositories/packages after host-neutral APIs, storage isolation, synthetic-host tests, and generic control-plane interfaces pass.

## Phase 7 — Capability Fabric, Integrations, and Multimodal Perception

Files, email, calendars, repositories, services, scientific tools, APIs, MCP/A2A compatibility, images, audio, video, telemetry, sensors, and model routing.

**Friday action:** Deliver the first signed Windows local-ingestion alpha using the shared kernel.

## Phase 8 — Autonomous Memory, Reflection, and Procedural Learning

Learning Curator, automated memory formation, belief revision, consolidation, source trust, skill synthesis, executable procedures, learned retention, representative replay selection, compression, archival, intentional forgetting, and training candidates.

**Friday action:** Earliest credible closed alpha. Friday's core differentiation requires automated selective learning, not only local chat.

## Phase 9 — Cognitive Core

World model, temporal and causal graphs, model of Rayan, generic host model, self-model, metacognition, uncertainty, social models, identity continuity, voice, and independent judgment.

**Friday action:** Personal-intelligence beta with stable host-specific behavior and the Personalization Inspector.

## Phase 10 — Planning, Curiosity, and Proactive Agency

Hierarchical goals, search, simulation, long-running missions, proactive research, automatic curriculum, specialist-agent teams, replanning, and resource-aware initiative.

**Friday action:** Proactive local-agent beta.

## Phase 11 — Computer Use, Autonomous Coding, and Self-Evolution

General desktop and terminal operation, repository agency, code generation, skill libraries, self-modifying agents, variant archives, automatic low-risk promotion, canaries, and rollback.

**Friday action:** Public beta with evaluated computer-use and signed local skill packages.

## Phase 12 — Scientific Discovery and Formal Intelligence

Hypotheses, experiments, simulation, statistics, optimization, CAD, scientific tools, formal solvers, theorem proving, neuro-symbolic reasoning, and evolutionary discovery.

**Friday action:** Expert and research capability packs.

## Phase 13 — Continual Model Adaptation and Self-Training

Learned rankers, routers, preference models, world models, adapters, automated dataset curation, training, champion/challenger promotion, lifelong neural learning, and machine-unlearning research.

**Friday action:** Host-specific adapters and model components become a standard product feature.

## Phase 14 — Operating Environment and Embodiment

Persistent desktop/mobile/edge operation, multimodal ambient interface, secure synchronization, sensors, smart devices, laboratories, robotics, and physical-world action.

**Friday action:** Friday Operating Environment preview and dedicated-system research.

## Phase 15 — Generalized Platform and Frontier Research

Reusable personal-AI platform, developer SDK, signed capability ecosystem, agent federation, distributed intelligence, optional open-source components, and active research on capabilities listed in `RESEARCH_FRONTIERS.md`.

**Friday action:** Platform launch, household/enterprise modes, optional privacy-preserving federation, and eventual AI operating-system distribution.

## Product-track rule

Friday milestones are mapped in `docs/FRIDAY_ROADMAP.md`. They do not create or renumber top-level A.L.I.C.E. phases.

## Architecture-change rule

Top-level capability domains remain stable to prevent planning churn. Their internal implementations are not frozen. A new requirement normally becomes a module, subphase, ADR, experiment, product-track milestone, capability entry, or frontier program.

No earlier phase is immune from migration. Change it when the final architecture requires it, while preserving useful released behavior through explicit compatibility profiles and evidence-backed migrations.

## Product-family parity rule

The consumer distribution internally codenamed Friday inherits the complete destination capability set of A.L.I.C.E. through the Personal Cognitive Kernel. A.L.I.C.E. may implement frontier work earlier, but generalizable successful capabilities enter a parity ledger and downstream productization plan. Every consumer installation selects its own assistant name and develops a distinct identity.

The host chooses the assistant identity during enrollment. This capability parity commitment is permanent at the destination-architecture level.
