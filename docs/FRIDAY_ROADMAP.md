# Consumer Product Roadmap (Internal Codename Friday)

**Status:** Cross-product roadmap mapped to the frozen A.L.I.C.E. capability phases
**Rules:** Milestones do not add or renumber A.L.I.C.E. phases. The consumer product shares A.L.I.C.E.'s complete capability destination. Each installation uses a host-selected assistant name.

## Milestone map

| Friday milestone | A.L.I.C.E. dependency | Deliverable |
|---|---|---|
| F0 — Product definition | Phase 4.5 | Vision, full capability parity, privacy promise, host-selected identity, product-brand clearance track |
| F1 — Shared foundation | Phase 5 | Kernel identity, Experience Ledger/storage/evaluation contracts, Mission Graph, Result Capsule, traceback, attention/workspace/speaker/guest schemas, dual-approval schemas, parity expansion |
| F2 — Cognitive Workspace | Phase 6 | Mission Canvas, adaptive windows, control plane, graph inspector, Result Capsule viewer, attention explanation, guest/trust UI, product-neutral model controls |
| F3 — Independent Product Readiness | Phase 6.5 gate | Independent build, versioned kernel pin, host isolation, dual-approval signing gate, migration, rollback, and parity evidence |
| F4 — Local ingestion alpha | Phase 7 | Windows app, host naming, hardware benchmark, multimodal local ingestion, connector permissions, local model runtime |
| F5 — Learning closed alpha | Phase 8 | Automated memory curation, beliefs, skills, correction, deletion, Identity Capsule |
| F6 — Personal intelligence beta | Phase 9 | User model, self-model, voice, judgment, uncertainty, personalization inspector |
| F7 — Proactive agent beta | Phase 10 | Goals, long-running local missions, curiosity, planning, background operation |
| F8 — Computer-use public beta | Phase 11 | Desktop/terminal action, skill packages, coding, evaluated local autonomy |
| F9 — Research and expert packs | Phase 12 | Scientific tools, formal reasoning, domain capability packs |
| F10 — Host-specific model adaptation | Phase 13 | Local rankers, routers, LoRA/adapters, challenger training, deletion-aware retraining |
| F11 — Friday environment preview | Phase 14 | Persistent service, cross-device continuity, voice, sensors, AI-native shell/OS layer |
| F12 — Platform launch | Phase 15 | Developer SDK, signed skill ecosystem, optional federation, enterprise/household modes |

## F0 — Product definition

Deliver now:

- Friday product vision;
- A.L.I.C.E.–Friday separation plan;
- privacy and non-access architecture;
- name and intellectual-property risk register;
- YC narrative and demo targets;
- product-line policy and validator.

Exit criteria:

- the product is not described merely as a local chatbot;
- each host instance has a distinct technical identity;
- developer non-access is an architectural property;
- the independent-repository boundary and Phase 6.5 readiness semantics are fixed.

## F1 — Shared foundation

**A.L.I.C.E. dependency:** Phase 5.0

During A.L.I.C.E. Phase 5:

- establish independently versioned host-neutral kernel contracts;
- define product and host identity, Experience Ledger, storage, evaluation, and migration contracts;
- define Mission Graph schemas and events;
- define Result Capsule and traceback state machines;
- define attention-decision, workspace-projection, speaker-context, and guest-grant schemas;
- add dual-approval governance and release-attestation schemas;
- test one synthetic A.L.I.C.E.-style host and at least two isolated synthetic Friday hosts;
- expand the parity ledger.

Friday product code remains in the Friday repository. The complete consumer UI is not required in F1.

Exit criteria:

- shared contracts contain no product-private state;
- two synthetic Friday hosts remain isolated;
- Friday can pin an explicit kernel contract version;
- production promotion cannot be represented without both required approvals.

## F2 — Cognitive Workspace

During A.L.I.C.E. Phase 6 and the parallel Friday product track:

- build the Mission Canvas and Mission Graph inspector;
- build adaptive multi-window composition with no empty fixed slots;
- build Result Capsule and traceback views;
- expose attention explanations and host workspace commands;
- expose speaker trust, guest mode, permission state, and sensitive-view hiding;
- build generic memory, learning, model, and capability controls;
- integrate each product independently against the same semantic contracts.

Exit criteria:

- A.L.I.C.E. and a synthetic Friday host interpret node state, Result Capsules, traceback, attention, and guest authority identically;
- owner-specific policy and branding remain outside shared components;
- Friday initializes without A.L.I.C.E. source or state.

## F3 — Phase 6.5 Independent Product Readiness Gate

Prove:

- Friday's repository and build are independent from A.L.I.C.E.;
- the kernel contract is independently versioned and pinned;
- no A.L.I.C.E. private state exists in Friday artifacts;
- at least two Friday hosts remain isolated through storage, cache, backup, restore, and deletion;
- release manifests support exact-artifact A.L.I.C.E. audit and Rayan approval;
- production signing rejects missing, mismatched, expired, or revoked approvals;
- emergency rollback works and emergency feature addition fails;
- migration, rollback, shared UI contracts, and parity tracking pass.

## F4 — Windows local-ingestion alpha

Recommended first product stack:

- Tauri desktop shell or an equivalent signed native Windows shell;
- React/TypeScript UI;
- Rust host process for permissions, storage, updates, and process control;
- Python sidecar initially for mature A.L.I.C.E. ML/retrieval components;
- llama.cpp-compatible local generation runtime;
- ONNX Runtime for embeddings, classifiers, vision/audio, and on-device training experiments;
- encrypted content-addressed local object store, database, vector index, lifecycle manager, and backup/restore verifier;
- DPAPI-protected local master key on Windows.

The architecture must allow later replacement of any runtime.

Alpha flow:

1. install signed application;
2. benchmark hardware;
3. choose offline-only or local-first mode;
4. choose the assistant name and create host keys plus the Identity Capsule;
5. select folders and connectors;
6. perform metadata-only preview;
7. approve ingestion scope;
8. build initial memory and preference model;
9. review what Friday learned;
10. run personalization baseline evaluation.

## F5 — Learning closed alpha

This is the minimum credible Friday launch cohort. Friday must automatically determine which interactions and files deserve:

- discard;
- session retention;
- episodic memory;
- semantic memory;
- derived belief;
- skill extraction;
- training-candidate status;
- quarantine;
- warm or cold archival;
- representative replay retention;
- verified deletion.

Closed-alpha gates:

- useful-memory precision and recall targets;
- no cross-host data leakage;
- complete source lineage;
- correction and deletion propagation;
- encrypted export/import;
- storage-pressure behavior that preserves protected records;
- successful backup and restore verification;
- offline operation;
- no mandatory vendor account.

## F6 — Personal intelligence beta

Friday becomes meaningfully different for each host through:

- preference prediction;
- stable voice;
- user and world models;
- independent judgment;
- confidence calibration;
- counterfactual recommendations;
- local personal evaluation suites.

Marketing may claim a distinct personal AI system. Claims of a separately trained model must correspond to actual host-specific learned parameters, not memory alone.

## F7–F8 — Proactive and computer-use beta

Friday begins completing local missions across applications. Permission prompts are replaced where appropriate by host-created standing mandates, visible action history, and revocable capability grants.

## F9–F10 — Expert packs and host adapters

Friday trains small local components first:

- memory utility classifier;
- source-trust model;
- preference ranker;
- tool router;
- style adapter;
- task or domain adapters.

Full base-model retraining is a research option, not a requirement for a distinct Friday instance.

## F11 — Friday Operating Environment

The application evolves into an AI-native environment through:

- persistent local service;
- secure boot-time activation;
- identity-aware application permissions;
- personal data fabric;
- local model scheduler;
- voice and multimodal shell;
- device and sensor integration;
- optional dedicated hardware image.

A true standalone operating system is considered after the OS-layer product proves demand and the team can sustain drivers, updates, security response, and hardware compatibility.

## F12 — Platform

Friday exposes:

- SDK and local APIs;
- signed capability packages;
- model packs;
- enterprise or family tenancy;
- optional privacy-preserving federation;
- third-party integrations with declared data flow;
- migration between compatible devices and models.

## Capability parity lane

Every A.L.I.C.E. phase has a parallel downstream productization lane:

1. A.L.I.C.E. or shared research proves a capability.
2. The capability receives a stable identifier and evaluation suite.
3. Owner-specific dependencies are removed.
4. It enters the Personal Cognitive Kernel or a signed product capability package.
5. Hardware, migration, privacy, and support work is completed.
6. It ships to host-named consumer instances.

A capability may be delayed in this lane. It may not be removed from the destination merely because it is powerful, experimental, or first appeared in A.L.I.C.E.

## Team handoff lane

Before a dedicated consumer team exists, the core team develops A.L.I.C.E., the kernel, and the consumer product together. After the team passes an independent-maintenance gate, Rayan may focus primarily on A.L.I.C.E.; the consumer team owns downstream implementation, support, and platform maintenance; production promotion remains subject to A.L.I.C.E. audit and Rayan approval.
