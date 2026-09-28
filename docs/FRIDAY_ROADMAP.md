# Consumer Product Roadmap (Internal Codename Friday)

> [!IMPORTANT]
> **OWNER-RATIFIED FLAGSHIP CAPABILITY RULE:** A.L.I.C.E. is the flagship and mandatory default capability upstream. Through at least completion of A.L.I.C.E. Phase 15, Friday must receive every transferable A.L.I.C.E. capability. Friday may gain a new capability only after A.L.I.C.E. has implemented, evaluated, approved, and gained it, unless MK Rayan records an explicit exact-scope owner override.
>
> This owner-ratified rule supersedes conflicting capability-order, team-independence, repository-creation, alpha/beta-release, or reduced-product language in this document.

**Version:** 3.0.0  
**Status:** Cross-product engineering and qualification roadmap  
**Rules:** Milestones do not add or renumber A.L.I.C.E. phases. They are internal construction/qualification checkpoints until the single Fable v1 consumer release gate passes.

## 1. Release doctrine

The old interpretation that F4, F5, F6, F7, or F8 could be progressively shipped as smaller consumer editions is superseded.

Before Fable v1:

- F4–F11 are **internal engineering and qualification milestones**;
- names such as alpha, closed alpha, beta, and preview may still exist in historical schemas or test channels, but they do not define smaller consumer capability tiers;
- an internal build may exercise only the capability under test, but it cannot be marketed or treated as the Fable v1 product;
- no partial-capability milestone is a consumer launch gate.

Fable v1 has one consumer capability boundary:

> **the full transferable personal cognitive foundation must be qualified as one connected entity.**

That foundation includes the complete Memory v4.x fabric, FBM, MFM, host/relationship/Fable-self development, native personal judgment, Mission Graph, Cognitive Workspace, procedural learning, governed personal-model evolution, correction/deletion/unlearning, multi-device continuity, inspection/export/restore/rollback, and provider/model replacement.

General feature models are the exception. In v1, fluent language generation, coding, research, simulation, vision, image editing, and similar broad feature work may use replaceable GPT/Claude-class APIs under the local Fable controller and egress policy. Those providers do not own the personal cognitive foundation.

F12 is post-v1 platform/ecosystem expansion. A developer SDK or marketplace is not required before the first personal Fable can ship.

## 2. Milestone map

| Friday milestone | A.L.I.C.E. dependency | Internal deliverable |
|---|---|---|
| F0 — Product definition | Phase 4.5 | Vision, full capability parity, privacy promise, host-selected identity, product-brand clearance |
| F1 — Shared foundation | Phase 5 | Kernel identity, Experience/memory/evaluation contracts, Mission Graph, Result Capsule, traceback, attention/workspace/speaker/guest schemas, dual-approval schemas |
| F2 — Cognitive Workspace | Phase 6 | Mission Canvas, adaptive workspace, control plane, graph inspector, Result Capsule viewer, attention explanation, guest/trust UI |
| F3 — Independent Product Readiness | Phase 6.5 gate | Independent build, versioned kernel pin, host isolation, dual-approval signing, migration, rollback, parity evidence |
| F4 — Ingestion and local-runtime qualification | Phase 7 | Signed Windows host, hardware planning, multimodal ingestion, connectors, local runtime, egress controls |
| F5 — Selective memory and learning qualification | Phase 8 | Full Memory v4.x formation/lifecycle path, beliefs, episodes, procedures, correction/deletion, Identity Capsule |
| F6 — Personal intelligence qualification | Phase 9 | Host, relationship and Fable-self development, world/social/causal state, native judgment, uncertainty, voice |
| F7 — Mission and proactive-agency qualification | Phase 10 | Goals, long-running missions, curiosity, planning, background operation, resource-aware initiative |
| F8 — Action, skill and self-evolution qualification | Phase 11 | Local action/controller path, tool/skill packages, coding/action integration, evaluated self-evolution and rollback |
| F9 — Expert feature integration qualification | Phase 12 | Scientific/formal/domain feature engines behind the personal controller; API-backed feature work remains allowed |
| F10 — Host-specific model adaptation qualification | Phase 13 | Local rankers, routers, adapters, challenger training, deletion-aware retraining and model continuity |
| F11 — Persistent environment and multi-device qualification | Phase 14 | Persistent service, cross-device continuity, voice/sensors, scheduler, private distributed placement |
| **Fable v1 Release Gate** | **F4–F11 + upstream acceptance** | **Complete transferable personal cognitive foundation; no reduced capability tier** |
| F12 — Platform/ecosystem expansion | Phase 15 | SDK, signed capability ecosystem, household/enterprise modes, optional federation and later platform distribution |

## 3. F0 — Product definition

Deliver:

- product vision;
- A.L.I.C.E.–Fable separation;
- privacy and non-access architecture;
- public-name/IP track;
- company narrative;
- product-line policy and validators;
- single full-v1 release doctrine.

Exit criteria:

- Fable is not described as a chatbot or memory plugin;
- each host instance has a distinct technical identity;
- developer non-access is an architectural property;
- no document defines a permanently reduced consumer edition;
- the first consumer release is bound to the full personal cognitive foundation.

## 4. F1 — Shared foundation

**A.L.I.C.E. dependency:** Phase 5.0

Build host-neutral contracts for:

- product and host identity;
- Experience/Event and Claim authority;
- storage and evaluation;
- Mission Graph;
- Result Capsule and traceback;
- attention/workspace;
- speaker/guest state;
- model/data lineage;
- release attestations;
- deletion and rollback;
- parity tracking.

Test at least one synthetic A.L.I.C.E.-style host and two isolated synthetic Fable hosts.

F1 is not a Fable release.

## 5. F2 — Cognitive Workspace

Build:

- Mission Canvas and Mission Graph inspector;
- adaptive multi-window composition;
- Result Capsule and traceback views;
- attention explanations;
- speaker trust and guest mode;
- memory/learning/model/capability controls;
- inspection of what the local personal foundation knows and why.

F2 is not a Fable release.

## 6. F3 — Phase 6.5 Independent Product Readiness Gate

Prove:

- independent repository/build;
- pinned host-neutral kernel interfaces;
- no A.L.I.C.E. private state in Fable artifacts;
- multi-host isolation through storage/cache/backup/restore/deletion;
- exact-artifact A.L.I.C.E. audit + Rayan approval path;
- production-signing rejection when approvals mismatch;
- emergency rollback without emergency capability addition;
- migration and rollback.

Phase 6.5 proves product independence. It does **not** authorize a partial consumer edition.

## 7. F4 — Ingestion and local-runtime qualification

Build and qualify the installation/runtime shell:

- signed native Windows application;
- host keys and Identity Capsule;
- hardware capability measurement;
- local/private service orchestration;
- authorized source selection;
- multimodal ingestion;
- connector permissions;
- source preview and custody controls;
- local model/runtime management;
- visible provider egress controls.

This milestone may use synthetic or controlled internal hosts. Calling such a build an alpha for engineering purposes does not make it a consumer Fable release.

## 8. F5 — Selective memory and learning qualification

Qualify the **full memory architecture**, not a memory-only product edition.

Required capabilities include:

- Experience/Event history;
- bitemporal Claim Authority;
- raw evidence/object storage;
- episodes/autobiographical memory;
- Cognitive Multi-Graph;
- associative graph retrieval;
- vector/multimodal retrieval;
- source-native/live retrieval;
- perceptual personal memory where authorized;
- Memory Resource Manager;
- Retrieval Orchestrator / Cognitive Recollection;
- Lifecycle Curator;
- procedural memory;
- correction/deletion/unlearning;
- model/dataset influence lineage;
- restore/rebuild/rollback.

The old phrase “minimum credible closed alpha” is superseded. F5 is an internal qualification milestone only.

## 9. F6 — Personal intelligence qualification

Qualify:

- host/user model;
- relationship model;
- Fable self/continuity;
- world/social/causal state;
- preferences and values;
- native personal judgment;
- uncertainty and confidence calibration;
- stable expression/voice;
- provider-swap continuity;
- causal intervention tests proving that personal state, not a fixed prompt or generic provider prior, changes judgment.

F6 is not a beta release. It is one required slice of the eventual v1 entity.

## 10. F7 — Mission and proactive-agency qualification

Qualify:

- hierarchical goals;
- Mission Graph continuity;
- long-running missions;
- proactive research/monitoring;
- replanning;
- resource-aware initiative;
- background operation;
- outcome capture and later learning.

These capabilities belong to the personal cognitive foundation. They are not postponed because an external feature model can draft text.

## 11. F8 — Action, skill and self-evolution qualification

Qualify:

- local desktop/terminal/tool execution under host authority;
- reusable procedural skills;
- signed skill packages;
- coding/action integration through replaceable feature engines;
- challenger generation;
- frozen evaluation criteria;
- canary promotion;
- rollback;
- preservation of identity/continuity across model changes.

General feature generation may still be API-backed in v1. The local Fable owns the decision, action authority, skill memory, evaluation, and promotion logic.

## 12. F9 — Expert feature integration qualification

Integrate scientific, formal, research, simulation, vision, coding, and other expert feature capabilities behind the same local personal controller.

Fable v1 does not require first-party frontier feature models. External feature engines remain replaceable suppliers behind privacy/egress controls.

## 13. F10 — Host-specific model adaptation qualification

Qualify personal parametric learning such as:

- memory utility and retention models;
- source-trust models;
- retrieval/routing models;
- preference/judgment rankers;
- adapters;
- personal perceptual banks;
- host/relationship/self updaters;
- challenger personal models.

Required controls:

- exact data/model lineage;
- representative replay;
- correction/deletion influence;
- champion/challenger evaluation;
- rollback;
- provider/base-model replacement.

There is no fixed count or maximum size for personal learned components.

## 14. F11 — Persistent environment and multi-device qualification

Qualify:

- persistent local service;
- secure boot-time activation where used;
- cross-device continuity;
- device identities and causal clocks;
- offline continuation and reconciliation;
- local/private model scheduler;
- voice and multimodal shell;
- sensors/integrations where authorized;
- one-machine, workstation, NAS, home-cluster and private-cluster placements of the same logical architecture.

A small machine may change placement or scheduling. It may not create a smaller cognitive edition.

## 15. Fable v1 Release Gate

The first consumer release is eligible only when a fresh authorized user corpus can build one connected Fable that:

1. does not use A.L.I.C.E. private data or weights;
2. contains the complete transferable personal memory architecture;
3. develops host, relationship and Fable-self state;
4. forms memory through MFM behind deterministic authority;
5. performs adaptive episodic, graph/associative, vector/multimodal, source-native, procedural, mission and personal-state recollection;
6. makes native personal judgments before replaceable feature generation;
7. checks/corrects provider output against its own verdict and expression contract;
8. learns from outcomes through governed updates;
9. maintains Mission Graph and Cognitive Workspace continuity;
10. supports governed personal-model evolution and rollback;
11. corrects/deletes/revokes influence across durable, execution and parametric state;
12. survives restart, restore, migration and device change;
13. preserves identity across provider/model replacement;
14. scales physical placement without removing logical cognitive planes;
15. passes privacy, egress, provenance, isolation, latency/resource and failure-recovery gates;
16. passes the product/comic behavioral suite.

No F4–F11 milestone by itself satisfies this release gate.

## 16. F12 — Platform/ecosystem expansion

After v1, expand into:

- developer SDK and local APIs;
- signed capability ecosystem;
- model/capability packs;
- household/enterprise tenancy;
- optional privacy-preserving federation;
- broader operating-environment distribution;
- later first-party frontier feature models.

These are platform expansions. They do not retroactively define what the personal foundation needed to be.

## 17. Capability parity lane

Every transferable A.L.I.C.E. capability follows:

1. A.L.I.C.E. implements and evaluates it.
2. It receives a stable identifier and evidence bundle.
3. Owner-specific dependencies are removed.
4. Host-neutral contracts/evaluations are generalized.
5. Fable productizes it without removing semantics.
6. Hardware/privacy/migration/support work is completed.
7. The exact Fable candidate is audited and approved.

Temporary implementation lag is allowed during development. Permanent omission from the destination is not.

## 18. Team handoff lane

Before a dedicated product team exists, the core project maintains A.L.I.C.E., the shared kernel and Fable together.

After a team passes independent-maintenance gates:

- it may own packaging, compatibility, support and downstream productization;
- A.L.I.C.E. remains the capability upstream;
- the full-v1 release doctrine remains binding;
- no team may redefine an internal qualification milestone as a smaller consumer intelligence tier without an explicit owner-ratified architecture change.
