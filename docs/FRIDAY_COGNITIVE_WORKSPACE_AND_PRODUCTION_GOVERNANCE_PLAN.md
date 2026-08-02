# Friday Cognitive Workspace and Production Governance Plan

> [!IMPORTANT]\n> **OWNER-RATIFIED FLAGSHIP CAPABILITY RULE:** A.L.I.C.E. is the flagship and mandatory default capability upstream. Through at least completion of A.L.I.C.E. Phase 15, Friday must receive every transferable A.L.I.C.E. capability. Friday may gain a new capability only after A.L.I.C.E. has implemented, evaluated, approved, and gained it, unless MK Rayan records an explicit exact-scope owner override.\n>\n> This owner-ratified rule supersedes conflicting capability-order, team-independence, or Phase 6.5 repository-creation language in this document.


**Status:** Owner-accepted product and governance direction
**Internal codename:** Friday
**Public product name:** To be selected separately
**Assistant identity:** Chosen locally by each host
**Final production authority:** MK Rayan
**Mandatory technical auditor:** A.L.I.C.E.
**Initial form:** Signed Windows application
**Long-term form:** Personal AI operating environment and possible standalone operating system
**Planning date:** 2026-07-31
**Suggested repository path:** `docs/FRIDAY_COGNITIVE_WORKSPACE_AND_PRODUCTION_GOVERNANCE_PLAN.md`

---

## 1. Purpose

This document defines:

1. how Friday receives the same generalizable Mission Graph and Cognitive Workspace capability as A.L.I.C.E.;
2. how Friday remains a separate product with separate repositories, host identities, private data, policies, and release maturity;
3. how development works before and after a dedicated Friday production team exists;
4. the permanent requirement that every Friday production feature be audited by A.L.I.C.E. and explicitly approved by MK Rayan before production promotion.

Friday is not a permanently reduced chatbot edition of A.L.I.C.E.

Friday is the distributable, host-neutral consumer product derived from the same personal-cognitive destination architecture.

Every installation creates a distinct host-owned intelligence.

---

## 2. Canonical product decisions

Unless Rayan explicitly revises them, future planning and implementation must preserve these rules:

1. Friday receives the same generalizable Mission Graph and Cognitive Workspace destination as A.L.I.C.E.
2. Friday is not one fixed assistant identity distributed to everyone.
3. Each host chooses the assistant's local name, voice, and identity.
4. Each installation has independent memories, beliefs, skills, evaluations, adapters, models, permissions, and Mission Graph state.
5. Friday and A.L.I.C.E. share capability through a host-neutral Personal Cognitive Kernel.
6. Friday never receives Rayan's private state, credentials, memories, personality state, or owner-specific tools.
7. Friday product code is never implemented inside the A.L.I.C.E. repository.
8. Friday has a separate repository from its first product implementation commit.
9. Before a dedicated Friday team exists, Friday and A.L.I.C.E. are developed together through shared contracts.
10. Generalizable production-ready A.L.I.C.E. UI capability enters Friday's implementation lane immediately.
11. A.L.I.C.E. may receive frontier experiments first.
12. Temporary productization lag is allowed; permanent capability omission is not.
13. A future Friday team may independently research, design, implement, test, and propose features.
14. A future Friday team may not independently ship production behavior.
15. Every production feature requires an exact-artifact A.L.I.C.E. audit attestation.
16. Every production feature requires Rayan's explicit production approval.
17. Either A.L.I.C.E. or Rayan may veto or return a candidate for revision.
18. Emergency rollback, disablement, containment, or reversion is allowed without introducing new capability.
19. An emergency fix that creates replacement production behavior still requires dual approval.
20. Only Rayan may amend the dual-approval rule.

---

## 3. Relationship to the existing product-family architecture

The current product-family direction already establishes:

- A.L.I.C.E. as Rayan's owner-specific flagship and frontier research deployment;
- Friday as a consumer distribution of the same evolving personal-cognitive architecture;
- destination capability parity;
- host-selected assistant identity;
- local-first private state;
- shared-kernel development before a dedicated consumer team;
- a machine-readable capability-parity ledger.

This plan extends those principles in two ways:

1. it makes the Mission Graph and Cognitive Workspace an explicit same-feature development program for both products;
2. it adds permanent dual-key production governance for Friday.

---

## 4. Product-family topology

The product family contains three independent identities.

### 4.1 Personal Cognitive Kernel

The Personal Cognitive Kernel owns host-neutral capability.

It should contain:

```text
Mission Graph schemas
Task-node state machines
Semantic-routing interfaces
Result Capsule contracts
Traceback Engine
Attention Engine contracts
Workspace projection contracts
Adaptive Compositor algorithms
Presence and speaker-state contracts
Guest-session primitives
Permission primitives
Experience Ledger contracts
Evaluation suites
Migration interfaces
Product and host scoping
```

It must not contain:

```text
Rayan's memories
A.L.I.C.E.'s personality or constitutional state
A.L.I.C.E.'s credentials
Friday customer data
Any host's voice profile
Product branding
Commercial account logic
Hard-coded host names
Owner-specific missions
```

Every persistent kernel record must carry:

- `product_id`;
- `host_instance_id`;
- `encryption_domain`;
- schema version;
- digest;
- provenance;
- retention class;
- storage tier;
- deletion lineage.

### 4.2 A.L.I.C.E.

A.L.I.C.E. owns:

- Rayan's private identity;
- Rayan's Mission Graph;
- Constitution and authority relationship;
- private memory and credentials;
- owner-specific integrations;
- frontier experiments;
- aggressive owner-approved autonomy;
- private UI experiments;
- research validation of generalizable capability.

### 4.3 Friday

Friday owns:

- consumer onboarding;
- host enrollment;
- host-selected assistant identity;
- product shell and branding;
- Windows installer and updater;
- generic Cognitive Workspace;
- product privacy defaults;
- hardware detection and model selection;
- local data onboarding;
- accessibility;
- consumer-safe authority profiles;
- product-specific integrations;
- support and diagnostics;
- signed downstream releases.

---

## 5. Revised repository topology

### 5.1 Rejected interim topology

Friday product source must not be placed inside:

```text
A.L.I.C.E/src/products/friday_incubator/
```

or any equivalent A.L.I.C.E. repository namespace.

The current public separation plan's interim single-repository arrangement is superseded by this owner decision once ratified through the normal governance process.

### 5.2 Permanent topology

```text
personal-cognitive-kernel/
    Host-neutral libraries, schemas, state machines, evaluators, and migrations

A.L.I.C.E/
    Rayan-specific flagship, private adapters, private policies, and research

Friday/
    Consumer shell, onboarding, UI, packaging, updates, and product integrations

friday-model-packs/
    Optional signed generic model, adapter, and evaluation manifests

product-family-governance/
    Optional future repository for cross-product release attestations,
    parity records, governance schemas, and signed release manifests
```

### 5.3 Timing

- Personal Cognitive Kernel extraction begins with Phase 5 host-neutral work.
- The Friday repository exists before the first Friday-specific product source commit.
- Phase 6.5 becomes an **Independent Product Readiness Gate**, not the first repository-creation event.
- Friday and A.L.I.C.E. pin explicit kernel versions.
- No product imports private source or state from the other product.

---

## 6. Same Cognitive Workspace capability

Friday initially receives the same generalizable UI and task-management architecture as A.L.I.C.E.

Shared capability includes:

```text
MissionGraph
Mission
MissionNode
MissionEdge
NodeStatus
ExecutionState
VisibilityState
RoutingDecision
ResultCapsule
TracebackTransition
AttentionDecision
WorkspaceProjection
WorkspaceLayout
SpeakerContext
GuestSession
GuestGrant
AuthorityRequest
```

### 6.1 Shared Mission Graph behavior

Friday must support:

- top-level missions;
- nested task nodes;
- child and sibling branching;
- reattachment to a more appropriate active node;
- independent mission creation;
- control-plane commands;
- immutable node identity;
- dotted navigation aliases;
- typed graph links;
- microbranch promotion;
- Result Capsules;
- automatic traceback;
- conflict handling;
- reopening completed nodes.

### 6.2 Shared attention behavior

Friday must support:

- host foreground and background commands;
- pinning;
- visibility limits;
- explainable ranking;
- protected security interrupts;
- deadline and blocker priority;
- result-ready priority;
- interruption-cost handling;
- layout stability.

### 6.3 Shared workspace behavior

Friday must support:

- dynamic multi-window layout;
- no empty fixed slots;
- pane and compact-card projections;
- Mission Canvas;
- dependency visualization;
- Result Capsule viewer;
- permission and trust state;
- background mission panel;
- future multi-monitor and spatial surfaces.

### 6.4 Shared voice and guest behavior

Friday must support:

- text and voice interaction;
- speaker diarization;
- speaker uncertainty;
- host-context recognition;
- stronger authentication for privileged actions;
- guest mode;
- scoped delegated guests;
- sensitive-view hiding;
- revocation and expiry.

---

## 7. Shared UI component strategy

Where practical, maintain product-neutral UI packages.

Suggested structure:

```text
cognitive-workspace-ui/
├── MissionCanvas
├── MissionTree
├── DependencyGraph
├── TaskWindow
├── TaskCard
├── ResultCapsuleView
├── AttentionExplanation
├── PermissionInspector
├── SpeakerTrustIndicator
├── GuestSessionBanner
├── BackgroundMissionPanel
└── WorkspaceCompositor
```

The shared component layer may be:

- part of the Personal Cognitive Kernel repository;
- a separately versioned shared UI repository;
- or a set of independent packages.

Required rules:

- component behavior is host-neutral;
- canonical mission state is not stored in the frontend;
- product branding is injected through configuration;
- A.L.I.C.E. and Friday may apply different visual themes;
- shared semantic behavior is tested once and integrated separately;
- product-specific behavior never mutates shared contracts without versioning and migration.

---

## 8. Product-specific differences

A.L.I.C.E. and Friday may differ in:

- branding;
- visual theme;
- onboarding;
- default authority profile;
- available integrations;
- hardware assumptions;
- release channel;
- experimental-feature exposure;
- support and diagnostic tools;
- commercial account behavior;
- product analytics policy;
- accessibility defaults.

They must not silently diverge in:

- Mission Graph meaning;
- node status;
- Result Capsule semantics;
- traceback rules;
- guest authority boundaries;
- protected interrupts;
- host ownership;
- product and host isolation.

```text
Shared:
Mission Graph
Typed relationships
Traceback
Result propagation
Attention explanations
Workspace projections
Guest authority enforcement

A.L.I.C.E.-specific:
Rayan's identity
Private mission types
Research modules
Frontier experiments
Owner-specific commands

Friday-specific:
Host enrollment
Consumer onboarding
Accessibility presets
Safe product defaults
Installer and updates
Support tools
```

---

## 9. Friday Mission Graph

Each Friday installation has its own host-local Mission Graph.

```text
1.0.0.0 — Organize a job search
2.0.0.0 — Plan a trip
3.0.0.0 — Repair a home computer
```

Nested work follows the same routing operations:

1. continue current node;
2. create child;
3. create sibling;
4. reattach to another node;
5. create new mission;
6. execute a workspace-control command.

The vendor must not maintain a centralized readable copy of private Mission Graph state.

The product may optionally synchronize encrypted state across host-owned devices, but decryption authority remains with the host.

---

## 10. Friday adaptive multi-window workspace

Friday follows the no-empty-slot invariant:

> The interface displays only useful task projections that currently exist.

| Visible work | Default composition |
|---|---|
| 1 task | Full workspace |
| 2 tasks | Focus-and-support split |
| 3 tasks | One primary plus two secondary panels |
| 4 tasks | Adaptive grid |
| 5–6 tasks | Primary and secondary panels plus live cards |
| 7–10 tasks | Command-center view |
| More than 10 | Highest-value set visible; remaining tasks stay in the graph |

The local host may command:

```text
Bring this forward.
Send this to the background.
Pin this task.
Show only this mission.
Do not interrupt me for this category.
Keep security alerts visible.
Use a maximum of three windows.
Restore automatic layout.
```

A.L.I.C.E.'s learned preferences do not become Friday defaults.

Friday starts with evaluated generic defaults and learns each host's preferences locally.

---

## 11. Friday attention policy

### 11.1 Layer 1 — Host overrides

The local host controls:

- pinning;
- foregrounding;
- backgrounding;
- visibility limits;
- layout lock;
- focus mode;
- interruption preferences.

### 11.2 Layer 2 — Protected interrupts

The following cannot be disabled by a learned ranker:

- immediate device or data threat;
- uncertain identity during a privileged action;
- destructive action awaiting authorization;
- security breach;
- failed migration;
- imminent deadline requiring intervention;
- integrity conflict affecting an active mission.

### 11.3 Layer 3 — Locally learned ranking

Initial order:

1. Host decision required.
2. Security or permission issue.
3. Critical-path blocker.
4. Imminent deadline or event.
5. Completed child result ready for integration.
6. Current host-engaged task.
7. Significant unexpected state change.
8. Important active execution.
9. Recently referenced support task.
10. Meaningful monitor update.
11. Waiting, unchanged, resolved, or reference task.

The vendor or production team must not remotely manipulate personal attention ranking for advertising, engagement maximization, or commercial prioritization.

---

## 12. Friday voice and guest trust

Friday distinguishes:

- speech content;
- speaker changes;
- likely speaker identity;
- authenticated host authority;
- capability authorization.

Trust states include:

```text
SPEAKER RECOGNIZED
SPEAKER UNCERTAIN
HOST CONTEXT RECOGNIZED
HOST PRIVILEGE VERIFIED
GUEST SESSION
DELEGATED GUEST SESSION
MULTIPLE SPEAKERS
```

Voice may personalize interaction.

Voice alone must not authorize high-consequence actions.

### 12.1 Guest behavior

When the current speaker is unknown or unauthorized:

- private Mission Graph titles are hidden;
- task previews are hidden or blurred;
- private memory is unavailable;
- private files are denied;
- external messages and commitments are blocked;
- purchases are blocked;
- persistent system changes are blocked;
- all guest-triggered actions are locally logged.

### 12.2 Scoped guest grant

```yaml
guest:
  identity: current_speaker
  capabilities:
    - general_questions
    - media_control
    - timers
  expires: 60_minutes
  private_memory: denied
  private_files: denied
  messaging: denied
  purchases: denied
  system_changes: denied
  delegable: false
```

Guest authority is:

- visible;
- scoped;
- expiring or session-bounded;
- immediately revocable;
- non-delegable;
- unable to expand itself.

---

## 13. Development model before the dedicated Friday team

Before a qualified standalone Friday team exists, development follows one coordinated product-family program.

### 13.1 A.L.I.C.E. track

Responsible for:

- frontier experiments;
- owner-specific validation;
- advanced Mission Graph behavior;
- private research;
- identifying generalizable capability;
- generating evaluated candidates.

### 13.2 Personal Cognitive Kernel track

Responsible for:

- neutral schemas;
- shared state machines;
- storage contracts;
- routing;
- traceback;
- attention;
- workspace projections;
- generic tests;
- multi-host isolation.

### 13.3 Friday track

Responsible for:

- consumer shell;
- host enrollment;
- product UX;
- installer;
- product defaults;
- hardware compatibility;
- accessibility;
- support and diagnostic design;
- consumer evaluation.

### 13.4 Working-parity process

For a generalizable Cognitive Workspace capability:

1. Define the capability jointly.
2. Place host-neutral contracts in the Personal Cognitive Kernel.
3. Test with at least:
   - one synthetic A.L.I.C.E.-style host;
   - one synthetic Friday host;
   - two simultaneously isolated Friday hosts.
4. Integrate into A.L.I.C.E.
5. Integrate into Friday.
6. Evaluate product-specific behavior independently.
7. Record status in the parity ledger.
8. Release each product only when its own gates pass.

A.L.I.C.E. may run an experimental capability first.

Friday productization planning begins immediately rather than waiting several phases.

---

## 14. Development model after the dedicated Friday team

A qualified Friday team may independently:

- set its own sprint schedule;
- perform product research;
- propose new features;
- develop Friday-only integrations;
- design alternative product experiences;
- improve accessibility;
- support new hardware;
- create experimental branches;
- run internal tests;
- maintain packaging;
- fix bugs;
- propose shared-kernel changes;
- prepare model packs;
- conduct separately approved user research.

The team does not need prior permission to think, research, prototype, or prepare candidates.

However:

> The Friday team cannot place a new feature or changed behavior into production without both A.L.I.C.E.'s exact-artifact audit approval and Rayan's explicit production approval.

The team may go its own way in research and implementation.

It may not bypass product-family production governance.

---

## 15. Permanent dual-approval production rule

Every Friday production feature requires two independent authorization artifacts.

### 15.1 Approval key 1 — A.L.I.C.E. Audit Attestation

A.L.I.C.E. audits the exact candidate.

The attestation is bound to:

- source commit;
- kernel version;
- dependency lock;
- build artifact hashes;
- model-pack versions;
- migration files;
- policy versions;
- evaluation results;
- deployment manifest;
- rollback manifest.

A.L.I.C.E. does not merely state that the feature "looks good."

It produces a structured, reproducible audit record.

### 15.2 Approval key 2 — Rayan Production Approval

Rayan reviews:

- feature purpose;
- product direction;
- user impact;
- autonomy implications;
- privacy implications;
- security implications;
- A.L.I.C.E.'s audit;
- remaining risks;
- rollout scope;
- rollback plan.

Rayan may:

- approve;
- approve with conditions;
- restrict the channel;
- restrict the host population;
- require additional evidence;
- reject;
- return for revision;
- defer.

### 15.3 Veto and revision

Either A.L.I.C.E. or Rayan may:

- block production;
- require revision;
- reduce scope;
- require additional audit;
- revoke an earlier approval before release.

Only a matching pair of approvals authorizes production promotion.

---

## 16. What counts as a production feature

Dual approval is required for any candidate that changes production behavior.

### 16.1 User-facing capability

- new interface function;
- new Mission Graph behavior;
- new voice behavior;
- new guest capability;
- new tool;
- new connector;
- new automation;
- new model;
- new agent behavior.

### 16.2 Authority or autonomy

- broader permissions;
- fewer confirmations;
- new background behavior;
- new external action;
- new self-modification ability;
- new computer-control ability;
- new authority profile.

### 16.3 Data and learning

- new stored data category;
- changed retention;
- changed deletion;
- new training pipeline;
- new model adaptation;
- new telemetry;
- new cloud processing;
- new cross-device synchronization behavior.

### 16.4 Architecture

- schema migration;
- kernel contract change;
- identity-format change;
- encryption change;
- key-management change;
- storage-layout change;
- permission-policy change.

### 16.5 Distribution

- signed public release;
- stable-channel update;
- production feature flag;
- remote activation;
- model-pack promotion;
- installer change with security consequence;
- mandatory dependency;
- service-side behavior affecting local hosts.

### 16.6 Commercial behavior

- account requirement;
- subscription enforcement;
- analytics;
- commercially influenced recommendations;
- marketplace capability;
- third-party integration with data access.

A purely non-behavioral documentation or visual correction may use a reduced review profile, but still requires traceable release authorization.

---

## 17. Actions allowed without production approval

The Friday team may independently:

```text
create branches
write code
build prototypes
conduct design research
run simulations
run unit tests
perform threat modeling
create migration candidates
build local developer artifacts
run isolated synthetic-host tests
run internal benchmarks
prepare documentation
submit proposals
open pull requests
create candidate model packs
```

These candidates must remain in:

- development;
- laboratory;
- sandbox;
- simulation;
- non-production test environments;
- isolated internal preview channels.

They may not become production behavior.

---

## 18. Emergency-response exception

The Friday team must be able to protect users during an active incident.

### 18.1 Allowed without prior dual approval

The team may:

- disable a vulnerable capability;
- revoke a compromised signing key;
- stop a rollout;
- roll back to the last approved release;
- isolate an affected service;
- block a malicious destination;
- quarantine a compromised model pack;
- disable an unsafe feature flag;
- issue a warning;
- preserve evidence;
- activate a previously approved recovery procedure.

### 18.2 Not allowed under the emergency exception

The team may not:

- introduce a new capability;
- broaden permissions;
- deploy an unapproved replacement feature;
- begin new data collection;
- change constitutional authority;
- weaken privacy controls;
- install a new model with changed production behavior;
- use emergency status to bypass normal governance.

A replacement fix that creates new behavior requires A.L.I.C.E. audit and Rayan approval.

Every emergency action creates an immutable incident record.

---

## 19. A.L.I.C.E. production audit standard

A.L.I.C.E.'s audit contains at least the following.

### 19.1 Scope verification

- What changes?
- Which products, channels, and hosts are affected?
- Is the change new, modified, migrated, removed, or disabled?
- Does the actual diff match the declared scope?

### 19.2 Architecture review

- Does the code belong in Friday, the kernel, or product-specific integration?
- Does it create forbidden A.L.I.C.E.–Friday coupling?
- Is it host-neutral where required?
- Does it preserve Mission Graph semantics?
- Does it preserve model independence?
- Does it create untracked parity debt?

### 19.3 Privacy review

- What data enters?
- Where is it processed?
- What leaves the device?
- What is retained?
- What is deleted?
- Can the vendor read it?
- Can separate hosts become linked?
- Does telemetry expose content or identity?

### 19.4 Security review

- threat model;
- dependency risk;
- supply-chain risk;
- permission boundaries;
- signing behavior;
- update behavior;
- rollback;
- abuse cases;
- guest exposure;
- speaker spoofing where relevant.

### 19.5 Authority review

- required autonomy class;
- host-granted permission;
- background behavior;
- external consequences;
- irreversible actions;
- confirmation policy;
- emergency stop behavior.

### 19.6 Learning review

- training input;
- labels;
- lineage;
- candidate promotion;
- deletion propagation;
- poisoning resistance;
- evaluation;
- model rollback;
- historical retention.

### 19.7 UI and attention review

- task-state truthfulness;
- explainable priority;
- host override;
- no-empty-slot behavior;
- attention-manipulation risk;
- guest privacy;
- accessibility;
- reduced motion;
- visibility versus execution semantics.

### 19.8 Evaluation review

- benchmark coverage;
- regression suite;
- adversarial tests;
- exact-commit results;
- hardware coverage;
- failure behavior;
- migration tests;
- rollback tests.

### 19.9 Product-family parity review

- whether the capability belongs in the kernel;
- whether it affects A.L.I.C.E.;
- whether divergence is intentional;
- whether parity debt is documented;
- upstream or downstream disposition.

### 19.10 Final A.L.I.C.E. determination

```text
APPROVED_FOR_RAYAN_REVIEW
APPROVED_WITH_CONDITIONS
RETURN_FOR_REVISION
BLOCKED_SECURITY
BLOCKED_PRIVACY
BLOCKED_AUTHORITY
BLOCKED_EVALUATION
BLOCKED_ARCHITECTURE
INSUFFICIENT_EVIDENCE
```

A.L.I.C.E. approval makes the candidate eligible for Rayan's decision.

It does not authorize production by itself.

---

## 20. Rayan production approval

Recommended approval record:

```yaml
approval_id:
feature_id:
candidate_version:
commit_sha:
artifact_hashes:
alice_attestation_id:
decision:
conditions:
rollout_scope:
approved_channels:
expiration:
approved_at:
owner_signature:
```

Possible decisions:

```text
APPROVED
APPROVED_CANARY_ONLY
APPROVED_CLOSED_ALPHA
APPROVED_WITH_CONDITIONS
REJECTED
REVISION_REQUIRED
DEFERRED
```

An approval is bound to exact artifacts.

A general statement such as "I like this feature" is not sufficient production authorization.

---

## 21. Cryptographic release enforcement

The dual-approval rule should be enforced by tooling.

### 21.1 Release manifest

```yaml
release_id:
product_id:
version:
commit_sha:
kernel_version:
schema_versions:
artifact_hashes:
evaluation_bundle_hash:
alice_audit_attestation:
rayan_approval:
migration_manifest:
rollback_manifest:
signing_identity:
release_channel:
```

### 21.2 CI/CD production gate

Production signing is blocked unless:

```text
A.L.I.C.E. audit status is eligible
AND
Rayan approval matches the exact candidate
AND
all required checks pass
AND
artifact hashes match
AND
approval has not expired or been revoked
```

### 21.3 Separation of authority

The Friday team must not control:

- A.L.I.C.E.'s audit-signing identity;
- Rayan's approval key;
- every production signing key;
- the audit-history deletion mechanism.

No one employee or service should be able to:

1. modify the feature;
2. approve it;
3. sign it;
4. erase evidence.

---

## 22. Independence and conflict controls

A.L.I.C.E. may help create a candidate and later audit it.

To reduce self-review risk, the audit must depend on:

- deterministic tests;
- separate verifier processes;
- reproducible builds;
- static analysis;
- adversarial evaluation;
- independent fixtures;
- exact artifacts;
- human-readable evidence.

High-risk features should normally require additional review.

Examples:

- encryption;
- identity and authentication;
- payments;
- health or legal actions;
- children or household accounts;
- unrestricted computer control;
- autonomous external communication;
- self-modification;
- model-weight promotion;
- vendor services receiving personal data;
- operating-system privilege;
- robotics and physical-world action.

A.L.I.C.E. remains a mandatory auditor, but not necessarily the only auditor.

---

## 23. Feature lifecycle

Every Friday production capability follows:

```text
IDEA
↓
PRODUCT PROPOSAL
↓
ARCHITECTURE CLASSIFICATION
↓
TEAM CANDIDATE
↓
SANDBOX IMPLEMENTATION
↓
TEST AND THREAT MODEL
↓
SHARED-KERNEL DISPOSITION
↓
A.L.I.C.E. AUDIT
↓
RAYAN REVIEW
↓
SIGNED RELEASE CANDIDATE
↓
CANARY OR LIMITED CHANNEL
↓
PRODUCTION
↓
POST-RELEASE EVALUATION
```

Possible side paths:

```text
RETURN FOR REVISION
REJECT
QUARANTINE
ROLL BACK
REMOVE
MIGRATE
PROMOTE TO SHARED KERNEL
KEEP FRIDAY-SPECIFIC
```

---

## 24. Capability-parity ledger

Add a dedicated Cognitive Workspace section to the product-family parity ledger.

Each capability records:

```yaml
capability_id:
title:
alice_status:
kernel_status:
friday_status:
shared_ui_status:
first_alice_version:
first_friday_version:
generalization_status:
privacy_review:
host_isolation_review:
alice_audit_status:
rayan_approval_status:
reason_for_lag:
responsible_team:
target_release:
evidence_bundle:
```

Initial capability identifiers:

```text
mission_graph.v1
semantic_router.v1
result_capsule.v1
traceback_engine.v1
attention_policy.v1
workspace_projection.v1
adaptive_compositor.v1
host_window_override.v1
speaker_context.v1
guest_session.v1
guest_grant.v1
legacy_chat_compatibility.v1
```

---

## 25. Phase-by-phase Friday plan

### Phase 5 / F1 — Shared foundation

Deliver:

- Personal Cognitive Kernel repository or independently versioned package boundary;
- product and host identity contracts;
- Mission Graph schemas;
- Mission Graph events;
- Result Capsule schema;
- Traceback state machine;
- attention-decision event schema;
- workspace-projection schema;
- speaker-context schema;
- guest-grant schema;
- synthetic-host tests;
- dual-approval governance schemas;
- parity-ledger expansion.

Do not build the complete consumer UI yet.

### Phase 6 / F2 — Cognitive Workspace

Deliver:

- shared Cognitive Workspace components;
- Friday desktop-shell prototype;
- host control plane;
- Mission Canvas;
- adaptive windows;
- Mission Graph inspector;
- Result Capsule viewer;
- attention explanation;
- guest-mode visual state;
- host workspace commands;
- product-neutral model and capability controls.

A.L.I.C.E. and a synthetic Friday host must use the same core workspace contracts.

### Phase 6.5 / F3 — Independent Product Readiness Gate

Prove:

- Friday repository is independent;
- kernel is independently versioned;
- Friday builds without A.L.I.C.E.;
- no A.L.I.C.E. private state exists in Friday;
- two synthetic Friday hosts remain isolated;
- release manifests support dual approval;
- production signing is gated;
- migration and rollback work;
- shared UI contracts pass;
- parity ledger is active.

### Phase 7 / F4 — Signed Windows alpha

Deliver:

- signed Windows application;
- local host enrollment;
- host-selected assistant name;
- local keys;
- local model runtime;
- speech-to-text;
- speaker diarization;
- guest mode;
- local ingestion;
- hardware benchmarking;
- Mission Graph desktop experience.

### Phase 8 / F5 — Selective learning closed alpha

Deliver:

- autonomous selective memory;
- Mission Graph learning signals;
- learned routing candidates;
- learned retention;
- Result Capsule-derived learning;
- Identity Capsule;
- deletion and correction;
- credible closed alpha.

### Phase 9 / F6 — Personal intelligence beta

Deliver:

- host-specific attention model;
- host model;
- self-model;
- learned interaction preferences;
- stable voice;
- uncertainty;
- Personalization Inspector;
- learned layout preferences.

### Phase 10 / F7 — Proactive missions

Deliver:

- background mission execution;
- hierarchical goals;
- autonomous branch creation;
- monitoring;
- proactive traceback;
- resource-aware attention;
- mission continuation across sessions.

### Phase 11 / F8 — Computer use and skill system

Deliver:

- desktop and terminal control;
- application-linked mission windows;
- generated skills;
- computer-use permissions;
- evaluated local autonomy;
- signed skill packages.

### Phase 13 / F10 — Host-specific model adaptation

Deliver:

- locally learned routers;
- attention rankers;
- task-routing classifiers;
- interaction adapters;
- deletion-aware retraining;
- host-specific model components.

### Phase 14 / F11 — Operating environment

Deliver:

- persistent local service;
- boot-time activation;
- cross-device Mission Graph;
- voice-first shell;
- AI-native desktop;
- application and file organization around missions;
- optional dedicated-system research.

---

## 26. Repository documents to amend

### `docs/FRIDAY_PRODUCT_VISION.md`

Add:

- same Cognitive Workspace direction;
- shared UI capability strategy;
- dual-approval production governance;
- production-team freedom to experiment without unilateral release authority.

### `docs/FRIDAY_ROADMAP.md`

Change:

- F1 to include Mission Graph and governance contracts;
- F2 to include the full Cognitive Workspace;
- F3 from repository creation to independent product readiness;
- team handoff to preserve dual approval.

### `docs/ALICE_FRIDAY_SEPARATION_PLAN.md`

Remove the interim Friday-incubator source layout.

Replace it with:

- Friday repository before the first Friday product commit;
- Personal Cognitive Kernel extraction in Phase 5;
- no Friday product source in A.L.I.C.E.;
- Phase 6.5 independent readiness gate.

### `docs/FRIDAY_ARCHITECTURE.md`

Add:

- Mission Graph plane;
- Cognitive Workspace plane;
- attention plane;
- presence and guest-trust plane;
- production-governance plane;
- A.L.I.C.E. audit attestation;
- Rayan approval;
- exact-artifact release gates.

### `docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md`

Add:

- initial working parity for UI capability;
- dedicated Cognitive Workspace parity lane;
- dual approval;
- independent-team development boundary.

### `docs/ROADMAP_GOVERNANCE.md`

Add:

- product proposals and sandbox implementation do not require prior production approval;
- production promotion requires dual approval;
- emergency rollback exception;
- only Rayan may amend the permanent rule.

### `docs/PERMISSION_MODEL.md`

Add:

```text
friday.feature.propose
friday.feature.test
friday.feature.audit
friday.feature.owner_approve
friday.release.sign
friday.release.promote
friday.release.rollback
friday.release.emergency_disable
```

### `docs/ARCHITECTURE.md`

Add:

- Product Governance Plane;
- A.L.I.C.E. Audit Attestation;
- Rayan Production Approval;
- exact-artifact promotion gates.

---

## 27. New Friday governance documents and policies

Recommended documents:

```text
docs/FRIDAY_COGNITIVE_WORKSPACE_AND_PRODUCTION_GOVERNANCE_PLAN.md
docs/FRIDAY_PRODUCTION_GOVERNANCE.md
docs/FRIDAY_DUAL_APPROVAL_STANDARD.md
docs/FRIDAY_FEATURE_LIFECYCLE.md
docs/FRIDAY_RELEASE_ATTESTATION_STANDARD.md
docs/FRIDAY_EMERGENCY_RESPONSE_POLICY.md
docs/FRIDAY_TEAM_INDEPENDENCE_AND_LIMITS.md
docs/FRIDAY_SHARED_UI_PARITY_STANDARD.md
docs/decisions/ADR-XXX-friday-independent-repository-and-dual-approval.md
```

Recommended machine-readable policies:

```text
policies/friday_feature_governance.json
policies/friday_dual_approval.json
policies/friday_release_channels.json
policies/friday_emergency_response.json
policies/friday_ui_parity.json
policies/friday_team_authority.json
policies/release_attestation_schema.json
```

Update:

```text
policies/product_lines.json
policies/capability_parity_ledger.json
policies/permissions.yaml
policies/capability_profiles.json
policies/phase_scope_registry.json
policies/authority_kernel_policy.json
```

Recommended machine rule:

```yaml
production_promotion:
  alice_audit_required: true
  rayan_approval_required: true
  exact_commit_binding: true
  exact_artifact_binding: true
  bypass_allowed: false

emergency_response:
  rollback_allowed: true
  disable_allowed: true
  containment_allowed: true
  new_capability_allowed: false
```

---

## 28. Required validators

Add validators such as:

```text
validate_friday_repo_separation.py
validate_friday_ui_parity.py
validate_friday_release_attestation.py
validate_dual_production_approval.py
validate_alice_audit_binding.py
validate_rayan_approval_binding.py
validate_emergency_change_scope.py
validate_product_family_data_isolation.py
validate_release_artifact_hashes.py
validate_parity_ledger_disposition.py
```

A production build fails when:

- A.L.I.C.E. attestation is missing;
- Rayan approval is missing;
- either approval points to another commit;
- artifact hashes differ;
- required evaluation is absent;
- a migration is unapproved;
- an emergency change introduces new behavior;
- Friday contains A.L.I.C.E. private data;
- product and kernel manifests disagree;
- parity disposition is missing.

---

## 29. Required tests

### 29.1 Repository separation

- Friday builds without A.L.I.C.E.
- A.L.I.C.E. builds without Friday.
- Kernel contains no private product data.
- No forbidden cross-repository imports.
- No product-private fixture is placed in shared evaluation data.

### 29.2 Host isolation

- host A cannot access host B Mission Graph;
- host A cannot access host B voice profile;
- guest grants remain host-local;
- cache, log, backup, and restore remain scoped;
- deletion does not cross host boundaries.

### 29.3 UI parity

- both products interpret node states identically;
- both produce valid Result Capsules;
- traceback semantics match;
- attention explanations conform to one contract;
- no-empty-slot layout holds;
- guest hiding follows the same trust contract.

### 29.4 Governance

- team candidate cannot reach production alone;
- A.L.I.C.E. approval alone cannot release;
- Rayan approval alone cannot release;
- mismatched approvals fail;
- revoked approval fails;
- expired approval fails;
- emergency rollback succeeds;
- emergency feature addition fails.

### 29.5 Release integrity

- reproducible build;
- signed manifest;
- exact dependency lock;
- exact artifact hashes;
- migration test;
- rollback test;
- canary behavior;
- post-release evidence linkage.

---

## 30. Independent-team qualification gate

A dedicated Friday production team is not considered independent merely because staff have been hired.

The team must prove competence in:

- repository maintenance;
- kernel compatibility;
- Windows packaging;
- hardware support;
- migration;
- rollback;
- incident response;
- security patches;
- privacy guarantees;
- test infrastructure;
- accessibility;
- release documentation;
- customer support;
- parity tracking.

Qualification transfers daily product responsibility.

It does not remove A.L.I.C.E. audit or Rayan approval.

---

## 31. Governance hierarchy

```text
MK Rayan
    Final constitutional and Friday production authority
            │
            ├── A.L.I.C.E.
            │     Mandatory technical, safety, privacy,
            │     architecture, and evaluation auditor
            │
            ├── Personal Cognitive Kernel maintainers
            │     Shared contracts, migrations, and evaluators
            │
            └── Friday production team
                  Independent product development,
                  maintenance, support, and proposals
```

The Friday team may disagree.

It may present evidence, appeal, revise, and resubmit.

It may propose changes to governance.

It may not silently bypass the release gate.

Only Rayan may ratify a change to the permanent approval rule.

---

## 32. Immediate implementation sequence

### PR 1 — Product-family governance amendment

In A.L.I.C.E. planning documents, add:

- Friday same-UI direction;
- separate-repository correction;
- dual-approval rule;
- emergency exception;
- revised Phase 6.5 meaning;
- capability-parity expansion;
- no production code.

### PR 2 — Machine-readable governance

Add:

- approval schemas;
- release manifest;
- product-line policy changes;
- parity-ledger fields;
- validators;
- policy tests.

### PR 3 — Personal Cognitive Kernel foundation

Add:

- Mission Graph contracts;
- Result Capsules;
- traceback contracts;
- product and host scope;
- synthetic-host fixtures;
- shared UI projection schemas.

### Friday repository initialization

Before the first Friday product source commit:

- create the independent repository;
- add governance pointer;
- add product manifest;
- add privacy boundary;
- pin an approved kernel version;
- configure dual-approval release protection;
- prohibit production signing until release gates exist.

No Friday product implementation is committed to A.L.I.C.E.

---

## 33. Source-alignment notes

This plan was aligned against the public `main` versions of:

- `README.md`;
- `docs/ROADMAP.md`;
- `docs/FRIDAY_PRODUCT_VISION.md`;
- `docs/FRIDAY_ROADMAP.md`;
- `docs/FRIDAY_ARCHITECTURE.md`;
- `docs/ALICE_FRIDAY_SEPARATION_PLAN.md`;
- `docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md`;
- `docs/PERMISSION_MODEL.md`;
- `docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md`.

Important amendment:

- the public separation documents currently place the formal repository split at Phase 6.5 and describe an interim Friday incubator inside A.L.I.C.E.;
- Rayan's newer accepted decision requires a separate Friday repository from the first Friday implementation commit;
- Phase 6.5 is therefore redefined here as an independent-product readiness and release gate;
- this change must be ratified through the normal ADR, policy-versioning, validator, migration, and exact-commit audit process.

---

## 34. Final governance statement

> Friday's production team is free to invent, investigate, implement, and argue for its own product direction. It is not free to place unreviewed capability into production. Production authority remains a two-key system: A.L.I.C.E. must attest that the exact candidate passed the required audit, and Rayan must explicitly authorize the exact release.
