# Friday Technical Architecture

## 1. System identity

Friday is a distributable product built on the Personal Cognitive Kernel. Each installation creates a cryptographically and behaviorally distinct `HostInstance`.

```text
Friday distribution
  + common code and model packs
  + locally generated host keys
  + locally ingested host data
  + locally learned memory/beliefs/skills/adapters
  = unique Friday host instance
```

## 2. Architectural planes

### Distribution plane

Contains signed installers, update manifests, common model manifests, schema migrations, and public evaluation assets. It must not contain host personal state.

### Local trust root

Creates and protects the host master key, instance identifier, recovery material, and signed local audit roots. On Windows, OS-bound key protection may wrap the application master key; bulk data remains encrypted with application-managed keys.

### Local data plane

Stores encrypted originals or references, extracted content, embeddings, memory, beliefs, experience, training assets, skills, adapters, and audit history.

### Local learning plane

Runs curation, consolidation, preference learning, local evaluation, adapter training, champion/challenger comparison, and deletion-aware rebuilds.

### Cognitive plane

Runs model routing, context construction, user/world/self models, planning, judgment, and conversation.

### Capability plane

Runs signed skills, local tools, OS actions, APIs, and connectors through declared manifests and host authority.

### Optional external-compute plane

Cloud or remote models are optional. The host sees exactly what data is leaving, may redact or transform it locally, and may disable the plane entirely. The vendor cannot silently convert local-first operation into cloud-required operation.

## 3. Storage design

Every record carries:

- host instance identifier;
- product identifier;
- source and provenance;
- sensitivity classification;
- encryption domain;
- content digest;
- retention class;
- active storage tier;
- deletion lineage;
- model/training influence identifiers;
- schema version.

Friday uses the same aggressive-capture/selective-retention architecture as A.L.I.C.E.: a compact permanent ledger, a policy-bounded raw buffer, utility-weighted durable stores, representative replay manifests, and encrypted hot/warm/cold/quarantine tiers. Full payloads are content-addressed. Deduplication is limited to the same host and encryption domain so equality information is not leaked across users.

The lifecycle manager predicts storage needs before ingestion or training, preserves free-space reserves, archives inactive data, verifies backups and restoration, and pauses low-priority capture or training before disk exhaustion. It never silently deletes authoritative evidence, active rollback state, owner-held records, or artifacts still referenced by a memory, evaluation, model, or deletion investigation.

No shared cache, vector collection, telemetry batch, blob namespace, backup set, or model-training directory may mix two host instances.

## 4. Host model stack

### General model layer

Replaceable open or licensed model selected for hardware and task.

### Contextual layer

Current conversation, retrieved memory, files, and tool state.

### Structured personal layer

Facts, events, preferences, beliefs, causal links, goals, relationships, and confidence.

### Procedural layer

Executable skills, workflows, tool preferences, and learned strategies.

### Parametric personal layer

Local rankers, routers, classifiers, embeddings, LoRA/adapters, or future host-specific weights.

### Identity layer

Voice, constitutional configuration, interaction style, owner relationship, and history of model replacements.

## 5. Network non-access architecture

- Core use requires no vendor login.
- Outbound network access is denied by product policy until a feature or host mission enables it.
- Every network-capable component declares destinations and data categories.
- The UI exposes a live and historical egress ledger.
- Crash reporting is opt-in and strips content.
- Update checks exchange only product/version/platform information.
- Host content, embeddings, memory, adapters, and keys are excluded from vendor telemetry.
- Optional sync uses end-to-end encryption with host-controlled keys.

## 6. Runtime strategy

Use a runtime abstraction rather than one mandatory engine.

Initial candidates:

- llama.cpp-compatible runtime for quantized language models across common CPU/GPU targets;
- ONNX Runtime for hardware-specific inference and small on-device training;
- optional Ollama or LM Studio adapters for advanced users;
- product-managed runtime for non-technical users.

The installer benchmarks:

- architecture and instruction set;
- RAM, available storage, storage tier performance, free-space reserve, and expected annual growth;
- GPU/NPU providers;
- thermal and power profile;
- expected context and training workload.

It then recommends a model pack and can switch models without replacing the host identity.

## 7. Desktop application strategy

The first Windows product should bundle all required components and present one installer. A practical migration path is:

1. Tauri or equivalent native shell;
2. web UI shared with A.L.I.C.E. where useful;
3. Rust process for security-sensitive local host functions;
4. Python sidecar for existing AI pipelines;
5. gradual movement of stable high-performance components into Rust/C++ libraries;
6. signed updater with rollback channels;
7. eventual Windows Store/MSIX option.

## 8. Identity Capsule

The export format contains encrypted logical sections:

```text
manifest
keys or wrapped recovery material
memory
beliefs
user/world/self model state
skills
adapters
model compatibility metadata
evaluation history
deletion and provenance ledger
```

The host may export a full capsule, a redacted capsule, or a model-only capsule. Import runs compatibility checks and never silently merges conflicting identities.

## 9. Local training

Training jobs require:

- curated input lineage;
- a declared objective;
- a held-out local evaluation set;
- resource, thermal, and storage budgets;
- a representative replay manifest;
- a current champion;
- rollback and deletion strategy.

Friday may train automatically during idle windows when the host enables continuous learning. A new adapter is promoted only when personal metrics improve without unacceptable regressions.

## 10. Vendor update boundary

Updates may modify software, common models, and evaluators. They may not silently overwrite host memory, personal adapters, constitutional choices, or learned identity. Migrations are previewable, versioned, and reversible when materially changing personal state.

## Host-selected identity layer

The runtime separates `product_codename`, `product_brand`, and `assistant_name`. The host-selected assistant name is encrypted personal state inside the Identity Capsule. It must not control storage schemas, encryption domains, API namespaces, or update channels.

## Full capability parity

The consumer architecture is not a reduced fork. All generalizable cognitive, learning, coding, scientific, proactive, multimodal, self-improvement, operating-environment, and frontier capabilities flow through the shared-kernel parity process. Hardware profiles and release maturity determine what is active on a particular installation.

## Upstream/downstream flow

```text
A.L.I.C.E. frontier experiment
        ↓
General capability contract + evaluator
        ↓
Personal Cognitive Kernel / capability package
        ↓
Consumer productization and hardware adaptation
        ↓
Host-named personal AI instances
```
