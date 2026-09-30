# Frontier Watch Recovery Intake — 2026-09-30

Status: research intake only; no production/main semantic change  
Branch: `research/frontier-watch`  
Purpose: recover qualifying findings surfaced after the 2026-09-27 baseline that were reported in watch runs but did not land because prior GitHub mutations failed.

## Governing baseline

This intake is subordinate to `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md` and `FRONTIER_WATCH_EXECUTION_GUARDRAIL_2026-09-27.md`.

It does **not** reopen selected physical placements. KurrentDB, NATS JetStream, XTDB v2, encrypted CAS, LadybugDB/NebulaGraph, Qdrant, Temporal, process-local L1 + Valkey, content-addressed lineage, and PyTorch + Accelerate remain selected defaults. Findings below change cognitive/model contracts only where stated.

F4–F11 remain internal qualification milestones. No finding here creates a partial consumer cognitive tier. The first consumer release predicate remains the full personal cognitive foundation after F11.

## 1. AutoMem — memory management as a learnable cognitive skill

Primary source: Wu et al., *AutoMem: Automated Learning of Memory as a Cognitive Skill*, arXiv:2607.01224  
Paper: https://arxiv.org/abs/2607.01224  
Reference implementation: https://github.com/autoLearnMem/AutoMem

Classification: **direct current capability improvement, future learned-policy implementation**

### Delta

A.L.I.C.E. already separates evidence, claims, episodic/procedural memory, working state, recollection, lifecycle curation, and governed evolution. AutoMem's useful delta is that **memory-control behavior itself can be trained independently of task-action behavior**: what to consult, what to write, how to organize memory, and when to use memory operations.

### A.L.I.C.E./Fable implication

Keep authority deterministic/governed, but permit a learned Memory Resource Manager / Recollection policy to propose operations such as consult/search, activate, write-proposal, update-proposal, and no-op. Learned memory control must never directly assert Claim truth or bypass provenance, identity separation, correction/deletion, or rollback.

### Evidence / limits / cost

The paper reports roughly 2–4x progression gains across Crafter, MiniHack, and NetHack while optimizing memory behavior without modifying task-action behavior. Evidence is promising but environment-heavy and not lifelong-personal-memory evidence. The meta-review/training loop adds trajectory-review and training cost.

### Action

Future-only. Preserve a learnable memory-control interface in MRM/Recollection contracts. Do not block current MFM/personality critical paths.

## 2. Proactive Memory Agent — retrieval is not behavioral activation

Primary source: Wu et al., *Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents*, arXiv:2607.08716  
Paper: https://arxiv.org/abs/2607.08716  
Reference implementation: https://github.com/yifannnwu/proactive-memory-agent

Classification: **direct current capability improvement**

### Delta

The paper isolates **behavioral state decay**: relevant state may still exist in storage or context yet fail to affect the next decision. A separate memory agent maintains structured memory and decides whether to inject a grounded reminder or remain silent. Selective intervention outperforms passive bank exposure, always-on injection, advisor-only guidance, and general retrieval.

### A.L.I.C.E./Fable implication

Make **retrieval** and **behavioral activation/intervention** separate operations. Cognitive Recollection may find relevant material; Cognitive Workspace should separately decide whether a bounded, provenance-grounded activation enters the current decision state. `no_intervention` is a first-class outcome.

Activation is transient context, never Claim Authority. Revoked/deleted material must be barred from activation and activation traces must remain lineage-auditable.

### Evidence / cost

Reported gains are +8.3 percentage points on Terminal-Bench 2.0 and +6.8 on tau2-Bench. Selective intervention reduces the case for indiscriminate context injection but adds a memory-policy inference path.

### Action

Carry the retrieval-vs-activation distinction into Recollection/Cognitive Workspace implementation and evaluate mission success, repeated-failure rate, stale/revoked activation, latency, and token overhead. No infrastructure benchmark.

## 3. Procedural Graphs — progress-localized connected procedural guidance

Primary source: Lu et al., *Procedural Graphs: Self-Evolving Execution Structures for LLM Agents*, arXiv:2609.09153  
Paper: https://arxiv.org/abs/2609.09153

Classification: **direct current capability improvement at design/contract level; future implementation item**

### Delta

A.L.I.C.E. already has Mission Graph, Cognitive Multi-Graph, procedural learning, executable skills, Experience/outcome lineage, Recollection, and governed promotion/rollback. The useful new piece is a **rebuildable procedural transition projection** where the active procedure/state is localized and connected neighborhood guidance is retrieved rather than treating procedural memories as independent top-k items.

The paper also evolves graph topology/attributes by contrasting failed and successful trajectories, admitting candidate edits only when held-out validation does not degrade.

### A.L.I.C.E./Fable implication

Procedural projections should be able to represent typed transitions, preconditions, expected effects, pitfalls/failure signatures, environment/tool/version scope, Experience/outcome provenance, validity/confidence, curator/model version, correction/deletion state, and rollback lineage.

Connected guidance remains a projection, not evidence or Claim Authority. Graph evolution must pass existing evaluation/promotion gates.

### Evidence / risks / cost

The paper reports gains across MultiChallenge, GDPval, and ALFWorld and shows localization can reduce guidance-token cost versus whole-graph guidance. It can still cost more than no graph. Self-evolution introduces evaluator dependence and potential topology drift.

### Action

When procedural learning reaches implementation, compare bounded connected-neighborhood guidance with flat top-k procedure retrieval; test localization failure, prerequisite preservation, stale tool/environment invalidation, deletion/revocation rebuild, and rollback. Do not reopen graph-engine selection.

## 4. Memory Is a Derivation — derivation integrity for memory formation/admission

Primary source: Liu and Zhao, *Memory Is a Derivation: The Distributed-Evidence Paradox in Long-Term Agents*, arXiv:2609.36130  
Paper: https://arxiv.org/abs/2609.36130

Classification: **direct current capability improvement; missing explicit invariant in proposal-to-Claim admission**

### Delta

A.L.I.C.E. already preserves Experience/evidence versus Claim Authority and requires provenance-bound memory proposals. The new requirement is to treat an admitted memory as a **derivation over a pre-write evidence horizon**, not merely a statement with valid individual citations.

Individually genuine citations can fail to license a composed relation: causality, temporal status, planned-vs-realized modality, subject/relationship binding, or another qualification may be introduced without support.

### A.L.I.C.E./Fable implication

Distinguish proposal citations from the evidence horizon searched by the admission gate. Support joint evidence sets. Preserve machine-inspectable material relations/qualifications. Prevent a projection derived from a source history from bootstrapping itself as independent authority. Keep `supported`, `contradicted`, and `unresolved/insufficient-evidence` distinct.

Bind admitted derivation bases to source IDs/generations so correction, revocation, deletion, and rollback can invalidate exact descendants.

### Evidence / limits / cost

The paper reports substantial citation insufficiency and difficult distributed-evidence composition failures across evaluated settings. It also shows that simply increasing verification obligations can sharply reduce valid retention, so this is **not** justification for a giant factuality checklist. Verification adds retrieval/inference cost and is model-dependent.

### Action

Before Stage-B MFM output is admitted to real Claim Authority, add focused contract cases for distributed-valid evidence, distributed-invalid composition, plan-vs-outcome, update-vs-contradiction, incomplete-citation recovery, search-limited unresolved, self-support rejection, correction/deletion invalidation, and verification-load control.

## Combined decision

These findings strengthen four distinct cognitive boundaries:

1. memory operation selection can become a learned skill;
2. retrieval and behavioral activation are different;
3. procedural recollection benefits from progress-localized connected structure;
4. persistent memory admission must preserve derivation integrity across distributed evidence.

None invalidates the selected Fable v1 architecture. None justifies a storage/backend tournament. None removes a logical cognitive plane. Future-only work must not block the current critical path.

## Recovery note

This file intentionally consolidates the missed Sep 28–30 watch findings. Earlier attempts to write separate intake notes failed before a GitHub commit was created. This recovery intake is the canonical branch record for those missed reports.
