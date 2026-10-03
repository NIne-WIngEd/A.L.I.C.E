# Frontier Watch Intake — Verified Mission State and Outcome-Gated Learning — 2026-10-03

Status: research/contract-impact note only; **no production/main semantic change authorized here**  
Branch: `research/frontier-watch`  
Governing baseline: `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md` and `FRONTIER_WATCH_EXECUTION_GUARDRAIL_2026-09-27.md`

## Decision summary

A newly surfaced cluster of primary work exposes one concrete gap in the current Phase-5 Mission Graph / Result Capsule contract:

> **An executor-produced result must not be sufficient authority to advance canonical mission progress to completed/succeeded. Completion needs an independently grounded outcome-verification receipt when the postcondition is externally checkable.**

The same boundary should protect procedural/environment learning:

> **Execution experience may be retained as evidence regardless of outcome, but it must not be promoted as a verified-success procedure or causal rule merely because the actor/executor says the task succeeded.**

This is a **direct current capability improvement / missing explicit invariant**, not architecture-invalidating evidence. It does not reopen any selected physical implementation and does not block the current MFM v1.6 source/teacher, authorization, corpus-admission, CPU-processing, or provenance critical path.

## Current A.L.I.C.E. state checked before judgment

Current inspected heads:

- A.L.I.C.E. `main@8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b`
- Fable `main@1805a01c73575378246cac8cbbb72cd891e8528e`
- frontier watch before this intake: `research/frontier-watch@a0e6d805a9e79d23d125f98dc021507e83eeac8c`
- active MFM v1.6, FBM, Gemma-source, Graphify/context, and continuity branches were inspected in this watch run.

The current architecture already has the right large-scale pieces: Mission Graph, Result Capsule, traceback, outcome/evidence lineage, Cognitive Workspace, failure localization, Experience/Event history, Claim Authority separation, deletion lineage, rollback, and environment/procedural learning.

The gap is narrower and visible in the current contract itself:

- `ResultCapsule` has `status`, output references, evidence references, source-event references, provenance, and deletion lineage.
- A `succeeded` capsule currently requires an output reference, but **does not require independently verified postcondition evidence**.
- `MissionNode` permits `completed/succeeded` state and checks lifecycle compatibility/transition legality, but the successor contract does **not** bind that transition to a verification receipt or distinguish actor/executor assertion from outcome-verification authority.
- Current Phase-5 tests prove tamper evidence, graph/scope integrity, output lineage, and valid lifecycle transitions, but do not test false self-completion.

This is not a reason to mutate main immediately. The finding should first become a focused contract-impact item and then be implemented when the Mission Graph/runtime execution lane reaches the relevant semantic change.

---

## 1. StructAgent — verifier-owned progress state

Primary source: Wenyi Wu et al., *StructAgent: Harness Long-horizon Digital Agents with Unified Causal Structure*, arXiv:2607.11388, submitted 2026-07-13.  
Paper: https://arxiv.org/abs/2607.11388  
Code: https://github.com/WenyiWU0111/StructAgent

Classification: **direct current capability improvement / contract invariant**

### Method

StructAgent maintains a compact task state with three typed surfaces:

- current requirements;
- useful durable values;
- verified evidence.

The planner and actor can propose progress. They cannot commit it. Each requirement remains Pending until a verifier-backed decision moves it to Verified; contradictory evidence can invalidate prior progress. The verifier uses structured/environment evidence where possible. A separate completion audit gates final DONE. Recovery operates from verified state rather than from the actor's narrative of what it believes happened.

This is materially different from merely adding a critic. It makes **progress authority** explicit.

### Evidence

Under matched OSWorld-Verified runner settings:

- Qwen3.5-9B: 27.0% -> 46.9%;
- Qwen3.5-27B: 31.6% -> 62.2%;
- MiniMax-M3: 75.2% -> 78.9%.

The same state discipline transfers to web tasks and a Minecraft instantiation where inventory evidence replaces desktop probes. The paper's verifier analysis also shows the verifier is not an oracle: structured verification improves completion checking, but residual verifier/planner/actor failures remain.

### What A.L.I.C.E. already does

A.L.I.C.E. already separates evidence from authority, stores outcome lineage, has Result Capsules, Mission Graph state, traceback, failure localization, and real-world outcome-verification as a design goal.

### Genuinely new implication

Make the Mission Graph authority rule explicit:

- executor/actor report = **proposed outcome evidence**;
- postcondition probe/auditor = **verification evidence**;
- canonical `completed/succeeded` mission state = permitted only through a verification-bearing transition for externally checkable outcomes;
- verification failure may keep state pending, mark it blocked, or invalidate previously committed progress;
- a later contradictory environment observation can revoke a previously verified milestone rather than leaving stale completion state.

A model-based verifier remains fallible, so verification provenance and evidence class must remain visible. Strong deterministic/source-native checks should outrank weak self-assessment where available.

### Cost / risk

Extra probes and verifier inference increase latency/tokens. Some environments expose weak or ambiguous postconditions. A verifier can be wrong or can accidentally mutate state if not constrained. Therefore the contract must allow `unverified/insufficient-evidence` outcomes rather than forcing binary success/failure.

---

## 2. LongHorizon-Harness — independent outcome audit and bounded execution context

Primary source: Ziyu Ma et al., *LongHorizon-Harness: Advancing Long-Horizon Agents for Real-World Tasks*, arXiv:2608.01964, submitted 2026-08-03.  
Paper: https://arxiv.org/abs/2608.01964

Classification: **direct current capability improvement; independent corroboration of the same missing invariant**

### Method

Its Manage-Execute-Audit loop separates:

1. a manager that maintains persistent task state and selects the next subtask;
2. a fresh-context executor that performs the state-changing work;
3. a read-only auditor that inspects the resulting environment independently.

The executor's report does not establish completion. The auditor verifies postconditions against the task contract before persistent task state advances. The raw execution trajectory need not become the canonical long-horizon state.

### Evidence

The paper reports:

- WeaveBench Qwen 3.7-Plus: 51.8% -> 80.7%;
- Terminal-Bench 2.1: 69.7% -> 77.2%;
- OSWorld 2.0: 2.8% -> 8.3%;
- Claude Opus 4.7 on an OSWorld 2.0 subset: 20.0% -> 34.3%.

The mechanism generalizes across different agent/model backends. The downside is substantial execution overhead: audit/manager loops and repeated bounded executor contexts cost materially more tokens and wall-clock time than the baseline.

### A.L.I.C.E. implication

Do **not** copy the harness as a mandatory three-model topology. The useful invariant is role/authority separation, which can be implemented with one or several models depending on hardware and risk:

- execution authority and outcome-verification authority are logically distinct even when physically colocated;
- mission state is compact/rebuildable and externally stored;
- raw action history remains Experience evidence rather than canonical mission truth;
- fresh/bounded executor context is an optimization, not a capability ceiling.

This integrates naturally with Mission Graph, Result Capsule, source-native/live reads, Temporal workflows, Experience/Event Fabric, and Cognitive Workspace.

---

## 3. RSIAgent — verified experience before procedural/environment-memory promotion

Primary source: Sibo Zhu et al., *RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments*, arXiv:2609.15364, submitted 2026-09-14.  
Paper: https://arxiv.org/abs/2609.15364

Classification: **direct future capability improvement to procedural/environment learning; current contract implication for promotion gates**

### Method

RSIAgent performs training-free environment adaptation through autonomous memory construction with curriculum, actor, and verifier roles.

- Broad Recursive Self-exploration builds coverage through parallel exploration.
- Deep Recursive Self-exploration targets hard cases, hidden constraints, boundary conditions, and missing environment knowledge.
- Verified successes/failures are consolidated into reusable environment memory.
- The final memory is frozen and reused at evaluation without changing model weights.

The useful design point for A.L.I.C.E. is not the exact three-agent topology or fixed exploration schedule. It is the **outcome-gated memory-consolidation boundary**.

### Evidence

On a four-task exploration-stage ablation, full broad+deep RSI reaches 74.54% mean partial score versus 65.52% for broad-only and 56.50% for deep-only. The paper reports broader gains on OSWorld-v2 and Agent's Last Exam.

Its own failure analysis is important: incomplete verification can admit wrong outcomes into memory, and verifier errors can then propagate into later behavior. The paper also notes substantial extra exploration compute, dependence on finite exploration/stopping budgets and memory quality, and incomplete isolation of every component.

### A.L.I.C.E. implication

A.L.I.C.E. already has procedural learning, FRESH-style failure/precondition/repair structure, Experience/outcome lineage, Lifecycle Curator, and governed promotion/rollback.

Add the explicit promotion invariant:

- failed, partial, blocked, and unresolved attempts can remain valuable Experience;
- they may yield failure lessons or hypotheses;
- they must not silently become verified-success procedures;
- environment/tool/version conditions and the exact verification basis travel with any promoted procedure;
- an `UNVERIFIED` outcome stays unresolved, not relabeled as success or failure for training convenience;
- later contradiction, correction, environment-version change, deletion, or rollback can demote/rebuild descendants.

This protects procedural learning from turning a verifier mistake into a durable self-reinforcing rule.

---

## Authority / provenance / identity / deletion compatibility

The finding strengthens rather than weakens current boundaries.

### Evidence vs Claim Authority

Mission-result verification does not make a Result Capsule into XTDB Claim Authority. A verified action result is still an outcome record. Promotion into world/person/relationship Claim state follows the separate Claim-admission path.

### Projections vs truth

Mission state and procedural memory remain derived projections over Experience, environment observations, and verification receipts. They do not replace source-native evidence.

### Identity separation

Host, source-person, A.L.I.C.E.-self, and relationship models remain separate. Verification of an action cannot silently promote an inference about a person or relationship.

### Correction / deletion / revocation

Verification receipts, Result Capsules, mission-state successors, procedural memories, evaluation artifacts, activation/context remnants, and any trained descendants must remain linked to deletion/correction generations. If supporting evidence is revoked or a postcondition is later invalidated, the dependent mission/procedural projection must be rebuildable or demoted.

### Rollback

A bad verifier or bad promotion gate must be reversible without rewriting Experience history. Preserve the attempted action and observed outcome; roll back the authority/projection decision.

---

## Focused validation when this contract lane becomes active

Do not start a general harness benchmark. Add only decision-bearing cases:

1. actor reports success but required artifact/postcondition does not exist -> mission remains uncompleted;
2. output exists but violates one acceptance criterion -> partial/unverified, not success;
3. deterministic/source-native evidence contradicts a visual/model verifier -> stronger evidence prevents completion;
4. verifier has insufficient access -> unresolved/awaiting verification, not forced failure;
5. a later action invalidates an earlier verified milestone -> prior completion is revoked/reopened with lineage;
6. verifier probe attempts a state-changing action -> reject/quarantine the verification receipt;
7. verified failed attempt -> Experience retained and failure lesson may be proposed, but no success-procedure promotion;
8. false-positive verifier result -> rollback removes its authority descendants without deleting the underlying Experience;
9. environment/tool version changes -> old procedure becomes scoped/stale until revalidated;
10. compare outcome-gated versus self-reported mission progress on a small long-horizon task set and measure false-completion rate, recovery success, extra tokens/latency, and valid-completion retention.

Stop after these tests answer the contract decision. No backend substitutions or all-pairs harness evaluation.

## Architecture decision

- **No selected physical stack change.**
- **No reduced Fable tier.**
- **No change to the current MFM v1.6 critical path.**
- **No main/production semantic mutation from this research intake.**
- When the Mission Graph runtime/result-propagation lane is implemented, require an impact review before changing `ResultCapsule` / `MissionNode` semantics.

## Screening notes

Other surfaced work in this pass was not promoted as a new architectural delta:

- RippleMem's anchor-driven associative expansion and missing-support targeting strongly corroborate the existing Associative Graph Compute + iterative Cognitive Recollection + evidence-sufficiency design. It is a future algorithm challenger, not a new plane or missing architectural invariant.
- Parametric Multimodal User Memory strongly supports grounded face/voice/perceptual identity memory and separation of perceptual identity from exact factual binding, but this direction is already explicitly present in current main architecture documents and therefore is not new for A.L.I.C.E.
