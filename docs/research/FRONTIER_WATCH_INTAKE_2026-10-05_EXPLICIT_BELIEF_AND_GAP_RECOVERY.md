# Frontier Watch Intake — Explicit Belief State and Gap-Aware Recovery — 2026-10-05

**Status:** research/contract-impact note only; no production/main semantic change  
**Baseline:** `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`  
**Branch:** `research/frontier-watch`

## Repo state checked before judgment

Current inspected heads:

- A.L.I.C.E. `main@8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b`
- Fable_Sleight `main@a6eba323dd27d3c58f232b351d09b86761b15935`
- frontier watch before this intake: `research/frontier-watch@3bea3318904ec2881d3f981d46a5b8d5bca91ba7`
- `alice-context@945245bbee4168fff46f863455a83bb219063017`
- `fable-builder-model@6861d76535c3418efbd2ee948c1056d57030cfe4`
- `research/graphify-context-substrate@a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15`
- `research/mfm-foundation-20260923@345dba7b47efd5b4707302c07508493777fcb465`
- active MFM v1.6 source/teacher branches through the 2026-10-01 heads.

The current critical path remains MFM v1.6 source/teacher authorization, corpus admission, provenance, and CPU processing. This finding does not block that path and does not reopen selected physical placements.

## Beyond Memory: Harnessing Long-Horizon Agents with Explicit Belief States

**Primary source:** Yu Luo et al., arXiv:2610.01415, submitted 2026-10-01.  
https://arxiv.org/abs/2610.01415

**Classification:** **direct current capability improvement / runtime contract refinement**. Not architecture-invalidating.

### What A.L.I.C.E. already has

A.L.I.C.E./Fable already has durable Experience/evidence, bitemporal Claim Authority, world/social/causal/mission state, Mission Graph, Cognitive Workspace, Recollection, provenance, uncertainty, correction/deletion lineage, rollback, and independently verified mission completion as a newly recorded contract direction.

The missing distinction is narrower: the runtime decision context does not yet have an explicit contract separating **current task belief** from durable truth while also separating **what remains unknown** from **what remains to be accomplished**.

### Genuinely new delta

PoS maintains a task-conditioned decision state

`B_t = (W_t, G, Δ_t^E, Δ_t^A)`

where:

- `W_t` is the task-relevant current-world estimate;
- `G` is the task objective/completion criterion;
- `Δ_t^E` is the set of epistemic gaps: information still unknown or insufficiently supported;
- `Δ_t^A` is the set of achievement gaps: world/task conditions that still need to be changed.

The world estimate uses Entity–State–Relation records with provenance, confidence, and supporting evidence. New belief updates are proposals first. A Belief Sentinel checks internal consistency and consistency/support against interaction evidence before the next decision state is committed.

PoS then detects **Belief Trapping** using gap persistence, progress stagnation, and belief recurrence. Traps are factorized as Static, Cycle, or Drift and paired with the blocked gap type. Recovery is correspondingly targeted: suppress a repeatedly ineffective transition, break a recurring loop, re-anchor drift to the active requirement, seek discriminative evidence for an epistemic gap, or require an actual state change for an achievement gap.

### Exact Fable implication

Add an explicit **runtime belief projection** contract to Cognitive Workspace / Mission execution.

This projection must be:

- transient and rebuildable;
- grounded in Claim Authority plus current Experience/evidence, source-native observations, and Mission Graph state;
- typed separately from Claim Authority;
- permitted to contain hypotheses and unresolved alternatives without promoting them to truth;
- split into epistemic and achievement gaps;
- versioned against the evidence/claim/correction/deletion generations from which it was built.

Candidate runtime-belief updates may be model-proposed. A consistency/evidence validator may reject or revise them. Neither proposer nor validator gains Claim-admission authority.

This complements, rather than replaces, the 2026-10-03 verified-mission-state finding:

- **externally checkable completion:** independent postcondition/outcome verification owns canonical mission-completion authority;
- **uncertain or inferential progress:** explicit belief + gap tracking maintains a coherent decision state until sufficient evidence or a verified state change exists.

### Evidence

PoS is evaluated on ALFWorld and LOCA-Bench for execution, plus RCA-100 and ClinDiag for diagnosis, using Qwen3.7-Plus, Kimi-K3, and GLM-5.3.

Reported overall results include:

- Qwen3.7-Plus ALFWorld: 72.39 strongest listed long-horizon harness baseline -> 88.81 PoS;
- Kimi-K3 RCA-100: 38.83 -> 51.46;
- GLM-5.3 ClinDiag: 46.85 raw trajectory / 45.53 LongHorizon-Harness -> 52.15 PoS.

The paper reports relative gains over the strongest same-backbone baseline reaching 22.68% on ALFWorld and 37.89% on RCA-100 joint accuracy.

Ablations are decision-relevant:

- removing consistency validation causes losses reaching 14.93 points on ALFWorld and 11.81 on LOCA-Bench;
- removing trapping diagnosis/recovery causes losses of 4.85–6.79 points on RCA-100 and 2.65–3.97 on ClinDiag;
- factorized recovery outperforms a generic recovery instruction;
- trapping modes differ by environment: cycles are prominent in ALFWorld, drift in LOCA/RCA, and static stagnation dominates ClinDiag.

This supports treating progress failure as structured state, not one generic retry/reflection signal.

### Authority, provenance, identity, deletion, rollback

**Authority:** runtime belief is a projection, not XTDB Claim Authority. Confidence inside the Workspace cannot override evidence/claim admission.

**Provenance:** every material State/Relation/hypothesis in the runtime belief should be traceable to the Claims, Experiences, observations, or inference receipts supporting it.

**Identity:** host, source-person, A.L.I.C.E.-self, and relationship entities must remain typed. A task belief may not silently transfer a property between them.

**Deletion/correction/revocation:** if a supporting source/claim generation is corrected, revoked, or deleted, dependent runtime belief and trapping/recovery state must be invalidated or rebuilt. A removed statement must not survive through an old Workspace summary or recovery constraint.

**Rollback:** bad belief updates or a bad recovery policy can be rolled back without rewriting canonical Experience history.

### Cost, risk, and limitations

PoS adds repeated belief construction/validation and recovery computation. Its natural-language state can be redundant and costly. The authors explicitly note omission/distortion risk and suggest future latent task-conditioned representations.

A.L.I.C.E. should not adopt a latent replacement that sacrifices inspectability, rebuildability, provenance, or deletion coverage. A compact learned representation may later accelerate the projection if an auditable source-linked surface remains available.

The paper also does not solve missing execution skill or domain knowledge. Better state maintenance cannot compensate for absent capabilities.

### Focused action

When the Cognitive Workspace / Mission runtime contract becomes active, add only decision-bearing validation:

1. current Workspace/Mission state versus explicit world-belief + epistemic-gap + achievement-gap state;
2. static/cycle/drift failure localization and targeted recovery;
3. false-confidence and unsupported-relation rejection;
4. current-versus-stale world state after updates;
5. host/source-person/self/relationship binding;
6. correction/deletion/revocation rebuild;
7. externally verified completion versus unresolved inferential progress;
8. token/latency overhead and whether it reduces repeated actions, drift, or mission divergence.

Do not start an infrastructure benchmark and do not create a new physical store.

## Screened, no new action

**Mem++ (arXiv:2610.02002)** is useful corroboration for source-native, non-destructive evidence retention and read-time temporal selection, but those decisions are already explicit in Fable and are already strengthened by JitMem/read-time reconstruction. Its optional supersession links may inform a future read projection; they do not replace bitemporal Claim Authority or source history.

Recent context-compression and KV-serving work was also screened. Nothing found in this pass justified reopening the selected serving/storage stack or changing the current MFM critical path.

## Decision

Preserve the selected architecture and critical path.

The new contract-level decision is:

> **Cognitive Workspace should maintain a provenance-grounded, rebuildable current-belief projection that explicitly distinguishes epistemic gaps from achievement gaps and can diagnose progress trapping, without becoming durable truth or completion authority.**

No production/main semantic change is made in this intake.
