# SHAPER — Skill-Harness Evolution Intake

**Reviewed:** 2026-09-23  
**Primary source:** https://arxiv.org/abs/2608.11350 (v2, 2026-09-10)  
**Paper:** *Self-Evolving Embodied Agents via Skill-Harness Evolution*  
**Status:** future research/design seed only. No canonical architecture, completed Stage G work, or current EIPM N0 implementation is changed by this note.

## Why this clears the frontier-watch bar

SHAPER freezes the planner and executor and instead optimizes two model-external artifacts from target-environment rollouts:

1. a reusable **skill**: procedural guidance for decomposition, command choice, recovery, evidence use, and stopping;
2. a **context-code harness**: executable code that selects and formats trajectory history, observations, evidence, and execution state before each planner call.

Rollouts are locally diagnosed, compressed into episode/batch summaries, and converted into a textual optimization signal. The same frozen foundation model is reused under a separate optimizer role. Skill candidates are evolved first; the selected skill is then held fixed while harness candidates are evolved. Generated harness code is sandbox-validated. Candidate artifacts are selected on held-out validation tasks and reused without per-task search.

The paper reports VLABench success increasing from 28.25% for the seed agent to 34.50% for full skill+harness evolution. On its 231-question ESI-Bench subset, micro accuracy increases from 32.5% to 49.8%. The ESI harness learns sparse evidence-aware visual memory rather than a fixed recency window, preserving task-critical historical observations and detecting low-information/action loops.

## What A.L.I.C.E. already has

Current canonical policy is broader than SHAPER in several ways:

- the lifelong-learning loop already turns trajectories and outcomes into candidate memory, belief, skill, model, or code changes, evaluates them, promotes/rejects them, measures real outcomes, and repeats;
- procedural evolution already permits synthesis, composition, testing, versioning, comparison, and retirement of executable skills;
- self-improving code and variant archives are already authorized under evidence and rollback;
- meta-learning already permits A.L.I.C.E. to improve its learning, evaluation, planning, routing, memory curation, storage, replay, and experiment-generation mechanisms;
- the Context Planner is already a first-class adaptive serving component;
- Track H already includes reusable skills, failure cases, learned routing/ranking, challengers, and continual-learning experiments;
- the architecture already requires trajectory/outcome capture, evaluation, rollback, lineage, and independent promotion gates.

Therefore SHAPER does **not** justify a new top-level plane, a rewrite of completed memory authority, or a change to current EIPM N0.

## Genuine delta

A.L.I.C.E. currently states that skills, code, context planning, and learning mechanisms are evolvable, but it does not make **the context-construction program itself a first-class co-evolved artifact paired with a procedural skill**.

SHAPER provides a concrete optimization pattern:

```text
trajectory + observable outcome
        ↓
local failure diagnosis
        ↓
episode / batch evidence summary
        ↓
failure attribution
   ┌───────────────┐
   ↓               ↓
skill defect    context/harness defect
   ↓               ↓
skill candidate  context-program candidate
   └───────┬───────┘
           ↓
sandbox + held-out validation
           ↓
versioned champion/challenger artifact
```

The strongest A.L.I.C.E.-relevant idea is not “freeze the model.” A.L.I.C.E. must retain parametric learning as another substrate. The useful idea is **explicit failure-localized co-evolution across external cognitive artifacts**, so a failure can be assigned to procedural policy, context/evidence routing, model behavior, tool/executor behavior, environment failure, or another subsystem before an update is proposed.

A second useful delta is **evidence-aware context retention learned from outcomes**. SHAPER demonstrates that a fixed recent-history window can discard old but task-critical evidence. Its evolved harness instead retains sparse informative observations from the full trajectory. This is directly relevant to Track F Context Planner and future multimodal episodic/context serving.

## Compatibility

Compatible with:

- Experience Ledger and outcome linkage;
- Learning Curator;
- procedural skill synthesis;
- Context Planner;
- multimodal/episodic retrieval;
- Evolution Laboratory champion/challenger evaluation;
- model-independent identity;
- rollback/versioning;
- future Phase 8, Phase 11, and Phase 13 learning.

Required A.L.I.C.E. differences:

- an evolved harness is a **derived executable artifact**, never evidence or claim authority;
- harness inputs must preserve evidence/provenance identities instead of converting model summaries into truth;
- failure diagnoses and textual summaries are proposals/derived analysis, not authoritative observations;
- code artifacts require version, lineage, permissions, tests, resource profile, rollback, and deletion/influence handling;
- no SHAPER-style frozen-model assumption becomes an architectural limit; external artifact evolution and parametric learning remain complementary;
- optimization must not collapse identity/source-person authority into task reward.

## Proposed future design contract

Add a **Skill–Harness Co-Evolution challenger** spanning Track F and Track H:

1. Treat procedural skill and context-construction policy/program as separately versioned artifacts with explicit interfaces.
2. Record exact rollout evidence, model/tool versions, selected context, actions, outcomes, costs, and failure receipts.
3. Run evidence-grounded failure localization before proposing a mutation. Candidate causes include skill, context/harness, retrieval, model, executor/tool, environment, evaluator, or unknown.
4. Allow isolated skill-only, harness-only, joint, and later parametric challengers so credit assignment can be measured rather than assumed.
5. Keep optimizer summaries linked to the underlying rollout evidence. Do not train/promote from an untraceable textual gradient.
6. Sandbox executable harness candidates and enforce capability/permission contracts.
7. Select candidates on held-out and historical replay suites. Include distribution shift, rare cases, corrections, identity continuity, and adversarial contexts.
8. Test learned evidence retention against recency-only, lexical/vector/graph retrieval, full-history, and current Context Planner baselines.
9. Measure whether the harness preserves old task-critical evidence without preserving stale, contradicted, revoked, or deleted evidence.
10. Permit external-artifact evolution and weight learning to compete or compose. Do not permanently freeze either substrate.
11. Require rollback to the exact prior skill+harness generation.
12. Feed successful generalizable patterns into the capability/parity process only after A.L.I.C.E. qualification.

## Impact by current work

- **Completed Claim/Experience authority:** no change.
- **Current Stage G memory-fabric qualification:** no required migration. At most, preserve rollout/context traces needed by the future challenger.
- **Current EIPM N0:** no architecture change. N0 should not be diverted into embodied-agent harness optimization.
- **Track F Adaptive Context:** future challenger and evaluation implication.
- **Track H / Phase 8 procedural learning:** direct future design implication.
- **Phase 11 self-evolution:** direct future design implication.
- **Phase 13 parametric learning:** comparison/composition implication; external evolution must not become a substitute for weight learning.

## Evidence, upside, and risk

**Evidence strength:** moderate. The method is evaluated on two embodied benchmarks with held-out tasks and artifact ablations. The ESI evaluation is a 231-question subset, not the full benchmark, and the paper explicitly leaves cross-embodiment transfer and real-robot validation to future work.

**Expected upside:** high for future procedural/self-evolution work. It gives A.L.I.C.E. a concrete way to learn not only *what procedure works* but also *what evidence/context representation makes the procedure work*.

**Runtime/storage cost:** low after promotion; the selected artifacts are reused. Evolution cost comes from rollouts, judging, summarization, candidate generation, and validation. A.L.I.C.E. should record actual local/cluster compute rather than rely on the paper's API-equivalent dollar estimate.

**Main risks:** optimizer overfitting to small validation sets; self-reinforcing bad diagnostics; summaries losing decisive evidence; harness code introducing hidden capability changes; context optimization retaining stale/invalid evidence; reward optimization conflicting with identity/authority constraints; and false credit assignment between skill, harness, model, and executor.

## Recommendation

Admit SHAPER as a **future Track F + Track H design seed**. Preserve the failure-localized skill/harness co-evolution pattern and evidence-aware context-learning idea. Do not copy the frozen-model constraint, do not mutate completed authority work, and do not interrupt current EIPM N0 for it.

Before implementation, benchmark the challenger against A.L.I.C.E.'s existing Context Planner and procedural-learning baselines with full provenance, rollback, correction/deletion, historical replay, and identity-continuity tests.
