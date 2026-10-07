# Frontier Watch Intake — 2026-10-07

Status: research/contract-impact note only. No production/main semantic change.

Repo state checked before judgment:
- A.L.I.C.E. main: `8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b`
- Fable_Sleight main: `a6eba323dd27d3c58f232b351d09b86761b15935`
- frontier-watch prior head: `3aedd37d7612427905b46db06c91ef1622ee29c1`
- alice-context: `945245bbee4168fff46f863455a83bb219063017`
- current MFM licensed/source-sequence lane: `research/mfm-v1-licensed-base-20260930@bc64e5f26f86f7c06ff44f85099455e5beb4afbb`
- active MFM v1.6 source/teacher, FBM, Graphify, Memory M2 bounded-serving/deletion, and shadow-adjudication branches were inspected.

The MFM source-sequence construction remains the current critical path. The findings below do not block it. The selected Fable v1 physical stack remains unchanged.

## 1. The Right Memory in the Wrong Context — retrieval admissibility

Primary: Zi Wang et al., arXiv:2610.07309, submitted 2026-10-05.  
https://arxiv.org/abs/2610.07309

Classification: **direct current capability improvement / missing explicit bounded-serving invariant**

The paper separates relevance from admissibility. A record may answer a query and still be ineligible because it belongs to another authenticated principal, violates a governing purpose/policy, or has the wrong lifecycle status for the query. Admissibility is three-valued: admissible, inadmissible, unresolved. Lifecycle is query-conditioned: a superseded record can be excluded from a current-state answer but required for a history query.

It also separates stored -> retrieved -> prompt-exposed -> disclosed. Routes are compared at matched required-evidence recall so a system cannot look safer simply by dropping needed evidence.

Reported top-20 reanalysis across 3,767 RHELM/MemOps queries: trusted namespace restriction raises anchor recall from 0.432 to 0.533, raises 80%-recall feasibility from 0.237 to 0.311, and reduces exact similarity evaluations by 98.3%. Across 1,523 benchmark-native paired cases, namespace routing is associated with judged-accuracy gains of 0.053–0.068 across three readers, although recall also changes. Under a 1% required-anchor false-denial limit, the tested text-only verifiers do not reliably detect violations; missing governance evidence should remain unresolved rather than be guessed from text.

A.L.I.C.E. already has ProductHostScope, authority namespaces, record versions/generations, host/source-person/self/relationship separation, deletion/correction lineage, Context Packets, and Retrieval Traces.

The missing explicit contract is query-conditioned prompt-entry admissibility. Current M2.4 ContextSelection records namespace, source kind, version, generation, stale flag, score and reason codes, but it does not require a verdict over authenticated principal/scope, active policy/purpose, query intent, lifecycle compatibility, and unresolved governance metadata. Retrieval and prompt exposure are also not first-class separate stages.

Required invariant:

> Relevance does not grant prompt-entry authority. A selected memory must pass query-conditioned admissibility before activation, or remain unresolved/excluded according to policy.

This is a serving projection, not Claim Authority. Historical data remains available when allowed; a superseded record is not globally forbidden. Correction, revocation or deletion must also invalidate dependent activation/runtime state.

Focused future validation:
1. relevant record from wrong principal/host;
2. wrong identity/relationship scope;
3. superseded record for current query vs required historical query;
4. revoked/forgotten record;
5. missing governance metadata -> unresolved;
6. graph expansion trying to reintroduce an excluded descendant;
7. matched-recall evidence retention;
8. explicit retrieved-vs-exposed audit;
9. deletion/correction invalidating dependent activation.

Evidence strength: moderate. The design distinction is strong; deployment generality is limited by two public sources, trusted namespace labels and small controlled verifier/disclosure studies.

No backend change.

## 2. Beyond Corrected Memory — execution evidence beyond outcome verification

Primary: Zhe Yu et al., arXiv:2610.08101, submitted 2026-10-06.  
https://arxiv.org/abs/2610.08101

Classification: **direct current capability improvement / extension of the Oct-3 verified-mission-state invariant**

The Oct-3 watch already established that executor self-report cannot authorize canonical mission completion. This paper adds a separate requirement: correct memory and a verified final state may still be insufficient to judge whether execution fulfilled the task duty.

The required evidence includes receipt/visibility of updates, whether an action depended on the state, actor authorization, historical-version eligibility, observation-window completeness, and whether a repair remained valid through its deadline. Missing evidence produces an undetermined result rather than presumed compliance or violation.

In 450 controlled pairs, 398 have opposite full-execution labels. Removing one of five execution-evidence factors makes 328/398 (82.4%) observationally identical; restoring the missing relationship separates 321/328 (97.9%). This is an information-preservation result: a later judge cannot infer a distinction the trace never retained.

CAVERT reports 87.1% accuracy / 83.5 macro-F1 on 720 executions, versus 80.6/74.8 for contract-prompted LLM diagnosis and 78.9/73.8 for rules. With the same pre-commit gate, its guidance improves AppWorld from 89/180 to 122/180 and LongCLI repair-without-regression from 51/180 to 75/180. A long-horizon cohort pays +24.6% tokens and +5 seconds/task.

A.L.I.C.E. already has Mission Graph, Result Capsules, Experience/Event history, tool/action/output lineage, traceback, verified-outcome doctrine, rollback and deletion lineage. The genuinely new delta is preserving enough execution structure to judge the duty, not merely the final postcondition.

Current ResultCapsule stores output/evidence/source-event references, provenance, deletion lineage and policy bindings. It does not itself require explicit proof of receipt/visibility, action dependence, actor authorization, version eligibility, observation-window completeness or effect validity at deadline.

Required invariant:

> Outcome verification and execution-consistency evidence are complementary. Final state establishes what happened; execution provenance must preserve enough structure to judge whether the action used allowed state and fulfilled the applicable duty.

Focused future validation:
1. update stored but never received;
2. update received but action used another state;
3. explicitly permitted version pin;
4. repair request exists but effect failed;
5. repair succeeds then rolls back before deadline;
6. incomplete relevant observation -> unresolved;
7. correct final state through a duty-violating path;
8. opposite execution-duty labels remain distinguishable from preserved lineage.

This extends, not replaces, the Oct-3 verification requirement. Do not copy CAVERT as a required architecture. No backend change.

## 3. LINEUP — shared parametric capacity with compact private correction

Primary: Songyuan Sui et al., *Does Every User Need a Private LoRA?*, arXiv:2610.02353, submitted 2026-10-01.  
https://arxiv.org/abs/2610.02353

Classification: **future challenger / direct future efficient-private-scaling idea**

LINEUP learns a shared bank of low-rank personalization factors, recalls factors from user history, calibrates the composition per query, and fits a small private correction code for each target user. In the paper's setup, the user code has 8 scalars versus 4.19M parameters for the evaluated private-LoRA baseline.

Across six LaMP tasks it reports the best mean on all 12 metrics. In the paper's analysis, 32 shared directions (0.8% of output dimension) capture 54.1–67.9% of held-out adapter-update energy versus 0.78% for random subspaces. Ablations show the largest loss when history-guided factor recall is replaced by random candidates; removing query calibration or the compact correction also hurts. At 1,024 users, the paper projects about 151.2 MB total adaptation storage for LINEUP versus 1.61 GB for PerFit and 8.60 GB for OPPU in the evaluated setup.

Fable implication is deliberately narrow:

> Parametric personalization capacity need not be duplicated per host if reusable host-neutral directions can be learned once while host-specific residual correction remains private, lineage-bound and governed.

The paper's 8-scalar code is not a Fable-person model, ontology or capability ceiling. Fable's evidence, memory, host/source-person/self/relationship state, Claim Authority and learned private history remain separate.

Any future use requires:
- no cross-host private-history leakage through the shared factor bank;
- per-host correction isolation;
- separate identity/relationship adaptation;
- factor-bank and host-state lineage/rollback;
- deletion/unlearning coverage if private data ever influences shared factors.

Evidence strength: moderate for task-level personalization efficiency, weak for lifelong personal cognition. The experiments are six task-isolated LaMP tasks on Llama-2-7B, not multi-year continual host development.

Future-only. Do not alter MFM/FBM now. Revisit only when personal-model storage/training becomes a measured bottleneck.

## Screened but no action

PACMI (arXiv:2610.05732) is **already covered/corroboration**: typed provenance, non-destructive historical preservation, cascading validity changes and current-vs-history retrieval agree with the selected Claim Authority, provenance/dependency lineage and M2.6 correction/deletion propagation. Its benchmark supplies the dependency graph, has 100 cases / 300 queries with six prior memories and fixed two-hop topology, and the cascade-only answer difference (3 vs 11 errors) does not meet its stated 0.05 threshold. Its four-state lattice is evidence, not a Fable state ceiling.

Recent KV/multi-device systems work does not reveal a current serving blocker beyond the already tracked CacheReforge/HEAR/cross-layer invalidation/federation direction. ForkPilot is useful corroboration for learned retrospective-search scheduling, already represented by MRM/Recollection plus AutoMem/MemAgent/PoS/Recuris/FRESH.

## Decision

Carry forward:
1. retrieval relevance != prompt-entry admissibility;
2. verified outcome != sufficient execution evidence for duty judgment;
3. future parametric personalization may amortize host-neutral capacity while retaining private host correction.

No architecture-invalidating evidence.
No physical-stack reopening.
No reduced Fable tier.
No main/MFM/FBM semantic change.
The MFM source-sequence construction remains the current critical path.
