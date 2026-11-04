# Frontier Watch intake — 2026-11-04

**Disposition:** Six substantive new research findings plus a conditional model-suppression challenger; three narrower papers screened. All primary papers below were submitted November 2–3, 2026 and surfaced in the November 4 screening. **No architecture-invalidating evidence, no established current MFM/identity-model repair, no selected implementation reopening, no new infrastructure tournament, no reduced Fable v1 capability, and no main/production semantics change.** All actions below are future or conditional, not blockers of checkpoint-26 assessment and native personal judgment.

## Mandatory baseline and current inspection (completed before screening)

- A.L.I.C.E. main `a1fe0a65df1d4143955eb1d3b35eb4ac590e7555` and Fable_Sleight main `1c28eb3c989105d6a3d205c06cd5e6e8634ad77b`. Both unchanged from November 3.
- Active MFM `research/mfm-v1-licensed-base-20260930@5653736f545c9c1c8d5d8006af0f1fa152eed9b1`; FBM `research/fbm-v1-licensed-base-20260930@bcd462b5bb93f41a56f79b5b63e3337c8e89bb45`; N0 `alice-eipm-v1-n0-full-envelope-joint-ddp-v2@5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2`; continuity `alice-context@945245bbee4168fff46f863455a83bb219063017`; Graphify `research/graphify-context-substrate@a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15`; Fable October 10 execution profile `Rayan/fable-current-stack-docs-20261010@35acb38c90a7e3b31ddfc149fe2cfebd8b2c6c6b`. No new active branch head reported.
- Read `docs/research/FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`, `docs/research/FRONTIER_WATCH_EXECUTION_GUARDRAIL_2026-09-27.md`, all prior intake inventory and latest November 3 note on `research/frontier-watch@a98559ebeae4510cb72ec694175f49ceb99bef2e`; main README and execution map `docs/ALICE_PHASE2_REPLACEMENT_AND_FABLE_V1_EXECUTION_PLAN_2026-09-27.md`; `docs/FABLE_PERSONAL_DEVELOPMENT_ARCHITECTURE.md`; Fable README and `docs/FIRST_RELEASE.md`; MFM `docs/MFM_CHECKPOINT10_FORMATION_DIAGNOSIS_2026-10-21.md`, `docs/MFM_OBSERVED_WORLD_TRAINING_2026-10-22.md`.
- Current qualified evidence: S5 language foundation finished 20 epochs / 45,760 steps. MFM joint formation checkpoint 10 produced zero complete grounded native formation decisions in nine cases. Checkpoint-10 numerical diagnosis measured partial learning and selected an unchanged 16-update continuation through checkpoint 26, preserving optimizer and source closure; no newer committed terminal success. Rill observed-world source/target material construction is separate from the pending checkpoint assessment and is not itself native learning. Conditional personality judgment remains unqualified.
- The full transferable personal cognitive foundation after F11 is the only first-release predicate. Canonical Experience KurrentDB, transport NATS JetStream, bitemporal Claim XTDB v2, encrypted CAS evidence, LadybugDB/NebulaGraph graph placement, Qdrant vector, Temporal workflow, local L1+Valkey ephemeral state and PyTorch+Accelerate training remain selected. Hardware adjusts placement, not logical cognition.
- Previously tracked rather than rediscovered: MemCalib, Jev-Mem, RPMem, JitMem, REALM, AdaMM, MUTE, PGMem, Recuris, FRESH, MemCryst, CALM, StateWise, COP, LEDGER, Heirloom, MemUse, MemHarmBench, StateGraph, MEMENTO, MemSidecar, SHIFT, and earlier October–November research. The entries below are distinct new primary sources.

## 1. MemShift — detect invalidation separately from accurately repairing derived personal values

**Primary:** Zichuan Xu et al., *MemShift: Separating Detection from Repair in LLM-Derived Personal Memory*, arXiv:2611.02338v1, submitted November 2. https://arxiv.org/html/2611.02338v1

**Classification:** Future challenger; direct future derived-state integrity/evaluation improvement, not a demonstrated current defect.

**Mechanism:** The benchmark contains defined derived personal values from long multi-app histories. It forks synthetic worlds under five shifts: source revision, deletion, user-definition change, writer swap and export. It separately checks detection/invalidation and correct recomputation. Gold values and dependencies are determined before any generated language; a reference implementation varies keys and recomputation. Static citation keys miss records which satisfy a definition only after a later insertion. Full source-dependency scope or executable typed definition tracks more changes.

**Evidence/ablations:** Source histories span months and millions of input tokens, but are generated worlds. Under the evaluated Qwen3.5-9B writer, only 16.3% of changed user-defined values are accurately recomputed after source revision; tested writers re-derive at most about one-quarter of definition-dependent changed values. Program-mediated repair over typed records substantially closes the gap to exact recomputation; on definition shifts, reported changes over the bare model writer are about +25.4 points for program-based repair and +26.9 for exact recompute. A static source-key approach pays ~3.64x token cost. Accuracy improvement after detected change does not guarantee semantic correctness.

**A.L.I.C.E. already:** Source-native reads, evidence/event contracts, XTDB bitemporal authority, derived analytic/host projections, provenance, deletion and rebuildability. **New:** Separation of (a) whether a derived value was invalidated from (b) whether its entire user-defined computation was accurately rebuilt, including records not previously cited. Avoid stale computed host or world attributes even when underlying events are updated.

**Authority and cost:** Dynamic provenance must record query/definition scope, actual source versions and owner permissions. Model-produced executable programs are untrusted, need sandboxed deterministic evaluation and versioned review; their outputs are proposals, not Claim Authority. Correction/revocation and deletion invalidate all derived traces including model-training inputs. Program validation and full dependencies add indexing/runtime/storage cost, with evidence currently synthetic.

**Action:** When analytic personal-state values and derived host aggregates are integrated, use one changed-definition or late-arriving-evidence case to distinguish invalidation detection from repair correctness. No new database, production migration, or MFM objective change.

## 2. EvoMemState — separate stable relational memory state from query-specific reasoning state

**Primary:** Zhishang Xiang et al., *EvoMemState: Evolving Memory State for Personal Agents*, arXiv:2611.03743v1, submitted November 2. https://arxiv.org/html/2611.03743v1

**Classification:** Future challenger to Recollection/MRM; incremental to CALM/PGMem/AgentMemRec, not a new plane.

**Mechanism:** Fact extraction plus evolving relational states expressing validity, temporal scope, contradiction and role. A reinforcement-learned policy produces task-specific reasoning state, conditioning how facts should affect the answer. It separates *complete retrieval* from *applicable use*, with a structured ontology built iteratively for its tests.

**Evidence/ablations:** Four dialogue/memory-use benchmarks and Qwen3-8B/GPT-4o-mini evaluations; on the reported Qwen3-8B average full method 59.9 versus 55.4 without relational state, 53.0 without reasoning state, 53.2 without RL. Memory completeness can still fail to yield correct downstream decisions. Ablation signals are positive but model-/dataset-dependent and partly model-judged.

**A.L.I.C.E. already:** Cognitive Multi-Graph and Claim temporal validity; source evidence; Recollection orchestration, MRM, Cognitive Workspace, native judgment. **New:** Learned query-specific *role of evidence* as an explicitly measured utility policy after retrieval. Do not copy its limited ontology or separate model count into Fable's full open-ended foundation.

**Authority and cost:** Retrieved salience, relational edge confidence and RL reward never establish canonical truth, subject identity, permission or revocation authority. Derive from accepted Claims and admissible sources at current versions; retain rollback/deletion lineage. Persistent relational updates, policy training, and inference have nontrivial compute and storage costs; long-term human evidence is weak.

**Action:** Conditional future use if complete but misleading evidence sets cause demonstrably poor judgment. Verify appropriate use under a corrected host preference versus temporally superseded memory. No current change.

## 3. Artifact-to-Capability — measure useful behavioral formation rather than artifact accumulation

**Primary:** Ziqian Ban et al., *Artifact-to-Capability: Diagnosing Breaks in Capability Formation of Self-Evolving Agents in a Controlled Environment*, arXiv:2611.04952v1, submitted November 3. https://arxiv.org/html/2611.04952v1

**Classification:** Direct future FloRA/Skill/Mission qualification improvement, not current learned capability proof.

**Mechanism:** A five-stage dependency chain: stored artifact → activated altered behavior → positive task effect → safe composition with prior competencies → cumulative extension in later versions. The diagnosis checks identical game states, full agent-game traces, and frozen version lineage. Crucially, an apparently good skill can be overridden by another valid skill in the integrated harness.

**Experiments/ablations:** Reflexion, Voyager and DGM with one base model in deterministic freestyle Gomoku. All show rising artifacts, behavioral changes and aggregate metrics. Reflexion mainly corrects illegal moves, not higher-quality legal choices; Voyager's control-order overrides immediate wins in all 20 tested conflict games despite functioning subskills; DGM gains task states in one version and later loses them. Replayable ground-truth rules make causal examples strong; transfer beyond board-game planning is not shown.

**A.L.I.C.E. already:** FloRA intervention plans, source-grounded identity/judgment differentiation, Mission outcome receipts, Experience Ledger and model lineage/rollback. **New:** Distinguishes learned useful decision impact from rule compliance, cross-component override and subsequent loss-only drift. No amount of artifact counting qualifies native personal model learning.

**Authority and cost:** An observed changed decision may still rely on unauthorized evidence; comparisons must use identical subject/source/Claim and permission versions, with exact frozen model/feature hashes and deletions across lineage. Controlled replay is bounded but real-world actions must not be replayed unsafely.

**Action:** When actual FloRA personal-learning evidence is available, check one genuinely new beneficial judgment with and without the learned artifact, one prior qualified judgment in full integrated execution, and successor preservation. This protects a real qualification, not a blanket new benchmark. Keep checkpoint-26 unchanged.

## 4. GroupAssistBench — real multi-person recall can excel while assistance fails

**Primary:** Zhanghao Hu et al., *One Assistant, Many Memories: Benchmarking Real-World Group Assistance When Recall Is Not Enough*, arXiv:2611.04867v1, submitted November 3. https://arxiv.org/html/2611.04867v1

**Classification:** Future social/relationship and Recollection-to-judgment evaluation challenger.

**Mechanism:** Real deployed group assistant corpus of 50 groups, 80,299 messages and 2,395 tasks; compare factual recall with open-ended assistance over the same source histories. Assign dependencies from one, distributed, or revised/conflicting multiple people. The evaluations anchor required evidence to actual utterances; response constraints and penalties are based partly on real feedback. Relational refinement and explicit assistance-constraint binding are the measured remedial methods.

**Evidence:** Strongest external memory recall 0.621 vs full-context 0.570, but its assistance score 0.247 vs full context 0.431. Full-context open assistance degrades from 0.505 single evidence to 0.303 conflicting. Some strong external memory workflows cost more than source-native retrieval; corpus domain and automated rubric judgments limit generality.

**A.L.I.C.E. already:** Source-person vs host vs relationship, social/temporal claims, source-native reads, privacy egress, Evidence→judgment decision packet. **New:** Evaluation of real multi-party, no-single-answer coordinated decisions where completeness of recall is a false proxy for usefulness.

**Authority and cost:** Every utterance belongs to its actual source and disclosure scope; a group's apparent consensus cannot rewrite the host's preferences or authorize another person's private material. Maintain corrections and subject-specific revocations; do not share group conversation traces without rights. Large transcript and model judge costs; partial real-world evidence stronger than purely synthetic dialogue.

**Action:** Future limited relationship/Mission integration test: two participants revise a shared commitment, the assistant must respond to the latest authorized constraint without collapsing identities. No role/store proliferation.

## 5. The Right Source Is Not Enough — irrelevant audio history influences latent answers after source blocking

**Primary:** HongJin Song et al., *The Right Source Is Not Enough: Understanding History Interference in Speech-LLM Memory*, arXiv:2611.05124v1, submitted November 3. https://arxiv.org/html/2611.05124v1

**Classification:** Future multimodal/activation memory and post-revocation influence-diagnostic challenger.

**Mechanism:** Contrast target speech evidence alone with full history containing distractor values; intervene on direct source attention, replace question-conditioned residual states, and isolate source-role versus source-value effects in low-dimensional activation subspaces. A wrong older candidate can continue influencing output after direct attention to the older source is blocked.

**Evidence:** Reproduced 165 erroneous MOSS-Audio cases over 122 episodes and 47 erroneous MiMo-Audio cases over 37 episodes, with causal interventions rather than plain retrieval ranking. Effects depend on model, layer and answer format, especially multiple-choice. Does **not** establish a general hidden-state contamination rate, real identity leakage, or a deletion erasure mechanism.

**A.L.I.C.E. already:** Perceptual evidence, versioned identity and subject separation, temporal Claim validity, working/activation memory, cross-layer deletion/unlearning, MUTE influence-echo watch. **New:** Source selection alone may not remove a distractor value already represented in downstream activations. A model's apparent source citation can be correct while its answer remains contaminated.

**Authority and cost:** All activation use must respect current source scope/Claim versions, revocations and reset rules; latent probes are diagnostics, not new authority. Fine-grained model hooks are expensive/model-specific and not warranted in current MFM.

**Action:** When actual speech personal memory is operational, one paired updated-audio-source query and reset/retained-old-history case to localize source selection versus activation contamination. Do not introduce activation surgery now.

## 6. MDSKILL — distill external procedural skill views into parametric competence

**Primary:** *MDSKILL: Learning to Internalize Skill Memory through Multi-View Distillation for Efficient Dynamic Agent Self-Evolution*, arXiv:2611.05313v1, submitted November 3. https://arxiv.org/html/2611.05313v1

**Classification:** Future procedural/parametric consolidation challenger, not an MFM lexical-formation repair.

**Mechanism:** Separate skill-augmented and unaugmented agent rollouts; dynamic utility selection and multiple functional skill views (state grounding, planning, execution control). Gain-gated on-policy self-distillation plus RL moves useful procedures into weights; additional retrieval becomes less necessary as a capability internalizes. Forward-only/ungated distillation can propagate noise; reverse/cross-view and gain gates are important.

**Evidence/ablations:** ALFWorld 93.8% vs no-distillation 81.3%; WebShop 82.8% vs no-distillation 46.9% in the reported experiment. Larger distillation weights degrade both. Additional runs use eight A100 GPUs, indicating a substantial training budget; tasks are controlled agent skills, not lifelong host-specific judgment.

**A.L.I.C.E. already:** Episodic Experience→procedural skill, MRM across external/parametric/activation memory, personal-model lineage and guarded promotion. **New:** One concrete student-teacher procedure for skill-form consolidation while testing whether resulting competence works without retrieval. Do not turn paper's three functional views into a capability ceiling.

**Authority and cost:** Skill success cannot make a false fact true or bypass permissions. Training examples need source and host identity filters, revoked/deleted skill exclusion, model generation hashes and rollback; learned weights may still reproduce forbidden behavior. Distillation may be expensive versus normal retrieval.

**Action:** Revisit only when actual useful learned procedures demonstrate reliance on external scaffolding and parametric consolidation is an identified cost/performance decision. No present MFM recipe change.

## 7. PEARL — guarded behavior suppression can survive some later retraining, but is not deletion

**Primary:** *Suppressing Misaligned Behaviors in Language Models, Durably*, arXiv:2611.05358v1, submitted November 3. https://arxiv.org/html/2611.05358v1

**Classification:** Future challenger / corroboration for post-correction reactivation tests, not cross-layer unlearning.

**Mechanism and evidence:** Prefix-expanded adversarial reinforcement learning trains across multiple contexts that might reactivate an unwanted learned behavior. On narrow-trigger Qwen3-4B, reported target misbehavior 1.6% versus 74.8% for a GRPO baseline without warmup; GRPO **with** warmup also reaches 1.3%, so this is not unequivocally better. Broad-trigger GPT-OSS-20B results maintain low failure rates under selected later fine-tuning. Results depend on synthetic organisms, evaluation prompts, RL data and reactivation schedules; many rollouts and GPU hours.

**A.L.I.C.E. already/new/action:** Existing correction, revoked-state, deletion and influence-echo contracts require later behavioral stability. Novel narrower test design for reappearance after further training. Suppression is not proven erasure and cannot replace canonical deletion, train-set removal, or retain-only comparisons. Only consider for a measured post-correction reactivation failure in a personal model. No current model change.

## Narrower screened, no current decision

- *Personas Are Not People: Identity as State, Not Attribute*, arXiv:2611.05292v1, November 3, https://arxiv.org/html/2611.05292v1. **Already covered/corroboration:** state-dependent identity is already an explicit host/source-person/self/relationship contract. Theoretical and limited ecological evidence does not create a new identity plane or prove existing learning.
- *Meta²: Precomputing Reusable Knowledge for Online Harness Evolution*, arXiv:2611.05139v1, November 3, https://arxiv.org/html/2611.05139v1. **Future conditional harness challenger:** precompute reusable adaptation knowledge, cost and interference trade-offs; prior CORAL/HarnessMM already tracked. No actual current harness blocker or reason for new qualification.
- *StateGraph*, *MemHarmBench*, *LEDGER*, *MemUse* and other earlier tracked papers were discovered again and excluded as repeats; no new revised evidence changing their decisions.

## Execution decision

No finding establishes a current defective A.L.I.C.E. contract, invalid Claim/Experience separation, unexpressible capability, or a new runtime blocker. No infrastructure/backend substitution or exhaustive comparative benchmark. Future-only candidates above do **not** block the selected native MFM checkpoint-26 assessment, actual observed-world formation extension, or native identity/conditional-judgment qualification. Any future attempt to act on a finding must preserve owner authority, source-person/host/self/relationship separation, source-native evidence, bitemporal Claims, deletion/correction/revocation lineage, and rollback. Internal F4–F11 milestones remain full-foundation qualifications; first consumer release after F11 only. No PR and no production/main mutation in this intake.
