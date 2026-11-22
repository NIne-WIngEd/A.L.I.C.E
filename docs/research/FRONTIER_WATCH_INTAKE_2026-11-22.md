# Frontier Watch intake — 2026-11-22

**Disposition:** Six previously untracked primary papers submitted **Thursday, November 19, 2026**, discovered after the November 21 watch intake. Five are **future challengers** and one is **already-covered/corroboration with a concrete conditional diagnostic**. No established current MFM/N0 repair, architecture-invalidating result, physical-stack blocker, new release tier, or production migration. Nothing here changes the selected critical path.

## Baseline verified before evaluating research

Verified fresh GitHub branches: A.L.I.C.E. `main@a1fe0a65df1d4143955eb1d3b35eb4ac590e7555`; Fable_Sleight `main@1c28eb3c989105d6a3d205c06cd5e6e8634ad77b`; `research/frontier-watch@7faaaf921923f618e58ba151fe44dc9c62d4fc16`; MFM `research/mfm-v1-licensed-base-20260930@5653736f545c9c1c8d5d8006af0f1fa152eed9b1`; FBM `research/fbm-v1-licensed-base-20260930@bcd462b5bb93f41a56f79b5b63e3337c8e89bb45`; N0 `alice-eipm-v1-n0-full-envelope-joint-ddp-v2@5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2`; `alice-context@945245bbee4168fff46f863455a83bb219063017`; `research/graphify-context-substrate@a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15`. No head changed since the November 21 note.

Read `docs/research/FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`, the watch execution guardrail and November 20–21 intakes; A.L.I.C.E. main README, full Phase 2/Fable execution plan and Personal Development Architecture; Fable main README, first-release specification and v1 execution profile; active MFM checkpoint-10 diagnostic/joint generation documents, continuity frontiers, branch directory and planned lanes. Historical vendor choices in Fable profile are superseded by current ALICE selected authority.

**Current engineering observation:** Last qualified checkpoint-10 native MFM formation **0/9**: eight copy-alignment failures and one parse failure, EOS 24–37 tokens, not context-budget exhaustion. The selected sixteen complete-nine-update continuation toward checkpoint 26 has no newer qualified generated result. Native personal conditional judgment likewise remains unqualified. These new papers do not demonstrate repairs to those failures.

**Authority and destination:** Full transferable Fable v1 cognitive foundation; Experience/evidence and bitemporal Claim Authority are separate; derived graph/vector/parametric/activation/cognitive projections are reconstructable and permission-bound. Host, source person, assistant self and relationship must remain distinct. Selected KurrentDB Experience, NATS JetStream edge/federation transport, XTDB v2 Claims, encrypted content-addressed evidence, LadybugDB local/NebulaGraph private scale-out graphs, Neo4j reference, engine-independent associations, Qdrant multimodal, source-native reads, Temporal, L1/Valkey and PyTorch+Accelerate are not reopened. F4–F11 remain internal gates, first consumer predicate is `full_personal_cognitive_foundation_after_f11`. Hardware adapts placement, not cognition.

## 1. DriftGuard — action-conditioned memory poisoning / drift risk from offline interventions

**Primary:** Pan et al., *DriftGuard: Online Memory-Drift Detection via Causal Attribution for LLM Agents*, arXiv:2611.25583v1, submitted November 19, 2026. https://arxiv.org/html/2611.25583v1

**Classification:** **future challenger** for Retrieval Orchestrator, governed Recollection and Mission execution, not a demonstrated current defect.

**Method:** From the same saved pre-action state, perform repeated offline paired suffix rollouts with and without one candidate memory (neutralized), aggregating four comparisons: action changed, attack removed, policy violation removed, terminal task recovery. Train a compact online risk model over task, recalled memory and current execution context to predict those intervention outcomes without repeating costly tool/agent rollouts during serving. Risky memories can be selectively rewritten at inference, but the method does not guarantee rewrites are correct.

**Evidence and ablations:** Under GPT-5.6 Luna, memory-poisoning detection AUPRC on MINJA/AgentPoison/MPBench **0.4993 / 0.6139 / 0.2415**; MPBench best compared Claude Opus 5 few-shot **0.1297**. Shuffling execution history reduces MPBench AUPRC **0.2415→0.0948**; deleting history gives **0.1423**. In the online attack defense study, example Mem2ActBench attack success drops **46.7→5.3%** for Direct Override and **77.4→3.3%** for Sleeper Memory. Tested 100-task added-runtime comparison: **1.52 s and 986 extra tokens per task**, vs an LLM memory auditor **8.53 s, 3127 tokens**; rewrite is ~80% of DriftGuard latency. Actual offline replay cost can be large and grows with candidate memories and repeated trajectories. Synthetic attacks, environment randomness, context shift, false positives, and a predicted risk score limit inference: online prediction is **not** a verified individual causal effect, and content rewrite is not an authoritative Claim correction or deletion.

**Already/new/impact:** ALICE already has source/permission-bound retrieval, Experience outcomes, trust/reliability projections, postcondition checks and correction/revoke lineage. New is *offline counterfactual supervision of memory's action-level harm with a cheap action-conditioned runtime estimator*, beyond static malicious-content scanning. Could detect harmful but superficially relevant procedural memories. **Future-only**; no current memory-security implementation repair established.

**Governance/action:** Never let a risk estimator silently mutate original evidence, claim adjudications, owner directives, tool permissions or training lineage. Any guarded output is a reversible derived read projection tied to source versions, host/speaker identity, current state, and revocation status; preserve audit and rollback, including missed attack cases and benign false-positive suppression. If real retrieved-memory-induced Mission misexecution is observed, compare intervention-derived risk for that exact failure with current provenance/permission and postcondition protections. No blanket guard, new model gate or backend change.

## 2. GLiDE — fast single-pass confidence-aware decision scoring

**Primary:** Allahyar et al., *GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers*, arXiv:2611.26509v1, submitted November 19, 2026. https://arxiv.org/html/2611.26509v1

**Classification:** **future challenger** for native decision-model mechanism and serving, not a qualified substitute for EIPM native personal judgment.

**Method:** Convert Qwen-derived hybrid transformer into option-scoring single-pass model, removing causal masks from full-attention layers, applying soft-target multi-teacher (including reasoning teacher) distillation and progressive training with a 255-option score head. Experiment with checkpoint weight averaging and *parameter-free recurrent depth*: rerun two intermediate layer blocks with damped fixed-point updates, adding compute but no parameters.

**Evidence/ablations:** The 27B V1.2 model reports Hugging Face Decision Index hidden score **71.52**, ECE **2.0%**; median **92.9 ms** on one RTX PRO 6000. Parameter average plus recurrent depth reaches **73.37**, ECE **2.1%**, **122 ms**. In a single-seed 125K-case ablation, dropping causal mask improves development score **60.42→61.78**; adding soft targets improves to **62.59** and held-out accuracy **89.16%**. Authors explicitly treat these as suggestive, not strong multi-seed evidence. Training lineage ~3.9B tokens from 3.4M questions; significant training/GPU cost. Calibration/latency benchmark is structured multiple-choice, not self/host/relationship-conditioned personal decisions or action policy safety.

**Already/new/impact:** EIPM N0 and future Fable personal-model judgment already own candidate comparison, uncertainty, provenance-linked verdict and reason/stance, with replaceable language generators. New candidate is distilling expensive reasoning into *non-generative calibrated selection* with potentially hardware-efficient serving. These benchmark results cannot qualify native learned identity judgment or justify dropping required expression/subject/relationship obligations. Future-only conditional on an actual local native decision fidelity/calibration/latency issue **after** model formation is proven.

**Governance/action:** Source and valid-time Claims still determine admissible evidence; verdict must contain identity and uncertainty/evidence obligations; confidence scores cannot authorize actions or override deletion. Treat model generations and training data as lineage-bound and revertible. No swap into current N0 or MFM.

## 3. AutoMaster — skill-guided teacher trajectories for model training

**Primary:** Chen et al., *AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher*, arXiv:2611.26510v1, submitted November 19, 2026. https://arxiv.org/html/2611.26510v1

**Classification:** **future challenger** for FBM teacher/data programs; not a current MFM repair.

**Method:** A controller diagnoses successful and failed teacher trajectories, designs natural-language skill instructions that shape how the *same fixed teacher* executes tasks, runs and evaluates alternative skills, then selects 512 trajectories for full-parameter SFT from same student start; skill context may remain in training demonstrations but absent at deployment. Experimental comparison holds model, teacher, 1000-task pool, dataset size and SFT recipe constant. Five outer-loop research rounds, three epochs SFT, significant parallel tool/teacher/GPU-cluster compute.

**Evidence/ablations:** Qwen3.5 4B/9B Terminal-Bench 2.1, BrowseComp Lite, Doc2Repo, six matched settings. Improvement over vanilla successful-trajectory SFT **+1.42 to +8.99 pp**. Terminal 9B: **26.59→35.58%**; Terminal 4B **22.47→28.09%**. Removing skill guidance or agent-directed data selection reduces result in all six; retained skill instructions at SFT also generally help despite absent at test. Yet gains vary by task and some ablations fall below vanilla SFT; 3 evaluation repeats do not establish source-grounded personal-learning generalization, and teacher instructions may create shortcuts.

**Already/new/impact:** FBM already has model/dataset build lineage, teacher evidence, source-bound training and independent qualification. New actionable future insight: task success alone is insufficient demonstration quality; skill-conditioned teacher rollouts can explicitly shape *how* a success was obtained. If a real FBM teaching-data failure arises, inspect trajectory-level behavior and consider a small comparison of skill-guided demonstrations to unchanged teacher on that family. Do not convert prompt compliance into personal-model learning proof.

**Governance/cost/action:** Teacher-generated demonstrations must remain attached to original tasks, source scopes, model/checkpoint/version, skills, reward evidence, rights, dataset retention/deletion and rollback; no host/source-person private identity used in general Fable products. No new training campaign or current recipe edit.

## 4. Seer — speculative tool-outcome planning without fake canonical experience

**Primary:** Zhou et al., *Seer: Guiding Small Deep-Research Agents with Shared Decision Models*, arXiv:2611.26423v1, submitted November 19, 2026. https://arxiv.org/html/2611.26423v1

**Classification:** **future challenger** for local research tool choice / Mission Graph inference-time planning and economic intelligence.

**Method:** From real search/visit tool-call pairs train Qwen3.5-9B tool-response simulator; from successful/failed complete real traces train independent 9B value estimator on discounted terminal correctness. At runtime propose up to 4–32 candidate actions, simulate possible responses, score imagined states via bounded depth-one tree search, execute *only chosen action* through real tool, commit only its actual response to Experience. Both models have LoRA training and additional runtime inference/token/VRAM cost; online speculative alternatives do not cost extra real tool calls.

**Evidence/ablations:** On GAIA Qwen3.5-4B accuracy **13.6→41.7%** and 9B **22.3→50.5%** with candidate width 32. Gains across 26 of 30 smaller-agent model/benchmark settings; larger widths sometimes hurt. Important ablation: replacing prompt evaluation with learned estimator generally improves; adding imperfect simulated responses can **decrease** accuracy at only width 4, so benefit is conditional on candidate coverage. Search/visit extraction includes shared giant 397B model in all comparison arms; compute/cost figures are not self-contained local deployment proof. Authors explicitly say tool calls saved versus increased inference cost and do not optimize context length.

**Already/new/impact:** ALICE already has Mission Graph, grounded live web/source reads, world state, outcome ledger and model routing. New candidate is a two-model *reusable shared, off-policy imagined branch scorer* that might improve small local research agents when live web access is expensive. No requirement for specific model counts or fixed candidate limits in Fable.

**Governance/action:** Hypothetical tool responses are simulations, never canonical Experience, source citations, Claim Authority, permissions or owner actions. Only actual tool results can enter source-grounded memory. If a real local-research action error or costly exploration pattern emerges, assess on the observed task and measured end-to-end cost. Never let speculative web content change a personal claim or authorization.

## 5. BiSpace — efficient long-sequence fact edits using separate semantic lookup and sparse latent injection

**Primary:** Che et al., *Decoupled Knowledge Retrieval and Compact Memory Enable Reliable Lifelong Model Editing*, arXiv:2611.25861v1, submitted November 19, 2026. https://arxiv.org/html/2611.25861v1

**Classification:** **future challenger** for external/activation/parametric scheduling and efficient governed personal update placement; **not** model-level unlearning nor native formation.

**Method:** Encode an edit query in external semantic space (Jina v5 Nano); robust retrieval through modern Hopfield-style association; store sparse masked hidden-layer value updates with indices and inject only after selected edit retrieval. Base LLM weights remain frozen. The external key and internal value are deliberately separate, which reduces vector/latent storage, not all system storage.

**Evidence/ablations:** ZsRE, CounterFact and WikiBigEdit with Llama3.1-8B, Qwen2.5-7B and Mistral-7B. Up to **16,000** edits; WikiBigEdit overall performance in a reported 16k comparison ~**93.47** vs HoReN **70.00** for one backbone. Per edit **4.5 KiB vs 44 KiB** (~10.2%), but the separate encoder has a fixed memory cost: reported total-storage crossover only near **14.7–14.8k** edits. Edit latency **1.39 s vs HoReN 0.98 s** at 5k; similar generation throughput. Hopfield retrieval improves one CounterFact setting **85.17 vs cosine 72.47**; random sparse dimensions can compete with gradient-selected ones when true edit is given. H100 training/evaluation cost and fact-edit benchmarks do not establish personal-values judgment, cross-layer deletion, identity safety or conceptual lifelong learning.

**Already/new/impact:** Fable's MRM, Qdrant, activation memory and governed adapters already cover distinct forms. New concrete serving technique for a measured long-sequence fact-update storage/fidelity problem, not a new cognitive plane. External latent patches are derived generations needing source/dependency attribution, correction re-index, validity filtering, revocation, cache invalidation and rollback. No current implementation action; no replacement of XTDB authority or Qdrant selected vector plane.

## 6. CompAct-Bench — downstream completion reveals evidence/state distortion from compression

**Primary:** Chen et al., *CompAct-Bench: A Benchmark for Working-Context Compaction in Long-Horizon Agent Tasks*, arXiv:2611.25673v1, submitted November 19, 2026. https://arxiv.org/html/2611.25673v1

**Classification:** **already covered/corroboration**, with a focused future diagnostic for Cognitive Workspace.

**Method:** Prescreen 200 agent tasks that the same executor solves three times before compaction (79 BrowseComp, 96 SWE-bench, 25 internal CompanyBench). At mid-run boundary, turn prior history into a target ~10%-length summary; restore environment state; resume the same executor under a matched remaining action budget. Three independent continuation runs. This tests whether a summary supports actual future execution, not only whether it semantically resembles source history.

**Evidence and limits:** 12 compactor models, mean post-compaction task completion **56.7%**, best **68.5%**, despite all original tasks pass^3. Two diagnosed failures: omitted necessary evidence leads to redundant tool use/budget exhaustion, **distorted task epistemic status** makes unverified ideas look concluded, triggering premature termination. An observational SWE ablation finds when a real code edit is already materialized, post-compaction success is **86.9%** versus **53.0%** without edits, showing live environment may carry state external to prompt. Caveats: prescreened trajectories, fixed powerful executors, one approximate retention target, private internal CompanyBench, and simulation of resume/replay cost; not proof that all Fable Missions fail at these rates. Costs dominated by long task executions and repeated trials; do not replicate full benchmark.

**Already/new/impact:** Prior Folio, Semantic Context Handoff, GrepMem, source-native and Cognitive Workspace already establish recoverable original logs. Useful *new diagnostic distinction*: compressed summaries may retain a fact while wrongly elevating its **verification / authority status**. A derived summary must preserve source IDs, unresolved hypotheses, terminal action state, environmental version, permission/deletion status and check receipt; never promote to Claims. Trigger only if a real Mission is prematurely closed or rework appears after compaction; perform one matched before/after replay to identify omitted vs distorted evidence. No new context backend.

## Screening exclusions / delta control

Previously tracked papers are not new: MemCalib, Jev-Mem, RPMem, JitMem, REALM, AdaMM, MUTE, PGMem, Recuris, FRESH and the daily intake through November 21 (e.g., LinEra, CLAM, Experience Rot, MindMem, MemCredit, Habit(us)Mem, EgoWitness, Some Memories Must Fade, MemFolder, GrepMem). Additional papers COVERAGE-TRACE and MemMend were screened as narrow summary/QA corroborations without a distinct current engineering action.

## Final implementation disposition

**Six genuine new research references.** All future-only or corroboration, none provides evidence of an existing A.L.I.C.E. model/contract defect or a selected physical-stack failure. Do not create architecture/prod changes, extra qualification gates, new training recipe, or a research infrastructure tournament. Complete native MFM formation from the presently selected source-bound continuation, then independently qualify native learned personal judgment. Treat research note as a future evidence register; preserve owner sovereignty and exact correction/deletion/rollback lineage. No PR requested.
