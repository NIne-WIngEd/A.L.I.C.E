# Frontier Watch intake — 2026-10-28

**Decision:** Seven distinct new or newly surfaced research items were assessed after verifying current main/active branch heads and the mandatory full-Fable-v1 baseline. Three are particularly useful: proactive-recall false alarms grow with memory size; task-evidence completeness and context contamination are different from ordinary top-k relevance; learning a constraint state is not the same as obeying its revocations. All observations fit the selected cognitive architecture. **No architecture-invalidating evidence, current production semantics change, infrastructure tournament, training-recipe diversion, or partial v1 release.**

## Mandatory comparison baseline and actual execution boundary

Read `docs/research/FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md` and `FRONTIER_WATCH_EXECUTION_GUARDRAIL_2026-09-27.md` on `research/frontier-watch`; the complete October 27 intake and recent October 24–26 findings; A.L.I.C.E. and Fable main READMEs; Fable `docs/FIRST_RELEASE.md` and the newer complete-v1 execution profile; current MFM checkpoint-10 diagnosis/observed-world plan, FBM continuation, personality conditional-learning note and FloRA continuity.

| Repository / stream | Verified head on 2026-10-28 | Delta |
| --- | --- | --- |
| A.L.I.C.E. main | `a1fe0a65df1d4143955eb1d3b35eb4ac590e7555` | unchanged |
| Fable main | `f8e9062ebd657f18c3a571cedc1206889d7fd53a` | unchanged |
| Fable current execution-profile documentation | `Rayan/fable-current-stack-docs-20261010@35acb38c90a7e3b31ddfc149fe2cfebd8b2c6c6b` | unchanged |
| MFM | `research/mfm-v1-licensed-base-20260930@5653736f545c9c1c8d5d8006af0f1fa152eed9b1` | unchanged |
| FBM | `research/fbm-v1-licensed-base-20260930@bcd462b5bb93f41a56f79b5b63e3337c8e89bb45` | unchanged |
| Graphify/context | `research/graphify-context-substrate@a06f61ea`; `alice-context@945245bb` | unchanged |
| Personality / FloRA | personality evidence `af89ba0fa9704cd21dcfa616cfc97180d46cf7e7`; FloRA continuity `fc6d82df8e4050ed5f0f73fe43d84b1e35463830` | no newer qualified result observed |
| Frontier-watch parent | `c8d9cb173f0e0c06174944b01f6e0bd1b67b5cea` | October 27 intake |

**Real execution boundary:** MFM checkpoint-10 native assessment failed 9/9 full source-grounded decisions (0/9 passed) despite better teacher-forced reference/copy/lexical losses; a continuation of 16 complete-nine updates toward checkpoint 26 had been dispatched but no newer branch evidence establishes its terminal state or learned native formation. Rill source/teaching material is constructed, not trained. N2 personality conditional-judgment still has a recorded 25/80 and 0/40 paired-correct gap with a later experiment under test; no newer verified learning success. No paper below warrants altering those current operations.

**Architecture authority:** Full transferable Fable v1 with canonical raw evidence/Experience/Event and bitemporal Claims, episodes, Cognitive Multi-Graph and associative graph compute, vector/multimodal and source-native retrieval, perceptual/host/source-person/relationship/self/world/social/causal/skill/Mission state, working/activation/parametric memory, MRM and Cognitive Recollection, Lifecycle Curator, durable workflows, multi-device/federation, complete deletion/influence-echo/rollback and governed personal model evolution. Selected defaults remain KurrentDB + JetStream, XTDB v2, encrypted content-addressed evidence, LadybugDB/NebulaGraph, Qdrant, Temporal, L1/Valkey and PyTorch+Accelerate. Hardware changes placement, never logical planes. F4–F11 internal only; first release `full_personal_cognitive_foundation_after_f11`.

## 1. The Price of Scale in Proactive Memory Recall — multiple-testing false-alarm law

**Primary:** [arXiv:2610.34507](https://arxiv.org/html/2610.34507v1), originally submitted **2026-10-25**, surfaced in this watch. **Classification: direct future proactive-Mission/MRM qualification improvement** (not current architecture defect).

**Mechanism:** When one proactive interruption is triggered by the maximum similarity over N stored memories, even unrelated cues produce higher maximum similarity as N grows. To hold a store-level false-alarm rate alpha, individual comparisons must operate at approximately `-ln(1-alpha)/N` false positives; the study derives the resulting hit-rate decay under stated score/independence/ROC assumptions. Evaluates cosine, lexical, normalized and learned gates plus small LLM judges in LongMemEval/LoCoMo streaming replays. A label-free cue-swap conformal mechanism calibrates the current cue against 199 other user cues on the same store; its finite-sample guarantee requires exchangeability.

**Evidence and ablations:** Threshold calibrated at N=8 to 5% false alarm grows to **87–100% by N=2048** for several static similarity gates. Holding false alarms near 5% with an oracle global cosine threshold lowers explicit bge-small hit rate from **70% (N=8) to 8% (N=2048)**. Per-cue normalization plus cue-swap at N=2048 gets **22% hits/5% false alarms** for explicit cues on bge-small, but only **6% hits/3% false alarms** on implicit cues; the 14B judge gives 28% hits/4% false alarms on one LongMemEval setting at higher inference cost. A blind sampled annotation finds about nine in ten false alarms are nonhelpful interruptions (moderate agreement, Fleiss kappa .63). These are tests of *proactive relevance*, not correctness/authorization of a Mission.

**Existing A.L.I.C.E. coverage/new delta:** Multi-plane MRM, contextual activation, Mission planning and perceptual/prospective memories already exist in target contracts. Newly demonstrated is that a fixed interruption threshold silently degrades with memory-store growth, even when retrieval rankings themselves remain useful. A user life is not a static-size index.

**Authority/identity/deletion:** Calibrate *which candidate deserves attention*, not what is true or permissible. Before proactive action, independently check active Claims, temporal validity, source-person/host/assistant identity, user permissions, previous completion and revocations. Deleted/retracted data must leave both active index and calibration inputs as required by lineage; calibration state must be rebuildable/versioned. Protect owner-controlled privacy: do not send private cue histories to outside calibration services by default.

**Costs/limits/evidence strength:** Strong mathematical explanation and controlled replay, with synthetic implicit cues, mixed simulated histories, English only and correlation/exchangeability failures (notably LoCoMo). Cue-swap costs ~199 comparator scores per calibration; LLM judges require extra inference. Real-user interruption harm not established.

**Concrete action:** When proactive retrieval is active, one size-stratified false-alarm/hit-rate test and an owner-allowed abstention/interrupt budget; no full benchmark reproduction, memory-size ceiling, or current MFM change.

## 2. Agent Memory as Recommendation / AgentMemRec — evidence-complete slates, not maximal similarity

**Primary:** [arXiv:2610.34673](https://arxiv.org/html/2610.34673v1), submitted **2026-10-26**. **Classification: future Cognitive Recollection/MRM policy improvement**.

**Mechanism:** Model each query as a constrained recommendation to select a *slate* of supporting evidence. Candidate generation combines lexical and embedding retrieval, a lightweight LambdaRank ranks items using context-aware features, and a budgeted context builder chooses turns without a generative LLM call at read time. Analysis factorizes candidate pool completeness, ranking completeness and answer fidelity; optional write-time records exist but the principal no-LLM variant uses raw source turns.

**Experiments/controls:** In the pre-fixed LongMemEval-S test subset needing >=2 evidence sessions (n=209), Qwen3-32B reaches **80.7%** and DeepSeek V4.1 Flash **89.4%**, vs best comparable memory 70.3% and 73.2% and a 64-LLM-call reranker 79.8%/89.1%. In LongMemEval-M (500 sessions, ~1.5M tokens), results are **70.0% / 81.3%**; 3.2x lower measured serving cost than 64-call reranker at broadly comparable accuracy, not a claim of zero total compute. Slate completeness is **0.940** for >=2 sessions but falls to **0.832** at >=3. Removing turn-context features costs 13.4 points completeness at >=2 and 31.1 at >=3; ranker improves evidential coverage. **Critical negative control:** adding ranked near-miss filler drops Qwen accuracy by **17.7 points**, by **39.4 points** for >=3 sessions; more context can worsen answers. Published/copied baselines and evaluator/provider effects warrant caution; raw-turn benchmark retrieval does not qualify personal cognition, multimodality or real-time source updates.

**Already covered/genuinely new:** Fable already has source-native retrieval, Cognitive Multi-Graph, associative recollection, semantic/sparse retrieval and MRM. This offers a concrete evidence-completeness and context-contamination decomposition, not a new database/graph store or replacement for structured memory. Different models have different tolerance for distractors; always filling the context budget would be wrong.

**Authority/provenance/deletion:** Every selected item retains original source/time/subject/authority metadata, is eligible under current permissions, and remains a provisional workspace input rather than an authoritative Claim. Nonauthoritative ranking features, cached slate and trained ranker must be invalidated/rebuilt after corrections or deletion; separation of host/source-person/assistant-self remains intact.

**Costs/risks/decision:** BM25, embeddings and CPU ranker plus labeled ranking training; lower marginal LLM-call cost but index/update costs and token contamination persist. In a demonstrated multi-evidence failure, measure pool gold coverage, complete slate coverage, answer accuracy, cost and distractor harm separately. No current data-plane or learning changes.

## 3. StateWise — learning a tombstone is not learning to obey it

**Primary:** [arXiv:2610.33766](https://arxiv.org/html/2610.33766v1), submitted **2026-10-24**, surfaced in Oct 27 announcement. **Classification: direct future native-judgment/Workspace revocation qualification improvement**, not a demonstrated current personality-model fix.

**Mechanism:** Represent evolving conversational instructions as typed constraint states with explicit tombstones for revoked rules. Stage I supervised training induces state and response tasks. Stage II self-trains with exact round-trip check over model-created trajectories. Stage III contrastive preference training uses examples of *enacting a revoked instruction* vs obeying its removal, balanced against continuing to obey still-active rules.

**Evidence and crucial negative ablation:** On Llama-3.1-8B, state prediction became almost perfect but pure supervised variants *roughly doubled stale-rule relapse*. With Stage III, relapse vs supervised self-training falls **8–22 percentage points**, conversation success improves **8–19 points** against base across tested settings/backbones. Llama seed-1 ablation: after self-training held-out relapse **0.246**, Stage III **0.094**. Label-shuffled training shows no gain; training only active-constraint contrasts worsens relapse to **0.284**. Benefits replicate on Qwen2.5-7B and paraphrases, but Llama seed-2 strict relapse-vs-base confidence interval reaches zero. **Important regression:** on unrelated IFEval, full adaptation drops strict instruction score from **0.725 to 0.660** (fails preregistered no-regression gate); interpolated weights partly reduce that loss. Training cost reported ~2 A100-40GB GPUs, 9.2k initial examples, 2,400 self-generated trajectories, 931 preference pairs, ~1h+68min+17min for three stages.

**A.L.I.C.E. implication:** Existing bitemporal Claim Authority, durable revocation lineage, Cognitive Workspace, native judgment and permission gates are already selected. The novel result is a demonstrated gap between *accurate state recollection* and *behavioral enforcement*, and evidence that targeted counterexample learning can correct it while risking general regressions. Complements the earlier State Integrity Collapse note rather than rediscovering it. The evidence is on formatting and constraint edits, not identity-safe personal relationships or full Mission authorization.

**Governance/action:** Never let inferred state or model weights overrule source-of-truth owner revocation; tombstones must propagate through workspace, cached context, skills, activation and model lineage with rollback/rebuild. When current native personal-judgment lane reaches appropriate relevance, test active-to-revoked-to-unrelated behavior and preservation of previously correct cases, anchored to the specific failing contract. No speculative retraining or changing running N2/MFM recipe.

## 4. TrawMem — role-adaptive complementary evidence inside a task workspace

**Primary:** [arXiv:2610.36918](https://arxiv.org/html/2610.36918v1), submitted **2026-10-27**. **Classification: future associative recollection / Cognitive Workspace challenger**.

**Method:** Source-ordered dialogue chunks grouped into event threads with LLM-generated thread heads, sequence edges and inter-thread semantic edges. At query time, select thread neighborhoods and assign task-specific soft roles: direct answer support, connecting/supporting facts, coverage. Greedy selection approximately maximizes a submodular objective of role coverage, relation coverage, query affinity and cross-thread coverage under a context budget, retaining original selected content and thread ordering.

**Experiments:** LoCoMo/MemGallery with 4 answer backbones (Qwen3.5-9B, Qwen3.8-27B, DeepSeek-V4-Flash, GPT-5.6-Luna). Macro F1 improvement 2.23–3.14 points, LoCoMo LLM-judge +4.07 and +4.31 over SimpleMem on Qwen9B/DeepSeek, using about 12.8%/5.1% fewer context tokens. Removing cross-thread selection causes the largest judge decline **1.86–2.62 points on LoCoMo and 1.91–3.13 on MemGallery**. Mean retrieved context about **2,216 tokens**. A failure case had a source event labeled merely 'last August' while benchmark demanded exact year: retrieved source *cannot establish* exact year.

**Existing/new/limits:** Multi-Graph, episodes and Recollection already support thread/association/cross-event workspace. New item is query-dependent support *roles* and explicit complementarity, distinct from maximizing relevance. Limited to benchmark conversational QA and judge-based scoring; LLM thread summarization costs, incorrect joins and missing original context matter.

**Governance/action:** Roles/edges remain derived, inspectable and deletable projections, never independent Claim truth; exact source spans, bitemporal validity and subject identity must be preserved. Consider role-aware selection only when actual multi-thread recollection misses necessary auxiliary support; no graph-engine migration or fixed role count.

## 5. ETAMem — event, affect, time as complementary retrieval views

**Primary:** [arXiv:2610.37443](https://arxiv.org/html/2610.37443v1), submitted **2026-10-27**. **Classification: future personal/social/recollection improvement**.

**Mechanism:** Generatively extract atomic memory units, classify factual/affective polarity and encode with type-aware semantic/affective embeddings; organize event, affect and time into complementary hierarchies; choose query-specific view budgets from retrieval-score entropy and perform combined lexical/dense beam search. If extracted memories prove insufficient, return to source dialogue.

**Evidence/ablation:** **84.68%** LoCoMo and **83.00%** LongMemEval-S in its reported evaluation, vs 82.14/80.60 without atomic units; 82.86/80.60 without tree; **84.48/74.20 without original-turn fallback**, an 8.8-point loss on LongMemEval-S that reinforces source-native retrieval. An alternative unified semantic organization scores 83.31/80.49. Qwen30B routing: three-view entropy 80.40 LongMemEval vs mean 75.00 or maximum 64.20. CloneMem experience/emotion/opinion accuracy 92.59% vs strongest listed 87.74%. Construction still costs **~63.41k tokens/session**, despite comparisons reporting savings over heavier baselines.

**A.L.I.C.E. coverage and risks:** Distinct host/source-person/self/relationship, temporal Claim/Episode/graph and perceptual social memory are already required, and views are not new cognitive planes. Affect labels are machine-inferred projections; sentiment polarity is not a person's enduring emotion, consent, preference, identity or authoritative approval. Summarization may drop exceptions and amplify false certainty. The number and content of memory views must remain open-ended. Costs: LLM formation, affect and semantic embedding models, hierarchical maintenance, retrieval.

**Action:** Future source-grounded affect and chronology retrieval test when a real personal-state gap appears. Do not add a new store or replace MFM.

## 6. ConMem — counterfactual probes of useful vs harmful memory influence

**Primary:** [arXiv:2610.33432](https://arxiv.org/html/2610.33432v1), submitted **2026-10-24**, newly screened. **Classification: future Lifecycle Curator evaluation challenger / corroboration of existing MemCalib, PGMem and authority scope**.

**Mechanism:** MemAlign: 1,079 synthetic multi-session cases with candidate memories that can be overgeneralized, stale, misattributed or sensitive. For each candidate, generate future tasks including relevant use, irrelevant invariance and privacy boundary conditions. Produce control/treatment responses, evaluate quality and audit whether memory actually caused the improvement/harm, then score a STORE/DO-NOT-STORE decision.

**Evidence/negative result:** Qwen3.8-Max implementation **75.16% accuracy, 64.15% balanced accuracy, 65.31% macro F1**; remove attribution auditor → **55.05/55.55/52.41**, demonstrating counterfactual quality judgment is insufficient. However, its rejection recall for *context misapplication* only **12.90%**, for *incomplete use* **16.67%**, for identity mis-inference **27.27%** and privacy overuse **27.27%**. Also **14 LLM calls per candidate** plus tie-breaking, so expensive and unreliable as an always-on write gate. Synthetic judges and scenarios do not establish causal effects in real life.

**A.L.I.C.E. action:** If actual Lifecycle Curator misapplication/over-retention defects appear, use a small contrasting-future evaluation to probe whether a *derived recollection* should be promoted/activated. **Never** turn counterfactual scores into authority to erase raw evidence or override explicit authorized correction/deletion; deletion remains governed and multi-layer. Do not block current MFM.

## 7. InferPriv — inferential egress remains after direct identifiers vanish

**Primary:** [arXiv:2610.34640](https://arxiv.org/html/2610.34640v1), submitted **2026-10-25**. **Classification: already covered/corroboration of Fable's first-release egress contract, with useful future evaluation design**.

**Method:** An honest-but-curious third-party recipient infers private user attributes from routine outward replies. Evaluate with and without sensitive memory, adjust for the base-rate inference control, measure top-k inference across multiple possible outputs alongside explicit disclosure and task completion. Five frontier models and privacy-aware prompts.

**Results:** Privacy-aware GPT-6.1-Sol direct leakage **0%**, yet single-guess multi-sample inferential leakage **21.0%** for default evidence records; suppressing explicit attribute names yields **20.6%**, nearly unchanged. Averaged across privacy-protected leading models, reported inferential leakage remains in the **16.2–25.2%** range despite nearly zero direct quotes. Increasing reasoning effort alone does not fix inference. Model-generated recipient inference and templated tasks limit real-world precision.

**A.L.I.C.E. implication:** Fable `docs/FIRST_RELEASE.md` already requires protection from sensitive *meaning* in outgoing text and cumulative inference across calls; this is not a new plane. When building egress, retain task-minimum disclosure, destination/purpose/payload consent, source permissions and recipient-specific risks. Test paired output without sensitive evidence and inspect what recipient can infer, not only literal redaction. No current implementation diversion.

## Screened but not promoted as new

- **Google 'sleep framework' blog (2026-10-27):** underlying paper arXiv:2606.03979 was available months earlier, and consolidation/forgetting challengers Cortex, HERD, MemVerse already cover the conceptual space. A new blog is not new experimental evidence.
- **SABench, CoSA, SkillLifeBench:** authorization and skill composition risks overlap previously tracked AuthorizationBench, I-HITL and procedural governance. Retain as context, not proof of current Mission implementation defect.
- **Context cache/prompt compaction studies:** potentially important *only with a measured serving workload bottleneck*; do not launch backend/model/harness tournaments.
- Previously tracked MemCalib, Jev-Mem/Jev-DLC, RPMem, JitMem, REALM, AdaMM, MUTE, PGMem, Recuris, FRESH, State Integrity Collapse, QEMem, ActionRAG, Memory Tax, ProPerAgent/ProMemAgent and I-HITL were not relabeled.

## Final action boundary

1. **No architecture-invalidating evidence.** Selected stores and full cognitive planes unchanged; no release segmentation.
2. **No current MFM/personality/FBM model recipe alteration.** Remain on evidence-gathering for native formation and conditional-judgment learning. Check actual terminal and native outputs, not a previously dispatched status.
3. **Future exact decisions:** proactive recall should calibrate false alarms against store size and cue distribution; actual multi-evidence failures should distinguish pool coverage, slate completeness, distractor harm and source-native fallback; Workspace/Claim integration should separately qualify remembering and obeying revocations; affect, privacy and authority projections must preserve subject/source lineage.
4. Do not schedule large benchmark campaigns until a real current capability failure demands a choice; no irreversible authority migration, deletion or model promotion.
