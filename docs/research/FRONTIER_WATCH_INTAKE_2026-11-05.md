# Frontier Watch intake — 2026-11-05

**Disposition:** Six new/newly surfaced substantive findings (Reglos, Route to Remember, Pro-Memory, LExMem/LExGraph, SumOne, rate-limited filtering) plus one conditional serving challenger (SEED). Screened FlyHash and the November 4 eLife version of the older complementary-learning-system study without reopening architecture. **No architecture-invalidating finding, no demonstrated repair for current MFM native formation or personality conditional judgment, no physical-stack reconsideration, no main/production change.** The first Fable consumer release remains `full_personal_cognitive_foundation_after_f11`.

## Mandatory repository-first comparison (before literature screening)

Verified branch heads November 5:
- A.L.I.C.E. `main@a1fe0a65df1d4143955eb1d3b35eb4ac590e7555`.
- Fable_Sleight `main@1c28eb3c989105d6a3d205c06cd5e6e8634ad77b`. PR #7 was already reported November 3; no newer merged head.
- MFM `research/mfm-v1-licensed-base-20260930@5653736f545c9c1c8d5d8006af0f1fa152eed9b1`.
- FBM `research/fbm-v1-licensed-base-20260930@bcd462b5bb93f41a56f79b5b63e3337c8e89bb45`.
- N0 `alice-eipm-v1-n0-full-envelope-joint-ddp-v2@5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2`.
- Context `alice-context@945245bbee4168fff46f863455a83bb219063017`; graphify continuity `research/graphify-context-substrate@a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15`.
- Fable execution-profile branch `Rayan/fable-current-stack-docs-20261010@35acb38c90a7e3b31ddfc149fe2cfebd8b2c6c6b`.
- Frontier-watch previous head `eeebc2ad801e1e990f4af49ace988db8dfdffbce`.
- Inspected the mandatory `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`, execution guardrail, November 3 and November 4 intakes, A.L.I.C.E. main's canonical Phase-2 replacement/Fable-v1 execution plan, Fable main README and FIRST_RELEASE, personal-development architecture, MFM input-construction plans and current continuity direction.
- Existing committed MFM checkpoint-10 evidence remains **0/9 valid complete source-grounded native formations**; the selected 16-update continuation toward checkpoint 26 has no newer committed terminal assessment on the checked head. Rill nine-cut observed-world inputs are evidence of construction, not proven learned formation. Personality native conditional judgment remains unqualified. Literature must not substitute for those missing model results.

**Selected physical direction is unchanged:** KurrentDB canonical Experience/Event behind first-party contracts; JetStream federation ingress; XTDB v2 Claim Authority; encrypted CAS evidence; LadybugDB local and NebulaGraph private-cluster graph; engine-independent associative compute; Qdrant multimodal; source-native/live reads; Temporal; L1/Valkey; content-addressed model lineage; PyTorch/Accelerate training. Fable's older September 27 execution-profile document has historical placements; newer selected direction is authoritative. Hardware changes placement, never capability. Internal F4–F11 are qualification milestones, not smaller releases.

## 1. Reglos — two different evolution loops, held-out harness improvement and explicit preference lineage

**Primary:** Sen Yang et al., *Reglos: A Self-Evolving Personal Memory Agent with Recursive Self-Improvement*, arXiv:2611.06049v1, submitted November 4, 2026. https://arxiv.org/html/2611.06049

**Class:** Future challenger for memory-write/read harness evolution and personal preference validity; corroborates bitemporal provenance.

**Mechanism:** One online loop extracts per-user facts and commitments, compares new/replacement/enrichment, retains append-only source-referenced validity and learned-at dates, and periodically rebuilds current profile. Retrieval fuses vector, keyword, graph and temporal/place/recency routes; the policy distinguishes current from superseded/forgotten. A second *offline* loop uses a coding agent proposing harness patches from failure traces, a different-model-family critic rejecting benchmark gaming, and a frozen two-part persona-disjoint evaluator. Evaluation holds the memory engine/store and model fixed when comparing harness patches. User memory and shared harness have different update scopes.

**Experiments and ablations:** Qwen3.8-27B writes memory; with Qwen answering, PersonaMem-v2 **61.3%** (best listed competing MemOS **53.8%**) and LongMemEval-S **84.0%** (EverOS **80.4%**); with GPT-5.6 Luna answering, **57.5%** and **87.2%**. Harness on its 1,317-question validation improves **54.2→60.7%** over accepted patches; full-option queries initially regressed until distinct-option wording was used. Raw retrieval remains a strong control. About **31%** of traced failures already had requisite evidence in context. Write effort per 32k-token persona history **225k–260k** task-model and **40k** embedding tokens; per-answer ~**6k** prompt tokens; ~**50M** writer tokens for 200 test personas. Long-history testing covers 40 personas rather than actual years.

**Existing A.L.I.C.E.:** bitemporal Claim authority, raw evidence, host/source-person separation, MFM candidate formation, Recollection, Lifecycle Curator, model/harness lineage and tests. **New:** concrete distinction between *per-host online memory change* and *cross-host offline harness evolution*, with persona-disjoint acceptance and measured examples of answer-option-specific retrieval and evidence-already-present failures. It provides a possible future method, not demonstrated native judgment or builder success.

**Compatibility and weaknesses:** Their "forget" mechanism records a tombstone and asks an LLM not to use a retained fact; this is **not** adequate for Fable's actual cross-layer deletion/erasure and influence-echo contracts. Their append-only user facts are not autonomous Claim Authority; source/person confusion remains the authors' hardest slice. Do not replace Claim authority with a single Postgres design, limit user memory to thousands of facts, or let a harness critic silently change permissions. Maintain versioned rollback, owner authorization and independent holdouts. Experiment cost high for exhaustive harness search.

**Action:** Future: when a specific Recollection or personal-answer harness failure emerges, require user-disjoint held-out benefit and no losses on revocation, source-person attribution or correction. No near-term harness search and no production edit.

## 2. Route to Remember — on-policy parametric memory, explicit verification and age-dependent routing

**Primary:** Zeyu Fang et al., *Route to Remember: Adaptive Memory Selection for LLM Agents*, arXiv:2611.05636v1, submitted November 3, surfaced November 5. https://arxiv.org/html/2611.05636

**Class:** Future direct capability challenger for MRM scheduling between external and parametric memory.

**Mechanism:** An LLM turns timestamped session memory points into paraphrase, counterfactual correction and unknown-answer QA examples. A per-user persistent LoRA student answers without context; a frozen base with recency-bounded text context scores *student-sampled* answers through on-policy self-distillation (OPSD). A verify-then-regenerate (VtR) reader retrieves both query- and **generated-answer**-anchored source candidates, checks the parametric claim against an updated merged text store (supported/conflicting/unsupported), and otherwise regenerates without adapter from current evidence.

**Evidence:** On HaluMem's **1,438 native questions** from **20 users** (109–136 sessions each), memory retrieval **47.8%**, extracted dump **51.9%**, plain SFT with identical data **36.9%**, OPSD **59.9%**, and VtR **66.1%**. Removing unknown-answer training drives a memory-boundary slice from ~**88% to 1%**; frozen teacher avoids ~**14-point** late collapse. Aged re-asks expose parametric degradation **59.9→32.4%** over 80 sessions while retrieval stays ~**47.8→40.1%**; retrieval overtakes weighted recall at older ages. Replay of stale answers harms dynamic updates. OPSD requires additional ~**76.5M input / 31.4M output** write-augmentation tokens beyond a common 30.2M/12M extraction; VtR adds verification. H200 used. Limitations: finite synthetic questions; old preferences and absent facts are failure cases, VtR quality depends on source store/correct verifier, unlearning never established.

**Existing A.L.I.C.E.:** parametric, external and activation planes and MRM; source-native/exact retrieval; bitemporal Claims, lineage and rollbacks. **New:** demonstrated training/verification recipe with explicit empirical *age and absence* asymmetry and answer-anchored retrieval. A raw weight prediction is **never** Claim Authority. Model confidence/similarity is not a substitute for current source validity.

**Action:** Future only, if a measured native personal-model generation supports reliable formation: compare a small actual authorized personal-state consolidation against its source-native and correction/absence controls; include revoked/deleted examples and model rollback. This is **not** a direct fix for the present MFM lexical native-formation failure and does not change its selected continuation.

## 3. Pro-Memory — proactive intention retention is separate from opportunity detection and actual enactment

**Primary:** *Prospective Memory for Proactive LLM Agents*, arXiv:2611.05731v1, submitted November 3; newly screened. https://arxiv.org/html/2611.05731

**Class:** Future direct Mission Graph/proactive capability improvement.

**Mechanism:** A standing future-directed intention retains original evidence and validity; a cue classifier decides whether a *new observation* enables this intention; selective enactment arbitrates with ongoing action, then records issued acts and retires one-shot obligations. Outcome experience may refine later triggers. Retaining an intention, observing a relevant cue, and executing are not conflated.

**Evidence/ablation:** On PM-Bench Claude Sonnet 4.5: Pro-Memory **0.760 Set-F1**, **0.100 false alarm**, vs RWM **0.7273** and **0.0875**; ProactiveBench F1 **0.7376** vs RWM **0.6609** and miss rate **0.0631** vs **0.2475**. For Haiku, removing explicit intention-evidence links lowers Set-F1 **8.45 points** and more than doubles false alarms; removing maintenance lowers by ~2 points; removing retirement increases repeated intervention. Text-only controlled tasks; *no real-world delegated execution or safety proof*. Cost: additional state reads/re-evaluation and LLM calls; sensitivity to cue accuracy and permissibility.

**Existing A.L.I.C.E.:** Mission Graph, prospective Mission nodes, Experience, Cognitive Workspace, permissions, durable workflow and MRM. **New:** an experimentally grounded division of *relevance, actionability, and issued-action lifecycle* for unsolicited help.

**Authority:** An inferred future desire is not owner authorization. Current Claim validity, participant identity, permission revocation, completed side-effect receipts and safe idempotency override cue scores; never retire a task merely because a tool was called if outcome confirmation is necessary. No fixed active-intention cap.

**Action:** Future qualification when a real proactive Mission lane is operational: one delayed authorized intention with an early irrelevant cue, one valid later cue, and one post-action no-repeat check. No present Mission stack change.

## 4. LExMem/LExGraph — recovering branches and revived hypotheses in long exploration

**Primary:** Chongyuan Dai et al., *Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks*, arXiv:2611.06414v1, November 4. https://arxiv.org/html/2611.06414

**Class:** Future Cognitive Multi-Graph/associative-recollection challenger and Mission-history evaluation.

**Mechanism:** Generate exploration graphs with support/refute/revise/revive branching, instantiate in research/software design, turn them into conversations. LExGraph itself, without the graph-generation oracle, segments actual dialogue turns, stores keyed summaries plus verbatim original turns, adds hierarchical and cross-branch edges, retrieves via BM25+dense RRF, expands neighbors and presents original evidence plus relationships to the answerer.

**Evidence:** LExMem includes **224 dialogues, 2,018 QA items** with dense/sparse/same-type relations. Overall ACC with Qwen3.8-27B **63.92** vs strongest relevant LightRAG **61.67**, vanilla **58.36**; GPT-6 Luna **64.00** vs LightRAG **63.46**. On external BEAM overall **0.472**, above compared alternatives (~0.436 next); evidence-count analysis shows retrieval degrades as dispersed support nodes grow. Compared with vanilla, LExGraph adds ~**1.2–1.3x** token cost and **1.1x** retrieval+answer latency, with extra indexing/edge-generation inference. Synthetic graph-derived QA, LLM judge and one seed; it measures recollection, **not improved actual exploration decisions**.

**Already:** A.L.I.C.E.'s Mission Graph, Experience causal branches, Cognitive Multi-Graph, derived topology, source-native and associative graph compute. **New:** exploration-specific tests for evidence across abandoned/revived branches where topical matching fails, not a novel canonical graph plane.

**Authority/action:** A revived hypothesis remains a hypothesis; branch edges are derived and rebuildable. Timestamp/source scopes and correction lineage must survive all links. When a real long-running research/software Mission cannot recover superseded/revived alternatives, test one actual failure against current graph plus exact-source retrieval. No LadybugDB/NebulaGraph replacement or benchmark tournament.

## 5. SumOne — training context compaction by masked reconstruction rather than task rewards

**Primary:** Jizhe Zhang et al., *SUMONE: A Simple and Efficient Summarizer for Plug-and-Play Agent Memory*, arXiv:2611.06445v1, November 4. https://arxiv.org/html/2611.06445

**Class:** Conditional future Cognitive Workspace / long-Mission compaction improvement.

**Mechanism:** A Qwen3.5-2B summarizer writes a compact version of expired interactions; a frozen Qwen3.5-9B reconstructs masked ~512-token trajectory blocks using the summary and retained context. A judge scores recovery/hallucination plus summary length; GRPO optimizes the summary writer without downstream task labels. Only the 2B summarizer runs at inference.

**Evidence/ablation:** ~5k web-search/software trajectories at 16k/32k; 600-step GRPO, 50% masking best among 25/50/75%. DeepSearchQA F1 **0.48 vs full 0.52**, input **591k vs 2.83M** tokens, estimated total cost **$0.038 vs $0.170**. SWE-bench Verified success **0.59 vs 0.63**, input **1.09M vs 4.1M**, cost **$0.071 vs $0.286** in main table (cost-saving figures vary by benchmark/settings; do not repeat headline 75% without its scope). LongMemEval-S compressed-memory score **0.47**, full-history **0.77**; summary-judge RL ablation **0.36**. Some cross-model and harness transfer; Codex harness gets **0.47**, the same as self-summary. Significant training judge/compute, 2B inference, and information-loss risk.

**Existing A.L.I.C.E.:** Workspace, Missions, canonical events and exact source recovery, MRM/context scheduler. **New:** one task-agnostic *training signal* for temporary context summaries, independently of Mission outcome; no new canonical memory type.

**Authority/action:** Summary is disposable, regenerable and subordinate to evidence/Claim scope. Never make summarization a substitute for retained evidence, revocation lineage, unknowns or mission receipts. Future only when real Mission context costs/quality are a blocker; validate a single affected workflow's accuracy/cost/rights after context reset, not generic compaction tournament.

## 6. Rate-limited filtering — a known-state rule can beat reading an entire history

**Primary:** Tianhao Guo and Chongyao Guo, *Agent Memory as Rate-Limited Filtering: Explicit State Outlasts Long Context Beyond a Reader-Dependent Horizon*, arXiv:2611.02361v1, November 2, newly surfaced. https://arxiv.org/html/2611.02361

**Class:** Already covered/corroboration, with a stronger future counterfactual-state validity test.

**Method:** Finite-state access-control benchmark with grants, revocations, re-grants, expiry, group roles and change windows; formal lower bound for **order-oblivious bounded extractive retrieval** in swap-paired histories, while replay plus exact transition rules maintains needed state. A model parses individual events; code applies updates. It also measures a reader-dependent length horizon and which components the reader can decode.

**Evidence:** 7–32B model full-log decisions fall toward **48–55%** at 256 events whereas correct current-state rendering enables **96–98%** for some reasoning readers. At 256 events the parsed-event rule system scores **80–89%**, beating model-maintained ledgers; strong 120B reader sees ~**0.87** log vs **0.99** compact state at 1,024 events, with seed uncertainty near the registered margin. On the independent second domain, some *state renderings themselves* are hard for readers; improvement is conditional on both correct state and an interpretable representation. Narrow authorized finite-state tasks, not universal personal cognition.

**Already:** Canonical replayable Experience/Event, XTDB bitemporal Claims, independent tool/Mission permissions and derived current-state projections. The paper does not prove histories are expendable or Claim Authority can be a text summary. **New:** controlled swap-pair evaluation of reinstatement/order and provenance sensitivity; stronger evidence against recency/similarity-based authority reconstruction. Storing exact state requires number-of-keys scaling; parser/transition error is the principal fidelity cost.

**Action:** No immediate test. At a genuine observed current-state permission/revocation bug, test the specific swap/reinstatement trace through existing contracts; do not replace stores or launch new replay benchmark.

## 7. SEED — conditional hybrid-model serving optimization

**Primary:** *Shallow Encoding Emits Deep Memories*, arXiv:2611.06084v1, November 4. https://arxiv.org/html/2611.06084

**Class:** Future conditional serving challenger, not personal memory architecture.

**Method:** In hybrid attention+recurrent models, shallow prompt-token execution plus learned deep write-path emitters reduce prefill; recurrent state decomposes into fixed rank-one sink plus low-rank content with periodic compression. Original decoder weights and attention caches remain unchanged. On five production hybrid families (30B–300B+), official-task quality is nearly retained; GB300/SGLang coding trace shows **1.2–1.4x** throughput per GPU on Qwen3.8-Flash-Next, state 2.0–3.9x smaller right after compaction (1.3–2.6x average). Model-specific emitters, calibration/copy probes, state compression and hybrid-kernel integration have substantial engineering overhead; not demonstrated on current Gemma/MFM pipeline or Snapdragon CPU/QNN.

**Existing/new/action:** First-party hardware-adaptive serving and replaceable feature models already fit the mechanism. Only revisit for measured throughput, recurrent-state-HBM or latency blocker with an actual chosen hybrid checkpoint. This cannot justify altering current MFM training, adding a new model plane, or pursuing serving comparisons without a decision to change.

## Narrow screened findings and exclusions

- *FlyHash for LLM Agent Memory: An Empirical Validation and a New Application*, arXiv:2611.05791, November 4. https://arxiv.org/abs/2611.05791. Finds a quality-per-bit advantage over SimHash under controlled labels **but HNSW still dominates practical ordinary-scale memory/latency**; negative evidence for any new vector-engine tournament. Qdrant stays selected.
- *A complementary learning system for continual episodic memory in large language models*, newly available November 4 as an eLife reviewed preprint; authors note earlier arXiv precursor from April 2025. Dedicated sparse episode LoRA and slow replay consolidation is relevant background but is not a newly proven full-model deletion or lifelong governed personal memory result. Track a materially stronger extension rather than treating re-publication as new architecture evidence.
- Prior MemCalib, Jev-Mem, RPMem, JitMem, REALM, AdaMM, MUTE, PGMem, Recuris, FRESH and Oct–Nov 4 intake papers are not rediscovered as new; none found with a stronger revision changing an implementation decision.

## Execution decision

**No architecture-invalidating evidence or direct present model/contract defect established.** None supplies a root-cause repair for the observed native MFM checkpoint-10 formation failure or unqualified personality judgment. Keep the original checkpoint-26 qualification/actual source-grounded learning work. All methods are future conditional, so no main change, infrastructure tournament, PR or new benchmark campaign. Future adoption requires observed issue and explicit provenance/Claim authority, subject separation, current permission and revocation, cross-layer deletion/influence echoes, full rollback, and held-out user/model evaluation. Fable remains the **full transferable personal cognitive foundation** after F11; no paper-specific fixed model/store/ontology ceiling.
