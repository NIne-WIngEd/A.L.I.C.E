# Frontier Watch Intake — 2026-10-02

Research/context note only. No production or main semantics changed.

Compared against:
- A.L.I.C.E. main `8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b`
- Fable main `1805a01c73575378246cac8cbbb72cd891e8528e`
- prior frontier-watch `5832fe6764ca5f307a42dade1e618494423efac9`
- current MFM v1.6 work `research/mfm-v1-licensed-base-20260930@ae7d680286b84a5755caeb1d2e82e0d09c5176ae`

The Sep 27 full-v1 baseline, Sep 30 recovery intake, Oct 1 context/memory-control intake, current MFM/FBM/Gemma/Graphify/context work, and Fable main were inspected before judgment.

The active MFM v1.6 source/teacher, authorization, corpus-admission, CPU-processing, and provenance path remains the critical path. The selected Fable physical stack remains unchanged.

## MemFit — arXiv:2610.00872

Primary source: *MemFit: Efficient Long-Term Agentic Memory*, Mitchell Piehl and Muchao Ye, submitted 2026-10-01.  
https://arxiv.org/abs/2610.00872

Classification: **direct future capability improvement / corroboration for source-native retrieval efficiency**.

MemFit stores each conversational turn verbatim with lexical and dense indexes. It avoids a per-turn LLM memory-construction call. A second layer creates segment summaries whose member episode IDs are retained as provenance. The summaries index the raw turns rather than replacing them.

Its retrieval path combines lexical and dense candidates, metadata signals, provenance expansion from segment summaries, pseudo-relevance feedback, cross-encoder reranking, and bounded chronological neighbors. Its multimodal mode makes image captions searchable while retaining the associated image for the reader.

The paper evaluates one configuration across LoCoMo, MemGallery, and LongMemEval-S. The authors report leading aggregate results in their comparisons and roughly 3.9x to 52.6x faster memory construction than compared systems.

A.L.I.C.E. already has the stronger authority separation: Raw Evidence/Object, Experience/Event Fabric, source-native/live retrieval, bitemporal Claim Authority, rebuildable projections, vector/multimodal retrieval, and graph/associative compute.

**New implication:** preserve the option for a compact summary or caption to act as a provenance-bound retrieval index without becoming remembered truth or replacing source-native evidence.

Summaries, captions, rerank outputs, and retrieval packs remain derived projections. Current-state truth remains a Claim Authority responsibility. Lifecycle changes must invalidate affected derived indexes. Generated visual captions remain perceptual hypotheses tied to the original artifact.

**Action:** when Recollection/perceptual-memory runtime work reaches implementation, test a raw-evidence-preserving retrieval lane with canonical source records, rebuildable summary/caption indexes, hybrid lexical+dense retrieval, and bounded reranking. Measure evidence hit rate, exact-source restoration, multimodal conflict handling, projection rebuild, latency, and local compute. No backend comparison is needed.

## FOCUS — arXiv:2609.37590

Primary source: *FOCUS: Training-Free Decision-Preserving Context Compression for LLM Agents*, Dixit et al., submitted 2026-09-29.  
https://arxiv.org/abs/2609.37590

Classification: **direct capability improvement to the future Cognitive Workspace / Mission execution contract**.

FOCUS works on whole reasoning-action-observation spans. Several future-plan sketches estimate which historical spans later decisions depend on. A defensive pass preserves low-frequency but high-impact state such as prior failures, constraints, and state transitions.

Across AppWorld, OfficeBench, 8-QA, WebVoyager, and tau2-Bench, the paper reports peak-context reduction up to 48%, cumulative dependency reduction up to 73%, and task-success improvement up to 8.9 percentage points over uncompressed execution. Its rollout ablation reports OfficeBench success rising from 71.6% with one plan rollout to 77.9% with three, with gains saturating after that.

A.L.I.C.E. already has Cognitive Workspace, Mission Graph, durable execution state, Experience/outcome lineage, source-native history, and failure/precondition/repair directions.

**New invariant:** semantic salience alone is insufficient for workspace compaction. A span can look semantically minor while still constrain the future mission path.

Workspace compaction remains a transient activation operation. Durable Experience, Claims, Mission state, and source-native history remain outside that operation. The paper's dependency score is an approximation rather than a proof of true causal importance. Its serving-cost claims also need local remeasurement under real prefix/KV caching.

**Action:** when Cognitive Workspace compaction is implemented, compare semantic/context baselines against decision-preserving retention. Measure repeated-error rate, mission-step divergence, constraint retention, stale influence, token cost, and KV/prefix-cache cost.

## LatentHarness — arXiv:2609.39740

Primary source: *LatentHarness: Learning Latent Actions for Memory and Reasoning via Counterfactual Policy Distillation*, Wang, Wang, and Liu, submitted 2026-09-30.  
https://arxiv.org/abs/2609.39740

Classification: **future challenger / direct future model capability improvement for activation-memory scheduling**.

LatentHarness gives a recurrent latent reasoner three internal actions: Think, Recall, and Exit. Its fast-weight memory contains input evidence and derived latent states. Counterfactual Policy Distillation compares the allowed actions at a latent state, while the write gate is trained by whether later recalls actually benefit from a previously written derived state.

On Ouro-1.4B/2.6B backbones, the paper reports gains over token-space and latent-space baselines. At 1.4B, the reported general and long-context averages are 55.8 and 49.7 while using about 55% of full-depth FLOPs. It reports 5.9x faster long-context execution than Memory-R2. Removing Recall lowers the long-context average to 42.0; always reading reaches 45.3, supporting selective recall.

A major remaining error is memory addressing. On wrong MuSiQue answers with at least three hops, a distractor outranks the supporting passage at the final recall in 72% of cases.

A.L.I.C.E./Fable already separates durable memory, activation/working memory, MRM, Recollection, parametric memory, model lineage, and authority.

**New implication:** future first-party models can learn whether the next useful operation is more computation, recall, or exit. Future-recall utility can also teach which derived activation states are worth retaining.

Fast-weight or activation memory must remain typed and non-authoritative. Source evidence, durable memories, derived states, hypotheses, and activation summaries cannot collapse into one truth plane. Activation state needs derivation and model lineage sufficient for rebuild and rollback.

**Action:** preserve a future interface for compute-vs-recall-vs-exit scheduling and typed ephemeral derived-state writes. When first-party personal-model work reaches this stage, compare latent scheduling with cheaper MRM signals and test multi-hop addressing, source attribution, rollback, and lifecycle invalidation. Do not block MFM v1.6.

## Retrieval-layer continual learning — arXiv:2604.27003

Primary source: *When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents*, Hu, Long, and Wang.  
https://arxiv.org/abs/2604.27003

Classification: **direct future memory-evaluation improvement; newly surfaced older paper**.

The paper studies frozen-backbone agents that accumulate external experience across sequential ALFWorld and BabyAI tasks. It varies experience representation, memory granularity, and retrieval frequency.

Its useful result is that stability/plasticity can move from weights into retrieval. Raw task-specific trajectories can cause negative transfer. More abstract procedural memories can transfer more safely. Fine-grained memory can create retrieval-diversity collapse. Step-level retrieval can improve within-task execution without necessarily improving transfer across task shifts.

A memory can therefore remain durably present yet become behaviorally inaccessible as newer experience competes for a limited retrieval budget.

**Action:** add a future sequential-memory qualification lane for forward transfer, backward transfer, hard-case negative transfer, retrieval diversity, dilution under growing experience, and prerequisite/provenance preservation. Correct retrieval competition through MRM/routing/representation rather than treating older durable memory as disposable.

## Decision

No finding invalidates Fable v1 or reopens a selected physical implementation.

Carry forward four hypotheses/contracts:
1. summary-as-index, not summary-as-truth;
2. decision/mission dependency as a distinct Cognitive Workspace retention signal;
3. learned compute-vs-recall scheduling with typed lineage-bound activation writes;
4. retrieval-layer stability/plasticity evaluation so durable memories do not become silently inaccessible.

All are future runtime/model/evaluation work and do not block the current MFM v1.6 corpus-admission and CPU qualification path.
