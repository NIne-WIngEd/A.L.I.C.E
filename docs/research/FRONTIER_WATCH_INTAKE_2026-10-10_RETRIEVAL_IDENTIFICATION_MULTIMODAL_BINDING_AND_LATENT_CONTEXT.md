# Frontier Watch — 2026-10-10: causal retrieval utility, multimodal history binding, query-local graph and residual context

**Status:** research intake only; **no `main`/production/MFM semantics modified**.  
**Mandatory baseline:** `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`; 2026-09-27 execution guardrail.  
**Previous watch head checked:** `5e9795e3e3543c70ec4e1ebf375857f674f87cfb`.  
**Publication window:** Friday 2026-10-09 arXiv new/cross-listed papers, plus a newly surfaced October revision.

## State inspected before judgments

- A.L.I.C.E. `main@8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b`; current Fable_Sleight `main@a6eba323dd27d3c58f232b351d09b86761b15935`; their README/product and full-v1 execution plan.
- All prior frontier-watch intakes through **October 9** and the full-v1 baseline. Existing deltas from MemCalib, JitMem, REALM, EPGM, MemTrace, active intent, PoS, MUTE, CacheReforge, MemCodex, MemAgent, DPPM, source-routing, query-conditioned admissibility, independence groups and verified Mission outcomes are not rediscovered.
- Active heads: `alice-context@945245bbee4168fff46f863455a83bb219063017`; `fable-builder-model@6861d76535c3418efbd2ee948c1056d57030cfe4`; `research/graphify-context-substrate@a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15`; `research/gemma4-v1-source-derivation-20260930@3e1328410dac52ba18be243cb9f7144e7a3964d1`.
- **Newer MFM head:** `research/mfm-v1-licensed-base-20260930@876adaefcce37e5009a5284a889badaad4eb25b0` (2026-10-10). Its TRACE111 records S2 **RUNNING**, planned from step **13,728 to 20,592** (epoch 6→9); S2 actual read-only-parent and optimizer/cursor/RNG resume are **not yet verified**; S1 complete 27-file Drive backup/catalog custody is still pending. Separate complete-input/capacity failures remain documented. No current watch finding interrupts exact MFM training/custody/source qualification.

## 1. Causal Memory Policy (CMP) — unobserved retrieval utility is not zero utility

**Primary:** Arman Behnam and Binghui Wang, *Causal Memory Policy: Making Memory Utility Identifiable by Intervening on Retrieval*, arXiv:2610.02070 (newly surfaced October revision). https://arxiv.org/html/2610.02070

**Classification:** **direct future capability improvement** to Memory Resource Manager (MRM), Lifecycle Curator and memory-policy evaluation; no current production change.

**Existing:** selective memory activation, non-destructive evidence and bitemporal Claim validity, source-native retrieval, learned memory-control/evolution candidates, provenance/deletion/rollback gates and irreversibility rules.

**New:** a memory that the ranking policy *never retrieves* cannot have its counterfactual usefulness identified by comparing store inclusion/exclusion. CMP explicitly randomizes a small retrieval-exposure slot with known propensities and estimates *query-conditional* utility. Critically, the study finds query-specific utility does **not** aggregate into a reliable universal retention score for unseen queries. **No retrieval/low observed utility is not evidence that source memory is safe to destroy.**

**Evidence/ablations:** about 54% of required memories on LongMemEval and 67% on LoCoMo lacked retrieval support. Randomized exposure raises required-vs-nonrequired discrimination AUC ~0.542→0.664. Removing exposure or cautious one-sided abstention increases lost required memories in its simulated reclamation setting; symmetrical exposure adds only 0.007 AUC and raises destructive errors. Per-query utility AUC ~0.784 on its own query degrades when pooled. Formal results depend on assumptions about randomized exposure, identifiable outcomes, and costs. These are benchmarks/diagnostics, **not validated lifelong retention rules**.

**Implication:** separate (a) relevance/selection and (b) causal evidence of utility; maintain `unknown/unexposed` distinctly from `demonstrated low value`. MRM may learn query-conditioned routing under governed budget. Lifecycle Curator must not convert non-retrieval, low salience or fitted aggregate utility into irreversible evidence/Claim erasure. Authenticated admissibility/privacy filtering applies *before* any experimental exploration; no random cross-host/source-person prompt injection. Correction and owner-authorized deletion remain enforced independently of utility.

**Cost/risk:** randomized exposure consumes context and can degrade answers or disclose otherwise ineligible material if badly implemented; propensity tracking/outcome logging adds runtime/state cost. **Action:** when MRM policy evaluation becomes current work, focused offline/synthetic or expressly authorized experiments; compare ordinary logging vs known-propensity exposure on held-out required memories, query distribution shift, privacy/admissibility failure and wrongful irreversible eviction. Do not deploy exploratory retrieval in production by default.

## 2. ReTurn — task-carrying and evidence-carrying multimodal history bind differently

**Primary:** Shuoyang Sun et al., *When History Helps and Hurts: Selective History Use across Multimodal Turns*, arXiv:2610.11948, surfaced in October 9 cs.CL listings. https://arxiv.org/html/2610.11948

**Classification:** **direct future perceptual-memory/Cognitive Workspace qualification improvement**.

**Existing:** source-native multimodal evidence, perceptual memory, active-intent/requirements projection, host/person/self scope, bitemporal state, Memory Resource Manager and Recollection, historical revisions and admissibility.

**New:** controlled separation between **history as the inherited task/question** (apply past question to *new* media while ignoring the old answer) and **history as the answer's evidence source** (apply *current* question to *old* media despite competing new media). Recall of past question/media is not the same as correct question–source binding or execution. ReTurn uses matched target-preserving Reconfirm/Reground and Retrieve/Rebind contrasts for visual/audio history.

**Evidence/limitations:** 7,000 tasks (3,500 matched pairs), 13 models, matched single-turn answerability. Median model-level open-ended accuracy 93.7% single-turn versus 72.3% conversational. On task-side probes some models recall the old question ≥96% but do worse applying it; on evidence-side probes swapping competing media changes answer behavior despite fixed historical target. Restating inherited question helps, while reweighting and supervised adaptation show mixed/partial recovery. Dataset relies on model-mediated construction/verification, limited turn horizons and benchmark media; not a real multi-year private history.

**Implication:** activation packets should distinguish `current_question_source`, `historical_task_reference`, `historical_evidence_source`, `target_media_record/version`, and temporal/lifecycle eligibility. Retrieved candidate relevance alone cannot substitute for correct task↔media binding. The selected historical image/audio object stays source-native; source-person/host/assistant-self binding and revocation/deletion propagate into media pointers, working context and downstream activation. Rejected/obsolete answers may be historical evidence but not active answer authority.

**Cost/action:** this is a compact focused future perceptual-memory evaluation rather than a new store or runtime requirement. When modality history enters live testing, add matched old-question/new-media and new-question/old-media cases with a contradictory competitor, corrected preference/identity and deleted source; measure target-binding accuracy, history-selection errors, modalities, token/latency and abstention. No present MFM diversion.

## 3. QGMem — query-local graph encoding as *activation representation*, not graph replacement

**Primary:** Yichen Liu et al., *Event-Centric Memory with Query-Aware Graph Augmentation for Long-Term Conversational Agents* (QGMem), arXiv:2610.11920; October 9 cs.CL listing. https://arxiv.org/html/2610.11920

**Classification:** **future model-interface challenger**.

**Existing:** full Cognitive Multi-Graph (LadybugDB local, NebulaGraph scale-out), associative graph compute, evidence/episode/Claim/temporal projections, hybrid multimodal retrieval, source-native and Cognitive Workspace. QGMem's event units and dynamic update traces largely corroborate existing contracts.

**New:** **learned query-local relational graph representation** at the *working/activation* boundary. After hybrid retrieval and rerank, it induces a small graph over activated evidence and injects a learned graph-token embedding alongside the source-grounded text into an accessible open-weight model; API-only models receive text serialization instead. It is **not** evidence that persistent global graphs can be removed.

**Evidence/ablations:** six conversational/long-history benchmarks, multiple backbones. Qwen3-8B six-benchmark unweighted macro 51.62; removing graph token drops 3.95 points, removing reranker 5.23, atomic units 7.43, and dynamic traces 2.08. LoCoMo answer score 82.77, LongMemEval-S 62.14, Fact-MH only 12.50. LoCoMo measured online end-to-end ~5.94 s/query and 1,208.79 online tokens; offline construction is amortized and not zero. Learned graph encoder/projector and its backbone-specific coupling add training/serving/lineage cost.

**Implication:** future Cognitive Workspace or native judgment models may benefit from graph-conditioned embeddings **in addition to** text/source pointers. A graph token is a rebuildable non-authoritative activation projection: its relations must be source-linked, valid for current time, correct subject, and admissible to the query. It cannot itself decide Claim truth or override identity, correction/deletion or rollback. Exact graph/model/encoder version and evidence generations must invalidate both graph token and KV state after changes. The **k=10** experimental activation size and fixed ontology are not Fable ceilings.

**Action:** when graph-to-model fusion is actually implemented, test same retrieved evidence with text-only linearization vs graph-token fusion on multi-hop and conflicting-history cases, plus deletion and cross-person relation leakage. No infrastructure tournament, no production rewrite now.

## 4. REMORY — residual soft activation across context compaction

**Primary:** Hanchen Xia et al., *REMORY: Learning Residual Memory for Context Compaction*, arXiv:2610.11287, October 9 cs.CL cross-list. https://arxiv.org/html/2610.11287

**Classification:** **future challenger** for Cognitive Workspace / learned activation compaction; distinct from previously recorded CLM/FOCUS by adding learned continuous soft memory to a *fixed readable summary*.

**Existing:** source-native durable evidence and Experience, compaction/reconstruction, MRM, working activation, model/adapter/KV versions, CacheReforge + MUTE influence-echo control.

**New:** a neural memory encoder produces a bounded soft-token *residual* accompanying the readable summary; train reconstruction, then continuation matching a frozen actor with access to full history. The actor is frozen but the memory network is not small: roughly **1.94B / 1.502B trainable parameters** in tested pairings.

**Evidence:** On SummHay same 92 queries, summary-only citation F1 **56.98** and joint **39.84** versus summary+REMORY **61.03** and **43.39**, roughly raw-context joint **43.41** using ~5.2% of raw input positions. Equal-budget extra text fails to reproduce the gain. Qwen3.8-27B Terminal-Bench 2.1 **71.9→76.4**, BrowseComp **74.0→77.0**; GLM-5.3-Flash likewise improves. Repeated tool outputs/errors decrease. These are within-actor comparisons but some agent benchmarks use single runs. Published frontier reference scores are not matched controls. Token-priced estimates omit memory encoder, features, tool time and all end-to-end privacy/rebuild cost; paper-trading anecdotes cannot establish causal performance.

**Implication:** learned residual activation can supplement but **never replace** readable workspace state or canonical evidence. Soft state must carry source-root, evidence/version, identity, policy, model and lifecycle generation lineage. Revocation/deletion/correction/rollback invalidates or rebuilds affected residuals and derived KV; opaque latent contents cannot quietly retain a deleted claim. Owner-authorized local/private inference/training is required for personal data.

**Cost/action:** memory model is substantial, not free or mobile-ready. If context compaction becomes a measured decision bottleneck, compare fixed-summary, summary+source recall and summary+residual at matched quality and complete hardware/latency/energy budgets, with exact-source attribution and deletion-echo checks. **Do not** integrate now or remove any full-v1 plane.

## Screened; no new action

- **MemoType / TriMEM**, arXiv:2610.11573 (Oct 9 new listing): learned memory/query type routing and tailored retrieval corroborate full MRM/Retrieval Orchestrator and existing learnable-routing research; fixed memory classes cannot cap Fable.
- **Gated Memory**, arXiv:2610.11270 (Oct 9 cross-list): utterance-level pre-extraction admission + privacy-aware enrichment supports existing formation/provenance and source-scope rules; +2.6% relative LoCoMo-10 result is too narrow to claim an additional MFM contract defect over current admitted-source program.
- **Compute Globally, Materialize Locally**, arXiv:2607.23693, Oct 9 revision: KV rows can retain source-conditioned value after earlier observation is dropped; strong mechanistic corroboration of already-recorded model/cache/source-generation lineage and MUTE post-deletion influence-echo controls. It does **not** authorize treating omission from prompt as deletion.
- **BFR**, arXiv:2610.09348: evidence-sufficiency acquisition after requirement decomposition is compatible with October 8 EPGM, distributed-evidence derivation, and current Recollection. Evaluation idea for future, not a new plane.

## Decision

**No architecture-invalidating evidence; no physical placement or backend reopening; no partial consumer tier; no main or MFM change.** Keep Experience/source evidence distinct from Claim Authority and transient learned memory. F4–F11 remain internal. First consumer predicate remains `full_personal_cognitive_foundation_after_f11`. Selected KurrentDB/NATS/XTDB/encrypted CAS/LadybugDB/NebulaGraph/Qdrant/source-native/Temporal/L1+Valkey/PyTorch+Accelerate unchanged.

Scite targeted search attempted but provider denied MCP access for lack of paid plan/trial. Primary full-text arXiv HTML was read through Exa and October 9 arXiv new listings were inspected directly.
