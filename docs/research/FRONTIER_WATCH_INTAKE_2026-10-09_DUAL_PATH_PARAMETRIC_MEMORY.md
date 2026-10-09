# Frontier Watch Intake — 2026-10-09 — Dual-Path Parametric Memory

**Status:** research-only future model challenger; no main/production/MFM semantics changed.  
**Mandatory baseline:** `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`  
**Watch head before write:** `08a704210242e22dbc43ea32cb3412cbe6c6e65d`

## Repo and execution state verified before comparison

- A.L.I.C.E. `main@8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b`.
- Fable_Sleight `main@a6eba323dd27d3c58f232b351d09b86761b15935`.
- Active MFM `research/mfm-v1-licensed-base-20260930@c1115be293691624931a9126b7bab0674e7101ec`; other v1.6 source/teacher/curriculum branches inspected.
- `alice-context@945245bbee4168fff46f863455a83bb219063017`; `fable-builder-model@6861d76535c3418efbd2ee948c1056d57030cfe4`; `research/graphify-context-substrate@a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15`.
- Read canonical full-Fable execution plan, personal development architecture, Memory Architecture v4.1, MFM foundation, watch guardrail and Oct 1–8 watch intakes before judgment.
- Most recent public MFM receipt `MFM_FULL_FIT_SESSION0_AND_STORAGE_RECOVERY_20261009_TRACE95_PUBLIC.json`: first **3-of-20-epoch** scheduled fit boundary ended at **step 6864** with trainer exit 0. **Full program 45,760 steps is not complete; original checkpoint and history custody on Drive remains incomplete; no restored/resumed trainer acceptance, learned capability, evaluation qualification, or product authorization.** Preserve actual trainer originals, finish authenticated custody/recovery, then resume only after exact optimizer/RNG/cursor/history restoration. No finding here interrupts this critical path.

## Qualifying new result: DPPM

**Paper:** Yuhao Chen et al., *DPPM: Dual-Path Parametric Memory for Personalized Language Models*, arXiv:2610.11776, submitted 2026-10-08.  
Primary: https://arxiv.org/abs/2610.11776  
Method/full text: https://arxiv.org/html/2610.11776  
Author implementation: https://github.com/isyuhaochen/DPPM (project MIT; compiler/backbones/datasets supplied separately and have their own terms).

**Classification: future challenger / direct future parametric personal-memory capability improvement.** It is **not** architecture-invalidating, not a new logical cognitive plane, and not a current MFM fix.

### Existing A.L.I.C.E. coverage versus new delta

A.L.I.C.E. already distinguishes canonical Raw Evidence, Experience/Event, bitemporal Claim Authority, Episodes, user/host/source-person/assistant-self/relationship state, working activation, source-native reads and governed parametric memory. The frontier watch already tracks RPMem's recurrent cross-session parametric memory, LINEUP's compact private corrections, MemCodex evolvable programs and CacheReforge model/activation lineage.

DPPM adds **a specific dual-path internal parametric-memory mechanism beyond RPMem**. A recurrent-only update can gradually attenuate early but still relevant evidence. A pure history pool can retain early evidence but loses event-order sensitivity, so it cannot represent that a user changed their mind. DPPM learns both in parallel:

1. A **direct Evidence path** encodes chronological history segments with a frozen Doc-to-LoRA compiler and applies learned *coordinate-wise content-weighted softmax pooling* across the segments. Old evidence remains directly represented without having to survive all later recurrent transitions.
2. An **ordered Delta path** processes the same segments chronologically through a gated delta-rule associative key/value state. A prediction residual is written according to learned keys and gates to reflect new/changed information.
3. Learned fusion and RMS-scale calibration combine the two; the frozen Doc-to-LoRA decoder produces a *history-conditioned LoRA adapter* for a frozen downstream language model. Only small memory modules train by answer cross-entropy; the compiler, decoder and backbone remain frozen.

Theoretical discussion is bounded: age-independent input sensitivity is only a weak direct-route argument, with pooled contribution still diminishing as history grows; it is **not** evidence that DPPM never forgets.

### Evidence and ablations

- Qwen3-4B, PersonaMem-v2: DPPM **54.22%** overall, RPMem **45.73%**; PrefEval: **86.79%** vs **82.47%**. These are task-level accuracy measurements, not full lifelong cognition.
- The release reports **527,364 learned memory parameters**. A matched-parameter RPMem control is provided to reduce the chance that gains are merely from adding parameters.
- The single-path Evidence and Delta ablations each lose some performance compared with their combination; shuffled/order-insensitive treatments cannot properly express opposite preference update orders.
- Experiments span Gemma-2-2B, Qwen3-4B and Mistral-7B, with PersonaMem-v2 and PrefEval histories. Five-seed means and a separately released Qwen seed-42 checkpoint are distinguished in the repository. A clean release smoke/validation on PrefEval reports 1,422/1,620 correct (87.78%) for one seed; **do not** misrepresent this as the paper's 86.79% five-seed mean.
- Authors report position-sensitivity and retention diagnostics favoring dual path on old evidence, but **latent sensitivity is not a deletion, provenance, or true temporal-validity guarantee**.
- The fast adapter-at-answer path does not include every upstream history-compilation, recomputation, source attribution, privacy, storage, and model-checkpoint cost. Frozen compiler and pretrained model licensing remains separate from project code.

**Evidence strength:** moderate for the specifically controlled cross-session text-personalization tasks and parametric retention-versus-revision hypothesis; weak for multi-year, multimodal, sovereign cross-layer deletion, native judgment evolution, or Fable-scale personalized cognition. This is a fresh preprint, not independently established as Fable-ready.

### Exact Fable implication and governance

For *future* parametric personal-memory / native-judgment development, keep a model-interface challenger allowing:
- history-derived **direct evidence accumulation** alongside a **chronologically updated revision/associative channel**;
- typed subject axes for host, source-person, assistant self and relationship, with no cross-subject latent blending granted epistemic authority;
- original source/version, authorized ingest scope, compiler/adapter/backbone versions, data-generation and evidence-root lineage for each derived training/adapter artifact;
- provenance-grounded source restoration for audit; latent memory and generated LoRA are learned projections, **never XTDB Claim truth or Experience authority**;
- explicit deletion/correction/revocation invalidation and clean rebuild/retrain as needed across compiled segments, pooled/Delta states, generated adapters and active KV caches. The paper does not itself establish unlearning or influence-echo freedom;
- versioned promotion/rollback gates for any personal-model evolution. A changed user preference must not silently overwrite the historical fact that the previous preference existed.

The Evidence/Delta **two-path method is an optional learned mechanism**, not a cap of two memory types, a replacement for full Fable planes, or a frozen-personality policy. Nor is DPPM's fixed 4,096-token segmentation a Fable ceiling.

### Action and cost

**Future only, not a blocker.** When parametric personal-memory training becomes the next engineering decision, compare (under matched frozen compiler/model/data/parameter budget):
- RPMem recurrent-only vs direct-evidence-only vs DPPM-like dual path;
- old-but-crucial constraints, opposite-order preference updates, contradictory/ambiguous evidence, subject attribution and held-out users;
- source-version revisions, correction/revocation/deletion rebuild and post-deletion influence echo; adapter/KV lineage on rollback;
- end-to-end host-private compilation/update, storage and inference latency, not only promptless answer tokens;
- behavioral *native judgment* causality and model swaps, not just multiple-choice personalization accuracy.

Do not train this challenger during the current MFM fit or launch any infrastructure tournament. The project code is MIT but separately downloaded compiler/backbone/datasets require independent commercial/license review before any use.

## Other candidates screened — no new decision

**CogMem**, arXiv:2610.11314, 2026-10-08, https://arxiv.org/abs/2610.11314. **Already covered/corroboration.** Its Person–Event–Concept–Claim–Fact graph, speaker/target attribution, temporally reconciled sourced claims, source-grounded graph-operator recollection and episode-to-fact consolidation support existing A.L.I.C.E. design. Its 150-pair lexical-overlap semantic-collapse probe (flat RAG misattributes a subjective belief as fact 64/150 vs typed 9/150) is a useful possible future focused regression case, not a missing logical plane. Do not import the fixed five-node ontology or fixed five controller-step budget.

**FlowState**, arXiv:2609.34565, v2 2026-10-08, https://arxiv.org/abs/2609.34565. **Already covered/corroboration.** Typed execution state, incremental state updates and progressive source-linked historical access align with the already recorded PoS / FOCUS / JitMem / Mission/Workspace and adaptive Recollection contracts. Its revision does not expose an additional implementation blocker.

**Agent-Controlled Forgetting**, arXiv:2610.10590, https://arxiv.org/abs/2610.10590. **Already covered/corroboration.** Reversible context curation of tool observations is not authenticated cross-layer deletion/unlearning; it fits the CLM/JitMem transient Workspace path. Token wins are workload-sensitive, and it must not be invoked as a reason to relax MUTE deletion or canonical evidence retention.

Scite search attempted but rejected by provider because the connected account lacks a paid plan/trial. Primary arXiv HTML and author GitHub implementation were read using Exa/GitHub.

## Decision

**Carry one genuinely new future model challenger:** parametric history evolution may benefit from parallel direct-old-evidence and ordered-revision routes. It strengthens but does not replace RPMem and existing cross-layer Fable architecture. **No selected store/engine reopening, no reduced consumer tier, no current MFM diversion, no main/prod semantic changes, no PR needed.** F4–F11 remain internal milestones; first consumer release is the full personal cognitive foundation after F11.
