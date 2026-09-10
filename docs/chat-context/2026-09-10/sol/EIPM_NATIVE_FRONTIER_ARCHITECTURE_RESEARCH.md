# A.L.I.C.E.-Native EIPM Frontier Architecture Research — 2026-09-10

**Status:** working frontier recommendation on `alice-context`; not canonical `main`; no A.L.I.C.E. weights created.

**Owner constraint:** third-party models may be disposable research, generation, or engineering tools. Third-party model weights must not be required by the permanent A.L.I.C.E. Elaina Identity / Personality Model (EIPM). The permanent EIPM weights must be A.L.I.C.E.-native and trained from A.L.I.C.E.-controlled initialization/training lineage.

**Supersession:** this note supersedes the earlier working recommendation that treated `Qwen3.8-27B + A.L.I.C.E. LoRA` as the permanent EIPM. Qwen may remain an optional disposable teacher/mechanics tool where licensing permits. Its weights are not part of the EIPM.

## 1. Research question

How much language understanding should live inside the EIPM itself, versus being supplied by A.L.I.C.E.'s cognitive fabric as structured state?

This matters because A.L.I.C.E. is explicitly not one model. The EIPM is the first learned identity component, not a replacement for the reasoning/model fabric, Memory Formation Model, Rayan Host Model, relationship model, A.L.I.C.E. continuity/self model, future world models, retrieval models, or specialist systems.

## 2. Frontier evidence

### Personalized alignment can be separated from text generation

- **PAD, ICLR 2025** explicitly decouples text generation from personalized preferences using a personalized reward model at decoding time. This is strong evidence that A.L.I.C.E.'s durable personality mechanism does not have to be a full generative LLM.
- **Persona-judge, ACL Findings 2025** uses a discriminative preference judgment process to control candidate generation rather than encoding all personalization into a permanently modified generator.
- **PCogAlign, ACL 2025** evaluates personalized *actions* using a cognition-aware, action-based reward model. This is closer to A.L.I.C.E.'s goal than surface-style imitation because A.L.I.C.E. must make Elaina-faithful judgments and choices, not merely sound similar.
- **PersonaAgent, ACL Findings 2026** separates personalized memory and personalized action. This converges with A.L.I.C.E.'s existing source/host/memory/action separation.

### Personalized judgment is multi-dimensional and context-dependent

- **P-Check, ACL 2026** finds that static/implicit user conditioning is insufficient and generates dynamic, scenario-specific evaluation criteria with preference-contrastive criterion weights.
- **AlignX, ACL 2026** models an intermediate psychological/behavioral preference space and reports strong gains from preference-bridged alignment. This supports an explicit intermediate identity-control representation between cognitive state and final generation.
- **AdaJudge, ACL 2026** argues that ordinary generative representations are poorly matched to fine-grained discrimination and improves reward modeling with discrimination-oriented representations plus adaptive multi-view pooling.
- **VL-MDR, ACL 2026** shows the value of dynamic multi-dimensional rewards rather than compressing every judgment into one scalar.
- **PRISM, ACL 2026** separates subjective preference structure from uncertainty rather than conflating both in a single score.

### Compact specialist judges can be strong

- **PairRM** is a roughly 0.4B pairwise model built specifically to compare candidate responses side-by-side. Its published results show that highly specialized discriminative models can be much smaller than general generators while remaining useful judges.
- **Skywork Reward V2** includes a 0.6B reward model whose reported average approaches an earlier 27B reward model. This does not make its inherited Qwen weights suitable for A.L.I.C.E.; it is evidence that the *function* of preference discrimination need not require a giant generator.
- **TS-Align** demonstrates that ranking capability from a large teacher can be distilled into a smaller effective reward model.

### Structured persona graphs matter when identity labels are scarce

- **ThinkPersona, ACL 2026** uses Persona Graphs containing life trajectories, values, relationships, and events, then grounds role behavior in those graphs. This maps closely to A.L.I.C.E.'s E0/persona-graph architecture.
- **PsyMem, TACL 2026** combines fine-grained psychological indicators with explicit memory-alignment training. It supports teaching the identity model how retrieved memory should affect behavior instead of treating memory as undifferentiated prompt text.
- **Dynamic Persona Coherence, ACL 2026** explicitly decouples stable identity from adaptive psychological state. This strongly supports A.L.I.C.E.'s distinction between a stable Elaina-derived anchor and changing Rayan/relationship/A.L.I.C.E.-continuity state.
- **Knowledge-Enhanced Hierarchical Heterogeneous Graph personality identification, AAAI 2025** reports strong low-label performance from graph structure and knowledge enrichment, including an experiment using only about 1% of its labeled data.
- **MemReward, 2026** uses a heterogeneous experience graph plus a GNN to propagate reward information under limited labels, further supporting graph-aware identity learning when labels are scarce.

### Pure hand-authored concept bottlenecks are too restrictive

- **Hybrid Concept Bottleneck Models, CVPR 2025** combines an explicit static concept bank with a learned dynamic concept bank to capture important concepts omitted by the human-defined ontology.
- **Incremental Residual Concept Bottleneck Models, CVPR 2024** similarly learns residual concepts when the explicit concept bank is incomplete.
- **Decoupling Concept Bottleneck Model, TPAMI 2025** separates explicit and implicit concept information to reduce distortion caused by incomplete concept sets.

For A.L.I.C.E., this argues against reducing Elaina to Big Five scores or a fixed human-authored trait vector. The EIPM should expose inspectable E0-grounded concepts while retaining a bounded learned latent residual space for nuance that the ontology misses.

### Full generative language competence from random initialization is the wrong v1 target

- **SmolLM** trained its 135M and 360M general language models on about 600B tokens each and its 1.7B model on about 1T tokens. This illustrates how much broad pretraining is normally required even for small general-purpose generators.
- **TinyStories** shows very small models can learn coherent generation in a deliberately restricted language/domain, but that does not establish broad adult language understanding from a tiny identity corpus.
- **BERTtime Stories** found synthetic story augmentation could give modest gains in some settings but overall harmed linguistic understanding in its constrained encoder experiments.
- **LEAF, ACL 2026** demonstrates that compact text encoders can acquire useful semantic representation through specialized distillation with modest infrastructure. This is a better analogue for the EIPM's language branch than building a full decoder-only LLM.

Conclusion: do not spend A.L.I.C.E.'s identity budget teaching a personality model the entire internet, coding, general world knowledge, or long-form language generation. The EIPM needs enough native semantic capacity to understand personality-relevant language and compare candidate actions/responses. The cognitive fabric should supply the larger world state in a structured form.

## 3. Recommended architecture

### 3.1 Model-agnostic canonical identity substrate (`I*`)

`I*` is not a neural checkpoint. It is the durable, inspectable identity source from which EIPM revisions can be rebuilt:

- E0 source evidence and hashes;
- Elaina persona/source graph;
- E-INF reconstruction with uncertainty;
- A-SYN behavioral completion with provenance;
- values, relationships, events, judgments, boundaries, communication/style evidence;
- clone-awareness and historical-truth rules;
- training examples and preference pairs;
- coverage map and lineage.

The EIPM is a learned projection of `I*`, not the canonical biography.

### 3.2 A.L.I.C.E. Cognitive Frame Protocol (`ACFP`)

The cognitive fabric should present the EIPM with a versioned, model-agnostic frame rather than a giant raw prompt. A frame may contain:

- actors/entities and their roles;
- current situation/event type;
- active mission, goals, constraints, conflicts, and stakes;
- evidence claims with provenance, temporal scope, and confidence;
- retrieved Elaina source/persona evidence;
- Rayan host state;
- A.L.I.C.E.–Rayan relationship state;
- A.L.I.C.E. continuity/adaptive state;
- social/emotional cues;
- candidate actions, candidate responses, or candidate reasoning summaries;
- expected consequences and uncertainty where available.

The ACFP must describe the world and evidence. It must **not** pre-decide `what Elaina would do`; that is the EIPM's job.

### 3.3 EIPM v1: hybrid structured + native semantic + explicit/latent identity model

Recommended learned blocks:

1. **A.L.I.C.E.-native semantic encoder**
   - bidirectional encoder trained from A.L.I.C.E.-controlled initialization;
   - reads short personality-relevant raw text spans, candidate responses, and ambiguous language that may be lost by schema extraction;
   - not a general-purpose text generator.

2. **Persona graph encoder**
   - graph transformer/GNN over selected persona evidence, relationships, values, events, precedents, and provenance edges.

3. **Structured-state encoder**
   - transformer/set-transformer style encoder over ACFP fields, temporal state, goals, relationships, and candidate actions.

4. **Explicit identity concept bank**
   - E0-derived interpretable dimensions covering actual Elaina values, boundaries, behavior, relationship modes, communication patterns, and judgment tendencies;
   - not a generic personality inventory imposed as the truth.

5. **Learned residual identity bank**
   - bounded latent dimensions trained to capture useful nuance missing from the explicit concept bank;
   - regularized so it complements rather than silently bypasses inspectable concepts.

6. **Cross-context fusion**
   - attention/fusion between semantic text, structured state, graph evidence, explicit concepts, latent residuals, and candidate behavior.

7. **Multi-head identity outputs**
   - candidate action/response preference score and pairwise/listwise ranking;
   - value-priority distribution;
   - stance: support / challenge / refuse / question / defer / mixed;
   - relationship/interpersonal posture;
   - emotional interpretation and response posture;
   - communication-style controls;
   - uncertainty/calibration;
   - evidence/precedent attention or pointers;
   - out-of-character / generic-assistant drift score.

Do not collapse all personality judgment into one scalar reward.

### 3.4 Identity Decision Packet (`IDP`)

The EIPM emits a portable packet to the rest of A.L.I.C.E. containing inspectable controls plus internal learned state. Example conceptual fields:

- stance;
- sparse value priorities;
- relationship posture;
- communication posture;
- uncertainty;
- supporting E0/E-INF/A-SYN/continuity references;
- relevant behavioral precedents;
- prohibited/contraindicated behavior patterns;
- candidate ranks;
- internal identity latent for A.L.I.C.E.-native downstream components.

Any replaceable reasoning/generation engine can consume the inspectable subset. Future A.L.I.C.E.-native generators can consume the richer learned representation directly.

## 4. Answer to the core language-understanding question

**Structured-first, but not structure-only.**

The cognitive fabric should carry most world knowledge, memory, mission state, host state, provenance, temporal state, and tool/research results in structured form.

The EIPM should still own a compact semantic-language branch because personality depends on nuance that a schema can lose: implication, politeness, sarcasm, emotional subtext, framing, wording differences, relationship register, and subtle differences between two candidate responses.

Therefore:

- **Do put inside EIPM:** personality-relevant semantic understanding, candidate comparison, judgment, values, boundaries, relationship posture, emotional interpretation, communication tendencies, uncertainty, and identity evidence use.
- **Do not put inside EIPM:** broad factual world knowledge, current web facts, coding competence, long-form general generation, tool knowledge, arbitrary scientific expertise, full host biography, Mission Graph storage, or canonical memory authority.

The EIPM becomes A.L.I.C.E.'s proprietary **identity/judgment policy**, not a miniature replacement for the entire model fabric.

## 5. Runtime loop

```text
request / event
    -> memory + mission + host + relationship + research/reasoning fabric
    -> ACFP
    -> EIPM identity decision
    -> Identity Decision Packet
    -> reasoning/generation/tool candidate(s)
    -> EIPM final candidate re-ranking when needed
    -> action / response
    -> outcome + Experience Ledger
```

For low-latency cases, the IDP can guide a single generation. For important/ambiguous cases, A.L.I.C.E. can request multiple candidates and let EIPM rank them.

This solves a critical model-replacement problem: the personality model remains A.L.I.C.E.-owned while the reasoning/generation engine can change.

## 6. Training objective

Because EIPM v1 is a discriminative/identity-policy model rather than a full generator, the earlier `SFT -> DPO` recipe is no longer the best primary objective.

Recommended stages:

### N0 — native semantic pretraining

Train the A.L.I.C.E. semantic encoder from random initialization on a legally cleared text corpus and A.L.I.C.E.-controlled tokenizer. Use objectives appropriate to bidirectional semantic representation (masked/denoising, contrastive/paraphrase, semantic relation, or equivalent). Third-party model weights are not loaded into the EIPM.

Optional teacher-generated labels/distillation may be used only where exact licenses/terms permit and the teacher remains disposable.

### N1 — identity representation learning

Use E0/E-INF/A-SYN/persona graph material for:

- explicit identity-concept supervision;
- graph relation/precedent learning;
- provenance/class prediction;
- contrastive same-identity versus plausible drift examples;
- memory-conditioned consistency;
- uncertainty supervision.

### N2 — judgment/preference learning

Train on chosen/rejected and listwise candidate sets using pairwise/listwise ranking objectives plus multi-task identity heads. Hard negatives should be plausible: generic assistant behavior, owner-pleasing sycophancy, wrong relationship stance, context-insensitive behavior, caricature/extreme traits, unsupported historical claims, or a response belonging to a plausible different person.

### N3 — calibration and owner-directed refinement

Calibrate uncertainty and ranking margins. Owner feedback after first activation becomes explicit versioned EIPM revision data. It does not silently rewrite historical E0 or ordinary host data into the Elaina anchor.

## 7. Initial size envelope

Do not freeze an exact parameter count before corpus materialization and mechanics benchmarks.

A sensible **research envelope is approximately 100M–400M total parameters**, not 7B–27B. The reason is functional: EIPM is a semantic encoder + graph/state fusion + identity/ranking heads, not a world-knowledge generator.

Recommended mechanics candidates are roughly small / medium / large within this envelope. Use the smallest model that preserves owner-evaluated identity fidelity and difficult contrastive ranking. Larger is not automatically better when identity labels remain limited.

The exact split among semantic encoder, graph encoder, fusion core, explicit/residual concept bank, and heads is an engineering decision after the real corpus inventory.

## 8. Key risk and mitigation

The biggest risk of structured-first architecture is **semantic loss before the EIPM sees the event**. A poor upstream frame could hide the nuance the personality model needs.

Mitigation:

- always allow selected raw text spans alongside structured fields;
- provenance-bind every extracted field to source spans;
- permit multiple candidate interpretations when ambiguity matters;
- train EIPM to detect frame/raw-text conflict;
- never allow upstream extraction to declare the desired personality answer;
- later replace generic extraction with the separate A.L.I.C.E.-native Memory Formation Model and other native semantic components.

The biggest risk of a pure learned black box is the opposite: identity may become impossible to inspect or correct. The explicit-concept + learned-residual design is intended to balance fidelity and inspectability.

## 9. Cross-model portability

Current research finds some concept representations can be linearly aligned across different LLMs, suggesting future portable latent bridges may be possible. Treat this as research only. Raw activation steering is architecture-dependent and can produce superficial behavior changes without guaranteeing identity fidelity.

The v1 portability contract is therefore the **ACFP + Identity Decision Packet + canonical `I*`**, not a hidden-state vector tied to another vendor model.

## 10. Startup / IP and teacher-model boundary

A.L.I.C.E.-native EIPM weights may be trained on rented GPUs. Renting compute does not make the model weights a provider dependency.

Third-party libraries and published algorithms may be used under compatible licenses, but the EIPM checkpoint, tokenizer/training lineage, identity data, and private model artifacts remain A.L.I.C.E.-controlled.

A permissively licensed self-hosted open model such as the exact Apache-2.0 Qwen3.8-27B checkpoint may be evaluated as a disposable synthetic-data or labeling tool. Pin exact model revision and license and preserve generation lineage. Do not require that model at EIPM runtime.

For proprietary hosted-model outputs, perform a terms review before using outputs as teacher/training data. In particular, current OpenAI terms include restrictions on using Output to develop models that compete with OpenAI. Therefore Astra/ChatGPT output should **not** be copied into the EIPM training corpus without explicit contractual clearance. The planned one-time Astra review should be treated as advisory only and should itself receive legal/terms confirmation before a commercial competing-model workflow relies on it.

This is an engineering/IP hygiene rule, not legal advice.

## 11. Compute implication

A 100M–400M discriminative EIPM is dramatically cheaper to train than a 27B generative foundation model. Initial private experiments should fit on 24–48GB modern GPUs; exact cost depends mostly on the amount of native semantic pretraining rather than the personality heads.

Do not buy a large 80–141GB training instance by default. Materialize the corpus, choose the language-pretraining volume, benchmark the selected model size, then rent only the smallest infrastructure that preserves the desired batch/precision/throughput.

## 12. One-review rule

Engineering tests, corpus statistics, unit tests, and small architecture/mechanics comparisons are not external semantic-validation tournaments and may occur before weight creation.

The owner-directed semantic review boundary remains one final independent review of the frozen corpus + model architecture before the first private identity-gradient run. If Astra is used for that review, confirm the applicable commercial terms first and do not turn Astra output into training examples.

After training, Rayan remains the decisive personality-fidelity judge.

## 13. Immediate next implementation step

1. Read-only materialization and exact inventory of canonical E0/router/MC10B/MC10C source material.
2. Measure real token counts, languages/modalities, dialogue density, candidate structure, and graph coverage.
3. Define `ACFP v0.1` and `IdentityDecisionPacket v0.1` schemas from the actual source and roadmap requirements.
4. Build the EIPM corpus compiler around those interfaces.
5. Build native semantic-pretraining corpus/license manifest.
6. Only then freeze parameter count, optimizer/training schedule, and cloud budget.

No MC10D restart. No third-party base checkpoint becomes the permanent personality model.

## Primary research references consulted

- PAD: Personalized Alignment of LLMs at Decoding-Time, ICLR 2025.
- Persona-judge, Findings of ACL 2025.
- Aligning VLM Assistants with Personalized Situated Cognition / PCogAlign, ACL 2025.
- ThinkPersona, ACL 2026.
- PsyMem, TACL 2026.
- Beyond Static Persona Consistency: Dynamic Persona Coherence, ACL 2026.
- PersonaAgent, Findings of ACL 2026.
- P-Check, ACL 2026.
- From 1,000,000 Users to Every User / AlignX, ACL 2026.
- AdaJudge, ACL 2026.
- PRISM, ACL 2026.
- PairRM / LLM-Blender.
- Skywork-Reward-V2, 2025.
- TS-Align, Findings of EMNLP 2024.
- Knowledge-Enhanced Hierarchical Heterogeneous Graph for Personality Identification with Limited Training Data, AAAI 2025.
- MemReward, 2026 preprint.
- Hybrid Concept Bottleneck Models, CVPR 2025.
- Incremental Residual Concept Bottleneck Models, CVPR 2024.
- The Decoupling Concept Bottleneck Model, TPAMI 2025.
- Cross-model Transferability among LLMs on the Platonic Representations of Concepts, ACL 2025.
- LEAF, ACL 2026.
- TinyStories, 2023.
- BERTtime Stories, CoNLL/BabyLM 2024.
- SmolLM technical release, Hugging Face.
- Current Qwen3.8-27B model license metadata and current OpenAI terms/service agreement.