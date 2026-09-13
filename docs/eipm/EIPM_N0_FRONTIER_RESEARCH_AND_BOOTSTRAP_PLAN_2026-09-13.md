# A.L.I.C.E. EIPM N0 Frontier Research and Bootstrap Plan — 2026-09-13

**Status:** research decision / implementation gate on `alice-eipm-v1-build`. Not canonical `main`. No permanent A.L.I.C.E. EIPM weights are created or authorized by this document.

**Supersession:** this document supersedes any earlier use of `~100M–400M` or `~400M` as an EIPM target, envelope, or ceiling. Those numbers were exploratory estimates for one specialist architecture hypothesis. **There is no ratified parameter count.** The permanent EIPM may be much smaller or much larger if measured capability and identity fidelity require it.

**Current project boundary:** Curated Frontier v2 and the private EIPM learning structures are complete enough to leave broad data generation and begin N0 design. The next irreversible action is **not** private personality training. It is selecting and qualifying the native semantic/judgment substrate that will later receive private identity learning.

---

## 1. What N0 must solve

A.L.I.C.E. is not one model. The EIPM is the durable Elaina-derived **identity and judgment policy**, not the world model, memory database, host model, mission system, tool planner, or general-purpose language generator.

N0 therefore should not attempt to teach a random-initialized model the entire internet merely so it can write prose. It must give the future EIPM enough **native semantic, pragmatic, social, relational, temporal, causal, and discriminative competence** to understand what an ACFP frame and its source spans mean, compare possible actions/responses, and preserve nuance that a symbolic schema can lose.

The target behavior after N0 is roughly:

> Given text, structured context, and relational/graph context, understand the meaning and distinctions required for later identity judgment without already pretending to know Elaina.

N0 is **identity-neutral**. E0/E-INF/A-SYN personality authority enters later in N1/N2.

---

## 2. Current private data footprint and what it implies

The current curated private source contains:

- 290 canonical E0 semantic units;
- 205 curated E-INF records;
- 1,965 curated A-SYN records across direct, base-policy, targeted, and contextual lanes;
- 2,460 canonical private learning records overall;
- 3,496 historical-UNKNOWN targets;
- 3,930 runtime preference pairs;
- 1,965 historical-provenance pairs;
- 205 E-INF uncertainty sets;
- 13,719 retained alternate branches that remain **unordered alternatives**, not rejects;
- 2,460 ACFP frames;
- 2,460 symbolic IDP targets;
- 2,830 identity-concept entries;
- persona graph: 5,290 nodes / 19,930 edges.

A sanitized direct inspection of the active E0/E-INF/A-SYN text-bearing fields is only on the order of **hundreds of thousands of natural-language tokens**, not billions. That is enough to specialize a competent semantic substrate, but it is not enough to safely create broad language competence from scratch by itself.

This creates a hard design implication:

1. **N0 must learn general semantic/pragmatic competence from public/licensed material before private identity learning.**
2. N1/N2 must use grouped, provenance-aware splits so mechanical siblings or multiple rows sharing the same underlying evidence cannot leak across train/evaluation partitions.
3. Model size cannot be chosen from private row count alone. A large native model can be valid if its general competence is learned during N0 and private identity adaptation is regularized correctly. Conversely, a larger model is not justified merely by availability.

---

## 3. Architecture conclusion after the 2025–2026 frontier review

### Decision A — EIPM v1 remains a specialist identity/judgment system

The best-supported v1 architecture is still **not** a full decoder-only conversational LLM.

Evidence converges from several directions:

- Modern encoder work shows strong NLU/retrieval performance-size efficiency versus much larger decoder models.
- R2END and LEAF show that semantic/reasoning capability can be distilled into dense encoders without preserving a teacher as an inference dependency.
- AdaJudge finds that generation-optimized representations are not automatically good fine-grained discriminators and improves reward modeling by explicitly learning discrimination-oriented representations plus adaptive multi-view pooling.
- ranking work shows that small encoders trained on high-quality ordinal data can match much larger models in discriminative settings.
- personalized alignment systems such as P-Check and AlignX benefit from explicit intermediate criteria/preference representations rather than only end-to-end generic generation.
- PersonaAgent and A.L.I.C.E.'s own architecture both separate memory/state from personalized action.

The permanent EIPM should therefore own the identity/judgment function while A.L.I.C.E.'s replaceable reasoning/generation fabric handles broad world knowledge and final prose generation.

### Decision B — structured-first, not structure-only

ACFP remains the primary runtime interface, but the EIPM must also see provenance-bound raw spans where nuance matters.

This is non-negotiable because personality-relevant distinctions include:

- implication versus literal meaning;
- affection versus dependency;
- teasing versus humiliation;
- privacy versus secrecy;
- loyalty versus blind defense;
- disagreement versus rejection;
- sarcasm, irony, register, politeness, indirectness, emotional subtext, and relationship-specific wording.

A pure symbolic frame would discard too much of this. A pure text black box would discard too much governance and provenance.

### Decision C — native bidirectional semantic core is the default baseline family

For v1, the primary student family should be a **from-scratch bidirectional encoder**, not a copied third-party checkpoint and not a decoder whose generation machinery is mostly unused.

ModernBERT is a useful architectural reference, not a base-weight dependency. Its useful ideas include long-context bidirectional attention, RoPE, modern gated MLPs, pre-normalization, efficient unpadding/packing, FlashAttention-compatible kernels, and alternating local/global attention. ModernGBERT demonstrates that scaling a dedicated from-scratch encoder to 1B parameters can remain parameter-efficient and can outperform substantially larger decoder-derived encoders on NLU in its domain.

This is a **family choice**, not a commitment to the exact ModernBERT layer count, hidden size, vocabulary, attention schedule, optimizer, or parameter count.

### Decision D — keep an architecture challenger track

Before freezing the permanent N0 architecture, compare the bidirectional Transformer baseline against at least one materially different challenger if implementation maturity permits:

- bidirectional/hybrid attention + state-space blocks; or
- a tokenizer-free/byte-level challenger; or
- another efficient long-context encoder architecture with equivalent provenance/control support.

The challenger is justified only if it improves the actual N0 competency frontier. Novelty alone is not a reason to adopt it. Current evidence still makes the modern bidirectional Transformer the lowest-risk primary baseline.

---

## 4. Recommended EIPM v1 learned topology

The current strongest architecture is:

```text
provenance-bound raw text spans ──> native bidirectional semantic encoder ──┐
                                                                          │
ACFP typed fields ────────────────> structured-state encoder ──────────────┤
                                                                          ├─> cross-context fusion
persona/evidence graph ───────────> relation-aware graph encoder ─────────┤
                                                                          │
explicit E0-grounded concepts ────> explicit identity concept bank ───────┤
learned missing nuance ───────────> bounded residual concept bank ─────────┘
                                                                                 │
                                                                                 v
                                                                      multi-view latent pool
                                                                                 │
                                  ┌──────────────────────────────────────────────┼──────────────────────────────────────────┐
                                  v                                              v                                          v
                          candidate ranking                           identity-control heads                      uncertainty/provenance
                                  │                                              │                                          │
                                  └───────────────────────────────> Identity Decision Packet <──────────────────────────────┘
```

### 4.1 Native semantic encoder

Requirements:

- bidirectional token interaction;
- efficient long-sequence handling;
- raw-span and candidate-response encoding;
- token-level outputs retained for evidence attribution;
- multi-vector representation support;
- no dependency on third-party inference weights;
- architecture scalable across the parameter ladder without redesigning the interfaces.

### 4.2 Structured-state encoder

Do not flatten ACFP into one JSON string and hope the text encoder learns the schema.

Encode field type, entity/actor identity, temporal scope, confidence, provenance class, relation role, mission/goal role, candidate identity, and missing/unknown status explicitly. Fields should be order-robust where order is not semantically meaningful.

### 4.3 Persona/evidence graph encoder

Use a relation-aware heterogeneous graph architecture over selected nodes/edges rather than serializing the entire graph into text. The graph branch should preserve edge type and provenance and should be able to expose which precedents/relationships contributed to the judgment.

Recent work showing that LLM embedding-space knowledge can be compressed into GNN representations supports optional **training-time** teacher alignment without requiring an LLM at runtime.

### 4.4 Explicit concept bank + bounded learned residual bank

Do not reduce Elaina to Big Five, DIAMONDS, or another generic psychology taxonomy.

The explicit bank should be derived from actual E0-supported values, boundaries, judgment tendencies, relationship modes, communication patterns, and repeated behavioral distinctions.

A learned residual bank is still necessary because any human-authored ontology will miss nuance. Residual concepts must be regularized and auditable enough that they complement rather than silently bypass the explicit representation.

### 4.5 Multi-view latent pooling

Do not rely on a single `[CLS]` vector or one scalar reward head.

Use multiple learned summary/evidence views or latent slots so different evidence can be preserved for semantic meaning, interpersonal stance, values, affect/appraisal, provenance/literal grounding, and candidate comparison. The exact number and interpretation of slots should be learned/ablated rather than hard-coded as personality traits.

AdaJudge's 2026 result is especially relevant here: adaptive multi-view pooling outperforms static readout for fine-grained judgment.

### 4.6 Multi-head, distributional output

The IDP should expose at least:

- candidate ranking / pairwise and listwise preference;
- stance: support / challenge / refuse / question / defer / mixed;
- sparse value-priority activations;
- relationship/interpersonal posture;
- emotional appraisal and response posture;
- communication posture/register;
- literal-grounding / unsupported-inference risk;
- provenance class/evidence sufficiency;
- uncertainty and ranking margin;
- generic-assistant/personality-drift risk;
- evidence/precedent pointers where available.

A single scalar is inadequate. PRISM's 2026 work is directly relevant: compressing subjective preference and uncertainty into one scalar makes the evaluator brittle. For A.L.I.C.E., uncertainty must remain a first-class output so the executive fabric can ask for more evidence, request more candidates, defer, or escalate.

---

## 5. N0 competency contract

N0 is complete only when the native student demonstrates the following competencies on held-out public or legally cleared tests. No single benchmark is authoritative.

### Core language semantics

1. lexical and compositional meaning;
2. paraphrase / semantic equivalence;
3. semantic textual similarity;
4. entailment / contradiction / neutrality;
5. negation, quantification, modality;
6. coreference, reference, deixis;
7. discourse relation and long-context consistency.

### Temporal and causal understanding

8. temporal order, duration, before/after, state change;
9. event causality versus correlation;
10. counterfactual and conditional relations;
11. cause/effect directionality and uncertainty.

### Pragmatics

12. implicature;
13. presupposition;
14. indirect speech acts;
15. literal versus intended meaning;
16. sarcasm / irony / teasing;
17. politeness, register, face-threat, indirect refusal;
18. conversational relevance and omitted-but-implied information.

PUB demonstrates that pragmatic performance varies substantially by phenomenon, so pragmatics must be measured as a vector, not one aggregate score.

### Social and emotional cognition

19. speaker intent and belief attribution;
20. theory-of-mind distinctions;
21. social/relationship-role inference;
22. emotion recognition from context;
23. appraisal: why an event would produce an emotion for a particular actor;
24. relationship-sensitive interpretation;
25. group/social norm reasoning without blindly importing stereotypes.

### Grounding and epistemics

26. literal-grounding versus plausible over-interpretation;
27. known / inferred / unknown distinction;
28. evidence sufficiency;
29. confidence calibration;
30. OOD/novel-situation uncertainty;
31. contradiction between structured frame and raw evidence;
32. provenance-aware fact versus interpretation versus hypothetical distinction.

PaCE is a particularly important warning for A.L.I.C.E.: stronger pragmatics can increase **pragmatic hallucination** if the model over-infers intent. Therefore N0 needs both pragmatic competence and a literal-grounding counterweight.

### Judgment and comparison

33. pairwise candidate comparison;
34. listwise ranking when real ordinal structure exists;
35. multi-criteria comparison;
36. robust comparison under surface-style perturbations;
37. abstention / tie / plural-plausibility when no unique ordering exists.

The 13,719 retained A.L.I.C.E. alternative branches must not be forced into negative labels simply to satisfy a ranking loss.

### ACFP and graph alignment

38. text <-> structured-field semantic consistency;
39. text <-> graph-node semantic consistency;
40. relation prediction and evidence linkage;
41. source-span attribution;
42. missing-field/UNKNOWN awareness;
43. raw-text-versus-frame conflict detection.

---

## 6. N0 objective stack

N0 should be multi-objective. No single teacher or loss should define the semantic space.

### Layer 1 — self-supervised native language learning

Primary source of general competence:

- masked-token / masked-span reconstruction;
- denoising variants where empirically useful;
- sentence/document order or discourse objectives only if they add measured value;
- long-context continuation-consistency objectives that remain bidirectional rather than converting the model into a decoder.

This layer establishes A.L.I.C.E.-owned representations without teacher dependence.

### Layer 2 — semantic contrastive learning

- paraphrase / non-paraphrase;
- STS ordering;
- sentence/paragraph embedding alignment;
- cross-view consistency under meaning-preserving perturbation;
- hard negatives that change meaning while preserving surface overlap.

### Layer 3 — relation / reasoning supervision

- entailment/contradiction;
- temporal and causal relations;
- reference/coreference;
- pragmatic relation labels;
- intent / belief / appraisal relations;
- literal-grounding and over-inference detection.

### Layer 4 — candidate discrimination

Use pairwise and listwise ranking only where the data actually has an ordering.

- listwise loss for clean graded tiers;
- pairwise loss where labels are noisy or comparisons are local;
- tie/plurality targets when alternatives are genuinely co-valid;
- uncertainty-aware margins rather than pretending every preference is deterministic.

### Layer 5 — representation distillation

Teacher guidance is optional and competency-specific:

- embedding-space alignment;
- relation-distribution alignment;
- pairwise/listwise rank distillation;
- calibrated soft targets;
- multi-teacher consensus/disagreement signals.

Do **not** distill unrestricted teacher hidden state or output style into every layer merely because it is available.

### Layer 6 — structure and graph alignment

- typed-field reconstruction/contrast;
- node/edge relation prediction;
- graph-text alignment;
- provenance/evidence pointer supervision;
- raw-span <-> ACFP field consistency.

---

## 7. Teacher strategy — replace “pick Qwen” with competency-routed qualification

There should be no single privileged N0 teacher.

### 7.1 Teacher classes

**Semantic / embedding teacher candidates**

- Qwen3-Embedding family;
- other strong permissively usable embedding models that win the exact qualification suite.

**Ranking teacher candidates**

- Qwen3-Reranker family;
- compact/open reward or ranking models used as references where licenses and task fit are acceptable.

**General pragmatic / reasoning teacher candidates**

- Qwen3.8-27B;
- OLMo 3.1 32B;
- gpt-oss-20b / gpt-oss-120b;
- additional open-weight challengers only after exact license/terms and model revision are recorded.

Qwen3.8, OLMo 3.1, Qwen3 Reranker, and gpt-oss currently expose permissive/open-weight licensing on their official cards. This makes them good **qualification candidates**, not automatic authorities.

### 7.2 Teacher selection procedure

For every N0 competency:

1. define a public/cleared held-out qualification slice;
2. run all eligible teachers blind to one another;
3. measure accuracy, calibration, pairwise consistency, perturbation robustness, and abstention behavior;
4. compare against authoritative dataset labels/human labels where available;
5. retain the best teacher or teacher ensemble **for that competency only**;
6. record exact model revision, tokenizer revision, inference config, prompt/template, and output hash;
7. preserve teacher disagreement instead of majority-voting it away when ambiguity is real.

### 7.3 Authority boundary

Teacher models may teach **general semantics and reasoning structure**.

They do not determine:

- what Elaina historically did;
- what is canonical E0 truth;
- whether an E-INF inference is historically true;
- whether an A-SYN policy should be promoted;
- what Rayan's current state is;
- what A.L.I.C.E. should remember as lived experience.

Those remain A.L.I.C.E./owner-governed authority classes.

### 7.4 Astra boundary

The planned final Astra use remains **advisory pre-weight review**, not automatic training-data generation. Hosted-model output must not enter a commercial model-training corpus unless the applicable terms/contract have been explicitly reviewed for that use.

---

## 8. Tokenizer decision — no frozen vocabulary size

The previous instinct to choose a BPE vocabulary by convention is rejected.

2026 tokenizer research reinforces that tokenization is an architectural co-design decision. TokLens measures fertility, characters/token, compression, normalized sequence length, single-token retention, and cross-lingual parity. MorphBPE further shows that respecting morphology can improve linguistic coherence and downstream performance without simply making tokens shorter.

### Required tokenizer bake-off

Train/compare candidate A.L.I.C.E.-owned tokenizers over the **exact legally cleared N0 corpus**, then audit them on the private EIPM text without using private text to alter public-corpus provenance unless deliberately approved.

Compare at least:

- standard BPE + byte fallback;
- Unigram/SentencePiece-like segmentation;
- morphology-aware BPE if tooling is mature enough;
- byte/tokenizer-free challenger only if the corresponding model architecture is actually being evaluated.

Sweep vocabulary sizes rather than freezing 32K/48K/64K by habit.

Evaluate:

- N0 public corpus compression;
- private E0/E-INF/A-SYN fertility;
- fragmentation of names, relationship language, slang, contractions, punctuation/emojis, and rare words;
- memory footprint of the embedding table;
- sequence-length inflation;
- robustness to misspellings/casing/romanization if present in source material;
- multilingual parity if the actual A.L.I.C.E. roadmap/source material requires multilingual behavior.

Tokenizer choice is frozen only after model-size and context-length tradeoffs are measured jointly.

---

## 9. Public N0 corpus strategy

### 9.1 Default foundation

Use a corpus with explicit source/license provenance as the first-choice foundation. The Common Pile v0.1 is a strong starting candidate because it was constructed from public-domain and openly licensed sources and exposes source-level collection/reproducibility information.

This does **not** mean “download all 8 TB and train blindly.” Build a task-relevant mixture.

### 9.2 Corpus mixture categories

The N0 mix should intentionally cover:

- high-quality general prose and dialogue;
- literature/narrative where legally usable, because social and emotional context matters;
- educational/explanatory prose;
- scientific and analytical prose for causal/epistemic distinctions;
- forum/conversational material only when provenance/license is acceptable;
- structured relation examples;
- high-quality task datasets for NLI, semantics, pragmatics, social reasoning, temporal/causal relations, emotion/appraisal, calibration, and ranking;
- generated curriculum examples only after teacher qualification and lineage capture.

Code should have low weight unless later evidence shows it benefits semantic robustness. EIPM is not the coding model.

### 9.3 License registry

Every dataset/source receives one of:

- `TRAIN_ALLOWED`
- `EVAL_ONLY`
- `RESEARCH_ONLY`
- `REJECT`

with:

- source URL/repository;
- exact revision/date;
- license text/id;
- attribution requirements;
- known restrictions;
- dataset-card caveats;
- extraction/filtering code hash;
- deduplication version;
- final sample count/token count;
- reason for inclusion.

FineWeb may be useful as a research comparison, but its ODC-By dataset wrapper and CommonCrawl origin do not automatically make every underlying webpage equivalent to explicitly open/public-domain training material. Do not mix it into the default commercial-quality foundation without a deliberate legal/product decision.

### 9.4 Contamination control

Before N0 training:

- hash/n-gram deduplicate against every public evaluation set we plan to use;
- hold back benchmark families before synthetic task generation;
- record generated-example ancestry;
- deduplicate near-duplicates across teacher-generated examples;
- keep a locked evaluation manifest separate from training construction.

---

## 10. Evaluation architecture

Do not repeat the old MC10D mistake of treating “more judges” as equivalent to more truth. N0 evaluation should be a diagnostic engineering suite.

### 10.1 Public N0 suite

Build a multi-domain matrix over the 43 competency categories above. Use multiple datasets per important competency and include perturbation/adversarial variants.

The 2026 audit of SocialIQa, FauxPas-EAI, and ToMi found duplicated/ambiguous/implausible items and strong sensitivity to surface phrasing. Therefore no social-reasoning benchmark may be accepted uncritically. Audited/clean subsets and hand-inspected local challenge sets are preferred.

### 10.2 Hidden A.L.I.C.E. generic challenge set

Before private N1 training, create an owner-hidden, personality-neutral set of difficult examples specifically targeting:

- sarcasm versus literal statement;
- teasing versus insult;
- support versus sycophancy;
- disagreement versus rejection;
- ambiguous intent;
- conflicting evidence;
- plausible-but-unsupported inference;
- temporal ambiguity;
- relationship-sensitive register;
- multiple co-valid actions;
- raw-text / structured-frame conflict.

This set tests whether N0 has the **machinery** Alice needs without teaching it Elaina's answers.

### 10.3 Grouped private evaluation later

When N1/N2 begins, split by **underlying evidence/behavior family**, not by random rows.

Rows sharing:

- the same E0 support cluster;
- the same A-SYN base policy;
- the same gap/coverage cell;
- mechanically transformed variants;
- near-duplicate scenarios;

must stay in the same partition. Otherwise evaluation leakage will make personality fidelity look much better than it is.

### 10.4 Owner evaluation remains final identity authority

Automatic metrics can measure consistency, calibration, provenance use, and held-out ranking. They cannot decide whether the resulting character actually feels like the intended Elaina-derived Alice. Final identity fidelity remains owner-evaluated.

---

## 11. Model-size selection — capability scaling, not a target number

### Hard rule

There is **no parameter ceiling and no parameter quota**.

Do not optimize the design to make the answer equal 400M, 1B, 7B, or any other round number.

### Scaling method

Use a staged scaling study:

1. implement one architecture family at a small mechanics scale;
2. train multiple capacities on controlled token budgets;
3. measure N0 competency-vector improvement, not only MLM loss;
4. fit empirical scaling curves for the competencies that still fail;
5. distinguish data/objective bottlenecks from capacity bottlenecks;
6. increase width/depth/latent capacity only when held-out results show capacity is the limiting factor;
7. continue upward until the marginal improvement no longer justifies cost **or** the N0 competency gates are met;
8. preserve the option to continue far beyond 1B/10B if the curve shows real unsolved capacity demand.

A practical experimental ladder may include models in the tens/hundreds of millions, around a billion, and several-billion scale. Those are **measurement points**, not limits. A 100B model remains architecturally permissible if evidence genuinely shows it is required; the project should simply demand proof because the compute/data burden rises enormously.

### What counts as proof that more parameters are needed

- training and validation losses remain capacity-limited rather than data-limited;
- semantic/pragmatic held-outs improve monotonically with capacity;
- difficult candidate-ranking gaps close with scale while calibration does not degrade;
- architecture/data/objective fixes have already been tested;
- owner-hidden generic challenges improve;
- later, private held-out identity fidelity improves without memorization/provenance leakage.

### What does **not** count as proof

- larger benchmark marketing numbers from unrelated LLMs;
- parameter count of the teacher;
- the size of Qwen/OLMo/gpt-oss;
- “bigger is usually better” intuition;
- fitting the private rows more completely.

---

## 12. N0 -> N1/N2 transition

The private identity corpus must not be poured into the whole network indiscriminately on day one.

The N1 mechanics pilot should compare controlled strategies such as:

- train identity heads/fusion first while semantic core is frozen;
- progressively unfreeze upper semantic blocks;
- low learning rate full-model refinement with strong replay of identity-neutral semantic examples;
- graph/concept branch training before broad cross-fusion;
- provenance/literal-grounding head stabilization before preference learning.

Selection criterion is held-out fidelity/generalization, not maximum train accuracy.

### N1 identity representation

Use E0/E-INF/A-SYN with different authority semantics:

- E0: source-grounded identity evidence;
- E-INF: inference with uncertainty and disconfirmation conditions;
- A-SYN: behavioral prior/completion, never historical memory;
- UNKNOWN: historical absence signal, not runtime blank;
- context-only/excluded material: conditioning only, not direct identity target.

### N2 judgment/preference

Train pairwise/listwise identity judgment only on comparisons that actually have preference authority.

Hard negatives should be plausible alternatives with a known reason they are wrong for the current context, such as:

- generic assistant response;
- sycophantic owner-pleasing response;
- emotionally flat neutrality where evidence says otherwise;
- caricature/exaggeration;
- wrong relationship stance;
- unsupported autobiographical claim;
- historically overconfident inference;
- a response that fits another context but not this one.

Do not convert the unordered 13,719 alternatives into rejects.

### N3 calibration

Calibrate:

- preference margins;
- epistemic uncertainty;
- OOD detection;
- provenance confidence;
- tie/plural-policy behavior;
- escalation thresholds.

---

## 13. Engineering lineage requirements

Carry forward the useful MC10 lessons even though the judge tournament is retired.

Every material N0/N1/N2 run must bind:

- source manifests and hashes;
- dataset license registry snapshot;
- tokenizer config + corpus hash;
- model architecture config;
- random seed(s);
- code commit;
- dependency lock;
- CUDA/PyTorch/kernel versions;
- hardware identity;
- teacher exact revisions and generation settings;
- train/eval split manifest;
- optimizer/scheduler config;
- checkpoints and checkpoint hashes;
- metric implementation version;
- promotion decision and owner approval where required.

Do not spend scarce GPU time on a target-scale run until controller, dataloader, checkpoint round-trip, resume, deterministic manifest, and evaluation harness have all passed on a tiny public mechanics run.

---

## 14. What is rejected now

The following shortcuts are explicitly rejected:

1. **Qwen + LoRA as the permanent EIPM.** Qwen may be a disposable teacher; its weights are not A.L.I.C.E.'s permanent identity.
2. **A fixed 400M target.** Superseded.
3. **One teacher for everything.** Teacher quality is competency-specific.
4. **One scalar personality score.** Identity judgment is multi-dimensional and uncertain.
5. **One `[CLS]` vector as the only semantic state.** Multi-view evidence is needed.
6. **Flatten the graph/ACFP into prompt text.** Structure and provenance deserve native encoders.
7. **Treat all synthetic rows as independent evidence.** Mechanical expansion is not new identity evidence.
8. **Treat all alternate policies as negatives.** Many are unordered/plausible alternatives.
9. **Random row-level train/test split.** It leaks behavior families.
10. **More external judges as validation.** Retired except the one planned final pre-weight Astra/owner review.
11. **Use hosted-model output as training data without terms review.** Not acceptable for a startup-bound permanent model.
12. **Train target-scale private weights before N0 architecture/data/tokenizer/eval freeze.** Not authorized.

---

## 15. Concrete next implementation sequence

### N0-R1 — competency/evaluation registry

Create a machine-readable registry for the 43 competency categories with dataset candidate, license status, train/eval role, contamination key, metric, and minimum quality gate. This comes first because architecture choice without a target competency vector is guesswork.

### N0-R2 — public data/license manifest builder

Build a manifest-only pipeline that inventories candidate public sources, licenses, checksums, and sample counts without training anything.

### N0-R3 — tokenizer bake-off harness

Train candidate tokenizers on a small legally cleared corpus subset. Report TokLens-style metrics plus private EIPM fertility/fragmentation statistics without exporting private text.

### N0-R4 — teacher qualification harness

Evaluate candidate teachers per competency. Preserve disagreement and exact lineage. Do not generate the full synthetic curriculum until this gate is passed.

### N0-R5 — architecture mechanics matrix

Implement tiny public-data versions of:

- modern bidirectional Transformer baseline;
- multi-view pooling;
- structured-state encoder;
- graph encoder;
- fusion;
- multi-head IDP outputs;
- one challenger architecture if feasible.

No private identity gradients.

### N0-R6 — scaling pilot

Train several model capacities × data budgets on public data, fit empirical scaling curves, and determine whether the first target-scale candidate should be hundreds of millions, ~1B, several billion, or larger.

### N0-R7 — N0 target freeze

Freeze:

- corpus versions/mix;
- tokenizer;
- student architecture;
- teacher routes;
- objectives and weights/schedule;
- model scale;
- context length;
- optimizer/scheduler;
- checkpoint cadence;
- evaluation gates;
- lineage manifest.

### N0-R8 — final pre-weight review

Give one concise frozen bundle to owner + Astra for material architecture/data mistakes. Fix material issues once.

### N0-R9 — train first permanent native semantic checkpoint

Only after the above.

Then proceed to N1/N2/N3.

---

## 16. Current decision summary

As of 2026-09-13, the strongest research-supported direction is:

- **EIPM role:** A.L.I.C.E.-native identity/judgment policy, not general-purpose LLM.
- **Runtime interface:** ACFP in, IDP out; raw provenance-bound spans available when nuance matters.
- **Text core:** scalable from-scratch bidirectional encoder family as primary baseline.
- **Other branches:** typed structured-state encoder + relation-aware persona/evidence graph encoder.
- **Identity representation:** E0-grounded explicit concepts + bounded learned residual concepts.
- **Fusion/readout:** cross-context fusion + adaptive multi-view latent pooling.
- **Output:** vector/distributional multi-head judgment + uncertainty, not one scalar.
- **N0 training:** self-supervised semantics + contrastive/relation/pragmatics/ranking/structure objectives + optional competency-specific distillation.
- **Teachers:** multiple qualified disposable teachers, no single Qwen dependency.
- **Tokenizer:** A.L.I.C.E.-owned and empirically selected; no fixed vocabulary size.
- **Public data:** provenance/license registry, open/public-domain-first foundation, contamination-controlled task mixture.
- **Scale:** unrestricted; chosen from measured capability scaling, not a 400M target.
- **Private data:** enters only after N0; grouped by evidence/behavior family; provenance semantics preserved.
- **External validation:** old tournament stays retired; final owner/Astra pre-weight audit remains.

This is sufficiently specific to begin the **N0 research/qualification tooling**, but not yet sufficient to start target-scale training.

---

## 17. Key references used for this decision

Primary/reference sources reviewed in this pass include:

- Warner et al., **ModernBERT**, ACL 2025 — https://aclanthology.org/2025.acl-long.127/
- Ehrmanntraut et al., **ModernGBERT**, 2025/2026 publication cycle — https://arxiv.org/abs/2505.13136
- Vujanic & Rückstieß, **LEAF**, ACL 2026 — https://aclanthology.org/2026.acl-long.2008/
- Han et al., **R2END**, Findings ACL 2026 — https://aclanthology.org/2026.findings-acl.1130/
- Miao et al., **AdaJudge**, ACL 2026 — https://aclanthology.org/2026.acl-long.440/
- Zhou et al., **PRISM**, ACL 2026 — https://aclanthology.org/2026.acl-long.563/
- Seo & Lee, **P-Check**, ACL 2026 — https://aclanthology.org/2026.acl-long.2011/
- Li et al., **AlignX**, ACL 2026 — https://aclanthology.org/2026.acl-long.1391/
- Cai et al., **ThinkPersona**, ACL 2026 — https://aclanthology.org/2026.acl-long.449/
- Cheng et al., **PsyMem**, TACL 2026 — https://aclanthology.org/2026.tacl-1.24/
- Li et al., **Trait Activation in Silicon / PD-LLM**, ACL 2026 — https://aclanthology.org/2026.acl-long.1792/
- Zhang et al., **PersonaAgent**, Findings ACL 2026 — https://aclanthology.org/2026.findings-acl.1315/
- Chen et al., **Compressing LLM Knowledge into Graph Representations**, ACL 2026 — https://aclanthology.org/2026.acl-long.1398/
- Bailleux et al., **Explanation Quality Assessment as Ranking with Listwise Rewards**, Findings ACL 2026 — https://aclanthology.org/2026.findings-acl.1800/
- Li et al., **PaCE / Pragmatic Hallucination**, Findings ACL 2026 — https://aclanthology.org/2026.findings-acl.959/
- Sravanthi et al., **PUB Pragmatics Understanding Benchmark**, Findings ACL 2024 — https://aclanthology.org/2024.findings-acl.719/
- Sravanthi et al., **ImpliedMeaningPreference**, Findings ACL 2025 — https://aclanthology.org/2025.findings-acl.1218/
- Mousavi et al., **social-reasoning benchmark audit**, Findings EACL 2026 — https://aclanthology.org/2026.findings-eacl.89/
- Alqahtani et al., **Stop Taking Tokenizers for Granted**, EACL 2026 — https://aclanthology.org/2026.eacl-long.394/
- Chiu, **TokLens**, ACL SRW 2026 — https://aclanthology.org/2026.acl-srw.18/
- Asgari et al., **MorphBPE**, Findings ACL 2026 — https://aclanthology.org/2026.findings-acl.2068/
- Kandpal et al., **Common Pile v0.1**, 2025 — https://arxiv.org/abs/2506.05209
- official Qwen3.8-27B / Qwen3-Reranker model cards — https://huggingface.co/Qwen
- official OLMo 3.1 model cards — https://huggingface.co/allenai
- official gpt-oss model card — https://openai.com/index/gpt-oss-model-card/
