# Frontier Watch intake — 2026-11-02

**Decision:** Eight primary-source findings from the October 29–30 submission window newly surfaced in this screening. The strongest deltas are (1) memory formation must distinguish an authorized change from a poisoned or third-party-supplied change without reducing normal retention; (2) personal learning must generalize a rule instead of copying feedback examples; (3) personal memory performance must be evaluated through changing state, higher-order inference, and actual decision behavior; and (4) source-grounded retrieval should establish the question's conditions, not only rank similar records. Cross-interface authorization and privacy-flow studies provide further scoped Mission/egress qualifications. **No architecture-invalidating evidence, selected-implementation reopening, current MFM recipe change, or new infrastructure benchmark.** These are future conditional methods and evaluations, not reasons to block ongoing model work.

## Required baseline, current heads, and execution inspection

Inspected this branch's mandatory `docs/research/FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md` and October 31/November 1 intakes; A.L.I.C.E. main README, `docs/ALICE_PHASE2_REPLACEMENT_AND_FABLE_V1_EXECUTION_PLAN_2026-09-27.md`, `docs/FABLE_PERSONAL_DEVELOPMENT_ARCHITECTURE.md`, Memory v4.1; Fable main README, `docs/FIRST_RELEASE.md`, its September 27 execution profile; active MFM's checkpoint-10 diagnosis and October 22 observed-world training note; the FBM/N0 and continuity branch heads. Verified directly through GitHub, before evaluating research:

| Reference | Verified head 2026-11-02 | Since November 1 |
|---|---|---|
| A.L.I.C.E. `main` | `a1fe0a65df1d4143955eb1d3b35eb4ac590e7555` | unchanged |
| Fable_Sleight `main` | `2b5fe3b3c4ded600e1e19a4940a87d9468470a03` | unchanged; previous README PR #5/#6 already tracked |
| MFM `research/mfm-v1-licensed-base-20260930` | `5653736f545c9c1c8d5d8006af0f1fa152eed9b1` | unchanged |
| FBM `research/fbm-v1-licensed-base-20260930` | `bcd462b5bb93f41a56f79b5b63e3337c8e89bb45` | unchanged |
| N0 `alice-eipm-v1-n0-full-envelope-joint-ddp-v2` | `5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2` | unchanged |
| `alice-context` | `945245bbee4168fff46f863455a83bb219063017` | unchanged |
| Graphify continuity | `a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15` | unchanged |
| `research/frontier-watch` before this intake | `dbf548468be3ecfb52826dd0439556906b9bf663` | November 1 intake |

**Actual training qualification, not project aspiration:** MFM checkpoint-10 achieved **0/9 valid native grounded formations**, despite lower teacher-forced losses. A 16-update continuation toward checkpoint 26 was dispatched, but **no new committed native generation result** establishes its completion or repairs the failure. N0/personality conditional judgment remains unqualified in reviewed engineering records. The Rill observed-world teaching extension has recorded source/rights/geometry construction evidence, **not trained native MFM success**. No finding below provides a demonstrated repair. The immediate priority remains the selected numerical/training continuation and behavioral qualification.

**Do not mistake an older Fable September physical profile for current owner selection.** Physical defaults remain KurrentDB canonical Experience events, NATS JetStream edge/federation ingress, XTDB v2 bitemporal Claims, encrypted content-addressed evidence, LadybugDB local and NebulaGraph distributed graph, engine-independent associative graph computation, Qdrant multimodal, exact/live source reads, Temporal, local L1/Valkey, signed/content-addressed lineage, PyTorch+Accelerate. Hardware changes placement only. F4–F11 are internal gates, full first release after F11, F12 post-v1. No cognitive plane or model count may be capped from a paper's benchmark.

## 1. Accept Changes, Reject Attacks — coupled correction and poisoning robustness

**Source/date:** Runze Zhao et al., *Accept Changes, Reject Attacks: Robust Memory Management for LLM Agents*, arXiv:2610.41056v1, submitted **October 29, 2026**. https://arxiv.org/html/2610.41056

**Class:** **Future challenger** for native formation admission and derived-state lifecycle, with a focused evidence-quality test. **Not** an established current MFM defect or qualified training remedy.

**Method and results:** One memory manager both writes ADD/UPDATE/NOOP actions and selects retrieved evidence. The authors distinguish standard inputs, legitimate fact supersession (explicit or implicit), and attacker-planted instructions/passages. Their evidence-conditioned accuracy (ECA) accepts a response only if both the answer is correct and selected memory contains the necessary current and attack-free supporting records. Static write-all and reject-all approaches necessarily trade stale-information failures against poisoning failures. Around one 41-turn LoCoMo conversation, authors synthesize 400 mixed episodes and ~21,656 training rows, supervised fine-tune Qwen3-4B/Qwen3.5-4B/Ministral3-3B, and test against Mem0, A-Mem and defenses. For Qwen3-4B, standard-category ECA **57.3%** (A-Mem 50.3%; untuned 40.6%), outdated ECA **28.5%** (best cited baseline 11.2%); out-of-distribution examples LongMemEval **36.7→61.7%**, BEAM-KU **6.2→35.0%**. All three categories improve over untuned on all three backbones; the system is still far from reliable.

**A.L.I.C.E. already:** evidence custody separate from Claim acceptance, bitemporal correction, source/subject authority, MFM proposal not canonical write, Lifecycle Curator, source-native verification, deletion/rebuild. **Genuinely new:** same training/evaluation protocol deliberately pairs *legitimate change* with *untrusted change* and scores the correctness of selected evidence separately from answer quality. This exposes an admission trade-off not captured by plain memory QA.

**Governance/cost/risk:** No model output can create authorized Claim, permission, or host identity. Treat a source-person or third-party assertion about the host as source evidence, not automatically as a host correction. Keep old historical evidence under applicable retention permission; delete/revoke via existing lineage. Synthetic supervised training adds generation+fine-tuning cost and inference over selected top-eight memory candidates. Experimental labels, judge and attacks are artificial and may miss complex real source poisoning.

**Action:** **Future only.** At actual native formation admission, one contrast set pairing authorized host change, third-party contradictory report, embedded prompt injection, unchanged fact, and valid evidence selection. Avoid adopting their SFT recipe until current checkpoint-26 native behavior is measured. No current training or production change.

## 2. CLR-Bench — memorizing corrections does not prove learning a rule

**Source/date:** Chenjie Qiu et al., *More Rules, Longer Lists? Benchmarking Self-Evolving Agents on a Hidden-Rule Curriculum*, arXiv:2610.40758v1, submitted **October 29, 2026**. https://arxiv.org/html/2610.40758

**Class:** **Future challenger** for FloRA native personal-judgment/procedural-generalization qualification.

**Method and evidence:** Twenty staged chains across ten domains hide a named grouping behind an opaque label. Feedback reveals one included and one excluded member; later tasks contain unseen members and close non-members, allowing the researchers to distinguish generalization from a growing lookup list. Base-model probes ensure the grouping was knowable given its name but not guessable from the initial feedback. Six self-evolving external-memory schemes scarcely improved on a controlled base model (at most **+5 points** over frozen); excluded only **6–12%** of unseen members compared with ~**20%** from simply keeping raw practice history. Their persistent states commonly copied **75–98%** of seen members without recovering the class. An explicit general inference convention raised scores **14.5–16.5 points**. Under a stronger GPT-6-Sol backbone, ACE and Dynamic Cheatsheet improved substantially, attaining **78%** accuracy and **59–60%** single-cell success against **53% and 8%** on the original certifying setup. Model competence materially changes findings.

**Already:** Experience/outcome→judgment→governed revision requirement; FloRA frozen/direct-personalization controls; procedural learning; source-native and model lineage. **New:** evaluation can specifically reject apparent learning explained by copying examples or accumulating exceptions, while requiring transfer to unseen covered cases.

**Governance/cost/risk:** Inferred rule is a revisable hypothesis, not Claim truth, scope or permission. Attribute it to training/outcome evidence and host-specific model generation; deletions/revocations invalidate associated derived policies. The 20-chain benchmark is artificial, source-class knowledge is a key assumption, repeated rollouts are costly, and its fixed rule schema is **not** an Fable ontology.

**Action:** **Future only.** During FloRA learned-judgment qualification, introduce one held-out unseen case, similar nonmember, and raw-history control to distinguish inference from memorized correction. Do not run a new general benchmark program now.

## 3. P-ContextBench — personal fact recall vs actual personal behavior

**Source/date:** Grant Nasir Brown et al., *P-ContextBench: From Personal Fact Recall to Personalized Agent Behavior*, arXiv:2610.40959v1, submitted **October 29, 2026**. https://arxiv.org/html/2610.40959

**Class:** **Future challenger** for full host/relationship/time-native judgment evaluation.

**Method and evidence:** Synthetic coherent digital people with 100 personas, 10,088 hidden facts, 18,261 messenger/calendar fragments, and **4,810 verified tests**, spread relevant evidence over changing app history. Five task families measure single-source recall, cross-source recall, current-state reconstruction, higher-level behavioral pattern inference, and whether to comply/warn/clarify/refuse under an implicit personal constraint. Cost-effective candidate LLMs score below **60%** on temporal/pattern/action-boundary tasks and under **10%** of cases are correct in three repeated trials; a stronger model reaches ~**75.5%** overall but **34.6%** fully consistent in repeated trials. Error analysis: **81.01%** of temporal task errors omit a needed current-state fact, **64.78%** of pattern-inference errors omit a supporting fact; constraint recognition and over-refusal also fail.

**Already:** host/self/source-person/relationship separation, temporal Claims and original evidence, native judgment, Mission and consent/owner sovereignty, FloRA held-out comparison obligations. **New:** unified *fragmentary evidence→inferred current host condition→actual appropriate decision* and repeatability test; a response can retrieve facts yet fail to act consistently.

**Governance/cost/risk:** Generated personal behavioral labels and scenario assumptions are not ground truth about real users; never infer protected traits or implicit approval from a synthetic profile. Use authorized source evidence and uncertainty; enforce separate authority for actions and independent correction. Constructing/judging thousands of synthetic cases is expensive, with possible demographic and evaluation-model bias. No paper warrants importing the full synthetic corpus or constraining Fable to its task taxonomy.

**Action:** **Future only.** Reuse just the task distinction when FloRA native judgments are testable: one changing host fact, one implicit pattern supported by multiple sources, one constraint-aware authorized decision, repeat under stochastic seeds and a provider swap. Current model qualification comes first.

## 4. KITE — compile question conditions into inspectable retrieval decisions

**Source/date:** Feng Li et al., *KITE: Retrieval as a Decision Procedure for Long-Term Conversational Memory*, arXiv:2610.41521v1, submitted **October 30, 2026**. https://arxiv.org/html/2610.41521

**Class:** **Future challenger** for Cognitive Recollection's conditional / temporal / multihop source selection; **not a Qdrant challenger or vector-removal proposal**.

**Method and ablations:** A governed, expandable topic index and typed entity/participant/temporal facets are built from conversation evidence; questions compile into inspectable retrieval conditions and deterministic lookup/ranking plans, backed by lexical exact-detail recovery. LoCoMo **93.5%** and LongMemEval-S **86.0%** with ~1.5–1.6k reader-context tokens. Importantly, under a frozen memory store, LoCoMo **93.5** vs hybrid **87.8** and dense **85.4**; LongMemEval **86.0** vs hybrid **78.0**, dense **76.9**. Removing facets lowers LoCoMo to **90.0**, indexed-only to **89.5**, lexical-only **81.2**; pure recency **60.6**. A governed topic-growth policy beats immediate uncontrolled expansion (93.5 vs 87.9). Accuracy is judged on finite QA benchmarks and may depend on schema conventions and correct query compilation.

**Already:** XTDB bitemporal authority, Cognitive Multi-Graph and associative compute, Qdrant dense/sparse, source-native lexical, MRM, Retrieval Orchestrator and derived role-aware context. **New:** index and retrieval are linked through an *explicit checkable question-condition plan*, not just better semantic ranking. This is not a fixed ten-topic memory model for all users.

**Governance/cost/risk:** Retrieved plan and topic confidence are projections, never Claim Authority; must apply identity, source, entitlement, valid-time, revocation and deletion filters before admission. Construction and LLM query interpretation add CPU/embedding/model inference and index maintenance; a bad plan can systematically omit valid evidence. Source-native fallback remains necessary.

**Action:** **Future only.** If a real current-vs-historical or multi-source query fails because similarity ranking does not satisfy required conditions, implement one derived plan-and-source sufficiency comparison over existing selected stores. No retriever/backend tournament.

## 5. Cross-Interface Authorization — scope/commit validity when tools can achieve equivalent effects

**Source/date:** Genliang Zhu and Chu Wang, *Cross-Interface Authorization for AI Computer-Use Agents*, arXiv:2610.41397v1, submitted **October 30, 2026**. https://arxiv.org/html/2610.41397

**Class:** **Future challenger** for operational Mission authorization and outcome receipts.

**Mechanism/evidence:** Normalize GUI/CLI/API/script/MCP commands to canonical effects; enumerate effect-bearing routes and sinks; bind a prepare ticket to route/effect/grant/current profile and check actual authoritative state again at commit. A receipt proves the sink result. Formal results are **profile-bounded**; a simulated reference maps 12 effect families, 5 interfaces and 60 registered routes, with **3,464 traces**, 26/26 semantic regressions detected, 24/24 coverage mutants and 720/720 normalizer mutants rejected. This is not an externally qualified guarantee for arbitrary unregistered real tools.

**Already:** owner action authority, Mission Graph, permission/scope/revocation contracts, Temporal durable workflow and verified Experience outcomes; previously tracked I-HITL/Concordat. **New:** an approved action reached through another interface, or committed after authority changes, can escape a correct single-entrypoint permission check. Authorization must follow *effect and final sink*, not model's route name.

**Governance/cost/risk:** Do not allow model-controlled approval or provider-confirmed success to substitute for independently verified owner grant and authoritative sink receipt; be careful with pending/irreversible effects. Canonical normalizers and commit checks add engineering and per-action costs; blind spots in route inventories invalidate formal assumptions.

**Action:** When real cross-interface actions ship, one focused matched-effect GUI-versus-CLI/API test with mid-flight revocation and receipt. No present tool gateway change and no audit of every Cartesian route/backend combination.

## 6. PrivacyFlow — anticipate what recipients can infer across an evolving task

**Source/date:** Robin Staab et al., *PrivacyFlow: Privacy in AI Agents through Information Flow Analysis and Consequence-Aware Risk Management*, arXiv:2610.40714v1, submitted **October 29, 2026**. https://arxiv.org/html/2610.40714

**Class:** **Future challenger / corroboration** to prior Task-Graph Contextual Integrity and InferPriv work, with a stronger dynamic-sequence test.

**Method/evidence:** PrivacyBench includes **5,873 steps in 986 task sequences** with changing environments and permitted disclosure sets. A deliberate planning stage traces provenance, recipient and downstream purpose, background knowledge and inferable private facts; it chooses minimal release, user clarification, or block before delegated execution. Across six evaluated models, fully privacy-preserving completed sequences increase **22–55 percentage points**. Tradeoff: additional planning/recipient inference and user confirmation improve privacy but may degrade task utility and reduce autonomy; human attention and judge assumptions matter.

**Already:** owner egress authority, data classification, Mission dependency graph and history, source/host/other-person separation, privacy minimization and cumulative leakage checks. **New:** a measurable multi-step *recipient-inference* risk under dynamic route changes rather than only direct leaks or fixed flows.

**Governance/cost/risk:** A model-estimated privacy risk cannot waive mandatory owner consent or authorize sharing; downstream inference may be hard to predict. Extra planner calls, simulation and approvals have nontrivial runtime/tokens/user burden. Benchmark is controlled, not proof of no leakage.

**Action:** **Future only.** One authorized A→B disclosure with an unauthorized downstream recipient plus a changed route; evaluate cumulative inferred sensitive content and user-attention cost. No additional privacy platform or mandatory always-on model calls now.

## 7. BlindToChange — triggering checks for unstated external changes

**Source/date:** Anirudh Phukan et al., *BlindToChange: AI Agents Can Find Changes When Asked But Do They Look When Needed?*, arXiv:2610.41435v1, submitted **October 30, 2026**. https://arxiv.org/html/2610.41435

**Class:** **Future challenger** for proactive world/source-native validation in Mission planning.

**Method/evidence:** Construct paired direct-vs-indirect requests around recent real-world changes from Wikidata. Agents can locate the change when directly asked, but fail when the user merely requests a recommendation requiring discovery that an assumed fact has become false. Gaps reach **27.5 points at high effort** in the paper's reported comparison; more than half of failures arise from never searching to test the latent assumption. Other harness comparisons show highly model-dependent gaps; stronger general searching doesn't ensure the relevant check gets triggered.

**Already:** public web research, freshness and provenance, source-native/live reads, Mission dependency validity. **New:** model should identify *which unstated prerequisites are high-impact and changeable* before assuming stored-world state remains applicable.

**Governance/cost/risk:** Prefer consequence/freshness-based checks instead of checking every possible fact, to protect privacy, latency and compute. New web evidence updates Claims through authority, not directly from search snippets; an external source can be wrong.

**Action:** **Future only.** On actual live Mission planning mistakes, test one hidden change to a decisive dependency plus a matched explicit-check control. No unconditional web-search or new scheduler.

## 8. KEEPER — optimizing memory summaries can reward persisting falsehoods

**Source/date:** Jinkyeong Choi et al., *KEEPER: Learning Control-Sufficient Textual Memory for Long-Horizon LLM Agents*, arXiv:2610.41482v1, submitted **October 30, 2026**. https://arxiv.org/html/2610.41482

**Class:** **Already covered/corroboration** of generated-memory-versus-teacher-forced qualification, with a conditional future training objective challenger.

**Method/limitations:** Compare auxiliary reward-preservation (RP), predictive preservation (PP), semantic anchoring (SA), update stability (US) during recursive memory supervised learning under a shared STEP-HRL agent, then evaluate some after offline RL on ScienceWorld/ALFWorld. RP and US improve task results relative to plain behavioral cloning; RP strongest and all-four combined worse than RP. One tracked baseline example shows false claims persisted despite corrective observations, reappeared after correction, or disappeared without proof of being fixed. **Authors do not establish a population-level reduction of false-memory persistence**, because they analyze one trajectory; shared action/memory parameters confound attribution and teacher-forced hidden-state objectives cannot validate the generated memory. Second backbone covers ALFWorld only.

**Already:** MFM generated-formation gate, Claim/Evidence distinction, correction/revocation lineage, validation beyond loss; checkpoint-10 0/9 formation despite improving loss. **New narrower lesson:** optimizing many theoretically complementary auxiliary objectives together is not necessarily better; evaluate *false persistence, valid retention and actual policy result separately*.

**Cost/governance/action:** Auxiliary heads add training expense and gradients, though removed for runtime; wrong corrections can be entrenched. Preserve source native truth and authorized corrections; never let recurrence establish Claim authority. **No current MFM objective changes**. Future focused native-generation diagnostics may inspect post-correction recurrence once a model actually generates valid outputs.

## Screened but no additional present action

- *Context as Infrastructure: Knowledge Compilation for Retrieval in Agentic Systems with Harnesses* (arXiv:2610.41172, October 30) reports little accuracy benefit for extra structural artifacts with capable BM25/grep harnesses but real context cost effects for claim–evidence units. **Already covered** by source-native evidence and query control; not an infrastructure reason. https://arxiv.org/html/2610.41172
- *ImplicitHop: Evaluating Knowledge Integration across Context and Memory* (arXiv:2610.41169, October 30) shows sharp composition failure from isolated parametric facts vs facts presented in context. **Corroborates** explicit graph operations/evidence assembly. No new model/graph engine. https://arxiv.org/html/2610.41169
- *Real Computation or Merely Naming?* (arXiv:2610.41604, October 30) finds many nominal cognitive components do not causally affect outputs in its audited systems. **Corroborates** current interventional behavior requirements and the distinction between 63 synthetic FloRA backbone tests and actual native judgment. Does not warrant a new cross-system audit campaign. https://arxiv.org/html/2610.41604
- *Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution* (arXiv:2610.41453, October 30) adds finite-model evidence for uncertain external effects and receipt-bound recovery. Existing Temporal/Mission/Experience contracts express it; **future challenger** if a real ambiguous-commit failure occurs. No current workflow change. https://arxiv.org/html/2610.41453

## Disposition and operational decision

**No architecture-invalidating evidence. No observed new committed engineering qualification. No selected database/transport/model-serving substitutions, ML recipe edits, production/main mutation or PR.** Research only; F4–F11 stay internal and the full personal cognitive foundation is the only consumer release after F11.

Current model work remains the critical path. Do not use research papers to bypass native generation, self/host/source-person authority, current revocations, historical evidence, reversible model/data lineage, or learned-behavior qualification. Each future finding is conditional on a *specific measured failure* or an already planned capability boundary and must be evaluated with the smallest test that can actually change the next decision.
