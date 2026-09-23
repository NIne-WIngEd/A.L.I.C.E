# Memory Use Calibration and Control Qualification

**Version:** 1.0.0  
**Status:** normative Stage G / adaptive-context qualification supplement  
**Authority boundary:** supplements the Stage G candidate matrix; does not transfer Claim, Experience, identity, or deletion authority to a model or retrieval controller

## 1. Why this contract exists

A correct memory store and a correct retriever are not sufficient. After memory reaches a reasoning model, the system must still decide how strongly each memory proposition should influence the answer or action. A retrieved item can be current and authoritative yet irrelevant to the present query. Another item may provide local context without being allowed to control the conclusion. A third may encode a current owner constraint that must materially govern the result.

The qualification target is therefore not simply retrieval precision. It is **calibrated memory influence**.

This contract is motivated by MemCalib (Cao et al., 2026, arXiv:2609.24259), which operationalizes proposition-level memory use as `IGNORE`, `BOUND`, or `CONTROL` and shows substantial over-use/under-use errors across frontier models. The paper also reports a calibration seesaw in common post-training methods: improving one direction can worsen the other. A.L.I.C.E. adopts the evaluation insight and counterfactual-test pattern, not the paper's authority model.

## 2. Canonical influence classes

For an invocation and a supplied memory proposition, A.L.I.C.E. evaluates an expected influence class:

- `IGNORE`: the proposition must leave no answer-specific footprint for this invocation.
- `BOUND`: the proposition may provide local support, context, ranking preference, or scoped adaptation, but cannot independently control a material conclusion.
- `CONTROL`: the proposition materially constrains or determines a conclusion, recommendation, action, refusal, or plan within its valid scope.

These are **invocation-scoped influence labels**, not truth labels and not memory-authority levels. A proposition's admissible influence depends on current query/task scope, provenance, authority, validity time, transaction state, uncertainty, contradiction state, correction/deletion/revocation state, and applicable policy.

The same underlying Claim or evidence may be `CONTROL` for one query, `BOUND` for another, and `IGNORE` for a third.

## 3. Authority precedence

Influence calibration cannot override the existing authority architecture.

1. Raw Experience/evidence remains evidence.
2. Claim Fabric remains authoritative for governed claims and current/historical truth semantics.
3. Graph, vector, episode, summary, host/source/self/relationship/mission, usage-aware retrieval, and other projections remain projections.
4. Retrieval rank, co-retrieval frequency, embedding similarity, graph reachability, model confidence, or repeated use cannot upgrade authority.
5. Deleted or revoked material cannot receive active influence merely because a model predicts that it would be useful.
6. An inference, prediction, reconstruction, or outside-source statement cannot become an owner statement through high influence.

Deterministic authority/policy gates establish what is eligible to influence an invocation. Learned components may estimate influence only inside that eligible envelope.

## 4. Required Stage G qualification

Every end-to-end memory/context configuration that can influence reasoning must be evaluated on proposition-level influence calibration.

The test corpus must include composite memory blocks and mixed retrieval sets containing, at minimum:

- current controlling constraints;
- relevant but locally scoped supporting facts;
- irrelevant but semantically similar memories;
- outdated and superseded memories;
- contradicted/disputed memories;
- uncertain inferences and predictions;
- source-person vs host vs A.L.I.C.E.-continuity material;
- temporary/session-only state;
- preferences whose applicability changes by task;
- malicious or misleading retrieved material;
- duplicate and near-duplicate evidence;
- graph/vector/summary projections paired with their authoritative sources;
- corrected, deleted, and revoked material in stale derivative stores;
- rare but decisive constraints mixed with many weakly relevant memories.

For every proposition, define the expected invocation-scoped influence class and the reason it is eligible for that class. Evaluation must report at least:

- exact influence calibration;
- over-use rate and severity;
- under-use rate and severity;
- results stratified by authority/provenance class, lifecycle state, temporal validity, identity namespace, retrieval source, and task type;
- critical failures separately from aggregate scores.

An aggregate score cannot hide a `CONTROL` proposition that was ignored or an `IGNORE` proposition that changed a material conclusion.

## 5. Counterfactual atom tests

For each important proposition, qualification must include exact or semantically controlled atom-level counterfactuals where feasible:

1. remove the proposition while keeping the rest of the context fixed;
2. restore it;
3. replace it with a controlled contradiction, outdated version, or lower-authority analogue where the contract permits;
4. compare answer/action changes and token-/decision-level influence;
5. verify that observed influence matches the expected `IGNORE`, `BOUND`, or `CONTROL` class.

These counterfactuals are evaluation evidence. They do not themselves grant authority to a learned judge.

## 6. Training and optimization rule

A future memory-use model or EIPM/context component may be trained to improve influence calibration, including with bidirectional counterfactual credit localization inspired by MemCalib-RL. However:

- over-use and under-use must be optimized and reported separately as well as jointly;
- a gain in one direction cannot silently compensate for regression in the other;
- held-out provenance groups, identities, temporal-update cases, corrections, and deletion/revocation cases are required;
- static rubric/judge performance must be checked against downstream behavior;
- evaluator disagreement and calibration must be measured;
- the training signal may shape model behavior but cannot redefine Claim authority or provenance.

No specific RL algorithm is permanently mandated. MemCalib-RL is a challenger/seed, not a ceiling.

## 7. Lightweight memory-control challenger

Jev-Mem (Jiang, Li, and Li, 2026, arXiv:2609.23986) provides evidence that frequent bounded memory decisions can be moved off an autoregressive reasoning model and into a lightweight structured controller. A.L.I.C.E. should evaluate this as a **System-One memory-control challenger** under the existing Context Planner / Retrieval Orchestrator / Memory Formation architecture.

Candidate bounded decisions include:

- memory/query typing;
- source/view routing;
- candidate scoring;
- retrieval-budget allocation;
- graph-expansion decisions;
- evidence-sufficiency estimates;
- adaptive stopping;
- redundancy filtering;
- proposal-level relation typing.

The challenger must use typed outputs/probabilities and should batch compatible decisions when possible. It must escalate uncertain, novel, authority-sensitive, contradictory, or open-ended cases to stronger deliberative reasoning or deterministic policy as appropriate.

### Qualification against current architecture

Compare the lightweight controller with the current/baseline control path on:

- authority correctness;
- evidence recall and citation-lock compliance;
- influence calibration;
- correction/deletion/revocation behavior;
- identity-boundary correctness;
- p50/p95/p99 latency;
- model calls and generated tokens;
- CPU/GPU/memory cost;
- adaptive stopping quality;
- failure under distribution shift and adversarial inputs.

A faster controller fails if it weakens authority, provenance, or evidence guarantees.

## 8. Parametric-memory lifecycle research seed

RPMem (Zhao et al., 2026, arXiv:2609.23466) introduces a model-independent recurrent latent memory decoded into backbone-specific LoRA parameters. Its most relevant architectural idea for A.L.I.C.E. is **backbone-lifecycle-independent parametric adaptation**: learned personal/procedural state need not be irreversibly coupled to one serving model generation.

This is a future Track H challenger, not a replacement for Claim/Experience memory. Any A.L.I.C.E. parametric-memory experiment must preserve:

- source/evidence lineage outside the latent state;
- rebuildability from authorized source datasets/manifests;
- correction/deletion/revocation propagation and machine-unlearning/retirement tests;
- versioned latent state, compiler, decoder, and target-backbone identities;
- rollback to a non-parametric authoritative path;
- explicit tests for consolidation order/path dependence;
- cross-backbone transfer without silently changing identity or memory influence;
- comparison against retrieval/context and ordinary adapter baselines.

Fixed-size latent memory is a performance substrate, not an authority store. If a latent state cannot be traced, corrected, retired, or rebuilt under A.L.I.C.E.'s governance, it cannot become the sole durable representation of personal memory.

## 9. Acceptance implications

Stage G/G2 memory/context qualification is incomplete until the implementation evidence demonstrates:

1. proposition-level `IGNORE` / `BOUND` / `CONTROL` calibration across governed memory classes;
2. separate over-use and under-use measurements with critical-case reporting;
3. counterfactual atom tests for decisive and high-risk memory influence;
4. no authority escalation through retrieval or model influence;
5. preserved citation-lock, correction, deletion, revocation, temporal, and identity boundaries;
6. any lightweight control challenger is evaluated for correctness before efficiency promotion.

RPMem-style parametric memory remains future Track H research and does not block Stage G closure unless a parametric-memory candidate is introduced into the active Stage G serving path.

## 10. Primary research sources

- Ruike Cao et al., **MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents**, arXiv:2609.24259, submitted 2026-09-21. https://arxiv.org/abs/2609.24259
- Dongming Jiang, Yi Li, Bingzhe Li, **Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents**, arXiv:2609.23986, submitted 2026-09-21. https://arxiv.org/abs/2609.23986
- Fanyu Zhao et al., **RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents**, arXiv:2609.23466, submitted 2026-09-20. https://arxiv.org/abs/2609.23466
