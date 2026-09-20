# A.L.I.C.E. Frontier Research Watch — 2026-09-20

**Branch:** `research/frontier-watch`  
**Baseline inspected:** current `alice-eipm-v1-build` plus current Stage G, memory, provenance, grounding, and EIPM design documents.  
**Status:** research intake only. This note does not authorize a production cutover, rewrite completed authority work, or silently change canonical architecture.

## Intake rule

A paper enters this register only when it creates a concrete A.L.I.C.E. architectural, validation, or model-design implication. Conceptual similarity alone is not enough.

## 1. Agent Zero Memory: Provenance-Aware Long-Term Memory for LLM Agents

**Primary source:** https://arxiv.org/abs/2608.29606  
**Date:** 2026-08-30  
**Impact:** current/later Stage G retrieval and context qualification; no completed Claim/Experience authority redesign.

### Relevant result

The paper uses three complementary memory representations: an episodic event timeline, an entity-event graph, and hierarchical documentary memory. Retrieval is intent-gated, source-routed, concurrent across the stores, hybrid lexical+dense, and can open source evidence on demand.

The most useful new formalism for A.L.I.C.E. is the paper's **citation lock**. Let `O` be the set of memory/evidence items actually opened by the reader during the current retrieval episode. A final answer may cite only evidence in `O`, and every atomic answer claim must be supported by that cited/opened set. If sufficient opened evidence cannot be assembled, the system abstains.

### What A.L.I.C.E. already has

A.L.I.C.E. already goes beyond the paper in authority semantics:

- raw Experience/Event evidence remains distinct from derived claims;
- Claim Fabric is append-only/bitemporal and carries validity, transaction time, evidence, conflict, correction, supersession, deletion, and authority;
- graph, vector, episodes, host/source-person/self/relationship/mission structures are projections or cognitive structures rather than automatic truth;
- Stage G already tests correction/deletion/revocation, historical/current truth, stale projection repair, replay/rebuild/restore, identity separation, and exact-source reconstruction;
- Phase 3 grounding already binds citations to authorized evidence and fails closed on inconsistent bindings.

Therefore the paper does **not** justify replacing A.L.I.C.E.'s architecture with a three-store design.

### Genuine delta

Stage G does not currently state a first-class **evidence-consumption invariant** for reasoning-time retrieval. Exact source reconstruction and citation binding are close, but they do not explicitly require the final support set to be a subset of evidence actually materialized/opened in the exact invocation that produced the answer.

### Proposed A.L.I.C.E. validation contract

Add a future/current Stage G challenger test, without changing authority ownership:

1. Every evidence/context object exposed to a reasoning invocation receives an invocation-scoped consumption/read receipt.
2. Let `O` be the evidence actually opened/materialized in that invocation.
3. Every externally asserted atomic claim must resolve to support in `O` or to an explicitly classified non-evidential operation.
4. Metadata, index hits, summaries, graph reachability, or latent retrieval scores alone cannot masquerade as opened source evidence.
5. If evidence sufficiency fails, the Context Manager must expand retrieval, ask, defer, or abstain rather than fabricate support.
6. Test citation-lock preservation across graph, vector, episode, claim, exact-source, and iterative-retrieval paths.
7. Measure whether an intent/memory-need gate and source routing reduce latency/cost without reducing authority/evidence correctness.

**Recommendation:** adopt as a Stage G retrieval/context **validation invariant**, not as a memory-topology rewrite.

---

## 2. Useful Memories Become Faulty When Continuously Updated by LLMs

**Primary source:** https://arxiv.org/abs/2605.12978  
**Latest revision reviewed:** 2026-08-29  
**Impact:** Stage G Memory Formation/consolidation qualification and future formation training; completed authority design is reinforced rather than invalidated.

### Relevant result

Across several agent settings, repeated LLM consolidation can make useful memory worse. The paper isolates the consolidation step and shows that the same experience set can produce different memory states depending on update order and grouping. It identifies failures from misgrouping experiences, stripping applicability conditions during abstraction, and overfitting narrow or near-duplicate streams. Raw episodic retention is often competitive with or more robust than forced continual consolidation.

### What A.L.I.C.E. already gets right

This is strong external support for decisions already present in A.L.I.C.E.:

- raw Experience/Event evidence is first-class and not overwritten by abstraction;
- the Memory Formation Model proposes rather than authorizes;
- deterministic authority/gating controls promotion;
- the Learning Curator can retain raw, compress, quarantine, promote, create a belief, or discard;
- episodes and derived abstractions remain separable from source evidence;
- correction/rebuild lineage is preserved.

There is no evidence here for undoing completed M2/Claim authority work.

### Genuine delta

Current Stage G tests include **reordering** and **duplicate consolidation**, but do not explicitly compare the *same evidence set* under multiple consolidation schedules/grouping policies and require bounded semantic/authority divergence. The paper shows that this path dependence is itself a major failure mode.

### Proposed Stage G tests

Add a **Consolidation Path-Dependence Suite**:

1. Feed the exact same authorized evidence under:
   - one-shot/static consolidation;
   - chronological streaming;
   - shuffled streaming;
   - entity/episode/task-grouped batches;
   - adversarially mixed unrelated experiences;
   - near-duplicate-heavy streams.
2. Keep raw evidence byte-identical and compare:
   - MemoryProposalBundles;
   - proposed claim scope and applicability conditions;
   - adjudicated claims;
   - episodes/projections;
   - downstream retrieval and decisions.
3. Require abstractions to retain provenance to the full contributing source set plus grouping rationale.
4. Detect applicability-condition loss and overgeneralization explicitly.
5. Make `RETAIN_RAW / NO_CONSOLIDATION` a valid learned proposal/action rather than treating consolidation as mandatory.
6. Measure divergence across schedules. Large unexplained divergence must fail or quarantine rather than silently become authority.
7. Include later correction/deletion propagation through every abstraction created under each schedule.

**Recommendation:** add these tests before Memory Formation/consolidation is considered qualified. No completed storage/authority migration is required.

---

## 3. Personalized RewardBench: Evaluating Reward Models with Human Aligned Personalization

**Primary source:** https://arxiv.org/abs/2604.07343  
**Version reviewed:** v2, 2026-07-30; COLM 2026  
**Impact:** future EIPM N1/N2 evaluation and possibly N0/N0B evaluation design; no current memory-fabric rewrite.

### Relevant result

The benchmark constructs high-general-quality response pairs whose key distinction is adherence to user-specific rubrics. State-of-the-art reward models still show a large personalization gap. The paper also finds that naively injecting raw user history/profile into a reward model can degrade evaluation, while a planner that converts history into structured personal rubric aspects recovers performance. It further evaluates whether static reward-model rankings predict actual downstream Best-of-N/PPO behavior, exposing the usual static-benchmark **proxy gap**.

### What A.L.I.C.E. already gets right

The current EIPM design is already directionally stronger than naive profile injection:

- EIPM is a specialist identity/judgment model, not the general generator;
- ACFP provides structured typed state rather than a raw-history dump;
- raw provenance-bound spans remain available when nuance matters;
- explicit identity concepts and bounded residual concepts represent identity-relevant criteria;
- multi-view latent pooling and multi-head/distributional outputs avoid a single generic scalar;
- candidate comparison is already a first-class objective.

### Genuine delta

Our N0 plan contains preference/ranking and personalization-relevant research, but it does not yet make **matched-general-quality personalized counterfactual pairs plus downstream proxy-gap validation** a named EIPM acceptance requirement.

### Proposed EIPM evaluation program

For N1/N2, build an Elaina-specific **Personalized Counterfactual Judgment Suite**:

1. For each identity/value/relationship distinction, construct candidate pairs that are matched as tightly as possible on generic correctness, helpfulness, relevance, fluency, safety, and style.
2. Make the discriminative variable an E0/E-INF-authorized identity criterion or relationship posture.
3. Include swapped response order, paraphrase/style perturbations, and generic-quality adversaries so success cannot come from position or surface cues.
4. Compare:
   - raw-history/profile conditioning;
   - ACFP/structured-state conditioning;
   - concept-bank conditioning;
   - combined raw-span + structured conditioning.
5. Keep provenance-grouped splits so sibling evidence does not leak.
6. Evaluate static pair/list ranking **and** downstream use: candidate selection, response posture, dialogue trajectories, and decision packets.
7. Require static EIPM metrics to correlate with downstream identity fidelity; a high pairwise score with weak downstream behavior is not sufficient for acceptance.
8. Preserve ties/plural-valid alternatives where identity evidence does not support a unique ordering.

**Recommendation:** record now and implement in N1/N2 evaluation design. Do not change N0 topology solely because of this paper.

---

## Papers screened but not admitted as architecture deltas

### SodaMem: Evidence-Grounded Temporal Graph Memory for LLM Agents
https://arxiv.org/abs/2608.08055

Directly relevant in topic, but its main design elements—source-grounded temporal facts, validity/update relations, hybrid retrieval, and evidence-first answering—are already covered more broadly by A.L.I.C.E.'s Experience/Claim bitemporal authority, provenance graph, projections, and retrieval fabric. Keep as external corroboration/benchmark candidate; no new architecture change from today's review.

Other memory/RAG systems surfaced during the scan were excluded when they primarily reproduced graph+vector retrieval, memory summaries, or ordinary long-context RAG without adding a concrete missing A.L.I.C.E. invariant.

---

## First-watch decisions

### Proposed current Stage G additions
- **Evidence Consumption / Citation Lock invariant**
- **Consolidation Path-Dependence / Schedule Sensitivity suite**

### Proposed future EIPM addition
- **Personalized Counterfactual Judgment + downstream proxy-gap evaluation**

### Explicit non-changes
- Do not replace Claim Fabric authority.
- Do not collapse Experience/Event evidence into learned summaries.
- Do not make graph/vector/episode projections authoritative.
- Do not force consolidation.
- Do not replace the current EIPM multi-view/structured design with a generic reward model.
- Do not change canonical plans until these deltas are reviewed and accepted through the existing governance process.
