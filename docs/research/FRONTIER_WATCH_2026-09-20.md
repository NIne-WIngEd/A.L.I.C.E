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

1. Feed the exact same authorized evidence under one-shot/static, chronological, shuffled, entity/episode/task-grouped, adversarially mixed, and near-duplicate-heavy schedules.
2. Keep raw evidence byte-identical and compare MemoryProposalBundles, claim scope/applicability, adjudicated claims, episodes/projections, and downstream retrieval/decisions.
3. Require abstractions to retain provenance to the full contributing source set plus grouping rationale.
4. Detect applicability-condition loss and overgeneralization explicitly.
5. Make `RETAIN_RAW / NO_CONSOLIDATION` a valid learned proposal/action rather than treating consolidation as mandatory.
6. Measure divergence across schedules. Large unexplained divergence must fail or quarantine rather than silently become authority.
7. Include correction/deletion propagation through every abstraction created under each schedule.

**Recommendation:** add these tests before Memory Formation/consolidation is considered qualified. No completed storage/authority migration is required.

---

## 3. Personalized RewardBench: Evaluating Reward Models with Human Aligned Personalization

**Primary source:** https://arxiv.org/abs/2604.07343  
**Version reviewed:** v2, 2026-07-30; COLM 2026  
**Impact:** future EIPM N1/N2 evaluation and possibly N0/N0B evaluation design; no current memory-fabric rewrite.

### Relevant result

The benchmark constructs high-general-quality response pairs whose key distinction is adherence to user-specific rubrics. State-of-the-art reward models still show a large personalization gap. Naively injecting raw user history/profile can degrade evaluation, while a planner that converts history into structured personal rubric aspects recovers performance. It also evaluates whether static rankings predict downstream Best-of-N/PPO behavior, exposing the proxy gap.

### What A.L.I.C.E. already gets right

The current EIPM design is already directionally stronger than naive profile injection: specialist identity/judgment ownership, ACFP typed state, provenance-bound raw spans, explicit plus residual identity concepts, multi-view latent pooling, multi-head outputs, and candidate comparison.

### Genuine delta and recommendation

Make **matched-general-quality personalized counterfactual pairs plus downstream proxy-gap validation** a named N1/N2 EIPM acceptance requirement. Compare raw-history, ACFP/structured, concept-bank, and combined conditioning; preserve provenance-grouped splits and plural-valid alternatives; require static ranking quality to predict downstream identity fidelity.

**Recommendation:** record now and implement in N1/N2 evaluation design. Do not change N0 topology solely because of this paper.

---

## 4. Retrieval-Driven Memory Reconsolidation for Long-Term LLM Agents (REALM)

**Primary source:** https://arxiv.org/abs/2609.16053  
**Date:** 2026-09-13  
**Impact:** future cognitive-graph/retrieval adaptation and a Stage G challenger/validation track; no completed Claim/Experience authority rewrite.

### Relevant result

REALM closes the memory lifecycle after retrieval. It organizes entity/event/episode/fact nodes in a heterogeneous cognitive graph; composes retrieval from seed, expansion, and filtering strategy atoms; and, after task feedback, modifies only the activated local topology using `add`, `strengthen`, or `weaken` edge operations. The paper reports 75.97% on LoCoMo and 65.11% on LongMemEval. Reconsolidation itself adds about +2.01 and +2.13 points average respectively, with larger gains on some multi-hop, preference, and knowledge-update slices. Randomized query-order experiments suggest the gain is not merely memorization of one evaluation sequence.

### What A.L.I.C.E. already does similarly or more broadly

A.L.I.C.E. already has a heterogeneous Cognitive Graph; adaptive context planning over claims, evidence, graph paths, vectors, episodes, sources, models, tools and agents; outcome observation and continuous revision; and explicit learned-retrieval/routing capacity. More importantly, A.L.I.C.E. has a stronger authority boundary: graph topology is a provenance-linked projection/cognitive structure and cannot override bitemporal Claim Fabric or raw Experience evidence.

### Genuine delta

The useful idea is **retrieval-driven adaptation of the retrieval substrate itself**. Current A.L.I.C.E. plans allow learned retrieval and continuous revision, but do not make successful/failed retrieval episodes a first-class governed signal for strengthening, weakening, or proposing associative graph edges. This is different from changing factual claims: it learns *how memories tend to become useful together*.

### Compatibility and conflict analysis

Compatible if implemented as a derived, versioned, rebuildable retrieval projection. Incompatible if retrieval co-activation can create factual/causal/identity authority, silently rewrite source relations, or bypass deletion/correction lineage. REALM's autonomous merge/skip and graph evolution are therefore too permissive for A.L.I.C.E.'s authority layer and must not be copied there.

The immediately preceding frontier finding on consolidation path dependence makes an additional guard important: retrieval-driven reconsolidation can itself create path dependence and popularity feedback. Frequently retrieved memories may become easier to retrieve again even when the initial retrieval was accidental or biased.

### Proposed future design/validation track

Create a **Usage-Aware Retrieval Projection** challenger rather than mutating the canonical Cognitive Graph:

1. Log invocation-scoped retrieval receipts: query/context fingerprint, candidates, opened evidence, selected support, answer/outcome feedback, and correction/deletion status.
2. Learn/propose associative accessibility edges or weights from repeated *successful* co-utilization; keep these edges explicitly typed as `retrieval_association`, never factual/causal authority.
3. Require minimum evidence/repetition or calibrated confidence before strengthening; allow decay/weakening after misleading retrievals.
4. Keep a frozen semantic/structural graph generation so the usage-aware projection is replayable, comparable, and rollback-safe.
5. Apply correction/deletion/revocation lineage to learned retrieval associations and prevent deleted evidence from leaving active accessibility influence.
6. Test cold-start, query-order shuffles, adversarial repeated queries, popularity loops, rare-but-critical memories, contradictory evidence, and distribution shift.
7. Compare static graph retrieval vs usage-aware projection on evidence recall, authority correctness, temporal/update questions, latency/token cost, and calibration.
8. Do not allow retrieval success alone to promote a claim, relationship, personality trait, or causal edge.

### Expected upside / risk / cost / evidence strength

- **Upside:** medium-high. Could make A.L.I.C.E.'s retrieval fabric improve through use, especially for recurring multi-hop evidence patterns and relationship/project contexts.
- **Risk:** high if authority boundaries are blurred; medium in a projection-only challenger. Main risks are popularity bias, self-reinforcing retrieval errors, privacy/deletion influence residue, and path dependence.
- **Compute/storage:** moderate. Local edge-weight updates are cheap relative to model inference, but the paper uses LLM-driven retrieval/reconsolidation decisions, so naive adoption adds inference cost. A.L.I.C.E. should evaluate smaller learned controllers or deterministic/statistical updates as challengers.
- **Evidence strength:** moderate. Two standard memory benchmarks and ablations support the mechanism, but evaluation is LLM-as-judge, the absolute LongMemEval gain over the strongest baseline is only 1.31 points, and the architecture has not established long-horizon safety under adversarial or authority-sensitive personal memory.

**Recommendation:** preserve as a future design seed and add a Stage G/G+ shadow challenger. Do **not** modify completed Claim/Experience authority or canonical graph semantics.

---

## Papers screened but not admitted as architecture deltas

### SodaMem: Evidence-Grounded Temporal Graph Memory for LLM Agents
https://arxiv.org/abs/2608.08055

Its source-grounded temporal facts, validity/update relations, hybrid retrieval, and evidence-first answering are already covered more broadly by A.L.I.C.E.'s Experience/Claim bitemporal authority, provenance graph, projections, and retrieval fabric.

### MemoryLACE / ROAM / Fortunate Recall
Their lifecycle/supersession/contradiction/atomic-role ideas largely reinforce A.L.I.C.E.'s existing claim/provenance/temporal authority and do not currently justify a topology change.

### EARM: The Retriever Should Remember
Experience-amortized reranking is promising for future efficiency research, but its low-rank relevance matrix assumes stable memory identities and recurring query-memory relevance. It is best treated as a challenger under the broader usage-aware retrieval track above rather than a separate architectural commitment.

### Selective Forgetting
Useful negative evidence: a simple extracted graph underperformed a matched flat vector baseline, while pruning reduced storage with bounded loss. This reinforces A.L.I.C.E.'s decision not to make graph serialization the sole memory representation and to retain raw evidence. Its heuristic pruning policy is too weak for A.L.I.C.E.'s authority/deletion semantics, so no direct adoption.

---

## First-watch decisions

### Proposed current Stage G additions
- **Evidence Consumption / Citation Lock invariant**
- **Consolidation Path-Dependence / Schedule Sensitivity suite**
- **Usage-Aware Retrieval Projection shadow challenger** (research-only; no authority mutation)

### Proposed future EIPM addition
- **Personalized Counterfactual Judgment + downstream proxy-gap validation**

### Explicit non-changes
- Do not replace Claim Fabric authority.
- Do not collapse Experience/Event evidence into learned summaries.
- Do not make graph/vector/episode/retrieval-association projections authoritative.
- Do not force consolidation or reconsolidation.
- Do not replace the current EIPM multi-view/structured design with a generic reward model.
- Do not change canonical plans until these deltas are reviewed and accepted through the existing governance process.
