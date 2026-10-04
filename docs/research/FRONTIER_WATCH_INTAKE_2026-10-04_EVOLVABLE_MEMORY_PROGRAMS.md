# Frontier Watch Intake — 2026-10-04 — Evolvable Memory Programs

**Status:** research/context note only; no production/main semantic change  
**Baseline:** `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`

## Repo state checked

Judged against A.L.I.C.E. `main@8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b`, Fable_Sleight `main@a6eba323dd27d3c58f232b351d09b86761b15935`, `alice-context@945245bbee4168fff46f863455a83bb219063017`, `fable-builder-model@6861d76535c3418efbd2ee948c1056d57030cfe4`, `research/graphify-context-substrate@a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15`, the active MFM v1.6 source lanes, and frontier-watch through `bd04f9c6147ab27178508f3bd45e50cffaeca19f`.

Current critical path remains MFM v1.6 source/teacher authorization, corpus admission, provenance, and CPU processing. Selected Fable v1 physical placements remain unchanged.

## MemCodex

**Primary source:** Wang & Liu, *MemCodex: Self-Programming Hierarchical Memory for Language Agents*, arXiv:2609.39765, submitted 2026-09-30.  
https://arxiv.org/abs/2609.39765

**Classification:** direct future capability improvement; not architecture-invalidating.

### New delta

A.L.I.C.E. already has the required logical planes, source-native evidence, authority separation, MRM/Recollection, lifecycle curation, rollback, and governed evolution. MemCodex adds a useful implementation principle: **memory-plane behavior can itself evolve as programs behind fixed runtime and provenance constraints.**

Its experimental text layers expose five methods: write-side `admit`, `index`, `propose`; read-side `score`, `route`. Evolution changes one method of one layer, or one read-layer activation decision, per child. Candidates must satisfy fixed memory constraints before an acceptance test, and accepted variants live in a rollbackable archive.

Transferable Fable rule:

> Memory behavior may evolve; authority, provenance, identity, deletion, rollback, and full-capability invariants may not.

The paper's fixed layer count is not a Fable ceiling. A layer toggle may inform query-time activation/routing; it may not remove a logical Fable cognitive plane.

### Source-native routing

Each derived item retains exact source turns. Derived summaries/graphs/skills can route the read back to source evidence rather than being injected as substitute truth.

Reported ablation:
- source-routing channel: 60.8 accuracy, 88.4 evidence recall, ~2.0k tokens/query;
- derived-text channel: 55.4 accuracy, 63.0 recall, ~3.9k tokens/query;
- content + routing: 59.6 accuracy, 88.7 recall, ~4.6k tokens/query.

This strengthens the existing Fable direction: projections can be navigation/index structures while source-native evidence remains recoverable. For A.L.I.C.E., source evidence still does not become Claim Authority merely because retrieval descends to it.

### Evidence, cost, limitations

Across the reported MemArena protocol, Qwen3.5-27B gains 5.6 average accuracy points over the strongest adaptive baseline; GPT-5.4-mini gains 3.6. The headline result uses 3.4x fewer context tokens and 2.1x lower query latency. Program evolution adds about 4 points over the initial architecture while reducing tokens about 17%.

Risks: offline program-search cost, evaluator dependence, task-specific evolution, incomplete held-out transfer, and more version/archive lineage. The latent Metis-27B path contributes part of the full Qwen result and is not a required Fable component.

### Future action

When MRM/Lifecycle/personal-model evolution reaches the active path, define an evolvable per-plane policy interface for bounded operations such as proposal formation, indexing, scoring, routing, adaptive stopping, graph expansion, activation packing, and procedure-promotion proposals.

Immutable gates must preserve Experience/evidence vs Claim Authority, projection vs truth, host/source-person/A.L.I.C.E.-self/relationship separation, exact source/model/data lineage, correction/deletion/revocation, rollback/rebuildability, owner governance, and full-Fable-v1 capability.

Focused validation only: fixed vs learned vs locally evolved control; derived-text vs source-routing; held-out/record-disjoint transfer; authority leakage; deletion/revocation rebuild; rollback; and measured token/latency benefit. No backend tournament.

MemCodex also requires replayed outcome agreement before admitting a skill, corroborating the 2026-10-03 watch invariant that executor self-report alone cannot establish verified procedural success.

## Verifiable Memory

**Primary source:** Sun et al., *Verifiable Memory: Learning Unified Memory Management with Local and Global Verifiers for Large Language Model Agents*, arXiv:2608.03137.  
https://arxiv.org/abs/2608.03137

**Classification:** future challenger / direct future training improvement.

This newly surfaced work adds a narrower training result beyond the already tracked learned-memory-control papers: future MRM training should not credit every memory operation only from terminal task success.

VerMem assigns:
- local operation credit for relevance, evidence grounding, immediate progress, and information fidelity;
- global trajectory credit for task outcome, evidence coherence/recall, terminal-memory consistency, and efficiency;
- separate hard constraint costs.

Its matched no-verifier control keeps the action space, SFT initialization, curriculum, task reward, constraints, and GRPO setup. Local and global credit are complementary in the reported ablation. Under one controlled budget setting, VerMem reaches the target success level at about 2,080 online tokens, roughly 20% fewer than AgeMem.

Important limitation: memory is reset after every episode, so this does not establish cross-session personal-memory retention. Training is evaluator-heavy and uses only two Qwen backbones. Privacy-sensitive/adversarial lifelong settings are not established.

**Fable implication:** when a learned memory-control policy is trained, test hierarchical local+global credit inside deterministic authority/provenance/deletion gates. Verifiers may train behavior; they may not redefine Claim Authority.

## Screened, no new action

Agent Zero Memory (arXiv:2608.29606) strongly corroborates layered provenance, source-native opening, timeline/graph/documentary views, and hybrid retrieval, but those directions are already explicit in Fable.

MeClear (arXiv:2609.09115) is an interesting MRM challenger for query-scoped suppression of harmful retrieved memories using cooperative attribution, but its operational conclusion is already covered by selective activation and influence-calibration work; sampled Shapley attribution is comparatively expensive.

Reversible Forgetting (arXiv:2608.18177) is compatible with Lifecycle Curator suppression/reactivation semantics but is mainly a conceptual framework and explicitly is not deletion/unlearning.

## Decision

No selected physical implementation is reopened. No Fable logical plane is removed. No current MFM v1.6 work is blocked.

The new future-facing decision is to preserve a path for **local, attributable, rollbackable evolution of memory-plane behavior behind non-evolvable Fable governance contracts**.
