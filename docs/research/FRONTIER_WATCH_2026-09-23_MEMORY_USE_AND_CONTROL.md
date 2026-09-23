# A.L.I.C.E. Frontier Watch — 2026-09-23 Memory Use and Control

**Status:** reviewed against canonical main after PR #91. Current-impact finding promoted through PR #92 / merge `5f9b9ccb2628eb2a5512d8624b9a78bcc9c7fe38`.

## MemCalib — arXiv:2609.24259 (2026-09-21)

MemCalib isolates a gap that A.L.I.C.E.'s existing retrieval/citation contracts do not fully measure: after correct memory is retrieved, the reasoning model can still over-use or under-use individual propositions. The paper labels ideal proposition influence as Ignore, Bound, or Control, reports directional over/under-use skew across frontier models, and shows that ordinary GRPO/on-policy self-distillation can improve one direction while worsening the other. MemCalib-RL separates nine ideal/actual channels and uses exact atom ablation plus target-aware counterfactual likelihood differences for finer credit localization.

A.L.I.C.E. already has stronger provenance/authority semantics, citation-lock, temporal/correction/deletion lineage, and projections-vs-truth separation. The genuine delta is proposition-level *influence calibration* after retrieval. This affects current Stage G/G2 qualification and future EIPM/context training, not Claim/Experience authority and not current N0 topology.

Promoted action: canonical `docs/MEMORY_USE_CALIBRATION_AND_CONTROL_QUALIFICATION.md` now requires invocation-scoped Ignore/Bound/Control calibration, separate over-use/under-use reporting, atom-level counterfactual tests, critical-case reporting, and authority-preserving training rules.

Primary source: https://arxiv.org/abs/2609.24259

## Jev-Mem — arXiv:2609.23986 (2026-09-21)

Jev-Mem separates high-frequency bounded memory-control decisions from expensive generative reasoning. Its lightweight System-One controller handles typing, relation decisions, routing, retrieval budgets, graph traversal, scoring, evidence sufficiency, and stopping; System Two is reserved for open-ended synthesis. It reports 0.777 LoCoMo judge score, 158 s construction time (6.6x faster than its fastest compared memory system), and 0.93 s average query latency (36.7% lower).

A.L.I.C.E. already has Formation Context Planner, Retrieval Orchestrator, Context Manager, deterministic authority gates, and planned learned routing. The genuine delta is making lightweight typed/probabilistic control a named challenger across both write and read paths, with efficiency measured only after authority/evidence correctness.

Promoted action: canonical qualification now defines a System-One memory-control challenger and requires comparison on authority correctness, citation-lock, influence calibration, identity/correction/deletion behavior, latency, model calls/tokens, resource cost, adaptive stopping, distribution shift, and adversarial inputs.

Primary source: https://arxiv.org/abs/2609.23986

## RPMem — arXiv:2609.23466 (2026-09-20)

RPMem compiles sessions into a model-independent fixed-size latent memory, recurrently consolidates that state with a learned gate, then decodes it into backbone-specific LoRA parameters. It reports near-constant update cost/memory footprint and reuse across five backbones; Qwen3-8B reaches 85.52% on PERMA, 5.32 points over the strongest parametric baseline and 12.98 over Full Context in the reported setup.

A.L.I.C.E. already requires replaceable backends/models, lineage, rollback, dataset/model registry, adapters/LoRA, and future parametric learning. The genuinely useful idea is lifecycle-independent *parametric* memory: persistent learned state can be decoupled from a serving-backbone generation. The paper's latent memory cannot replace A.L.I.C.E.'s authoritative evidence/Claim layers because opaque fixed-size state does not by itself satisfy provenance, deletion, correction, reconstruction, or authority requirements.

Promoted action: canonical qualification records RPMem-style lifecycle-independent parametric memory as a future Track H challenger with mandatory external lineage, rebuildability, correction/deletion/revocation tests, versioned compiler/decoder/backbone identities, rollback, consolidation path-dependence testing, and comparison against non-parametric/adaptor baselines. It does not block Stage G unless such a candidate enters the active serving path.

Primary source: https://arxiv.org/abs/2609.23466

## Screened but not separately promoted

- LIMBO (arXiv:2609.14138): adaptive cost-aware memory/replay budget allocation is useful but is subsumed by the Jev-Mem challenger plus A.L.I.C.E.'s already-promoted adaptive Context Planner/resource-aware control direction.
- ARM (arXiv:2609.24417): differentiable fixed-size routed KV memory is promising for model-serving efficiency, but it is an internal attention/cache mechanism rather than durable governed personal memory. Preserve as a later model-serving challenger; no current architecture contract change.
- ACLArena (arXiv:2609.23989): routed LoRA experts plus replay reinforce A.L.I.C.E.'s multi-substrate continual-learning direction. It does not yet justify replacing the current EIPM/N0 training plan without direct evidence on the current architecture and tasks.

## Explicit non-changes

- No change to Experience/evidence vs Claim authority.
- No projection becomes truth authority.
- No N0 topology/objective change from these papers.
- No latent/parametric memory becomes the sole durable personal-memory representation.
- No efficiency controller may trade away provenance, correction/deletion/revocation, identity separation, or citation-lock correctness.
