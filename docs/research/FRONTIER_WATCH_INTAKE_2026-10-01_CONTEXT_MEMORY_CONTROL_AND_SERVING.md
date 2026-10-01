# Frontier Watch Intake — 2026-10-01

Status: research/context note only. No production/main semantics changed.

This intake was judged against the current A.L.I.C.E. main, Fable main, active MFM/FBM/Gemma/Graphify/context branches, the 2026-09-27 full-v1 baseline, and the 2026-09-30 recovery intake. The current MFM v1.6 CPU/privacy/corpus-admission path remains the critical path. The selected KurrentDB/NATS/XTDB/CAS/LadybugDB-NebulaGraph/Qdrant/Temporal/L1-Valkey/PyTorch-Accelerate physical direction is unchanged.

## Context Language Models — arXiv:2609.37725

Classification: **direct current capability improvement to Cognitive Workspace / activation-context design; future implementation**.

The paper makes live model context a model-managed artifact rather than an append-only harness product. It reports better long-horizon task results with lower context-compute cost, supports learning context-management policy in text or weights, and introduces Suffix Cache Reuse for cheaper post-edit serving.

A.L.I.C.E. already has Working/Activation Memory, Cognitive Workspace, MRM, Recollection, source-native retrieval, mission state, provenance boundaries, and model/data lineage. The genuine delta is to make context transformation itself a first-class learned operation.

For Fable, model-managed context must remain a governed transient projection. A model may eventually propose compact/offload/restore/reorder/annotate operations, but those operations cannot mutate Experience/Event history, Claim Authority, identity/relationship authority, Mission Graph authority, deletion/revocation lineage, or durable workflows. Context must remain rebuildable from durable state. Deleted or corrected material must not survive through summaries, notes, or cached activation state.

Action: preserve a governed context-transform interface for the future Cognitive Workspace/MRM implementation. Validate authoritative-fact retention, source-native restoration, correction/deletion invalidation, mission-state preservation, rebuild parity, and compute savings. Do not block current MFM work.

## MemAgent — arXiv:2609.32521

Classification: **direct current capability improvement to MRM/Recollection design; future learned-policy implementation**.

MemAgent routes across heterogeneous memory providers at three different lifecycle points: long-term read selection before execution, working-memory injection during execution, and selective storage after execution. Its ablations support separating these decisions and show that storing into every provider can pollute later retrieval. The paper reports a 10-point average accuracy gain over no memory across GAIA, WebWalkerQA, and xBench-DS, with under 0.3% routing overhead in its setup and fewer task steps.

A.L.I.C.E. already has heterogeneous logical memory planes. The genuine delta is evidence for content-aware, lifecycle-specific routing rather than one universal retrieval/write policy. For Fable these are logical cognitive routes, not competing physical databases.

Action: keep `read-route`, `activation-route`, and `write-proposal-route` distinct and learnable. Hard authority, provenance, identity, correction/deletion, and Claim-admission gates stay outside the learned router.

## CacheReforge — arXiv:2609.30884

Classification: **future serving challenger with a direct lineage/invalidation invariant**.

CacheReforge studies KV caches that outlive LoRA/adapter updates. It tracks which adapter version produced each cached layer and selectively recomputes stale regions. The reported Qwen experiments recover much of the current-model behavior while avoiding most full-prefill cost.

The important Fable implication is broader than this algorithm: **activation/KV state has model-version provenance**. Reuse must not be keyed only by token/session identity. Persistent activation state should be bound to the exact model/adaptor lineage and to relevant source/deletion generations. Rollback, correction, revocation, or unlearning must not silently reuse activation produced under an incompatible lineage.

Action: add the future serving invariant `no activation/KV reuse across an unrecognized model-lineage or deletion epoch`. Exact invalidation/rebuild remains the conservative default for deletion/correction until influence-echo tests establish a safe recovery path.

## PaMER — arXiv:2609.27286

Classification: **future challenger / corroboration for learned MRM scheduling**.

The paper finds that pre-action hidden states predict compression and recall needs better than simple context-length/turn controls. It also shows that a compact recent working state preserves much of this signal and that selected raw historical evidence is useful when long-range dependencies return.

The result is predictive, not causal, and the current online implementation does not use the learned recall probe as its actual retrieval gate. Therefore this does not justify a new model or current-path change.

Action: when a trainable MRM exists, test hidden-state features as optional scheduling inputs against cheaper observable features. Raw/source-native evidence remains the support layer; latent state only helps decide when to look.

## ReaLMem / ChronoProfiler — arXiv:2609.19167

Classification: **direct future evaluation improvement; optional salience challenger, not authority**.

ReaLMem evaluates authentic multi-year personal photo/video archives with first-person annotations across factual recall, persona inference, and predictive personalization. The reported benchmark contains 2,508 sessions from seven participants and exposes a persistent predictive-personalization ceiling. ChronoProfiler uses timestamped evidence plus support/duration/recency to derive a temporal-stability salience prior; it improves persona/longitudinal results in the paper, while predictive-assistance effects are mixed.

A.L.I.C.E. already has bitemporal Claim Authority, host/preference state, Episodes, multimodal/perceptual memory, source-native evidence, and update lineage. The genuine delta is a stronger real-world qualification target and a narrow hypothesis that temporal persistence/reaffirmation can help retrieval/activation salience.

Action: when perceptual personal memory enters qualification, add a ReaLMem-like lane for source-native multimodal evidence, longitudinal preference evolution, conflict/update handling, and predictive-personalization calibration. Temporal stability stays a reversible derived feature and never determines truth/currentness.

## Combined decision

No finding invalidates Fable v1 or reopens a selected backend. The current critical path is unchanged.

The new cross-layer contracts to preserve are:

- governed, rebuildable model-managed activation context;
- separate learned read, activation, and write-proposal routing;
- model/version/deletion lineage on activation and KV state;
- latent-state scheduling as advisory only;
- authentic multimodal longitudinal personal-memory evaluation.
