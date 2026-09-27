# Agentic Source-Native Retrieval and Low-VRAM Model Streaming — Frontier Review

**Date:** 2026-09-27  
**Status:** current Stage G qualification impact + future serving research seed  
**Current N0 impact:** no topology, objective, or training-path change

## Executive conclusion

Two separate trends were reviewed:

1. stronger models increasingly perform useful just-in-time, source-native retrieval with grep, file reads, structured queries, and progressive disclosure rather than receiving fixed top-k embedding matches automatically;
2. AirLLM trades model residency for layer/expert streaming, making very large open-weight models fit in small VRAM at potentially severe I/O and latency cost.

Neither trend justifies deleting A.L.I.C.E.'s vector plane or changing the current N0 training route.

The correct A.L.I.C.E. response is **retrieval pluralism plus measured routing**: make agentic no-vector/source-native retrieval a first-class Stage G challenger, and let exact workload evidence decide when lexical, vector, graph, temporal, source-native, hybrid, or dynamic routing should serve a query. Low-VRAM model streaming should be evaluated as a future serving profile, not used as a substitute for current native full-parameter N0 training.

## 1. What actually happened in Claude Code and Cursor

### Claude Code

Anthropic's own April 2026 account says the first internally released Claude Code used RAG backed by a vector database that pre-indexed the codebase. Anthropic later replaced that path with a Grep tool so Claude could find and assemble its own context. The reason given is not that semantic similarity became mathematically useless; the harness became more effective when the increasingly capable model could search progressively and inspect the live environment itself.

Primary sources:

- https://claude.com/blog/seeing-like-an-agent
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- https://code.claude.com/docs/en/how-claude-code-works

Anthropic's broader context-engineering guidance is explicitly hybrid rather than anti-retrieval: it describes embedding-based pre-inference retrieval, just-in-time source retrieval, and hybrid designs as task-dependent choices. It also names runtime exploration's downside: extra latency/tool use and the possibility of dead ends.

Therefore the statement "Claude Code never used a vector DB" is historically false if it includes Anthropic's internal first release. The useful claim is narrower: current Claude Code is designed around model-directed progressive source discovery rather than a mandatory semantic code index.

### Cursor

Cursor staff confirmed in July 2026 that the older dedicated semantic code-search path had stopped adding meaningful value for their current agents and that agents now lean on Instant Grep and normal file reads. Staff also said Cursor no longer needed to maintain the old semantic codebase index for that search path.

Primary source:

- https://forum.cursor.com/t/indexing-tab-missing-in-settings/165908/8
- https://forum.cursor.com/t/codebase-indexing-settings-tab-hidden-by-disable-codebase-indexing-feature-gate-please-re-enable/165859

This is evidence about **codebase retrieval under Cursor's current models and harness**, not a universal result for lifelong personal memory.

### Toffu

The owner supplied a September 2026 LinkedIn post from Toffu's founder saying Toffu disabled its former embedding/Pinecone memory flow and moved toward explicit instructions, named/keyword notes, agent-driven conversation/file searches, and direct live-account reads.

That claim is not fully independently verified by Toffu's public site. The current public subprocessor page still says Pinecone powers semantic search/recommendation and may store customer-data embeddings, while Toffu's Memory documentation still describes automatic memory checking. The most plausible interpretations are that the public page is stale, Pinecone remains for a non-memory feature, or the transition is partial.

Public sources:

- https://toffu.ai/subprocessors
- https://toffu.ai/academy/memory

Do not treat the LinkedIn post as proof that Toffu has globally removed every vector use.

## 2. Why code retrieval does not settle A.L.I.C.E. memory

Code is unusually favorable to source-native search:

- paths, filenames, symbols, imports, type names, error strings, and stable textual identifiers are high-information;
- the working tree itself is the current source of truth;
- exact source can be opened cheaply after a match;
- source mutation makes stale precomputed indexes a real operational cost.

Personal cognitive memory has different workloads:

- paraphrases with little lexical overlap;
- emotional or behavioral resemblance;
- implicit preference and relationship patterns;
- semantically related events separated by months or years;
- multimodal similarity across images/audio/video;
- temporal, causal, provenance, identity, and authority questions;
- incomplete recollection where the query may not contain the historical words at all.

Those cases can favor vector, graph, temporal, episodic, or learned retrieval even when grep dominates a code-navigation task.

## 3. Direct empirical evidence is mixed even for code

### CodeGrep

*CodeGrep: An RL-Trained Retrieval Agent for LLM Coding Agents* (arXiv:2608.05886) trains a 14B agent specifically to perform multi-turn parallel grep/glob/read retrieval. On all 500 SWE-Bench Verified cases it reports 27.0% resolved versus 25.8% for the no-retrieval baseline, with fewer rounds and tokens on resolved tasks. Its retriever precision substantially exceeds tested BM25 and embedding baselines.

Primary source:
- https://arxiv.org/abs/2608.05886

This supports learned/model-routed source-native search.

### Deep Agentic Search

*Deep Agentic Search for Repository-Level Code Question Answering* (arXiv:2608.01507) finds the opposite result on a read-only repository QA setting: semantic/vector retrieval answers 65.2% correctly versus 46.2% for deep agentic search and costs less than half per correct answer. The largest agentic failure category occurs at planner-to-subagent handoff.

Primary source:
- https://arxiv.org/abs/2608.01507

This prevents a universal "better models made vectors obsolete" conclusion. Retrieval performance is a function of task, corpus, model, tool interface, training, budget, freshness requirements, and evaluation metric.

## 4. A.L.I.C.E. architectural implication

A.L.I.C.E. already has the right top-level destination: lexical, vector, graph, symbolic, source, temporal/claim-aware, and adaptive Context Planner paths coexist as projections/tools rather than truth.

The missing qualification requirement was explicit competition with a **no-vector agentic source-native strategy**.

The canonical Stage G change in this branch therefore requires matched evaluation of:

- exact/lexical retrieval;
- iterative agentic source-native search with query reformulation and direct reads;
- semantic/vector retrieval;
- hybrid lexical + vector retrieval;
- graph/temporal/claim-aware retrieval;
- dynamic routed combinations;
- explicit no-vector/no-semantic-index profiles.

The test must measure final supported reasoning quality, complete evidence retrieval under context budget, exact wording, vocabulary mismatch, paraphrase, multimodal cases, temporal/identity/relationship cases, latency, tool calls, token cost, index maintenance, freshness, traceability, deletion/revocation behavior, and cold start.

The architectural rule is:

> **Do not keep an index because it exists. Do not delete an index because another product stopped benefiting from it. Measure marginal value on A.L.I.C.E.'s actual workload and route accordingly.**

For live mutable state, direct reads from the current system of record should be evaluated against stored snapshots. Memory remains valuable for historical state, continuity, provenance, trend, and reasoning, but must not impersonate a fresher reachable authority.

## 5. AirLLM — what the 4 GB claim really means

Primary project:
- https://github.com/lyogavin/airllm

Current project claims include full-precision 70B inference on roughly 4 GB VRAM, very large MoE inference through expert streaming, and September 2026 low-VRAM training support.

### Mechanism

For dense inference, AirLLM splits a model into layer-wise shards. During a forward pass it loads the needed layer from disk, executes it on GPU, and advances rather than holding all model weights resident in VRAM.

For sparse MoE models, newer paths can stream selected experts rather than a whole layer, which can make total-parameter size much less relevant to peak VRAM.

This is a valid way to lower **resident GPU weight memory**. It does not remove the model's storage or data-movement cost.

### Practical trade

Every generated token still needs the applicable weight path. When weights are repeatedly streamed from disk/host storage, inference can become I/O-bound.

Current public issue evidence includes:

- issue #364: a user reports 28.6 seconds/token for a 3B model on Windows, a 16 GB RTX 5060 Ti, and NVMe storage;
- issue #298: explicitly identifies layer-loading disk I/O as a source of GPU idle time and proposes double buffering;
- issue #325: notes that unconditional layer streaming can waste available VRAM and make even models that fit the GPU run much more slowly;
- issue #295: challenges the reproducibility of several historical headline VRAM/performance claims.

Issues:
- https://github.com/lyogavin/airllm/issues/364
- https://github.com/lyogavin/airllm/issues/298
- https://github.com/lyogavin/airllm/issues/325
- https://github.com/lyogavin/airllm/issues/295

These are user reports/criticisms rather than controlled independent benchmark results, so they should constrain confidence rather than be treated as universal measured performance.

### Training is not current N0 training

AirLLM's September 2026 training support streams **frozen base weights** and keeps trainable adapters on the GPU. That is useful for LoRA-style adaptation of pretrained giant models.

Current A.L.I.C.E. N0 v0.2 is different:

- native/random initialization;
- approximately 140M parameter operating point;
- full semantic/judgment foundation training;
- no inherited third-party model weights;
- current qualification targets the complete joint path rather than frozen-base adapters.

Therefore AirLLM is **not a fix for the current N0 P100 training/runtime path**. Replacing the N0 route with AirLLM adapter training would change the ownership/training problem rather than merely reduce VRAM.

## 6. Where AirLLM could help later

AirLLM-style streaming is worth a future challenger in:

- offline/batch inference with a model too large to reside in available VRAM;
- rare specialist-model use where latency is secondary to access;
- shadow/challenger evaluation;
- enormous sparse MoE models where only a small expert subset activates;
- Phase 13 host-specific LoRA/adapter research over large public pretrained models;
- degraded/offline profiles where the alternative is no access to the model.

It should be compared against at least ordinary GPU residency, CPU/GPU offload, quantized llama.cpp/MLX-style serving, Transformers/Accelerate offload, vLLM-class serving where applicable, and remote acceleration.

Required measurements:

- task quality and numerical equivalence where claimed;
- first-token latency and tokens/sec;
- prefill vs decode cost;
- sequence length / KV-cache pressure;
- SSD and host-memory bandwidth;
- storage footprint and preprocessing time;
- GPU utilization and idle fraction;
- energy;
- concurrency/batching;
- cold vs warm start;
- failure/recovery behavior;
- cost per useful result.

Low VRAM by itself is not a serving-quality metric.

## 7. Current decisions

### Change now

- Make no-vector agentic source-native retrieval a formal Stage G challenger.
- Require workload-specific matched retrieval strategy evaluation before selecting a default.
- Preserve full retrieval traces and citation-lock behavior.
- Explicitly test direct live-source reads for mutable current facts.

### Preserve

- Claim/Experience authority.
- vector/multimodal plane as an optional qualified projection.
- Cognitive Graph and temporal retrieval.
- exact/lexical retrieval.
- adaptive Context Planner and dynamic routing.
- current N0 architecture and full training route.

### Future research seed

- low-VRAM layer/expert streaming as a serving challenger;
- AirLLM as one implementation candidate, not an architectural dependency;
- adapter training via streamed frozen weights as a Phase 13 experiment, not N0 replacement.

## 8. Bottom line

The industry signal is real: stronger agents can increasingly **search for their own context**, so systems should stop blindly injecting fixed top-k semantic matches when the model can cheaply inspect a live structured source.

The stronger A.L.I.C.E. conclusion is not "remove vectors." It is:

```text
query / situation
       ↓
Context Planner
       ↓
what kind of evidence problem is this?
       ↓
┌──────── exact / lexical ────────┐
├──────── agentic source-native ──┤
├──────── semantic / vector ──────┤
├──────── graph / temporal ───────┤
├──────── live authority source ──┤
└──────── hybrid / learned route ─┘
       ↓
opened source evidence
       ↓
authority + citation-lock checks
       ↓
reasoning
       ↓
outcome → retrieval learning
```

Retrieval infrastructure should earn its place continuously through measured marginal value.
