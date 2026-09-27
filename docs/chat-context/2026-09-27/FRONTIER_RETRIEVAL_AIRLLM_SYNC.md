# Frontier sync — agentic source-native retrieval and AirLLM

**Date:** 2026-09-27
**Canonical main merge:** `b2234ab114d23b3e028143bb1b9e4a3d11e2db8a`
**PR:** #94

## Binding continuity points

1. Anthropic's own April 2026 Claude Code account states that the first internal Claude Code used RAG with a vector database before moving to Grep/model-directed context discovery. Do not repeat the claim that Claude Code historically never used a vector database.
2. Cursor staff confirmed that its current code-agent path moved away from the older dedicated semantic code search toward Instant Grep and direct file reads. Treat this as code-domain evidence, not proof that semantic retrieval is obsolete for personal memory.
3. Current Toffu founder material supplied by the owner claims its memory flow moved away from automatic embedding/Pinecone recall. Public Toffu pages still list Pinecone for semantic search/recommendation, so treat total removal as unverified/possibly partial or documentation-lagged.
4. Canonical Stage G now requires an explicit no-vector agentic source-native challenger and matched comparison against lexical, vector, hybrid, graph/temporal/claim-aware, and dynamically routed retrieval. Retrieval infrastructure must earn workload-specific marginal value.
5. Live mutable operational state should be tested through direct current-source reads/reconciliation instead of assuming a stored snapshot is current.
6. AirLLM is a future low-VRAM model-serving challenger. Its key trade is resident VRAM versus repeated model-weight I/O/latency.
7. AirLLM's September 2026 training path streams a frozen pretrained base and trains adapters. **It is not the current N0 training regime.**
8. Current N0 remains native/random-init full foundation training. Do not replace the active N0 P100/DDP qualification route with AirLLM/LoRA merely to avoid the current runtime work.
9. AirLLM-style layer/expert streaming may later be compared for offline/batch specialist inference, very large sparse MoE access, challenger evaluation, or Phase 13 adapter research.

Full canonical research note:
`docs/research/AGENTIC_SOURCE_NATIVE_RETRIEVAL_AND_AIRLLM_2026-09-27.md`
