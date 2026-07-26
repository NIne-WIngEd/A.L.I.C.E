# Phase 3.3 — Read-Only Grounding Bridge

**Status:** P3.3 implementation milestone
**Phase 1 dependency:** Frozen, read-only evidence layer
**Phase 2 dependency:** Authoritative Memory Core, read-only cited-answer packets
**Phase 3 dependency:** P3.0 contracts, P3.1 model abstraction, P3.2 private state

## Purpose

P3.3 converts already-selected Phase 1 evidence and already-authorized Phase 2 memory answer packets into the provider-neutral `ConversationGroundingPacket` contract.

The bridge does not retrieve by itself. It does not call a model. It does not write memory. It does not authorize its own reads.

## Trust boundaries

```text
Phase 1 context package
        |
        | validated read-only evidence
        v
P3.3 Phase 1 adapter
        |
        | uncertain external claims
        v
ConversationGroundingPacket

Phase 2 MemoryAnswerSubmission
        |
        | exact authoritative claims and memory_sources citations
        v
P3.3 memory adapter
        |
        | unchanged claim text, digest, status, confidence, and token
        v
ConversationGroundingPacket
```

All source text remains untrusted data. `ConversationGroundingPacket.render_for_model()` places it inside explicit untrusted-data delimiters.

## Phase 2 adapter

The memory adapter consumes one deterministic `MemoryAnswerSubmission` created by Phase 2.

It preserves:

- answer outcome;
- claim order;
- claim ID;
- authoritative text;
- content SHA-256;
- knowledge status;
- confidence;
- data classification;
- exact `memory_sources` source reference;
- exact Phase 2 citation token.

It rejects:

- denied read authorization;
- `HIGHLY_SENSITIVE` and `SECRETS` content;
- claims above the authorization scope;
- altered claim text or digest;
- citations bound to another memory;
- non-supporting provenance relations;
- duplicate or inconsistent citation identifiers;
- invalid conflict or empty-outcome shapes through the Phase 3 contract.

P3.3 does not repeat Phase 2 retrieval or independently decide whether a corrected, deleted, stale, conflicting, or unauthorized memory is eligible. Those decisions remain in Phase 2. The bridge only accepts the deterministic cited-answer result and fails closed if its binding is inconsistent.

## Phase 1 adapter

The Phase 1 adapter accepts an in-memory grounded-context package with its original guardrails.

It requires:

- contiguous `S1`, `S2`, ... citation IDs;
- exact `[S1]`, `[S2]`, ... tokens;
- unique source-content SHA-256 values;
- preserved provenance;
- matching `source_count`;
- unresolved contradiction groups;
- private-output-only and no-action guardrails;
- an explicit ordinary-classification authorization.

Direct Phase 1 snippets are not promoted into authoritative personal facts. They enter conversation grounding as:

```text
knowledge_status = external_claim
confidence = 0.5
outcome = uncertain
```

The confidence value means the evidence is available for bounded conversation use pending claim validation. It does not assert that the source statement is true.

If a package contains an unresolved contradiction group, the packet outcome is `conflict`. An empty package becomes `insufficient_evidence`.

## Packet merging

Approved packets may be merged deterministically.

Outcome precedence is:

```text
conflict
uncertain
answerable
insufficient_evidence / not_applicable
```

A denied packet cannot be merged with claim-bearing packets. Duplicate claim IDs and citation tokens that point to different logical sources are rejected.

## Conversation-state integration

P3.3 can derive `ConversationStateReference` records for P3.2.

Only metadata is retained:

- citation ID;
- source kind;
- source reference;
- exact citation token;
- classification;
- packet timestamp.

Source text and hidden reasoning are not copied into reference rows. The full packet receives a deterministic SHA-256 digest for `set_turn_context()`.

## Explicit non-capabilities

P3.3 does not enable:

- web access;
- tool calling;
- external actions;
- memory writes;
- memory candidate creation or promotion;
- ordinary highly sensitive retrieval;
- secret storage;
- model generation;
- response validation;
- orchestration;
- CLI or web-interface behavior.

## Next milestone

P3.4 should compile constitutional dialogue behavior into a versioned system contract. P3.5 should then assemble the explicit orchestration state machine around P3.1–P3.4.
