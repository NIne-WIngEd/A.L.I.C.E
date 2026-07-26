# Phase 3.2 — Private Conversation State

**Status:** P3.2 implementation milestone
**Phase 1 dependency:** Frozen, read-only evidence layer
**Phase 2 dependency:** Authoritative Memory Core, read-only from ordinary conversation state
**Owner:** MK Rayan

## 1. Purpose

P3.2 creates durable, inspectable state for local A.L.I.C.E. conversations without turning conversation history into authoritative personal memory.

Conversation state records what occurred in a session:

- session identity and retention;
- ordered user and assistant messages;
- selected memory and Phase 1 evidence references;
- provider and model identity;
- generation attempts and terminal outcomes;
- interruption and resume transitions;
- metadata-only reasoning status;
- validation outcomes;
- sanitized lifecycle events.

It does not infer truth from conversation history. It does not promote messages into Phase 2 memory. Any future durable-memory proposal must use the Phase 2 candidate, assessment, authorization, and promotion path.

## 2. Security boundary

P3.2 remains a no-tool, no-action milestone.

The state layer cannot enable:

- web access;
- tool calling;
- external actions;
- authoritative memory writes;
- automatic memory promotion;
- `HIGHLY_SENSITIVE` ordinary conversation storage;
- `SECRETS` storage;
- hidden chain-of-thought persistence.

The public repository contains only schema, policy, service, inspection, and synthetic tests. The live SQLite database belongs outside the repository under the private vault.

Default location:

```text
C:\ALICE_Vault\conversation\alice-conversation.sqlite3
```

The location is resolved from a versioned public policy. Runtime path validation rejects a vault inside the repository, a database outside the approved vault root, and path traversal.

## 3. Retention model

P3.2 supports two explicit retention modes.

### 3.1 `session_only`

This is the default.

When the session is closed:

1. every turn must already be terminal;
2. session, turn, message, reference, generation, and event rows are deleted atomically;
3. foreign-key cascades remove dependent content;
4. a sanitized tombstone preserves only the session ID, retention mode, deletion time, and aggregate row counts;
5. the purged session ID cannot be reused.

The tombstone contains no message content, content digest, citation, source reference, provider response, or failure text.

### 3.2 `retained`

A retained session remains inspectable after normal close. It may later be explicitly deleted after all turns are terminal. Explicit deletion uses the same atomic purge and sanitized tombstone path.

Conversation retention is not memory authority. Retained messages remain conversation records only.

## 4. Lifecycle

Text diagram:

```text
session active
    |
    v
turn received
    |
    v
context ready
    |
    v
generating
    |-------------------|
    |                   |
    v                   v
completed          interrupted
                        |
                        v
                  context ready
                        |
                        v
                   new attempt
```

A nonterminal turn may also transition to `cancelled` or `failed` with a sanitized machine-readable code.

Only one nonterminal turn may exist in a session at a time.

### 4.1 Interruption and resume

An interruption:

- closes the active generation attempt as `interrupted`;
- increments the turn interruption count;
- records only a sanitized reason code;
- marks the session and turn as interrupted.

Resume returns the same turn to `context_ready`. A new generation attempt receives the next contiguous attempt index. Interrupted provider output is not stored as an assistant message.

### 4.2 Completion

Turn completion requires:

- the active request ID;
- exact provider and model identity;
- an accepted or abstained validation outcome;
- an assistant message exactly equal to the visible model response;
- a non-cancelled model finish reason;
- an assistant classification no more sensitive than the session classification.

The response digest is bound to the stored assistant-message digest.

## 5. Data model

Schema version 1 defines:

- `conversation_schema_migrations`;
- `conversation_sessions`;
- `conversation_turns`;
- `conversation_messages`;
- `conversation_turn_references`;
- `conversation_generations`;
- `conversation_state_events`;
- `conversation_session_tombstones`.

SQLite requirements:

- foreign keys enabled;
- WAL journal mode;
- `FULL` synchronous durability;
- trusted schema disabled;
- transactional migration;
- exact required-table verification;
- SQLite integrity verification.

## 6. Messages and classifications

Each turn stores at most:

- one user message at ordinal 0;
- one assistant message at ordinal 1.

Each message preserves:

- message ID;
- turn ID;
- visible content;
- SHA-256 content digest;
- ordinary data classification;
- UTC timestamp.

Ordinary state accepts only:

- `PUBLIC`;
- `INTERNAL`;
- `PRIVATE`.

A message or reference cannot exceed the session classification. `HIGHLY_SENSITIVE` and `SECRETS` fail before ordinary storage.

## 7. Source and grounding references

A turn may preserve ordered references to:

- authoritative memory;
- authoritative memory source provenance;
- a Phase 1 chunk;
- a Phase 1 source;
- a deterministic grounding packet.

References preserve identifiers and optional citation/digest metadata. They do not copy source content into the conversation-state tables.

P3.2 stores references but does not perform retrieval. Live retrieval and grounding orchestration remain P3.3 responsibilities.

## 8. Generation metadata

Each generation attempt preserves:

- generation ID;
- exact request ID;
- provider and model;
- contiguous attempt index;
- lifecycle status;
- metadata-only reasoning status;
- validation outcome;
- finish reason;
- response SHA-256 when completed;
- sanitized failure code;
- start and completion times.

No database column stores hidden reasoning, a scratchpad, chain-of-thought content, or provider thinking text. Schema verification rejects known prohibited reasoning-content column names.

## 9. Inspection and integrity

Inspection is metadata-safe by default. Message content is returned only when the caller explicitly requests content-inclusive inspection through the local application boundary.

The deterministic integrity report verifies:

- exact schema and SQLite integrity;
- contiguous turn ordering;
- no more than one nonterminal turn;
- session/turn interruption consistency;
- exactly one user message per turn;
- exactly one assistant message for completed turns;
- message SHA-256 binding;
- session classification bounds;
- reference limits and classification bounds;
- contiguous generation attempts;
- valid metadata-only reasoning statuses;
- completed-generation/assistant-message digest binding.

Tampering is reported rather than silently accepted.

## 10. Policy limits

The P3.2 public policy currently bounds:

- maximum message length: 100,000 characters;
- maximum turns per session: 10,000;
- maximum references per turn: 256.

These are application safety limits, not model context limits. Model context and output budgets remain governed by the P3.1 model policy.

## 11. Non-goals

P3.2 does not implement:

- user-query classification;
- Phase 1 retrieval;
- Phase 2 memory retrieval;
- cited grounding-packet construction;
- model orchestration;
- constitutional prompt compilation;
- final response validation;
- a text CLI or web interface;
- automatic conversation-to-memory formation.

## 12. Exit criteria

P3.2 is complete when:

1. private conversation databases are forced outside the repository;
2. schema initialization is versioned, transactional, and idempotent;
3. ordered session, turn, message, reference, and generation state is preserved;
4. only ordinary classifications can enter the ordinary state store;
5. no hidden reasoning content can be persisted;
6. interruption and resume preserve deterministic attempt lineage;
7. visible assistant content is bound to the completed model response;
8. session-only close physically purges conversation content;
9. retained sessions remain inspectable and can be explicitly deleted;
10. sanitized tombstones contain no conversation content;
11. tampered message content fails integrity verification;
12. existing Phase 1, Phase 2, and Phase 3 regression suites remain passing.

## 13. Next milestone

P3.3 will build grounded conversational orchestration on top of these contracts. It will retrieve read-only Phase 2 memory and Phase 1 evidence, construct deterministic grounding packets, call the P3.1 model boundary, validate the visible response, and record the resulting P3.2 turn state.
