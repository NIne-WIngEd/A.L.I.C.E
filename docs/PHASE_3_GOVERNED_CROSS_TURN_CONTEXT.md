# Phase 3 P3.8 — Governed Cross-Turn Context

**Status:** Implementation milestone
**Depends on:** P3.2 private conversation state, P3.5 orchestration, P3.6 response validation, and P3.7 local runtime

## Purpose

P3.8 lets a later conversation turn use eligible earlier turns from the same private session. The context is assembled before model generation. It stays bounded, deterministic, integrity checked, and read-only.

This milestone closes the main continuity gap left by P3.7. The local runtime can now carry forward approved conversation history instead of sending only the current user message.

## Context lifecycle

```text
private session state
        ↓
session-integrity verification
        ↓
eligibility filtering
        ↓
whole user/assistant turn pairs
        ↓
newest contiguous suffix within policy limits
        ↓
current user message
        ↓
model request and P3.6 response validation
```

## Eligible history

A prior turn is eligible only when all of these conditions hold:

- it belongs to the same session as the current turn;
- it is earlier than the current turn;
- its state is `completed`;
- it has exactly one user message and one assistant message;
- it has exactly one completed generation outcome marked `accepted` or `abstained`;
- the session passes the existing P3.2 integrity verification.

Failed, cancelled, interrupted, rejected, incomplete, and malformed turns are excluded. Rejected model text is never available because P3.6 does not persist it as an assistant message.

## Whole-turn selection

P3.8 never slices message text and never includes only one side of a conversation turn. It selects complete user/assistant pairs.

Selection uses the newest contiguous suffix of eligible pairs. Older pairs are dropped first. If the newest remaining pair cannot fit a policy limit, selection stops rather than skipping that pair and cherry-picking older material.

The default limits are:

- 12 prior turns;
- 24 prior messages;
- 12,000 prior-message characters.

The current user message is outside those prior-history limits and remains the final request message.

## Request contract

`ModelRequest` now permits this shape:

```text
prior user
prior assistant
prior user
prior assistant
current user
```

The request contract enforces complete prior pairs, distinct message identifiers, distinct prior turns, and a current-turn user message in the final position.

Internal session, turn, message, request, and generation identifiers are not rendered to the model. The existing model adapters still render only message roles and content.

## Deterministic digest

Every assembled context receives a SHA-256 digest derived from:

- the context-policy version;
- ordered message roles;
- ordered message-content digests;
- ordered data classifications.

The digest excludes raw message text and internal identifiers. Identical context content therefore produces the same digest even when stored under different session or turn identifiers.

## Resume behavior

An interrupted turn is not included as prior context. When that turn is explicitly resumed, P3.8 deterministically rebuilds the same eligible completed history and appends the interrupted turn's original user message once. It does not duplicate the current user message or expose interrupted model output.

## CLI integration

The P3.7 local runtime automatically receives governed history through the orchestrator. No new user command is required.

`:inspect` now adds metadata-only context diagnostics:

- policy version;
- context digest;
- included turn and message counts;
- included character count;
- omitted and excluded turn counts;
- truncation status.

Diagnostics do not display message text or internal identifiers.

## Failure handling

Context assembly fails closed. A turn is failed before generation when session integrity cannot be verified or when context cannot be assembled under the policy contract.

Only sanitized codes are propagated:

- `context_integrity_failed`;
- `context_assembly_failed`.

Private state details are not included in the public orchestration or CLI error message.

## Security boundaries

P3.8 does not enable:

- cross-session context;
- rejected or failed output reuse;
- hidden-reasoning persistence or display;
- semantic summarization;
- live retrieval;
- web access;
- tool calling;
- external actions;
- memory writes or promotion;
- automatic response repair;
- provider fallback.

Conversation history remains private session state. It is read only for ordinary context assembly and is still governed by the session's retention mode.

## Testing

The P3.8 tests cover:

- strict policy parsing and anti-weakening checks;
- single-turn backward compatibility;
- ordered prior-pair request validation;
- deterministic newest-suffix selection;
- independent turn, message, and character budgets;
- no partial messages or partial turns;
- accepted and abstained history;
- exclusion of rejected, failed, interrupted, and cancelled turns;
- cross-session isolation;
- integrity-tampering rejection;
- deterministic identifier-independent digests;
- resume consistency;
- sanitized orchestration failures;
- metadata-safe inspection;
- local CLI continuity.

## Deferred work

P3.8 does not summarize long sessions or promote conversation content into authoritative memory. Semantic context compression, governed memory formation, live retrieval, repair, fallback, and graphical interfaces remain separate milestones.
