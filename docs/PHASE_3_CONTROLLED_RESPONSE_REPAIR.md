# Phase 3 — Controlled Response Repair

**Status:** Implementation milestone
**Milestone:** P3.9
**Package version:** `alice_conversation 0.10.0`

## 1. Purpose

P3.6 rejects generated responses that violate the response-validation contract. P3.9 adds one narrow recovery path after that rejection.

A repair is not a generic retry. It is a separate generation attempt triggered only by a deterministic validation rejection. It uses the same provider, model, grounding packet, capabilities, and governed cross-turn context.

The repository policy keeps repair disabled by default. A caller must explicitly supply or enable the P3.9 repair policy before the second generation is permitted.

## 2. Controlled lifecycle

```text
initial model response
        ↓
P3.6 deterministic validation
        ├── accepted or abstained
        │       ↓
        │   complete the turn
        │
        └── rejected
                ↓
        record rejected response digest
                ↓
        policy permits one repair?
                ├── no → fail the turn
                │
                └── yes
                        ↓
                sanitized repair request
                        ↓
                one same-model generation
                        ↓
                P3.6 validation again
                        ├── accepted or abstained → complete the turn
                        └── rejected → fail the turn
```

## 3. Repair request boundary

The repair request receives only:

- the unchanged constitutional system contract;
- the unchanged ordered conversation messages;
- the unchanged grounding packet;
- the unchanged capability contract;
- sorted and unique validation issue codes;
- metadata-only digests that bind the original request, rejection, context, and grounding.

It does not receive the rejected response text. It does not receive hidden reasoning. It cannot add retrieval, tools, web access, external actions, memory writes, or provider fallback.

The repair request and generation IDs are derived from one deterministic SHA-256 digest:

```text
repair-request:<sha256>
repair-generation:<sha256>
```

## 4. State behavior

The existing generation-attempt schema already supports P3.9. No database migration is required.

For a successfully repaired turn:

- attempt `0` is marked failed with validation outcome `rejected`;
- only the original response SHA-256 is retained;
- the turn returns atomically to `context_ready`;
- attempt `1` is the controlled repair generation;
- only attempt `1` may complete the turn and create the assistant message.

Rejected response text is never inserted into conversation messages or inspection output.

A completed turn still has exactly one completed generation and exactly one assistant message. Earlier rejected attempts remain metadata-only audit records.

## 5. Limits

The P3.9 policy enforces:

- exactly one repair attempt;
- no provider or model change;
- no context or grounding change;
- no capability expansion;
- at most 64 sanitized validation issue codes;
- at most 4,096 repair-directive characters;
- at most 1,024 repair output tokens;
- at most 2,048 combined requested output tokens;
- at most 900 seconds of total elapsed generation time.

The time and output limits cover the original generation and the repair attempt together.

## 6. Failure behavior

Repair is not attempted after:

- provider failure;
- model timeout;
- model-budget failure;
- cancellation;
- conversation-state integrity failure;
- context-assembly failure;
- request or response protocol failure.

An interrupted repair attempt cannot be resumed into a third generation. A second validation rejection fails the turn with `response_repair_exhausted`.

## 7. Inspection

Metadata-safe inspection may expose:

- whether repair was attempted;
- generation-attempt count;
- original and repair statuses;
- validation outcomes;
- response digests;
- repair-request digest;
- whether both attempts used the same provider and model.

Inspection does not expose user text, grounding text, rejected output, repaired output, or hidden reasoning.

## 8. Scope boundaries

P3.9 does not add:

- repeated retries;
- provider fallback;
- alternate-model routing;
- live retrieval;
- web access;
- tool calls;
- external actions;
- memory writes or promotion;
- semantic conversation summarization;
- a graphical interface.

## 9. Verification

The P3.9 targeted suite covers:

- explicit policy enablement and disabled-by-default behavior;
- exact one-attempt enforcement;
- deterministic repair digests and IDs;
- issue-code-only repair prompts;
- same-model, same-context, and same-grounding guarantees;
- rejected-text non-persistence;
- atomic generation-state transitions;
- successful completion after repair;
- terminal failure after a second rejection;
- cancellation, interruption, model timeout, total-time, and output-budget behavior;
- idempotent replay after successful repair;
- metadata-safe inspection;
- adversarial policy weakening.
