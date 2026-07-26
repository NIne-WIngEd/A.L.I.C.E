# Phase 3 P3.5 — Controlled Conversation Orchestration

**Status:** implementation milestone
**Phase 1 dependency:** frozen read-only evidence layer
**Phase 2 dependency:** authoritative Memory Core, read-only from ordinary conversation
**Phase 3 dependencies:** P3.0 contracts, P3.1 model abstraction, P3.2 private state, P3.3 grounding bridge, P3.4 constitutional dialogue

## Purpose

P3.5 composes the completed Phase 3 foundations into one controlled single-turn lifecycle:

```text
user message
    -> private conversation state
    -> trusted constitutional system contract
    -> optional prebuilt read-only grounding
    -> exact model-registry resolution
    -> recorded generation attempt
    -> validated provider-neutral model response
    -> one assistant message or a recorded terminal/interrupted outcome
```

P3.5 is orchestration, not a release of autonomous agency. It does not add tools, web access, external actions, memory writes, memory promotion, live retrieval, provider fallback, or hidden reasoning persistence.

## Public components

- `ConversationOrchestrationPolicy` validates the fail-closed P3.5 policy.
- `ConversationTurnCommand` provides deterministic IDs and one user message.
- `ConversationResumeCommand` explicitly resumes one interrupted turn.
- `ConversationOrchestrator` coordinates the existing state, constitutional, grounding, registry, and model boundaries.
- `ConversationTurnResult` returns the accepted assistant message and provider-neutral response.

## State ownership

The orchestrator does not create a parallel state machine. It calls the P3.2 `ConversationStateService` transitions directly:

1. `start_turn`
2. `set_turn_context`
3. `start_generation`
4. exactly one of:
   - `complete_turn`;
   - `interrupt_turn`;
   - `cancel_turn`;
   - `fail_turn`.

Every generation attempt records the exact request ID, provider, model, attempt index, reasoning-persistence status, finish reason, validation outcome, response digest, and sanitized failure code.

## Trusted and untrusted inputs

The trusted P3.4 constitutional system contract is compiled before orchestration and supplied through `ModelRequest.system_contract`.

A P3.3 `ConversationGroundingPacket`, when supplied, remains prebuilt read-only data. It is validated, digested, and converted into metadata-only P3.2 references before generation. P3.5 does not perform live retrieval and does not treat grounding as instructions or authorization.

## Model boundary

The requested provider and model must resolve through the explicit P3.1 registry. P3.5 performs no fallback and no automatic retry.

The response must:

- validate as a `ModelResponse`;
- preserve the request ID;
- preserve the exact provider and model identity;
- contain user-visible non-empty content;
- use an approved finish reason.

Only then can one assistant message be committed atomically.

## Failure and interruption behavior

P3.5 maps model failures to public sanitized codes:

- cancellation: `model_cancelled`;
- interruption: `model_interrupted`;
- timeout: `model_timeout`;
- budget failure: `model_budget`;
- provider failure: `provider_failure`;
- model configuration failure: `model_configuration`;
- protocol failure: `model_protocol`;
- unexpected orchestration failure: `orchestration_internal`.

Provider messages, transport details, stack traces, and private exception text are not written to conversation state.

An interrupted attempt remains immutable. Explicit resume changes the turn back to `context_ready` and creates the next contiguous generation attempt. The original grounding packet must be supplied again and must match the stored packet ID and SHA-256 digest.

## Idempotency

A completed turn can be replayed only when the supplied user message, assistant-message ID, request ID, generation ID, provider, and model match the recorded completion exactly.

The orchestrator returns the existing result without calling the model again. Conflicting idempotency keys fail closed. This prevents duplicate assistant messages and duplicate completed generation attempts.

## Deferred work

P3.5 does not implement:

- final answer-to-grounding validation;
- automatic retry or provider fallback;
- live Phase 1 or Phase 2 retrieval orchestration;
- CLI or web UI;
- tools, web access, or external actions;
- memory candidate creation or promotion.

Those require separate policy, implementation, adversarial evaluation, and release gates.

## Test contract

The deterministic P3.5 suite covers:

- policy weakening and deferred-feature rejection;
- successful grounded and ungrounded turns;
- constitutional request construction;
- exact provider/model metadata;
- metadata-only grounding references;
- cancellation, timeout, budget, provider, protocol, and internal failures;
- sanitized failure persistence;
- interruption and explicit resume;
- contiguous attempt indexes;
- exact-grounding resume checks;
- completed-turn idempotency;
- duplicate-assistant prevention;
- state-integrity verification.
