# Phase 3 P3.1 — Governed Model Abstraction

**Status:** implementation milestone
**Phase:** 3 — Conversational A.L.I.C.E.
**Boundary:** local inference only; no tools, web, external actions, or memory writes

## 1. Purpose

P3.1 connects the provider-neutral P3.0 contracts to a real local model without
moving governance into the model. The model remains a replaceable component.
Deterministic application code continues to own policy, authorization, budgets,
response validation, and failure handling.

P3.1 does not implement conversation persistence, live memory retrieval,
constitutional prompt compilation, orchestration, or a user interface.

## 2. Components

### `model.py`

Defines:

- the provider-neutral `ConversationModel` protocol;
- cooperative `CancellationToken` behavior;
- bounded-budget, protocol, configuration, timeout, and provider errors;
- sanitized `ProviderFailure` metadata;
- the deterministic test adapter.

### `registry.py`

Registers adapters by exact `(provider, model)` identity. Duplicate identities,
missing methods, empty identities, and unregistered selections fail closed.

### `model_policy.py`

Loads `policies/conversation_model_policy.json` and validates the P3.1 provider
allowlist. The initial allowlist is deliberately small:

- `deterministic-test/fixed-response-v1`;
- `ollama-local/qwen3:8b`;
- `ollama-local/qwen3:4b-instruct`.

The default local model remains `qwen3:8b` because it is the currently validated
Phase 1 local model. P3.1 does not claim equivalent behavioral calibration for
conversation use. That requires later evaluation.

### `transport.py`

Provides an injectable JSON HTTP boundary. Normal tests use fake transports and
never require a running model. The production transport uses only the Python
standard library and bounds response size.

### `ollama.py`

Translates a validated `ModelRequest` into one non-streaming Ollama `/api/chat`
request and translates the response back into `ModelResponse`.

## 3. Local-only boundary

The public policy permits only an explicit loopback URL:

```text
http://127.0.0.1:11434
```

The parser rejects:

- remote hostnames or IP addresses;
- HTTPS/cloud endpoints;
- embedded credentials;
- query strings and fragments;
- preconfigured API paths;
- omitted or invalid ports.

This is intentionally stricter than general Ollama compatibility. Remote model
providers remain outside P3.1.

## 4. Request construction

The adapter constructs this conceptual request:

```text
system contract
+ explicitly delimited untrusted grounding, when present
+ user-visible conversation messages
```

It then sends:

```json
{
  "model": "qwen3:8b",
  "messages": [],
  "stream": false,
  "think": false,
  "options": {
    "temperature": 0.0,
    "num_predict": 1024
  }
}
```

The adapter never adds a `tools` field.

## 5. Budget enforcement

P3.1 uses deterministic application-side limits:

- request timeout: 600 seconds;
- maximum rendered context: 32,768 characters;
- maximum requested output: 4,096 tokens;
- maximum HTTP response: 2,000,000 bytes.

The character limit is an implementation budget, not a claim about tokenizer
accuracy or the model's native context window. A tokenizer-aware budget may be
added later behind its own deterministic tests.

## 6. Response validation

A response is accepted only when:

- HTTP status is successful;
- the JSON root is an object;
- generation is complete;
- provider model identity exactly matches configuration;
- the finish reason is `stop` or `length`;
- the response role is `assistant`;
- visible response content is non-empty;
- no tool calls are returned;
- no hidden thinking is returned;
- no images are returned.

Unexpected response surfaces are rejected instead of ignored. This prevents a
provider from silently expanding the approved P3.1 capability surface.

## 7. Errors

Provider and transport failures expose sanitized structured metadata:

- provider;
- model;
- failure code;
- user-safe message;
- retryability;
- HTTP status when applicable.

Raw request content, private grounding, credentials, and full provider response
bodies are not included in the structured failure.

## 8. Cancellation and timeout semantics

Cancellation is cooperative:

1. checked before request construction;
2. passed to the transport boundary;
3. checked immediately after the transport returns.

The standard-library blocking request cannot be forcefully interrupted at an
arbitrary instruction boundary. The configured timeout is the hard bound while
that call is active. A future asynchronous transport may improve mid-request
cancellation without changing the provider-neutral model interface.

## 9. Testing

Normal tests use an injected fake transport and cover:

- exact payload construction;
- grounding delimiters;
- context and output budgets;
- provider registry selection;
- cancellation before and after transport;
- timeout propagation;
- structured provider failures;
- malformed JSON;
- model identity mismatch;
- incomplete responses;
- tool-call rejection;
- hidden-thinking rejection;
- image rejection;
- policy tampering and remote endpoint rejection.

A live integration test exists but is skipped unless explicitly enabled:

```powershell
$env:ALICE_RUN_OLLAMA_INTEGRATION = "1"
py -m pytest tests\phase3\test_conversation_ollama_integration.py -q
```

The optional test uses the installed local model and may be slow on the current
CPU-only Ollama runtime. It is not part of the deterministic required suite.

## 10. Exit condition

P3.1 is complete only when:

- all Phase 3 tests pass;
- Phase 2 plus Phase 3 tests pass;
- the complete repository suite passes;
- the branch contains no unrelated changes;
- required GitHub checks pass through the protected-main workflow.
