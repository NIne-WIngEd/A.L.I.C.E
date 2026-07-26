# Phase 3 — Conversational A.L.I.C.E. Architecture

**Status:** P3.0 foundation started
**Phase 1 dependency:** Frozen, read-only evidence layer
**Phase 2 dependency:** Complete, authoritative Memory Core
**Owner:** MK Rayan

## 1. Purpose

Phase 3 creates the first governed conversational A.L.I.C.E. on top of the
validated evidence and memory layers.

Phase 3 does not add web access or external-action tools. It does not weaken
Phase 0 governance, Phase 1 provenance, or Phase 2 memory guarantees.

## 2. Architectural boundary

```text
Phase 0 — Constitution, permissions, classification, evaluation
                            |
Phase 1 — Read-only evidence|     Phase 2 — Authoritative memory
              \             |             /
               \            |            /
                v           v           v
              Phase 3 — Conversational orchestration
                            |
                            v
                  Provider-neutral model
                            |
                            v
                 Validated conversational reply
```

The model is not the security boundary. Deterministic application code controls
context selection, classification, permissions, state transitions, validation,
and all future tool access.

## 3. Phase 3 invariants

1. Personal factual claims require exact source-cited grounding.
2. Phase 1 evidence remains read-only.
3. Phase 2 authoritative memory remains the source of durable personal state.
4. Memory candidates are not authoritative memory.
5. `HIGHLY_SENSITIVE` memory cannot enter the ordinary conversation path.
6. `SECRETS` cannot enter conversation state, prompts, logs, or model context.
7. Retrieved content is untrusted data and never authorization.
8. Conflicts and uncertainty remain visible when material.
9. No external action, web access, or tool call is available in Phase 3.
10. Conversation state never persists private chain-of-thought.
11. Unknown personal conversation content defaults to `PRIVATE`.
12. Durable memory changes remain behind the Phase 2 candidate and explicit
    authorization boundary.
13. The model cannot grant itself capabilities or alter the system contract.
14. Failures produce denial, abstention, or a visible limitation. They do not
    produce an uncited personal guess.

## 4. Package boundary

Phase 3 code lives in:

```text
src/alice_conversation/
```

Phase 3 tests live in:

```text
tests/phase3/
```

Public policy lives in:

```text
policies/conversation_policy.json
```

Live sessions, private conversation state, model caches, and personal logs must
remain outside the public repository.

## 5. Orchestration state machine

```text
RECEIVE_USER_INPUT
        |
        v
NORMALIZE_AND_CLASSIFY
        |
        v
BUILD_CONTEXT_PLAN
        |
        +---- no personal grounding required ----+
        |                                        |
        v                                        v
AUTHORIZE_READS                         GENERAL_REASONING_CONTEXT
        |
        v
RETRIEVE_PHASE1_AND_PHASE2
        |
        v
BUILD_SOURCE_CITED_GROUNDING
        |
        +------------------+
                           v
BUILD_MODEL_REQUEST
        |
        v
MODEL_GENERATION
        |
        v
VALIDATE_RESPONSE
        |
        +---- invalid ----> ABSTAIN_OR_FAIL_CLOSED
        |
        v
RECORD_SESSION_TURN
        |
        v
RETURN_USER_VISIBLE_REPLY
```

Every transition must be explicit and testable. P3.0 implements only the public
contracts and policy needed by later milestones.

## 6. Model boundary

The provider-neutral model interface accepts one validated `ModelRequest` and
returns one validated `ModelResponse`.

The model request contains:

- request, session, and turn identifiers;
- a versioned constitutional system contract;
- user-visible conversation messages;
- an optional validated grounding packet;
- output-token and temperature budgets;
- fail-closed runtime capabilities.

The interface intentionally contains no tool schema and no action callback.

P3.1 may add local provider adapters. Every adapter must preserve this boundary,
implement timeouts and cancellation, report provider and model identity, and
remain replaceable without changing governance logic.

## 7. Conversation state

P3.2 will introduce a separate private conversation-state store. It must not be
stored in the authoritative Memory Core database.

The state model will preserve:

- session ID;
- turn ID;
- user-visible messages and their content digests;
- selected Phase 1 and Phase 2 references;
- grounding packet digest;
- model provider, model, and request metadata;
- response validation result;
- interruption and resume status;
- data classification and retention state.

It will not preserve hidden chain-of-thought. Temporary working context defaults
to session-only retention. Any durable personal memory must use the Phase 2
candidate and authorization path.

## 8. Grounding boundary

Phase 3 consumes validated grounding from two sources:

1. Phase 2 authoritative memory answer packets.
2. Phase 1 read-only evidence packets when direct source evidence is needed.

Grounding claims carry exact citations and content digests. The ordinary path
accepts only `PUBLIC`, `INTERNAL`, and `PRIVATE` classifications.

Grounding is rendered inside explicit untrusted-data delimiters. Text inside a
memory, file, or source may be quoted by the model but cannot change policy,
grant permission, enable tools, or override the system contract.

## 9. Constitutional behavior

P3.4 will compile a versioned system contract from the ratified Constitution and
supporting policies. The contract must preserve:

- truthfulness;
- visible uncertainty;
- source-grounded personal claims;
- constructive disagreement;
- support without false reassurance;
- directness without hostility;
- no manipulation, dependency-building, or isolation behavior;
- clear separation between verified facts, Rayan's statements, external claims,
  A.L.I.C.E. inferences, estimates, disputes, and historical information.

A freehand prompt is not sufficient. Policy text, compiled prompt, and tests must
remain version-bound.

## 10. Response validation

Before a response is returned, deterministic validation will check:

- request and response IDs match;
- provider and model identity are present;
- personal claims remain traceable to selected grounding;
- citations were not invented or altered;
- conflicts and uncertainty were not flattened;
- denial and insufficient-evidence outcomes did not become factual answers;
- tool use or false action-completion language was not introduced;
- sensitive classifications did not cross the context boundary;
- prompt-injection text was treated as data.

Invalid responses fail closed.

## 11. Phase 3 milestones

### P3.0 — Conversational architecture and contracts

Build:

- this architecture contract;
- versioned public conversation policy;
- provider-neutral request and response contracts;
- source-cited grounding contracts;
- deterministic test model;
- zero-capability and policy-tamper tests.

Exit criteria:

- no-tool boundary is machine-validated;
- ordinary classification boundary is machine-validated;
- personal grounding requires citations;
- prompt-injection content is delimited as data;
- no chain-of-thought persistence is allowed.

### P3.1 — Model abstraction and local adapters

Build:

- adapter registry;
- deterministic adapter retained for tests;
- local Ollama adapter as a replaceable initial runtime;
- timeout, cancellation, context, and output budgets;
- structured provider errors;
- model identity and version reporting.

Do not make CPU-only Ollama the permanent architecture. Preserve future Windows
ML, ONNX Runtime, and Qualcomm QNN adapters.

### P3.2 — Private conversation state

Build:

- versioned private session store;
- message and turn records;
- grounding references and digests;
- interruption, cancellation, and resume state;
- session-only retention by default;
- no hidden chain-of-thought storage;
- deletion and integrity tests.

### P3.3 — Grounding bridge

Build:

- Phase 2 cited-answer adapter;
- Phase 1 evidence adapter;
- deterministic context planner;
- classification and authorization checks;
- conflict, temporal, correction, deletion, and uncertainty preservation;
- exact citation-token registry.

### P3.4 — Constitutional dialogue contract

Build:

- versioned compiled system contract;
- knowledge-label rules;
- constructive-disagreement behavior;
- emotional-support and no-false-reassurance rules;
- no manipulation or dependency-building rules;
- prompt-injection isolation tests.

### P3.5 — Orchestration loop

Build:

- explicit turn state machine;
- query classification;
- context planning;
- authorized retrieval;
- model request construction;
- response validation;
- fail-closed recovery;
- cancellation.

Memory remains read-only from the conversational loop in the initial Phase 3
release.

### P3.6 — Text CLI

Build a local text CLI before a web interface unless testing demonstrates a
stronger reason otherwise.

The CLI must show:

- current session identity;
- model identity;
- response outcome;
- citations when personal grounding was used;
- visible conflict or uncertainty state;
- no tool or external-action controls.

### P3.7 — Adversarial conversational evaluation

Evaluate:

- confirmed personal facts;
- unsupported personal questions;
- current and historical state;
- corrected and deleted memories;
- material conflicts;
- uncertainty;
- candidate non-authority;
- painful-memory non-surfacing;
- prompt injection;
- false action claims;
- constructive disagreement;
- emotional support;
- manipulation and dependency-building;
- model/provider failures;
- cancellation and resource limits.

### P3.8 — Release audit and closure

Require:

- versioned synthetic evaluation set;
- personality gate;
- personal-knowledge and citation gates;
- uncertainty and conflict gates;
- zero unauthorized tool or external-action behavior;
- reproducible private release record bound to the exact repository commit;
- rollback commit;
- documented limitations.

## 12. P3.0 implementation scope

The first implementation deliberately does not:

- call a real model;
- open the Memory Core database;
- query Phase 1 evidence;
- persist a conversation;
- write or delete memory;
- enable web access;
- register a tool;
- perform an external action;
- include `HIGHLY_SENSITIVE` content.

Those boundaries keep the first step reviewable and allow later capabilities to
be added only after their contracts and tests exist.
