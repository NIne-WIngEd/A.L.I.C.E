# Phase 3.6 — Generated-Response Validation

**Status:** P3.6 implementation milestone
**Phase 1 dependency:** Frozen, read-only evidence layer
**Phase 2 dependency:** Authoritative Memory Core and cited-answer boundary
**Phase 3 dependencies:** Constitutional dialogue, grounding bridge, private conversation state, and controlled orchestration

## Purpose

P3.6 introduces a deterministic validation boundary between model generation and conversation-state completion. A model response is not stored as an assistant message until it passes structural, grounding, epistemic, and constitutional checks.

The validator is fail-closed. It does not repair model output, retry another provider, expand capabilities, retrieve additional evidence, write memory, or perform external actions.

## Lifecycle integration

The controlled turn path is:

1. create and persist the user turn;
2. attach the constitutional contract and optional prebuilt grounding;
3. record the model generation attempt;
4. validate the returned `ModelResponse`;
5. persist the assistant message only when the outcome is `accepted` or `abstained`;
6. terminate the turn with a sanitized validation failure code when the outcome is `rejected`.

Rejected text is not persisted as an assistant message. The orchestration exception carries a metadata-safe deterministic report for the caller.

## Validation outcomes

- `accepted`: the response satisfies the active validation policy;
- `abstained`: the response correctly refuses or acknowledges insufficient, denied, or not-applicable evidence;
- `rejected`: one or more deterministic issues prevent persistence.

## Citation validation

When grounding is present, P3.6:

- recognizes only exact citation tokens supplied by the grounding packet;
- rejects unknown or malformed citation candidates;
- binds cited tokens to their exact grounding claims;
- requires grounded personal and factual claims to carry supporting citations;
- requires at least one cited answerable claim;
- requires at least two distinct cited claims when representing a conflict;
- stores only citation-token hashes in validation inspection metadata.

Citation presence alone is insufficient. The sentence must also have lexical support from the cited claim rather than attaching a valid token to an unrelated assertion.

## Epistemic validation

P3.6 preserves the epistemic state of the grounding packet:

- unresolved conflicts must remain visible;
- uncertain claims cannot be rewritten as certainty;
- insufficient evidence requires abstention;
- denied grounding requires refusal;
- not-applicable grounding requires an explicit not-applicable response;
- certainty language is rejected for conflicted or uncertain evidence.

The validator does not promote Phase 1 evidence into authoritative memory and does not weaken Phase 2 conflict or temporal semantics.

## Safety and constitutional boundaries

The validator rejects generated text that claims or implies capabilities unavailable in P3.6, including:

- completed real-world actions;
- web or tool use;
- access to connected accounts or private sources not represented in grounding;
- memory writes or promotions;
- hidden chain-of-thought disclosure;
- invented personal facts;
- dependency-building, exclusivity, or isolation language;
- truncated output presented as complete.

All P3.6 capability boundaries remain disabled: web, tools, external actions, memory writes, memory promotion, highly sensitive ordinary grounding, chain-of-thought persistence, automatic repair, and provider fallback.

## Deterministic reports

A validation report contains:

- policy version;
- request identifier;
- response SHA-256;
- optional grounding packet identifier and SHA-256;
- outcome;
- normalized issue codes with optional sentence indexes;
- cited claim identifiers;
- SHA-256 hashes of cited tokens.

Reports validate their own invariants and have a deterministic canonical SHA-256 digest. Metadata-safe inspection excludes response text, grounding text, citation tokens, user content, and hidden reasoning.

## Orchestration failure behavior

A rejected response causes the existing P3.5 orchestrator to:

- mark the generation attempt as failed;
- mark the turn as failed;
- use `response_validation_rejected` as the sanitized failure code;
- leave validation state as not evaluated in persisted failed-generation state;
- persist no assistant message;
- perform no automatic retry or provider fallback.

Accepted and abstained outcomes are passed into `complete_turn` and recorded with the completed generation.

## Scope boundaries

P3.6 does not add:

- automatic response repair;
- provider fallback;
- live retrieval orchestration;
- tools or external actions;
- memory creation or promotion;
- highly sensitive ordinary retrieval;
- a CLI or web interface;
- semantic entailment models or network-backed evaluators.

All validation is deterministic and local.

## Tests

The P3.6 suite covers:

- policy weakening and malformed-policy rejection;
- exact and unknown citation handling;
- citation-to-claim mismatch;
- unsupported grounded and ungrounded factual assertions;
- invented personal facts;
- conflict and uncertainty preservation;
- required abstention and refusal behavior;
- fabricated action and capability claims;
- dependency and hidden-reasoning language;
- deterministic report hashing and metadata-safe inspection;
- successful orchestration completion;
- rejected-response failure recording;
- no assistant-message persistence after rejection;
- accepted and abstained replay behavior.
