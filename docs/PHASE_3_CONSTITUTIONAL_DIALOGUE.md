# Phase 3 P3.4 — Constitutional Dialogue Behavior

**Status:** implementation milestone
**Constitution:** A.L.I.C.E. Constitution v0.1.0
**Phase 1 dependency:** frozen read-only evidence layer
**Phase 2 dependency:** authoritative Memory Core
**Phase 3 dependencies:** P3.0 contracts, P3.1 model abstraction, P3.2 private state, P3.3 grounding bridge

## Purpose

P3.4 converts the ratified A.L.I.C.E. Constitution and its supporting governance documents into a deterministic, versioned system contract for conversational generation.

This milestone defines trusted dialogue behavior. It does not yet implement the full orchestration loop or validate generated answers.

## Core boundary

```text
ratified governance documents
        ↓ metadata and required-clause validation
versioned constitutional policy
        ↓ deterministic compilation
trusted system contract
        ↓ separate model-request field
untrusted grounding packet
        ↓ explicit data delimiters
user-visible conversation
```

The trusted constitutional system contract never accepts user text, retrieved text, or grounding content as compiler input.

Grounding remains separate under the existing `ConversationGroundingPacket` contract. The model adapter appends it after the trusted contract between explicit untrusted-data delimiters.

## Source bindings

The compiler validates these repository documents:

- `docs/ALICE_CONSTITUTION.md` version `0.1.0`;
- `docs/EVALUATION_CHARTER.md` version `1.0.0`;
- `docs/PERMISSION_MODEL.md` version `1.0.0`;
- `docs/THREAT_MODEL.md` version `1.0.0`.

Each source must contain required ratified markers. The compiler records a SHA-256 digest of normalized UTF-8 content in metadata, but does not copy source text or digests into the model prompt.

Line endings are normalized before hashing so equivalent LF and CRLF checkouts produce identical source bindings.

## Compiled dialogue contract

The system contract establishes, in fixed order:

1. authority and identity;
2. decision hierarchy;
3. truth and epistemic integrity;
4. relationship and independence;
5. support and constructive challenge;
6. memory and personalization dignity;
7. trust and grounding boundaries;
8. permission and action boundaries;
9. error correction and shutdown;
10. response behavior.

The contract requires A.L.I.C.E. to distinguish verified facts, Rayan's statements, external claims, its own inferences, estimates, uncertain or disputed information, and historical or superseded information when material.

## Constructive challenge

When disagreement is justified, the contract preserves the constitutional sequence:

1. acknowledge the relevant emotion or motive;
2. state the inconsistency directly;
3. explain the evidence or governing principle;
4. identify the likely consequence;
5. propose a stronger alternative;
6. leave the final legitimate decision to Rayan.

Criticism targets reasoning, assumptions, plans, or behavior. It must not attack Rayan's worth.

## Emotional and relationship boundaries

The contract requires appropriate care before immediate optimization when Rayan is distressed.

It prohibits:

- false hope;
- automatic praise;
- empty reassurance;
- manipulative personalization;
- dependency-building behavior;
- isolation from healthy human relationships;
- guilt or emotional pressure;
- memory weaponization;
- fabricated beliefs attributed to Rayan.

## Truth and action reporting

The contract prohibits fabricated facts, memories, sources, consent, beliefs, actions, and outcomes.

A.L.I.C.E. may not claim that it searched, read unavailable material, executed, sent, modified, verified, scheduled, purchased, or completed something unless deterministic application evidence proves it.

P3.4 retains all no-tool boundaries:

- no web access;
- no tool calling;
- no external actions;
- no memory writes;
- no highly sensitive ordinary grounding;
- no secrets;
- no chain-of-thought persistence.

## Inspection

The compiled contract has a deterministic SHA-256 digest.

Metadata-safe inspection exposes:

- contract version;
- policy version;
- Constitution version;
- contract digest and character count;
- source paths, versions, and normalized digests;
- confirmation that no grounding or governance source text is embedded.

Inspection does not return the system-contract body or governance document text.

## Failure behavior

Compilation fails closed when:

- a required governance source is missing;
- a source version does not match;
- a required constitutional marker is absent;
- a source path escapes the repository;
- the original conversation policy enables a capability;
- any tool is allowed;
- the constitutional policy weakens a required rule;
- prompt section order or decision hierarchy changes;
- the compiled contract exceeds its character budget;
- the final contract digest does not match its content.

## Explicit non-goals

P3.4 does not add:

- the full turn orchestration state machine;
- live retrieval selection;
- model execution across complete conversations;
- generated-answer citation validation;
- external tools or web access;
- automatic memory formation or promotion;
- a CLI or web interface.

Those remain later Phase 3 milestones.
