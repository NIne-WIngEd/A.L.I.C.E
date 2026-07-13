# A.L.I.C.E. Memory Policy

**Version:** 1.0.0  
**Status:** Ratified for Phase 0  
**Authority:** A.L.I.C.E. Constitution v0.1.0  
**Owner:** MK Rayan  
**Effective date:** July 13, 2026

## 1. Purpose

A.L.I.C.E. is intended to understand Rayan's life deeply. This policy allows broad personal continuity while preventing untraceable, inaccurate, excessive, or insecure memory.

Personal facts belong in inspectable memory stores—not hidden inside model weights whenever a changeable, correctable record is possible.

## 2. Memory categories

### Working memory

Temporary context required for the current task or conversation.

Default retention: session only, unless promoted under this policy.

### Profile memory

Relatively stable information such as communication preferences, education, skills, values, and long-term constraints.

### Episodic memory

Dated events, experiences, decisions, milestones, conversations, and outcomes.

### Project memory

Objectives, files, decisions, collaborators, commands, failures, current state, and next steps for a named project.

### Goal memory

Short-, medium-, and long-term goals, dependencies, deadlines, progress, and status.

### Procedural memory

Repeatable methods and workflows Rayan uses or has approved.

### Relationship memory

Relevant information about important people and the nature, history, and current status of relationships.

### Reflective memory

Carefully labeled patterns or interpretations inferred from multiple events. Reflective memories are always inferences and never silently become facts.

## 3. Eligibility for durable memory

Information may become durable memory when at least one condition is met:

- Rayan explicitly asks A.L.I.C.E. to remember it;
- it is likely to remain useful across future conversations;
- it materially affects an active project, goal, commitment, or safety boundary;
- it is a major life event or verified milestone;
- repeated corrections show that retaining it prevents recurring errors.

Trivial, fleeting, speculative, or contextless details should remain temporary unless Rayan asks otherwise.

## 4. Required memory record

Every durable memory must contain, when technically possible:

- unique memory ID;
- content;
- memory category;
- source reference;
- source date or applicable time range;
- ingestion date;
- knowledge status;
- confidence;
- data classification;
- people and projects involved;
- validity status;
- verification date;
- superseding or conflicting memory IDs;
- whether Rayan explicitly confirmed it;
- retention and deletion state.

Allowed knowledge statuses:

- `verified_fact`
- `rayan_statement`
- `external_claim`
- `alice_inference`
- `estimate`
- `uncertain`
- `disputed`
- `historical`
- `superseded`

## 5. Source and provenance

A.L.I.C.E. must preserve enough provenance to answer:

- Why is this remembered?
- Where did it come from?
- When was it true?
- Is it current?
- Is it confirmed?
- What changed it?
- Which answer used it?

A summary may point to a source segment, but the source must remain separable from the derived memory.

## 6. Accuracy and contradiction handling

A.L.I.C.E. must not silently choose whichever memory supports a preferred answer.

When records conflict:

1. preserve both records;
2. compare dates, source quality, directness, and confirmation;
3. mark likely historical or superseded information;
4. ask Rayan when the distinction affects an important conclusion;
5. avoid presenting the unresolved claim as confirmed.

A later statement does not automatically erase the historical record. It may change current-state status while retaining chronology.

## 7. Sensitive memory

Memories classified `HIGHLY_SENSITIVE` require:

- encryption at rest;
- restricted retrieval;
- purpose-based access;
- no unnecessary inclusion in cloud-model context;
- explicit disclosure when materially used;
- stronger logging and deletion verification.

Secrets are never personal memories. Passwords, API keys, private keys, session cookies, recovery codes, and authentication tokens belong only in a dedicated secret manager.

## 8. Retrieval rules

Memory retrieval must be:

- relevant to the current request or approved proactive task;
- proportional to what is needed;
- filtered by access and data classification;
- resistant to instructions embedded inside stored content;
- traceable to the memories actually used.

A painful or intimate memory must not be surfaced merely because it is semantically similar.

## 9. Memory modification

Rayan may request an addition, correction, reclassification, consolidation, archival action, or deletion in natural language.

A.L.I.C.E. must:

1. identify the targeted memory or memories;
2. describe the proposed change when ambiguity exists;
3. apply the change through the memory service;
4. update derived indexes;
5. preserve an audit event without preserving deleted sensitive content unnecessarily;
6. confirm what changed and disclose any technical limitations.

## 10. Deletion

Deletion must remove the targeted memory from:

- the primary memory database;
- vector and full-text indexes;
- caches;
- derived summaries used as active memory;
- future retrieval results.

Backups may retain encrypted copies until scheduled expiry. A.L.I.C.E. must disclose this limitation and ensure deleted memories are not restored into active systems.

## 11. Model training

Personal memories must not be used for fine-tuning by default.

Any training use requires a separate, explicit dataset approval that identifies:

- exact records or source collections;
- intended behavior;
- redaction plan;
- storage location;
- model provider;
- deletion limitations;
- evaluation and rollback plan.

Facts that may change should remain in memory/RAG rather than model weights.

## 12. Portability and inspection

Rayan must be able to export memories in a documented, machine-readable format and inspect them through a human-readable interface.

The system must support searches by:

- topic;
- person;
- project;
- date range;
- memory category;
- confidence;
- sensitivity;
- confirmation status;
- conflict or supersession status.

## 13. Release gate

Persistent memory may not enter production until:

- record schemas are versioned;
- source attribution works;
- correction and deletion tests pass;
- sensitive-data retrieval is access-controlled;
- backups are encrypted;
- prompt-injection tests against stored content pass;
- Rayan can inspect and modify memory without editing the database manually.
