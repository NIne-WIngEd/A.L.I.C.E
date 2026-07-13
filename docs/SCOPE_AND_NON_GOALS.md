# A.L.I.C.E. Scope and Non-Goals

**Version:** 1.0.0  
**Applies to:** First implementable release (`v0.1`)  
**Owner:** MK Rayan

## Product statement

A.L.I.C.E. v0.1 will be a local-first, text-based personal assistant prototype that can answer questions using an approved personal knowledge vault while showing the sources and memory records used.

## In scope for v0.1

- local text conversation;
- explicit user authentication for access to private memory;
- ingestion of a small, approved pilot dataset;
- document parsing and chunking;
- structured memory records with provenance;
- vector and metadata retrieval;
- answers grounded in retrieved personal sources;
- memory inspection, correction, supersession, and deletion;
- uncertainty and conflict handling;
- read-only web research through an approved tool;
- activity and retrieval logs with privacy controls;
- permission gateway implemented before external-action tools;
- automated tests for memory, permissions, privacy, and prompt injection.

## Out of scope for v0.1

- sending email or messages;
- changing calendars or accounts;
- purchases or financial transactions;
- unrestricted terminal or administrator access;
- continuous background monitoring;
- computer-wide control;
- production self-modification;
- autonomous code deployment;
- model training on the complete life archive;
- ingestion of the entire 14 GB dataset;
- voice-first interaction;
- mobile application;
- custom operating-system kernel;
- representation of A.L.I.C.E. as conscious, human, or infallible.

## Phase 1 pilot-data limit

The first ingestion experiment must use a deliberately selected, reviewable subset rather than the complete archive.

Recommended pilot:

- 50–200 files;
- less than 2 GB total;
- multiple data types;
- known duplicates and contradictions for testing;
- no active credentials;
- no identity documents in the first run;
- a written ground-truth answer set.

## Success condition for v0.1

A.L.I.C.E. can answer a defined set of questions about Rayan and one technical project, cite the exact personal sources used, recognize outdated or conflicting records, and correctly process memory corrections and deletions without performing unauthorized external actions.
