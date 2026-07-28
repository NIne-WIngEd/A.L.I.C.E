# A.L.I.C.E. Scope and Non-Goals

> **HISTORICAL RELEASE SCOPE:** This document describes only the first implementable `v0.1` release. It is retained for reproducibility and does not define A.L.I.C.E.'s destination architecture, current roadmap, permanent non-goals, or capability ceiling. Current authority is defined by `docs/ALICE_CONSTITUTION.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, capability profiles, and ratified ADRs.

**Version:** 1.0.0
**Applies to:** First implementable release (`v0.1`) only
**Scope kind:** Historical compatibility document
**Capability ceiling:** false
**Owner:** MK Rayan

## Product statement

A.L.I.C.E. v0.1 was defined as a local-first, text-based personal assistant prototype that could answer questions using an approved personal knowledge vault while showing the sources and memory records used.

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

The following were deferred from the first release. They are intended future directions or research programs under the current roadmap, not permanent prohibitions:

- sending email or messages;
- changing calendars or accounts;
- purchases or financial transactions;
- unrestricted terminal or administrator access;
- continuous background monitoring;
- computer-wide control;
- production self-modification;
- autonomous code deployment;
- model training on the complete life archive;
- ingestion of the entire initial dataset;
- voice-first interaction;
- mobile application;
- custom operating-system kernel;
- unsupported representation of A.L.I.C.E. as conscious, human, or infallible.

## Phase 1 pilot-data limit

The first ingestion experiment used a deliberately selected, reviewable subset rather than the complete archive.

Recommended pilot:

- 50–200 files;
- less than 2 GB total;
- multiple data types;
- known duplicates and contradictions for testing;
- no active credentials;
- no identity documents in the first run;
- a written ground-truth answer set.

## Success condition for v0.1

A.L.I.C.E. could answer a defined set of questions about Rayan and one technical project, cite the exact personal sources used, recognize outdated or conflicting records, and correctly process memory corrections and deletions without performing unauthorized external actions.
