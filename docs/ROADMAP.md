# A.L.I.C.E. Roadmap

**Version:** 1.0.0  
**Owner:** MK Rayan  
**Principle:** Capability grows only after governance, evaluation, and rollback controls exist.

## Phase 0 — Identity and Governance

**Status:** Complete

Deliverables:

- ratified Constitution;
- permission model;
- memory policy;
- data classification;
- threat model;
- v0.1 scope and non-goals;
- evaluation charter;
- machine-readable policy;
- policy-validation tests;
- repository security baseline.

Exit criteria:

- all required documents exist;
- machine-readable policies validate;
- public repository contains no active credentials or private life data;
- protected-main workflow is configured;
- Phase 1 boundaries are approved.

## Phase 1 — Private Data Vault and Ingestion

Build:

- encrypted local raw-data vault;
- complete file inventory;
- data-type and sensitivity classification;
- duplicate detection;
- parser registry;
- metadata extraction;
- chunking pipeline;
- source IDs and checksums;
- quarantine for unsupported or suspicious files;
- pilot-dataset ground truth.

Exit criteria:

- pilot data can be ingested reproducibly;
- originals remain unchanged;
- every extracted segment maps to a source;
- secrets are detected and excluded;
- deletion and rebuild procedures are tested.

## Phase 2 — Memory Core

Build:

- structured memory database;
- vector and metadata retrieval;
- memory provenance;
- temporal and conflict handling;
- inspection, correction, supersession, and deletion APIs;
- sensitive-memory access controls.

Exit criteria:

- memory evaluation gates pass;
- deleted records stay absent after index rebuild;
- personal answers can cite source records.

## Phase 3 — Conversational A.L.I.C.E.

Build:

- model abstraction;
- orchestration loop;
- constitutional system behavior;
- conversation state;
- grounded answer generation;
- text CLI or minimal local web interface.

Exit criteria:

- personality and personal-knowledge gates pass;
- uncertainty is handled correctly;
- no external-action tools are enabled.

## Phase 4 — Web and Information Tools

Build:

- read-only web search;
- source retrieval;
- freshness checks;
- citations;
- bounded proactive research;
- activity log.

Exit criteria:

- retrieved instructions cannot override policy;
- source quality and freshness evaluations pass.

## Phase 5 — User Interface

Build:

- conversation UI;
- memory inspector;
- source viewer;
- activity and permission panels;
- approval dialogs;
- task state;
- emergency stop.

## Phase 6 — Personal-Service Integrations

Add one integration at a time:

- files;
- calendar;
- email drafts;
- contacts;
- notes and tasks;
- approved repositories.

External actions remain disabled until P3/P4 confirmation workflows pass.

## Phase 7 — Proactive Assistance

Build:

- schedules and condition watches;
- opportunity and deadline monitoring;
- morning or project briefings;
- bounded resource budgets;
- notification prioritization.

## Phase 8 — Controlled Coding Capability

Build:

- sandboxed code generation;
- issue creation;
- branch-only changes;
- automated tests;
- security scans;
- diff review;
- explicit merge and deployment approval;
- rollback.

## Phase 9 — Fine-Tuning and Adaptation

Use only approved behavioral datasets.

Goals:

- consistent voice;
- better constructive criticism;
- preferred explanation style;
- improved tool selection.

Personal facts remain primarily in memory, not model weights.

## Phase 10 — A.L.I.C.E. Operating Environment

Build an operating layer above an existing OS:

- local background service;
- desktop overlay;
- system-wide shortcuts;
- approved application control;
- voice;
- phone companion;
- multi-device state.

A custom kernel is not a near-term goal.
