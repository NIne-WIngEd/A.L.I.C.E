# ADR-001: Foundational System Principles

**Status:** Accepted  
**Date:** July 13, 2026  
**Decision owner:** MK Rayan

## Context

The project began as a small voice-assistant prototype. Its new goal is a persistent personal AI system that may eventually hold extensive personal memory and operate tools. A monolithic chatbot or model trained directly on the entire life archive would be difficult to inspect, correct, secure, or update.

## Decision

A.L.I.C.E. will be designed as a system of separable components:

1. **Identity and Constitution** — who A.L.I.C.E. is and how it behaves.
2. **Memory** — what it knows about Rayan, with provenance and lifecycle controls.
3. **Intelligence** — interchangeable reasoning and language models.
4. **Orchestrator** — task planning and component coordination.
5. **Permission Gateway** — deterministic authorization before tool execution.
6. **Tools** — narrowly scoped capabilities.
7. **Interface** — text, visual, voice, and approval surfaces.
8. **Evaluation Laboratory** — tests, traces, release gates, and rollback evidence.

Additional decisions:

- local-first for highly sensitive data;
- RAG and structured memory before personal fine-tuning;
- default-deny permissions;
- no direct model-to-production-tool access;
- source attribution for personal knowledge;
- versioned and reversible changes;
- private data stored outside the public code repository;
- self-improvement means proposing and testing changes, not silently deploying them.

## Consequences

Positive:

- facts can be corrected or deleted;
- models and providers can change without losing memory;
- tool access can be independently secured;
- failures are easier to isolate;
- development can proceed incrementally.

Costs:

- more engineering than a single prompt or chatbot script;
- explicit schemas, logs, tests, and migrations are required;
- some convenience is delayed until permission and evaluation gates exist.

## Rejected alternatives

### Train one model on the entire archive first

Rejected because mutable personal facts would become difficult to inspect, update, or selectively delete.

### Give the model unrestricted tools

Rejected because language-model output is not a reliable authorization mechanism.

### Build a custom operating-system kernel first

Rejected because it adds large engineering cost without first proving memory, reasoning, permissions, and usefulness.
