# Project-Wide Capability Barrier Audit

**Version:** 1.0.0
**Audit target:** public `main` plus the local branch at application time
**Date:** July 27, 2026

## Confirmed barrier classes

The public baseline contained capability restrictions in:

- root scope and security documents;
- the Constitution, permission model, threat model, and evaluation charter;
- historical ADRs still readable as active architectural decisions;
- conversation policy, orchestration policy, CLI policy, constitutional policy, and compiled system prompt;
- conversation contracts that rejected all enabled capabilities;
- information contracts and policies that rejected live networking, private context, authenticated browsing, JavaScript, form submission, background work, memory writes, code execution, provider fallback, and larger budgets;
- tests that described phase-local defaults as values that must remain disabled;
- exact phase, version, provider, tool, command, schema-key, and numeric-limit assertions.

## Resolution model

This migration uses four remedies:

1. Replace active project-level governance with Constitution 1.0 and mission-scoped autonomy.
2. Convert foundational contracts to capability-profile authorization.
3. Mark released phase implementations and tests as compatibility profiles rather than universal ceilings.
4. Run a full local tracked-file audit and fail the migration when an unresolved active barrier remains.

## Local Phase 4.5 coverage

The public repository cannot reveal unpushed local work. `scripts/audit_capability_barriers.py` scans the actual local Git index and working tree when the package is applied. It therefore covers Phase 4.5 files that did not exist in the public snapshot.

## Interpretation

The audit does not remove validation, evaluation, provenance, rollback, or credential custody. Those mechanisms increase the reliability of autonomous improvement. It removes the assumption that capability itself is a defect.

## Friday generalization extension

The audit now includes product and host isolation. Phase 1–4 code may be migrated when it assumes one owner, one product, one storage root, or one personal identity. The Phase 6.5 separation gate cannot pass until synthetic multi-host tests prove that shared-kernel components do not mix A.L.I.C.E. and Friday state.
