# Mega Architecture Migration

**Version:** 1.0.0
**Date:** July 27, 2026
**Authority:** A.L.I.C.E. Constitution 1.1 and ADR-003 through ADR-007

## Purpose

This migration consolidates every architecture decision approved during the July 27, 2026 redesign into one authoritative repository state. It supersedes the earlier Roadmap 2.0, Governance 1.0, Project-Wide Capability Unblocking, Friday Product Architecture 2.x, and Consumer Product Architecture 3.0 delivery packages.

## Final decisions

- A.L.I.C.E. is a model-independent, continuously learning personal cognitive system.
- Rayan retains final constitutional authority; A.L.I.C.E. is independent in judgment and subordinate in purpose.
- The protected authority kernel is minimal and exists to preserve ownership, truthful state reporting, provenance, credential custody, stop/rollback authority, and recoverable known-good states.
- Phases 1–4 are released compatibility baselines, not immutable architecture.
- Capability activation is mission- and profile-driven, with open-ended capability names and resource dimensions.
- Continuous learning may produce memory, beliefs, skills, code, rankers, adapters, and future model-weight updates.
- Self-coding, self-evaluation, self-modification, and evidence-based production promotion are intended capabilities.
- The Personal Cognitive Kernel begins extraction at Phase 5.0; formal product separation occurs at the Phase 6.5 gate.
- `Friday` is an internal development codename. Every consumer host selects the assistant's name and develops a private identity.
- The consumer product has the same ultimate capability destination as A.L.I.C.E.; temporary productization lag is allowed, permanent omission is not.
- A.L.I.C.E. remains Rayan's frontier deployment. Generalizable successes flow through the parity ledger into the shared kernel and consumer product.
- Research capabilities that remain beyond present engineering reach stay in the Research Frontiers Register for long-term work.

## Migration discipline

The migration uses a separate Git worktree based on `origin/main`, so the active Phase 4.5 worktree and any uncommitted local changes are not switched, reset, or overwritten. The active worktree is audited read-only using the new scope registry. Unexpected unscoped barriers stop the migration. The script does not automatically register or waive them.

## Compatibility rule

A released module may intentionally preserve a narrow historical behavior. Such a module must be explicitly bound to a named compatibility profile and a successor runtime. Compatibility status preserves reproducibility; it does not create a permanent capability ceiling.
