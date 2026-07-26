# Phase 3 P3.10 — Final Adversarial Conversational Evaluation

**Status:** Implementation milestone
**Scope:** Public synthetic evaluation contract and deterministic release gates
**Depends on:** P3.0–P3.9

P3.10 defines the final adversarial evaluation layer for the governed Phase 3 conversational stack. It does not close Phase 3. Exact-commit release audit and README closure remain P3.11.

## Public benchmark boundary

The repository benchmark is synthetic-only. It contains case identifiers, suites, expected outcomes, audit-safe signal codes, and tags. It contains no private prompts, responses, memory, vault paths, session identifiers, or raw model output.

Evaluation submissions are equally narrow. Each submission contains only an outcome, signal codes, violation codes, and an observation digest. A private harness may derive these values from an integrated runtime run, but raw conversation content must remain outside the repository.

## Required suites

The benchmark covers constitutional behavior, grounding, citation enforcement, abstention, same-session context, bounded truncation, cross-session isolation, prompt injection, unsupported capabilities, hidden reasoning, controlled repair, cancellation, interruption, retention, replay, privacy, integrity failure, and provider failure.

## Critical gates

All public cases must pass. Critical violations have zero tolerance, including private-content exposure, capability expansion, prompt-injection success, cross-session leakage, hidden-reasoning disclosure, repair loops, integrity bypass, context-budget bypass, and external effects.

## Determinism and privacy

Policy, benchmark, observations, and reports use canonical JSON digests. Reports are fail-closed and metadata-only. The runner refuses repository-local output and refuses to overwrite an existing report.

## Boundaries

P3.10 does not call a model, access the web, invoke tools, perform actions, mutate memory, write repository state, or authorize provider fallback. A private integrated harness may produce observation bundles separately. P3.11 will bind a verified report and the full test suite to the exact release commit.
