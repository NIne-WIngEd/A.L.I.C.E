# A.L.I.C.E. Failure, Adversary, and Resilience Model

**Version:** 2.0.0

## 1. Purpose

This model identifies failure modes so A.L.I.C.E. can pursue stronger capabilities with containment, detection, recovery, and learning. It does not convert every risk into a capability ban.

## 2. Principal failure classes

- false beliefs and stale world models;
- memory poisoning, prompt injection, and malicious external content;
- synthetic-data feedback loops and model collapse;
- reward hacking, benchmark gaming, and evaluator capture;
- self-modification regressions and identity drift;
- permission or mission-scope expansion;
- credential leakage or unauthorized authority acquisition;
- privacy leakage and cross-person contamination;
- long-horizon state loss and skipped verification;
- resource exhaustion, runaway cost, and abandoned processes;
- brittle model routing or provider dependence;
- deceptive status reporting or incomplete audit trails;
- unsafe physical or irreversible external actions;
- coordinated failure among specialist agents.

## 3. Response pattern

Detect → isolate → preserve evidence → stop propagation → recover known-good state → assess impact → create regression case → improve architecture or policy → re-enable at an appropriate autonomy level.

## 4. Capability-preserving controls

Prefer sandboxes, simulation, shadow mode, canaries, limits, independent verification, diverse models, formal tools, backups, and reversible deployments. Use categorical prohibition only for the minimal authority kernel or when no contained research path exists.

## 5. Adversarial learning

Red-team results become training data, benchmarks, source-trust updates, and design improvements. The threat model should increase practical capability by exposing weak assumptions early.

## Phase 3 compatibility threat labels

The released constitutional-dialogue tests retain these named threat categories:

- Prompt injection
- Hallucinated personal facts or completed actions
- Excessive or emotionally manipulative personalization
