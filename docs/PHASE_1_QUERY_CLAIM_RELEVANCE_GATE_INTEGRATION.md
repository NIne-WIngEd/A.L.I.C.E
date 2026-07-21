# Phase 1.11 — Query–Claim Relevance Gate Integration

The frozen-threshold relevance holdout validated the local
`cross-encoder/ms-marco-MiniLM-L6-v2` as a conservative irrelevance rejection
filter at threshold `-9.76161`.

Calibration V2:
- precision: 1.0
- recall: 0.666667
- F1: 0.8

Frozen-threshold holdout:
- precision: 1.0
- recall: 0.75
- F1: 0.857143
- false positives: 0

The holdout was small (six claims from three query IDs), so the filter is
integrated conservatively and remains independently auditable.

Pipeline order:
1. evidence-constrained claim generation;
2. personal claim validity contract;
3. query-claim non-irrelevance rejection gate;
4. existing evidence-support/entailment gate;
5. final grounded-response verification.

The relevance gate cannot establish factual support. It only rejects claims
whose question-claim score falls below the frozen threshold. Claims that
survive still have to pass the existing evidence-support gate.
