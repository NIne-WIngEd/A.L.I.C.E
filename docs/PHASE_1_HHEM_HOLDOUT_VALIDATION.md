# Phase 1.11 — HHEM Frozen-Threshold Holdout Validation

This patch adds a human-reviewed claim-level holdout for the P1.11 grounded-response verifier.

## Why

The first 12-item human calibration set selected an HHEM threshold of `0.984156`.
That same sample cannot provide an independent estimate of performance at the
selected threshold.

The holdout workflow therefore:

1. freezes `0.984156` before holdout labels are reviewed;
2. reconstructs the current claim candidate pool;
3. excludes every calibration `item_id`;
4. samples a new stratified claim-level holdout with a distinct seed;
5. reuses the existing loopback-only blind review UI;
6. scores HHEM only at the frozen threshold;
7. compares HHEM with Qwen and FEVER NLI;
8. does not change the production gate automatically.

## Independence scope

This is a claim-level holdout. Calibration item IDs cannot appear in the
holdout. Query IDs may overlap because the pilot semantic benchmark is small.
A later, stronger validation can use a query-disjoint benchmark when more
human-approved benchmark questions are available.

## Production decision

A passing holdout marks HHEM as a candidate for production-gate review only.
`production_gate_changed` remains `false`; a separate explicit change is
required after reviewing the results.
