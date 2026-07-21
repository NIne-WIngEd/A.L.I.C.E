# Phase 1.11 — Verifier Ensemble Diagnostics

This diagnostic compares fixed Boolean combinations of the three existing
claim-support judges on the completed HHEM holdout:

- Qwen auditor
- HHEM at its already-frozen threshold
- FEVER NLI

The analysis preserves the project's existing binary semantics:

- human `supported` is positive;
- human `partially_supported` and `unsupported` are negative;
- Qwen is positive only for `supported`;
- HHEM is positive only when its frozen-threshold decision is true;
- FEVER is positive only for `keep_entailment`.

Evaluated fixed rules:

- Qwen only
- HHEM only
- FEVER only
- Qwen AND HHEM
- Qwen AND FEVER
- HHEM AND FEVER
- all three
- any two of three

This step is diagnostic only. It does not alter the production gate.

Important: selecting the best-performing rule from this holdout would itself
tune the ensemble rule to the holdout. A rule selected after inspecting these
results must be validated on a fresh independent set before production use.
