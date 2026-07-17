# P1.11 — HHEM Human-Calibrated Verifier Experiment

The existing 12-item blind human calibration showed that FEVER-NLI confidence
does not separate supported from unsupported claims in this personal-data
sample. The six highest FEVER entailment scores were all human-labeled
unsupported.

This package tests HHEM-2.1-Open on the exact same already-labeled calibration
bundle. It does not change the production response pipeline.

## Model

`vectara/hallucination_evaluation_model`

The model is downloaded locally and pinned to a resolved Hugging Face revision.
Its repository uses custom model code, so loading requires
`trust_remote_code=True`. Private A.L.I.C.E. text is never uploaded.

## Fair comparison

Each HHEM pair is:

- premise: the same evidence windows shown during the blind human review;
- hypothesis: the claim that the human reviewed.

This avoids changing the evidence after human labeling.

Human binary labels are defined as:

- `supported` -> positive;
- `partially_supported` -> negative;
- `unsupported` -> negative.

This is intentionally strict because the production hard gate should only pass
fully supported claims.

## Metrics

The experiment reports:

- HHEM ROC-AUC;
- HHEM average precision;
- best-F1 threshold;
- best-accuracy threshold;
- best threshold meeting 0.90 support precision, if one exists;
- Qwen auditor binary metrics on the same items;
- FEVER-NLI binary metrics on the same items.

The threshold sweep is diagnostic only. The same 12-item sample is used for
threshold discovery, so any promising HHEM threshold still requires holdout
validation before production.

## Production safety

This patch does not replace FEVER-NLI, Qwen, or any production hard gate.
It only evaluates HHEM against the existing human gold labels.
