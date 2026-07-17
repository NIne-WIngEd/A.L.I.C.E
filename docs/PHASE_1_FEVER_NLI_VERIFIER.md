# P1.11 — Fact-Verification-Aware NLI Verifier

The AFM claim remained overwhelmingly neutral after full passages, combined passages, explicit owner attribution, and compact claim-focused windows.

This patch performs one controlled verifier-model comparison before changing generation or lowering safety thresholds.

New verifier: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

The policy explicitly configures the model label order as `entailment, neutral, contradiction`. Thresholds remain unchanged. Claim-focused cited-evidence windows remain enabled. No private data is uploaded.

If the AFM claim remains strongly neutral, the next step is atomic claim decomposition/repair.
