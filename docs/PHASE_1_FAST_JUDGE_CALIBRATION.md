# P1.11 — Fast Human Calibration of Competing Support Judges

The latest seven-case claim-support audit produced a major evaluator
disagreement:

- every audited final claim had already passed the FEVER/ANLI NLI gate;
- the Qwen-based claim-support auditor subsequently labeled many of those
  claims unsupported.

The next step is not another generator or verifier change. It is a small,
blind human calibration set.

## Goals

1. Reuse the existing claim-support audit output.
2. Reconstruct the current cited evidence locally.
3. Re-score those existing claims with the current FEVER/ANLI verifier only.
4. Select a small stratified sample, default 12 items.
5. Review each claim and cited evidence in a local click interface.
6. Compare human labels against:
   - the Qwen claim-support auditor;
   - the FEVER/ANLI NLI gate.
7. Recommend:
   - FEVER as the hard gate;
   - Qwen as the hard gate;
   - or disagreement escalation.

## No expensive regeneration

This workflow does not run Qwen response generation or the four-hour
claim-support audit again.

It reconstructs context for the benchmark questions and runs the local NLI
verifier against the already-audited claim texts.

## Stratification

The sample prioritizes disagreement-heavy buckets:

1. FEVER high-confidence + Qwen not-supported
2. FEVER borderline + Qwen not-supported
3. FEVER low + Qwen supported
4. FEVER high-confidence + Qwen supported
5. FEVER borderline + Qwen supported
6. FEVER low + Qwen not-supported

It also prefers query diversity.

## Blind local review

The local browser UI shows:

- benchmark question;
- claim;
- cited evidence windows;
- source filename/family/owner-relation metadata.

It intentionally does not show the Qwen or FEVER verdict before the human label
is entered.

Labels:

- Supported
- Partially supported
- Unsupported

The server binds only to `127.0.0.1`.

## Metrics

The evaluation reports:

- exact three-class accuracy;
- binary full-support accuracy;
- support precision, recall, and F1;
- Cohen's kappa;
- confusion matrices;
- judge disagreement count.

The automated recommendation requires at least 8 human-labeled items.

This is a calibration aid, not an automatic replacement for ground truth.
