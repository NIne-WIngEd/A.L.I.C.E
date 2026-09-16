# Fusion frozen challenge failure + source-anchored repair — 2026-09-16

## Observed frozen challenge

Magnolia job 575620 completed successfully with exit 0:0. The challenge was frozen before results, contained 96 public synthetic evaluation-only rows across 12 families, supplied no gradient, and was never used for training.

No candidate checkpoint passed all frozen ratification gates.

Step 240 remained the strongest candidate:
- decisive view accuracy: 1.0
- macro routing similarity: 0.9227234901239475 (gate 0.92, pass)
- worst-family routing similarity: 0.8476811423897743 (gate 0.85, fail)
- fused semantic cosine: 0.9443701598793268 (gate 0.88, pass)
- contextualized view preservation cosine: 0.8126243089297505 (gate 0.88, fail)
- disagreement geometry MAE: 0.04909189363631109 (gate <=0.15, pass)
- missing-view max routing weight: 0.0 (pass)

Worst routing family was `evidence_corrects_stale_consensus`.

Status: `FAIL_NO_CHECKPOINT_ELIGIBLE_FOR_RATIFICATION`.

## Root-cause analysis

The original full-scale fusion architecture had a non-destructive residual path for contextualized token outputs but exposed only a learned summary-token projection as `view_summaries`. The preservation objective therefore forced one learned vector to serve two distinct jobs:
1. become contextually transformed by cross-view fusion;
2. reconstruct/preserve the ratified parent view.

Training-dev preservation looked strong, but the independently authored challenge dropped to ~0.81. This is an interface/architecture problem, not evidence of a parameter-count ceiling.

## Architecture repair

Added `SourceAnchoredCrossContextFusion` in:
`src/alice_personality/n0/cross_context_fusion_anchored.py`

It leaves all existing fusion parameter tensors and learned behavior intact. It adds zero trainable parameters. It exposes:
- `contextualized_view_summaries`: learned cross-context representation;
- `source_view_summaries`: exact parent summary anchor for every available view;
- `source_view_tokens`: exact parent token anchor for every available token;
- existing `view_summaries` remains a backward-compatible alias for learned contextualized summaries.

Missing views receive zero source anchors so stale unavailable state cannot leak forward.

This separates non-destructive evidence retention from contextualized fusion. Later adaptive pooling may use either/both channels. No capacity ceiling is introduced.

## Failure-driven repair curriculum

Added `build_n0_v02_cross_context_fusion_repair_curriculum_v0_3.py`.

It combines the original 320-row public fusion curriculum with 192 newly authored public synthetic repair rows across 8 families. These are a training tranche, never a coverage cap.

New repair rows target general principles exposed by the failed challenge:
- current verified correction over stale prose/state;
- multi-hop revision/supersession chains;
- direct semantic authority when typed state is weak/unknown;
- verified structured authority over hedged prose;
- relational reasoning with semantic view missing;
- semantic+structured agreement when evidence view is missing;
- equal unresolved conflict;
- consensus with noisy low-authority duplicate evidence.

Frozen challenge rows, entities, and text templates are not reused.

## Repair optimization

Resume from the actual full-scale fusion step240:
SHA256 `9b2a7a120539e9c6a254ea5a47d30a0c9c827d76c18dc46822ee171303bb8d14`.

No smaller model or generic fallback is introduced.

Repair objective:
- fused semantic: 0.35
- view routing: 0.40
- contextualized/source soft alignment: 0.10
- disagreement geometry: 0.15

Exact source preservation is structural now and is tested for zero error rather than approximated solely by a cosine-loss weight.

Semantic/structured/evidence parent modules remain frozen only during this fusion repair run. This freeze is not permanent. End-to-end/progressive unfreezing remains allowed if later fidelity evidence requires it.

## Ratification integrity

The first frozen challenge is retired from future ratification because its failure categories informed this repair. It may remain diagnostic evidence but may never be used as the untouched ratification set for the repaired model.

If the repair succeeds on its independent train/dev tranche, build a new independently authored untouched fusion challenge v0.2 before any fusion ratification.

Private identity data/gradient remain CLOSED.
