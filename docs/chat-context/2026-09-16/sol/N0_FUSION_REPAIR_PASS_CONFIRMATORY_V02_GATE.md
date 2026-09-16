# N0 Fusion Repair Pass + Confirmatory Challenge v0.2 — 2026-09-16

## Repair result

Magnolia job 575621 completed 0:0 in 00:02:06 on gpu001.

The full-scale source-anchored fusion repair resumed from original fusion step240 (SHA256 `9b2a7a120539e9c6a254ea5a47d30a0c9c827d76c18dc46822ee171303bb8d14`). No smaller/pilot replacement was introduced.

Repair step240 / total fusion update count480 is the preselected winner:
- fusion SHA256: `4d51494beb788f74ddc03590da05ea00f0e36574294649c2cd5d438f9472e577`
- macro routing similarity: 0.9885677920171508
- worst-family routing similarity: 0.9807335833708445
- decisive-view accuracy: 1.0
- fused semantic cosine: 0.9693831051699817
- contextualized/source cosine: 0.9529840919408905
- disagreement geometry MAE: 0.023420968729624292
- missing-view max weight: 0.0
- source-anchor summary max abs error: 0.0
- source-anchor token max abs error: 0.0
- parents mutated: false
- private identity gradient: false

Repair comparison status: `PASS_FULL_SCALE_REPAIR_READY_FOR_NEW_UNTOUCHED_CHALLENGE`.

## Architecture lesson

The v0.1 frozen challenge revealed that source preservation and contextualized fusion must be distinct channels. The repaired architecture exposes exact parent token/summary anchors while leaving the learned contextualized representation free to diverge when useful. Source anchors add zero parameters and impose no capacity ceiling.

The retired v0.1 challenge is diagnostic-only after informing repair. Its rows, entities, and text templates remain forbidden for training and cannot ratify the repaired checkpoint.

## Confirmatory challenge v0.2 design

A new independent challenge is required before fusion ratification.

Key statistical correction versus v0.1: repair step240 is selected **before** challenge v0.2 is frozen. v0.2 evaluates only that one candidate once. The challenge is not allowed to choose among checkpoints.

Build-branch files:
- `configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.2.json`
- `scripts/eipm/n0/build_n0_v02_cross_context_fusion_frozen_challenge_v0_2.py`
- `scripts/eipm/n0/run_n0_v02_cross_context_fusion_frozen_challenge_v0_2_prepare.sh`
- `scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge_v0_2.py`
- `scripts/eipm/n0/run_n0_v02_cross_context_fusion_frozen_challenge_v0_2_eval.sh`
- `scripts/eipm/n0/magnolia_p100_n0_v02_cross_context_fusion_frozen_challenge_v0_2.sbatch`

Coverage: 136 rows, 17 families, 8 rows/family. New entity namespace (`Cipher ...`) and independent text templates. No original fusion rows, repair rows, or retired challenge rows are reused.

Important new capability coverage includes:
- historical query over superseded state;
- current state through three updates;
- temporal-successor latest and previous-state queries;
- correction against stale majority;
- support aggregation;
- derived-from chains;
- causal chains;
- unresolved three-source conflict;
- semantic and structured authority;
- reliability reversal;
- missing-view conflict resolution;
- missing-view historical retrieval;
- lexical high-overlap/low-authority distractors;
- consensus with one stale source;
- exact source-anchor integrity.

`temporal_successor` direction was cross-checked against the canonical evidence curriculum: successor/newer -> previous/older (`1->0`, `2->1`).

Frozen numeric gates before any v0.2 result:
- worst-family routing >= 0.90
- macro routing >= 0.94
- decisive-view accuracy >= 0.95
- fused semantic cosine >= 0.90
- source summary anchor max abs error == 0.0
- source token anchor max abs error == 0.0
- disagreement geometry MAE <= 0.12
- missing-view max weight <= 1e-6

Contextualized/source cosine is report-only, not a hard gate, because contextualized representations are intentionally allowed to diverge from source anchors when that improves reasoning.

If v0.2 passes: write the fusion ratification manifest and advance to adaptive multi-view latent pooling.
If v0.2 fails: retire v0.2 and perform failure-driven analysis without training on its rows.

Private N1 identity gradient remains CLOSED.
