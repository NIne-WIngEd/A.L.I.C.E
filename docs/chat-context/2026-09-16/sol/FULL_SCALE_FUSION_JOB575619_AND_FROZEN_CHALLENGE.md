# Full-Scale Fusion Job 575619 + Frozen Challenge Gate — 2026-09-16

## Observed run

User pulled build head `4b58455a5c82e9ca69b2b3fdd2f50d41d5bcd03b` and ran the audited CPU/no-ceiling gate.

CPU gate:
- 17 tests passed, 2 harmless PyTorch nested-tensor warnings.
- 320-row public synthetic fusion curriculum compiled, 240 train / 80 dev, 10 families.
- no accidental capability ceilings gate passed.
- view_count_ceiling=none.
- field_node_edge_runtime_ceilings=none.
- semantic token-level representation=true.
- finite checkpoint vocabularies migratable=true.
- no gradient performed.

GPU training:
- Magnolia job 575619.
- node gpu001.
- COMPLETED 0:0.
- elapsed 00:01:59.
- full-scale fusion model: 116,981,764 trainable fusion parameters.
- parents unchanged.
- no private identity data or gradient.

Checkpoint results:
- step80: gate FAIL. Macro routing 0.9149191, min family 0.8066376, decisive acc 1.0, semantic cosine 0.9556014, preservation 0.9289137, disagreement MAE 0.0498663.
- step160: gate PASS. Macro routing 0.9784053, min family 0.9643085, decisive acc 1.0, semantic cosine 0.9641312, preservation 0.9486324, disagreement MAE 0.0437194.
- step240: gate PASS and provisional winner. Macro routing 0.9835126, min family 0.9738669, decisive acc 1.0, semantic cosine 0.9652088, preservation 0.9507348, disagreement MAE 0.0398966, missing view max weight 0.0.
- step240 fusion SHA256 `9b2a7a120539e9c6a254ea5a47d30a0c9c827d76c18dc46822ee171303bb8d14`.

No ratification yet. Required next action from comparison: untouched cross-context fusion challenge.

## Frozen untouched fusion challenge created in repo

Build branch additions:
- `configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.1.json`
- `scripts/eipm/n0/build_n0_v02_cross_context_fusion_frozen_challenge.py`
- `scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge.py`
- `scripts/eipm/n0/run_n0_v02_cross_context_fusion_frozen_challenge_prepare.sh`
- `scripts/eipm/n0/run_n0_v02_cross_context_fusion_frozen_challenge_eval.sh`
- `scripts/eipm/n0/magnolia_p100_n0_v02_cross_context_fusion_frozen_challenge.sbatch`

Challenge properties:
- 96 rows.
- 12 families.
- independently authored public synthetic evaluation templates.
- no training curriculum rows/entities/text templates reused.
- challenge training use forbidden.
- no gradient.
- private identity data/gradient false.
- thresholds frozen in config before any challenge result is observed.

Frozen gates:
- family min routing >= 0.85
- macro family routing >= 0.92
- decisive view accuracy >= 0.90
- fused semantic cosine >= 0.88
- view preservation cosine >= 0.88
- disagreement geometry MAE <= 0.15
- missing view max weight <= 1e-6
- all required.

Families cover novel vocabulary/lexical transfer, semantic decisive under weak state, structured decisive under hedged prose, multi-hop supersession, evidence correction against stale consensus, unresolved equal conflict, reliability-conditioned routing, all three missing-view combinations, low-confidence distractor resistance, and three-way temporal conflict.

Evaluation candidates are exactly step80/160/240. Any candidate must pass the frozen challenge gate before ratification. Among passing candidates selection remains capability-first: worst family -> macro -> decisive -> semantic -> preservation -> disagreement -> earlier if tied.

Challenge rows may not be trained on for this ratification decision. A failure triggers failure-driven analysis/new independent work rather than re-training on the frozen challenge and re-grading it.

Authoritative stack state on build branch updated to `FULL_SCALE_FUSION_TRAINED_STEP240_PROVISIONAL_FROZEN_CHALLENGE_REQUIRED`.

## Execution next

1. Pull latest `alice-eipm-v1-build`.
2. Run CPU freeze script through udocker with NVIDIA disabled. It materializes challenge/manifest/freeze receipt with exact Git revision and SHA before results.
3. Do not edit/pull after freeze.
4. Submit one-P100 frozen challenge job.
5. Read result. If PASS, write ratification manifest and advance to adaptive multi-view latent pooling. If FAIL, do failure-driven analysis without using frozen challenge rows for training.

Private N1 identity gradient remains CLOSED.
