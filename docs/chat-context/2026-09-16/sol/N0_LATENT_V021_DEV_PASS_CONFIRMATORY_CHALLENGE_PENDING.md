# N0 Latent Pool v0.2.1 — Dev Pass, Confirmatory Challenge Pending

## Training result
- Magnolia job 575676 completed successfully, exit 0:0.
- Full-scale competitive latent pool v0.2.1, fresh latent initialization.
- Winner selected on train/dev only: step 360.
- Winner SHA256: `503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`.
- Comparison status: `PASS_COMPETITIVE_LATENT_POOL_READY_FOR_UNTOUCHED_CHALLENGE`.
- This is NOT ratification.

## Winner dev metrics
- best_slot_semantic_cosine = 0.9751548366621137
- family_min_best_slot_semantic_cosine = 0.9590210765600204
- pooled_semantic_cosine = 0.9682566747069359
- family_min_pooled_semantic_cosine = 0.9450539797544479
- mean_pairwise_offdiag_slot_cosine = 0.42469215393066406
- mean_max_offdiag_slot_cosine = 0.99395751953125
- mean_centered_slot_effective_rank = 0.20624157402198762
- mean_disagreement_weighted_view_specialization = 0.9575104223241909
- mean_min_available_view_best_slot_semantic_cosine = 0.9226469276472926
- mean_min_available_view_best_slot_attention = 0.9999999990686774
- mean_min_channel_best_slot_attention = 0.7197942254133523
- missing_view_attention_max = 0.0

## Interpretation
v0.1-style complete slot collapse is resolved on the development split. A high `mean_max_offdiag_slot_cosine` does not by itself imply collapse: one highly similar slot pair is allowed. The set-level indicators are the authoritative anti-collapse signals: mean pairwise cosine, centered effective rank, semantic view recoverability, disagreement-conditioned specialization, and missing-view safety.

Do not add a rule saying every pair of slots must be dissimilar. Redundant slot pairs may be legitimate when the overall representation remains diverse and useful.

## Confirmatory challenge
A new untouched public identity-neutral challenge is required before ratification. Candidate selection is already locked to step 360; the challenge may not select step 120/240/360 post hoc.

Build-side authoritative artifacts:
- `configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0.1.json`
- `scripts/eipm/n0/build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1_1.py`
- `scripts/eipm/n0/eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.py`
- `scripts/eipm/n0/run_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1_1_prepare.sh`
- `scripts/eipm/n0/run_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.sh`
- `scripts/eipm/n0/magnolia_p100_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.sbatch`

The earlier `...builder_v0_1.py` and `...prepare_v0_1.sh` are superseded drafts and must not be executed.

Challenge: 160 rows, 20 families, new `Constellation ...` entities/templates, no private identity data, no training authorization. The evaluator rebuilds the full public parent stack and ratified fusion from challenge text/fields rather than using the training caches.

Frozen gates are capability-oriented and permutation-free. `mean_max_offdiag_slot_cosine` is diagnostic only, not a gate. Set-level slot diversity is gated through mean pairwise cosine and centered effective rank. Source-view semantic recoverability and disagreement-weighted specialization are hard gates.

Counterfactual drop is tested only where one view is genuinely uniquely necessary. Authoritative counterfactual families:
- novel_semantic_authority
- novel_structured_authority
- missing_structured_evidence
- semantic_reliability_reversal

Do not require counterfactual drop in relation/history families where redundant structured/semantic evidence can legitimately preserve the answer.

If the untouched challenge passes, write a latent-pool ratification manifest and advance to multi-head Identity Decision Packet scaffolding. If it fails, do not train on the challenge and retest it; diagnose capability failure, build independent repair data/architecture, and create a new untouched challenge.

Private identity gradient remains CLOSED. N0 remains incomplete.
