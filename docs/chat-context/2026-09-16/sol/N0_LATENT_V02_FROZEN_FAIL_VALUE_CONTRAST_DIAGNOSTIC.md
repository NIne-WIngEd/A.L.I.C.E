# N0 latent pool v0.2 frozen challenge: valid fail, value-contrast diagnostic required

Date: 2026-09-16

## Immutable observed result

Magnolia job 575682 completed successfully with exit 0:0. The preselected step-360 latent candidate (SHA256 `503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`) was evaluated on frozen challenge v0.2. No checkpoint selection, training, parent mutation, or private gradient occurred.

The challenge correctly applied counterfactual removal before parent-cache construction and before ratified cross-context fusion. The overall result remained `FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION`.

All non-counterfactual gates passed. The only failed gate was `counterfactual_family_min_drop`, caused by `missing_structured_evidence` with mean absolute target-cosine drop `-0.009834006428718567`. Overall counterfactual mean target-cosine drop was `0.08747776597738266`.

Strong unchanged capability metrics included best-slot semantic cosine 0.968523, family minimum 0.928698, pooled semantic cosine 0.958509, family pooled minimum 0.935944, centered effective rank 0.204377, disagreement-weighted specialization 0.968399, source-view semantic recoverability 0.884086, pairwise slot cosine 0.431778, and missing-view attention 0.0.

## Post-result measurement audit

The v0.2 intervention point is now causally valid, but the counterfactual score remains an absolute cosine to a full target sentence. For a target and stale alternative that share subject, attribute, syntax, and almost all wording, removing the decisive value can leave absolute sentence cosine nearly unchanged. Therefore the sole failed gate does not yet isolate a capacity bottleneck.

The historical v0.2 result must not be reclassified as a pass.

## Next diagnostic

Run `scripts/eipm/n0/diagnose_n0_v02_latent_pool_counterfactual_value_contrast_v0_1.py` via its guarded Magnolia launcher. This is diagnostic only and has no ratification effect.

For each required counterfactual row it constructs a matched stale-value foil by replacing only the current value in the target summary. It measures normal and ablated target-vs-foil margins for best-slot and pooled latent outputs, plus the required source-view target-vs-foil margin.

Interpretation:
- If the required source is value-discriminative and the latent target-vs-foil margin falls after pre-fusion ablation, the absolute target-cosine gate was insensitive.
- If the required source is value-discriminative and the latent target-vs-foil margin does not fall after pre-fusion ablation, treat the behavior as a real causal-grounding defect and build an independent repair. Consider scaling only if subsequent evidence isolates capacity as the bottleneck.

No gradient, training, ratification, production promotion, or private identity work is authorized by this diagnostic.
