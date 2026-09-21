# N0 job 575986 — P2A frozen semantic authority valid FAIL

**Date:** 2026-09-21  
**Status:** valid model/capability failure at P2A; Production P2 remained closed; no rerun/tuning authorized  
**Implementation source:** `alice-eipm-v1-n0-frozen-semantic-authority-v3@0a8ac74aa6fa73259c03dc4d8754a42a6fde2307`  
**Magnolia job:** `575986`  
**Slurm:** FAILED / exit `42:0` / elapsed `00:05:04` / node `gpu001`  
**stderr:** empty

## Clean preflight

The run reached the P100 and passed the complete source/artifact lineage preflight before the governed evidence root was created.

Confirmed in stdout:

- exact source revision `0a8ac74aa6fa73259c03dc4d8754a42a6fde2307`;
- corrected P1 checkpoint SHA `91f2c78dcc35967064189af4a7110cdee83e5652fb037d3d88f58500ac3aaab9`;
- preserved job 575958 Production P2 failure;
- preserved job 575962 P2S-v1 failure;
- preserved job 575966 P2S-v2 failure;
- original frozen final validation reused;
- selected repaired DualEndpoint graph SHA `3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`.

This is not an infrastructure failure.

## P2A result

Status:

`FAIL_QSRE_FROZEN_SEMANTIC_AUTHORITY`

Production P2 authorization remained false.

Precommitted metrics:

| Metric | Observed | Gate |
| --- | ---: | ---: |
| Production core single-relation top-1 | 0.4068181813 | 0.95 |
| auxiliary seen relation top-1 | 0.2708333433 | 0.95 |
| auxiliary holdout relation top-1 | 0.2604166567 | 0.85 |
| held-out factor macro accuracy | 0.3666666706 | 0.90 |

All four gates failed.

### Component diagnostics

Production core:
- joint preference ≈ 0.17045;
- principle alignment ≈ 0.13864;
- semantic projection ≈ 0.15909–0.18636;
- token evidence ≈ 0.50682–0.52955;
- equal-weight combined ≈ 0.40682–0.41591.

Auxiliary holdout:
- joint preference = 0;
- semantic projection ≈ 0–0.0104;
- principle alignment = 0.125;
- token evidence ≈ 0.21875–0.30208;
- combined ≈ 0.26042–0.29167.

Factor combined macro ≈ 0.36667. Parameter-free token evidence was individually stronger for several factor families:
- role = 0.875;
- traversal ≈ 0.6667;
- modifiers ≈ 0.6667;
- direction = 0.5;
- control = 0.5.

## What 575986 falsifies

It falsifies the final-package assumption that the four frozen surfaces, used with the current P2A runtime geometry, already provide sufficiently discriminative zero-shot relation/factor semantic authority for N0 closure.

It does **not** by itself prove that the frozen semantic backbone contains no useful relation/operator information.

Important counter-evidence remains:

- job 575801: final-layer token late interaction role accuracy 0.6875 versus 0.5625 for mean pool and semantic projection;
- later layerwise audit evidence found causes/supports directional signal in intermediate token layers;
- P2S-v2 job 575966 recovered Production-core relation accuracy to 0.9477 at its best precommitted observation and later 1.0 fit, while held-out relation/factor transfer remained sub-gate.

Thus the semantic/operator boundary is reopened, but immediate semantic-backbone retraining is not yet causally justified.

## Source-level mismatch exposed by postmortem

The original N0 teacher objective trained:

- listwise candidate preference over the original curriculum prompt + candidate pair;
- semantic projection on selected preferred prompt/candidate representations;
- principle alignment between candidate-side semantic projection and the row-level rationale projection;
- contrastive alignment of selected candidate semantics to rationale representations grouped by principle tag.

P2A repurposes those surfaces differently:

- a new `JOINT_MATCH_PROMPT` is prepended before runtime queries;
- semantic projection is treated as a symmetric sentence-embedding space between independently encoded query and schema text;
- runtime schema meaning is placed on the rationale side of the principle-alignment surface.

Those are plausible distribution/task-geometry mismatches. Job 575986 shows the repurposed heads fail badly, but does not separate head/task mismatch from deeper representation insufficiency.

Equal-weight z-score fusion is not the primary explanation because no individual frozen learned surface approaches the gates. Fusion does hurt some Production-core rows relative to token evidence, but token evidence alone is still far below closure thresholds.

## Preserved architecture

575986 produced no evidence against:

- corrected P1 executor;
- ordered remaining-query-evidence coverage;
- continuous relation hypotheses;
- Binder v2 as first exact sparse structural support;
- dynamic runtime relation schema and zero relation-ID parameter axis;
- the selected repaired evidence graph;
- structured state, fusion, latent pool, and original frozen final validation;
- no-ceiling doctrine;
- private identity separation.

Those remain preserved until a diagnostic localizes otherwise.

## Anti-loop decision

Do not:
- rerun job 575986;
- lower P2A gates;
- tune LR/steps/width/batch;
- change equal-weight fusion and immediately retry;
- train P2;
- create P2S-v3;
- delete or rewrite the `qsre-n0-frozen-semantic-authority-v3` evidence root.

The next scientific work is one zero-gradient semantic-base localization pass that distinguishes:

1. representation deficiency;
2. readout/prompt/task mismatch;
3. fusion/calibration error.

No owner GPU run is authorized by this handoff.

## Smallest next investigation

Use the exact frozen checkpoint and exact P2A relation/factor examples. Produce one diagnostic matrix, without optimization, that compares:

- original P2A prompt geometry;
- original teacher-style prompt/candidate geometry;
- raw per-component rankings before and after z-score normalization;
- token late interaction at every hidden layer;
- candidate-conditioned token interaction using the existing parameter-free mechanics;
- pairwise source/target and relation-description interventions;
- per-relation/per-factor confusion;
- oracle best-single-component and simple non-learned fusion upper bounds.

The decision after that single audit:

- if token/intermediate-layer views cleanly recover the targets, redesign the semantic-authority readout/interface without retraining the backbone;
- if recoverable signal exists only after learned P2S adaptation and no frozen representation/readout exposes held-out semantics, reopen semantic representation objectives/data;
- if one frozen component is already strong and fusion alone destroys it, repair calibration/fusion without touching the backbone.

One result, one architecture decision. No hotfix chain.
