# N0 Full-Scale Frontier Fusion Policy — 2026-09-15

## Owner correction

For A.L.I.C.E. N0 and later EIPM stages, do not use a reduced-capability model as a temporary "pilot" with the intent to build the real model later.

A bounded training/evaluation run may limit optimizer steps, wall time, checkpoint count, or GPU consumption. Those are experiment controls only. The architecture being optimized must be the actual full-capability architecture intended for the stage.

The word `pilot` is deprecated for new model components because it ambiguously suggests a toy or reduced model. Historical artifact paths containing `pilot` remain unchanged for lineage only.

## Architecture selection doctrine

Do not start from a generic fallback architecture merely because it is common or easy. Select the strongest frontier architecture justified for A.L.I.C.E.'s actual role and constraints. Scale or change it only when measured capability evidence identifies a real bottleneck or a superior architecture.

Capability, personality fidelity, nuance preservation, and correct evidence use outrank parameter minimization or latency. Efficiency is a secondary selector among designs that already satisfy capability requirements.

No hard parameter ceiling applies.

## Fusion correction

The initial cross-context fusion v0.1 used a single flattened Transformer stream. No fusion gradient was performed on that architecture.

It is superseded by v0.2 before training. The active full-scale fusion architecture is:

- three preserved streams: raw semantic, typed structured, relation/evidence;
- per-view self-refinement;
- repeated bidirectional cross-attention between heterogeneous views;
- query-conditioned and reliability-conditioned cross-view gates;
- explicit view identity and reliability;
- missing-view support;
- residual preservation of parent representations;
- distinct contextualized view outputs retained for later adaptive multi-view latent pooling;
- disagreement retained rather than forced away.

Current governing config: `configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json`.

## Objective bug fixed

`soft_distribution_cross_entropy` previously masked target mass assigned to unavailable views and renormalized the remainder. That could silently turn invalid supervision into apparently valid training data. It now raises when any positive target mass is assigned to an unavailable view.

## Current stage

Evidence specialist remains ratified at expanded evidence adapter step80 + dual-endpoint relation repair step80. Fusion gradients remain closed until the corrected full-scale fusion mechanics and curriculum CPU gate passes.
