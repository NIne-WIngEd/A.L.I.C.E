# N0 job 575986 — P2A semantic localization v1

**Date:** 2026-09-21  
**Status:** zero-gradient diagnostic package; no architecture change, optimizer, P2, TEST, or semantic-backbone retraining authorized  
**Parent source:** `alice-eipm-v1-n0-frozen-semantic-authority-v3@0a8ac74aa6fa73259c03dc4d8754a42a6fde2307`  
**Source evidence:** Magnolia job `575986`, `FAIL_QSRE_FROZEN_SEMANTIC_AUTHORITY`

## Why this audit exists

Job 575986 is a clean P2A model/capability failure. All four precommitted gates failed before Production P2.

The failure falsifies the claim that the current four-surface P2A readout already supplies sufficient zero-shot schema authority. It does not yet distinguish:

1. frozen representation deficiency;
2. prompt/readout/task-geometry mismatch;
3. fusion/calibration loss.

Earlier evidence prevents skipping this distinction:
- job 575801 showed token-level relation-role signal stronger than pooled/projection views;
- later layerwise audits found useful relation signal at intermediate layers;
- job 575966 showed a learned schema matcher could recover much stronger Production-core and held-out semantics than P2A, though it still failed the broad semantic gate.

## Diagnostic contract

The audit uses the exact job-575986 semantic checkpoint and exact P2A relation/factor examples.

It is deliberately CPU-only and zero-gradient.

It compares:

- current P2A meta-prompt joint preference;
- teacher-native prompt/candidate joint preference;
- current semantic projection;
- current P2A principle alignment;
- teacher-native prompt/candidate principle alignment;
- current deterministic mixed-layer token evidence;
- token evidence separately for every frozen hidden-state output.

For every task it records:
- component top-1 accuracy;
- confusion matrices;
- every-layer token accuracy;
- the best-layer token upper bound;
- current four-surface accuracy;
- teacher-native four-surface accuracy;
- teacher-native + best-layer diagnostic upper bound;
- per-row any-component oracle upper bound.

The oracle quantities are diagnosis only. They cannot become a production selector.

## Exact source reconstruction

Before interpretation, the audit must reconstruct job 575986's four headline P2A metrics to within `1e-6`.

If it cannot reconstruct the source failure, the diagnostic is invalid and stops.

## Decision after the audit

### Representation/readout survives

If one or more frozen intermediate token views approach the existing gates while pooled/trained heads remain weak, the semantic backbone contains useful information and the P2A interface/readout is the failed abstraction.

The next architecture decision should use a shared multi-layer token/schema interaction rather than another pooled semantic head.

### Prompt/task geometry is a major source

If the same frozen trained heads improve materially when evaluated under their original teacher-style prompt/candidate geometry, P2A repurposed those surfaces outside their training contract.

The next design should not treat arbitrary trained heads as universal semantic metrics.

### Fusion/calibration is dominant

If an individual frozen component is already strong but the equal-weight candidate-zscore fusion destroys that performance, the next change belongs at the calibration/fusion boundary.

### Semantic representation objectives must reopen

If no frozen layer/readout exposes broad held-out relation/factor semantics near the existing gates, while learned P2S adaptation remains substantially stronger, the evidence is sufficient to reopen the public N0 semantic representation objective/data rather than adding another matcher head.

## Anti-loop

This audit does not authorize:

- rerunning 575986;
- lowering gates;
- changing LR, step count, batch size, or width;
- P2S-v3;
- Production P2;
- semantic-backbone retraining before the audit is interpreted;
- private identity gradients;
- N0 completion.

One audit result -> one architecture decision.
