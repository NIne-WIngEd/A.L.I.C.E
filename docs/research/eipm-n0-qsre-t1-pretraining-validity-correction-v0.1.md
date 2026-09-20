# QSRE T1 Pretraining Validity Correction v0.1

**Date:** 2026-09-20  
**State:** v0.47  
**Branch:** `alice-eipm-v1-qsre-t1-executor-training`

## Decision

The v0.46 one-run GPU eligibility is retracted before execution.

This is **not** a learned-model failure. No T1 gradient run has occurred.

Two pretraining validity issues were found during the full-workload audit:

1. The v0.1 T1 curriculum allowed operation/family-specific answer-position shortcuts.
2. The original forward test only established that relation-order changed latent state. Because every supported node could receive `q_step` through the GRU, that test did not prove ordered graph traversal.

## Corrected T1 mechanics

`PATH_FOLLOW` now maintains an explicit oracle focus frontier.

At each relation step:
- the edge must be in oracle support;
- its relation must match the current relation-program step;
- its SOURCE endpoint must be on the current path frontier;
- only endpoints of active edges are updated;
- the next frontier is the TARGET endpoints reached by the active step.

Final PATH_FOLLOW readout is restricted to the reached final frontier. A dead path fails closed.

T1 also requires oracle structural support to be exact edge membership (0/1). Direct field-only support is rejected because unary/support discovery is outside the T1 executor causal question.

## Corrected curriculum

v0.2 remains 504 public TRAIN/DEV rows across nine operation families.

Rows are now organized as same-graph causal pairs:
- SOURCE vs TARGET;
- requested relation A vs relation B;
- SOURCE-plural vs TARGET-plural;
- r1->r2 vs r2->r1;
- reliability metadata swap;
- temporal metadata swap;
- FALLBACK vs DEFER applicability;
- outside-support distractor metadata intervention with invariant target.

Field and edge order are deterministically permuted by causal group.

Exact curriculum SHA-256:

`155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063`

## Production-real representation preparation

T1 does not rerun the semantic backbone.

Preparation reuses the qualified cache:

`5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823`

Only `field_semantic` is reused. TRAIN curriculum fields draw from the source TRAIN pool; DEV fields draw from the source DEV pool. Paired causal rows receive bit-identical field representations. Parent global field scores never enter T1.

## Training boundary

Trainer source and eligibility logic may exist under v0.47, but training execution is closed.

Before one P100 run can be authorized:
1. corrected GitHub pretraining CI passes;
2. Magnolia CPU real-cache preparation passes and produces an immutable receipt;
3. a new state explicitly authorizes exactly one governed T1 run.

No automatic rerun or hotfix is authorized.
