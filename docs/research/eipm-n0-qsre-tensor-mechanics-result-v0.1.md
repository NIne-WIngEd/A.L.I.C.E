# QSRE Tensor Mechanics Result v0.1

**Date:** 2026-09-20  
**GitHub Actions run:** `35497962294`  
**Result:** SUCCESS

The tensorized QSRE mechanics skeleton passed all static and runtime CPU tests.

## What was exercised

- dynamic tensor interface validation;
- 257 fields and 513 edges, beyond the historical 64/256 operating hints;
- variable relation-anchor and role-anchor counts;
- masked sparsemax with exact zeros;
- adaptive support cardinalities;
- empty-mask fail-closed behavior;
- explicit SOURCE/TARGET incidence;
- support-local readout that excludes a globally dominant outside-support distractor;
- edge-induced field support;
- independent RELATIONAL / FALLBACK / DEFER control;
- field-permutation stability.

## Boundary verification

The source contains no:

- `nn.Parameter`;
- learned projection;
- optimizer;
- backward;
- CUDA training path.

The branch-level clean-sheet guard also remained green at the same frontier.

## Interpretation

The QSRE factorization has now passed three increasingly concrete layers:

1. deterministic interface/oracle contract;
2. pure reference mechanics;
3. PyTorch-shaped tensor mechanics.

The next justified layer is **T1 architecture implementation with parameters present but gradient still closed**.

T1 must test only executor/readout competence under:

- oracle query operator;
- oracle structural support;
- frozen upstream semantic/evidence representations.

The next source implementation is not allowed to include learned support selection or natural-language operator extraction. Those remain T2/T3 causal questions.

No training is authorized by this receipt.
