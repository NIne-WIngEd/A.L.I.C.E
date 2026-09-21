# N0 Production QSRE single-launch Magnolia frontier

**Date:** 2026-09-20
**Experiment branch:** `alice-eipm-v1-qsre-production-core-v1`
**Qualified source HEAD:** `b32fabe49f206c2d71e17df5197a62b2a37c8c43`
**Full preflight:** GitHub Actions `35546280865` — SUCCESS

## What changed before launch

The Production QSRE/N0 pipeline was not launched immediately after implementation. A final source and lineage qualification found and corrected two infrastructure/evaluation problems before any production gradient:

1. `ALICE_N0_EXPECTED_REVISION` was required inside the governed runner but was not forwarded across the Magnolia uDocker boundary. `magnolia_udocker_exec.sh` now forwards it explicitly.
2. The final QSRE causal ablation originally compared augmented evidence tensors against shorter base evidence tensors. That could confound the measured QSRE contribution with fp16 attention shape effects. The ablated path now carries the exact same two extra evidence-token positions as the integrated path, but those positions are zero and invalid. The only causal difference is QSRE content/validity.

## Static qualification

The preflight now:
- runs `bash -n` on the full pipeline runner, P100 sbatch and uDocker wrapper;
- verifies the P100 request and uDocker execution path;
- verifies `ALICE_N0_WORKDIR` and `ALICE_N0_EXPECTED_REVISION` cross the container boundary;
- parses every P0-P5 Python CLI and proves all required arguments are wired by the shell runner;
- verifies production and final dynamic schemas are both materialized;
- proves the frozen final validation is created before P1;
- proves final-only artifacts never enter P1/P2/P3/P4;
- compiles the full production/final-validation Python stack;
- reruns production mechanics/metric tests;
- rebuilds the production curriculum byte-for-byte and verifies SHA-256;
- rebuilds the final native validation deterministically;
- proves P4 contains no oracle focus execution path;
- reruns the no-accidental-capability-ceilings audit.

All checks passed in workflow `35546280865`.

## One owner launch

The next authorized action is exactly one Magnolia P100 job using:

- `scripts/eipm/n0/magnolia_p100_n0_v02_qsre_production_full_pipeline_v1.sbatch`
- `scripts/eipm/n0/run_n0_v02_qsre_production_full_pipeline_v1.sh`

The owner must update the local experiment branch to exact HEAD `b32fabe49f206c2d71e17df5197a62b2a37c8c43`, export that same value as `ALICE_N0_EXPECTED_REVISION`, and submit from the repository root.

The job freezes the final native validation before P1, then executes P0 -> P1 -> P2 -> P3 -> P4 -> native P5/final validation. Failure stops the chain. Existing evidence must be preserved. No automatic rerun, hotfix, LR search, step search or width search is authorized.

## Completion semantics

Only `PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE` with `n0_complete=true` and `n1_authorized=true` closes N0. Any earlier stage failure or final gate failure is evidence to localize; it does not authorize a tuning loop.

Private identity data and private identity gradient remain closed throughout N0.
