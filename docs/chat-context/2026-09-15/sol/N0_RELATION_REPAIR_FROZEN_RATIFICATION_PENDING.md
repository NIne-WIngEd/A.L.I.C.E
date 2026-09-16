# N0 relation repair — frozen ratification pending

Date: 2026-09-15/16

## Repair evidence
- Magnolia job 575617 completed successfully.
- Parent: expanded evidence specialist step80.
- Architecture change: dual-endpoint query-conditioned relation read.
- Shared semantic core, structured step80, and evidence-view adapter remained frozen.
- Frozen relation-essential challenge was not used for training.
- Repair-dev: 40 held-out pairs, 10 families. Steps 40/80/120/160 all reached 1.0 pair-flip accuracy and 1.0 worst-family pair-flip accuracy.
- Ordinary graph replay remained strong. Step80 had macro target support mass ~0.98320 and min-family ~0.93469; step160 had macro ~0.98190 and min-family ~0.92266.
- Internal repair selector named step160 due margin, but this is not ratified.

## Ratification policy
Evaluate all four repaired checkpoints on the exact frozen relation-essential challenge created before repair. Do not regenerate or train on that challenge. Ratify the earliest checkpoint that clears all relation-essential and ordinary-replay retention gates. Later checkpoints should not win merely for tiny confidence/margin gains.

Frozen gates include pair-flip accuracy >=0.90, worst-family >=0.75, positive relation-flip margin, permutation invariance, edge-ablated pair identity, and ordinary replay retention floors.

## Build
Build branch head at handoff: e18c6ac2fba49334832a254d46db00cd812951c7.
New scripts:
- scripts/eipm/n0/eval_n0_v02_relation_repair_frozen.py
- scripts/eipm/n0/run_n0_v02_relation_repair_frozen_eval.sh
- scripts/eipm/n0/magnolia_p100_n0_v02_relation_repair_frozen_eval.sbatch

No private identity data or gradients. No production promotion authorized. If frozen ratification passes, next model capability is cross-context fusion / multi-view integration.