# N0 v0.2 First Tranche PASS → Held-Out Dev Challenge

**Date:** 2026-09-14

## Completed run

Magnolia job `575583` completed successfully on `gpu001` in 25m52s using 2 × Tesla P100-PCIE-12GB. The first native `alice-n0-semantic-v0.2` 500-step multitask tranche trained from random initialization only. No private identity gradient was used.

Observed fixed-eval trajectory:

- core top1: 0.359375 random → 0.9765625 step250 → 0.9765625 step500;
- core full invariance: 0.325581 → 0.976744 → 0.976744;
- voice top1: 0.50 → 1.00 → 1.00;
- voice full invariance: 0.375 → 1.00 → 1.00;
- MLM NLL: 10.826638 → 7.069161 → 6.660788;
- MLM perplexity: 50,344.18 → 1,175.16 → 781.17.

The sole fixed-core failure at both trained checkpoints is `N0V02-FIX-ALIGN-01`. Its negative margin worsened from about -0.330 at step250 to -0.697 at step500. Meanwhile fixed-suite score margins grew strongly despite no step250→500 accuracy gain.

Interpretation: real learning occurred and the native v0.2 path is healthy. However the 43-core + 8-voice fixed base suite saturated too quickly to justify more gradient updates or N1 promotion by itself.

## Initial-path failure resolved

The first submission looked under `/root/rayan-compute/...` because `$HOME` inside udocker is `/root`. The owner locally changed the v0.2 workdir to derive from the mounted repository path. Upstream now carries the same fix, so future scripts use `$(dirname "$ROOT")/rayan-n0/n0-v02` rather than container `$HOME`.

## Current decision

Do **not** blindly continue the same 500-step objective mix yet.

Before another gradient update, compare step250 and step500 on the full governed teacher **dev** split that was never used for training:

- 255 dev rows total;
- 5 independent dev scenarios for each of 51 competencies;
- deterministic candidate-order rotations;
- top1, separation, full order invariance, margin distribution, and score magnitude.

This is a much stronger immediate generalization check than the tiny fixed suite and costs only a short one-P100 evaluation job.

## Next executable

After pulling the current `alice-eipm-v1-build`, submit:

`sbatch scripts/eipm/n0/magnolia_p100_n0_v02_teacher_dev_challenge.sbatch`

The output comparison will be written under:

`$HOME/rayan-compute/rayan-n0/n0-v02/teacher-dev-challenge-v0.1/teacher_dev_challenge_comparison.json`

No private data and no training are involved.

After this result, choose the better checkpoint. Then author genuinely novel cross-competency/hard-negative challenge cases and only generate repair teaching for observed failures. Preserve both step250 and step500 checkpoints.
