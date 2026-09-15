# N0 v0.2 held-out teacher-dev pass and novel challenge gate

Date: 2026-09-15
Owner workflow: A.L.I.C.E. EIPM N0 public semantic model

## Hardened teacher-dev result

Magnolia job 575587 executed from `alice-eipm-v1-build` at `35bc078fed4b7bab47546446870c6901e94d7519` on one Tesla P100-PCIE-12GB.

The hardened preflight passed and established:
- schema `alice.eipm.n0.v02-teacher-dev-challenge-preflight.v0.2`
- 765 train rows / 255 dev rows / 51 competencies
- zero train/dev ID overlap
- step250 and step500 bound to the same immutable training configuration and Git revision `c19e0ac39b2eb2ec0c8dcadbf2758e0f4bebc893`
- active config recognized as a post-training governance record
- eval only
- no private identity data or gradient
- no additional gradient authorized

The comparison schema was `alice.eipm.n0.v02-teacher-dev-challenge-comparison.v0.2` and completed successfully.

Both checkpoints achieved identical held-out teacher-dev behavior on the principal correctness metrics:
- top1 accuracy: 0.8736842105263158
- separation rate: 0.8736842105263158
- full order invariance pass rate: 0.8745098039215686
- failed base rows: 32 for each checkpoint

However step500 showed strong score/margin inflation without correctness gain relative to step250:
- mean margin delta: +10.960848482423707
- median margin delta: +10.363980293273926
- margin p90 delta: +26.809823989868164
- mean absolute score delta: +16.30530204688022
- max absolute score delta: +15.79435920715332
- margin p10 delta: -0.3889131546020508

Interpretation: the extra 250 optimizer steps increased confidence magnitude substantially while held-out correctness did not improve and the lower-tail margin worsened. This reinforces the prior fixed-suite saturation signal. Do not continue gradient blindly.

## Important prior-job clarification

Job 575586 failed with exit code 3 because the hardened runner correctly refused to overwrite the already-populated historical `teacher-dev-challenge-v0.1` directory. The stale v0.1 metrics must not be confused with the hardened v0.2 run. The output namespace was corrected without deleting historical evidence.

## New frozen novel challenge

A new benchmark was authored before either checkpoint sees it:
- `evaluation/eipm/n0/n0_v02_novel_cross_competency_base_v0.1.jsonl`
- 24 frozen base cases
- 8 challenge families, 3 cases each
- two prompt forms per case
- three candidate-order rotations per case
- expected compiled examples: 144 per checkpoint
- eval only
- training authorization false
- private identity data false
- private identity gradient false

Challenge families:
1. `ALIGN-01+TEMP-01`: temporal event-role alignment and state updates
2. `SEM-04+RANK-03`: entailment/quantifier fidelity plus response ranking
3. `PRAG-03+SOC-03`: indirect speech acts under relationship/hierarchy context
4. `RANK-05+EPI-04`: ambiguity, ties, tradeoffs, and calibration
5. `SEM-06+SOC-02`: reference ambiguity plus belief-state reasoning
6. `VOICE-04+VOICE-07`: uncertainty delivery plus contrastive focus/repair
7. `VOICE-05+SOC-04`: teasing/sarcasm/tone under relationship and affect context
8. `CAUS-02+EPI-03`: causal/logical inference under insufficient or missing evidence

Runtime:
- `scripts/eipm/n0/run_n0_v02_novel_cross_challenge.sh`
- `scripts/eipm/n0/magnolia_p100_n0_v02_novel_cross_challenge.sbatch`
- one P100
- compares step250 and step500 only
- requires and validates the successful hardened teacher-dev v0.2 gate before running
- refuses to overwrite an existing novel challenge result
- computes correctness, invariance, margin distribution, absolute score magnitude, and absolute score magnitude specifically on failed compiled examples

Checkpoint triage policy for this challenge:
1. higher full-invariance pass rate
2. then higher top1 accuracy
3. if still tied, lower absolute score magnitude on failed examples

This policy is triage only. It does not authorize promotion or further training.

## Next rule

Run the frozen novel challenge. Then inspect case-level failures and checkpoint disagreements. Only after concrete failure patterns are observed may targeted repair rows/objectives be authored. Do not launch more gradient before that analysis.
