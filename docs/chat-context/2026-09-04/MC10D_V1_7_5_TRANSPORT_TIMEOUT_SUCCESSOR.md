# MC10D v1.7.5 Transport-Timeout Successor — 2026-09-04

## Observed v1.7.4 boundary

v1.7.4 successfully re-bound the frozen MC10D authority and exact neutral task set, then passed the Magnolia CPU capability preflight.

Preserved science:
- effective pool: 287
- valid replacements: 63
- deferred slots: 1
- slot 63 deferred
- slot 64 resolved; regeneration forbidden
- A-SYN accepted: 0
- A-SYN promoted: 0
- model training: false
- pointwise screen: not started
- MC8: sealed-pending

Neutral task set SHA256:
`7D91C291A65A61D16D5CC0623D58B49D856944F8466FEAD8DBB866EC26B867AA`

The v1.7.4 runtime Slurm file was uploaded:
`/homes/01/mxrayan/rayan-compute/jobs/rayan-mc10d-v174-runtime-860f751f.sbatch`

Independent read-only probes established:
- no v1.7.4 Slurm job existed;
- no runtime cache existed;
- no related remote process existed;
- Slurm primary and backup controllers were UP;
- `sbatch --test-only` reported the runtime script schedulable;
- the local blocked child was `ssh.exe` running the read-only remote SHA256 verification of the uploaded 3 KB runtime script.

Therefore the v1.7.4 stop is classified as:
`READ_ONLY_SSH_SHA256_TIMEOUT`

No runtime download, model pull, judge inference, Drive checkpoint write, repair regeneration, or scientific mutation occurred.

## v1.7.5 release

Package:
`ALICE_MC10D_PROVIDER_NEUTRAL_JUDGE_CONTINUATION_v1.7.5.zip`

SHA256:
`2E5554036D1E4A9D72BA07D6BB58BFF8BE930DB17C2028398999408000ADB6BE`

Launcher:
`Start-ALICEMC10DProviderNeutralJudgeContinuationV175.ps1`

Launcher SHA256:
`AEC67E9CDC3251605195D6FBF9ADC638D46434D89AE977A77E8921B09EE4D453`

Build audit SHA256:
`85FC368C127BB1912AB2E068E413C219986B4E2C0282A326C0F13AEA10556EBD`

Parent v1.7.4 SHA256:
`E025528FFE4637D0E665174BA342B8155F0AC40A4ECB6A27AD6972051929795F`

Frozen scientific workload SHA256 remains:
`F1F79C15EA3052E420AF769E806F9DDBA148C9057671749482A03F353741F6CC`

## Adapter-only changes

- bounded SSH reads;
- 30-second remote SHA probe;
- 120-second SCP bound;
- 45-second sbatch response bound;
- SSH keepalive and one connection attempt;
- exact v1.7.4 timeout-boundary migration;
- remote v1.7.4 no-submission reconciliation before mutation;
- SCP timeout reconciliation by remote SHA;
- sbatch timeout reconciliation by immutable job name;
- no blind resubmission after ambiguous sbatch timeout;
- fresh provider execution names under v1.7.5;
- same neutral run/task scientific identity.

## Scientific behavior unchanged

- Gemma and GLM remain complete 16-task family qualifications.
- Partial Q12 resume remains unauthorized.
- Frozen judge models/digests/prompts/tasks/parser/budgets/thresholds remain unchanged.
- Gemma and GLM are submitted concurrently at the family level.
- Kaggle GPU is not used.
- Magnolia P100 is not used.
- Unauthorized A100 is not used.
- No model substitution.
- No A-SYN acceptance/promotion/training.
- No pointwise screen yet.
- MC8 remains sealed-pending.

## Next action

Stop any still-running hung v1.7.4 local controller/ssh child. Do not delete the remote v1.7.4 runtime .sbatch or Vault state. Run only the v1.7.5 launcher. Preserve all state on exit 75/76/nonzero.
