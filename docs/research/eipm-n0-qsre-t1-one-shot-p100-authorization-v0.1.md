# QSRE T1 One-Shot P100 Authorization v0.1

**Date:** 2026-09-20  
**State:** v0.48  
**Stage:** T1 — executor competence with oracle operator + oracle structural support

## Evidence now satisfied

Corrected pretraining CI:

- run `35502096941`
- conclusion: SUCCESS
- source head: `31b89390a2d6e6677092a0f4107fa0e9aacbcdc4`

Magnolia real-cache preparation:

- source cache: `5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823`
- curriculum: `155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063`
- curriculum receipt: `55f7bc40a5d5f1ff5c134e1f4ad00c1788748f317d7c0b118114c111e794b60d`
- prepared cache: `03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564`
- preparation receipt: `b92ddc17749f03230f3feff1d70942ec2fa5d893638574d8c81ee2957608e9db`

The preparation receipt proves:

- 360 TRAIN rows;
- 144 DEV rows;
- paired causal representations are bit-identical;
- TRAIN/DEV source representation pools are isolated;
- direct field support is zero;
- oracle edge support is exact membership;
- no TEST;
- no private identity;
- no optimizer/gradient/GPU in preparation.

## What one T1 run may answer

Only:

> Can the corrected QSRE executor/readout learn the public relational workload when operator and structural support are already correct?

It does not test:
- natural-language operator extraction;
- learned support discovery;
- semantic backbone learning;
- parent graph learning;
- private identity.

## Authorization

Exactly one P100 training run is authorized.

The run must:
- use the exact prepared-cache SHA above;
- use contract v0.2;
- stop at the first fully eligible DEV checkpoint;
- otherwise stop at max_steps and return a genuine T1 failure receipt;
- never open TEST/challenge;
- never auto-rerun.

A failure is evidence about T1 executor competence. It is not permission for LR tuning, longer training, or architecture hotfixes.
