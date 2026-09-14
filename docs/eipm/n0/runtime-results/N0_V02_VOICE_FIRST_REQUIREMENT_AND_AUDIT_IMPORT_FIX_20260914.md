# N0 v0.2 Voice-First Requirement and Teacher-Audit Import Fix

**Date:** 2026-09-14  
**Status:** active; rerun one upgraded teacher-bank audit  
**GPU used:** no  
**Model training performed:** no  
**Private identity gradient:** no

## Trigger

The owner clarified that A.L.I.C.E. will be used primarily through voice and that the personality model must control tone and emotion, using governed Elaina behavior as the later identity reference.

The first teacher-bank audit attempt also failed immediately with:

`ModuleNotFoundError: No module named 'alice_personality'`

## Root cause

The audit wrapper exported `PYTHONPATH=$ROOT/src` on the Magnolia host, but `magnolia_udocker_exec.sh` did not forward `PYTHONPATH` into the validated Debian udocker runtime. The failure occurred before curriculum auditing and did not indicate a semantic-data problem.

## Concrete repair

`magnolia_udocker_exec.sh` now forwards `PYTHONPATH` using the existing environment-forwarding mechanism. No additional runtime qualification or network retry is warranted.

## Voice-first design change

Voice expression is now first-class in EIPM design while preserving subsystem boundaries:

- EIPM decides expressive intent / delivery policy.
- Replaceable TTS renders that policy acoustically.
- Raw acoustic timbre or speaker cloning is not the EIPM identity objective.
- N0 learns generic public voice-expression reasoning only.
- Elaina-specific expressive tendencies remain N1/N2 identity material and stay behind the existing private-gradient authorization boundary.

The active voice contract is:

`docs/eipm/EIPM_VOICE_FIRST_INTERACTION_CONTRACT_v1.0.md`

## Public N0 voice competencies

Eight supplemental competencies were added without mutating the frozen original 43-competency readiness suite:

- VOICE-01 affect-congruent delivery
- VOICE-02 relationship-sensitive warmth
- VOICE-03 stakes/intensity calibration
- VOICE-04 uncertainty in delivery
- VOICE-05 teasing/sarcasm/playfulness vs hostility
- VOICE-06 empathy/vulnerability without overacting
- VOICE-07 emphasis/pause meaning
- VOICE-08 turn-taking and repair

A separate eval-only benchmark was added at:

`evaluation/eipm/n0/n0_v02_voice_readiness_base_v0.1.jsonl`

A 40-row public teacher shard (4 train + 1 dev for each voice competency) was added at:

`training/eipm/n0/sol_curriculum_voice_expression_v0.4b.jsonl`

The active teacher-bank registry is now v0.4b with 383 registered rows across 51 competencies. Full multitask training remains closed until both the row floor and coverage gate are met.

## Next action

Run exactly one upgraded CPU teacher-bank audit against v0.4b. Do not rerun the obsolete v0.4a-only audit and do not allocate a GPU for this step.
