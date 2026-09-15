# N0 v0.2 Teacher-Dev Challenge — Stale Namespace Collision

**Date:** 2026-09-15

## Observation

Magnolia job `575586` was submitted after pulling build head `6dbf9685ca7621c329e13588ea148d7c47883841`.

The owner then found:

- no `challenge_preflight.json` under `teacher-dev-challenge-v0.1`;
- an existing `teacher_dev_challenge_comparison.json` with schema `alice.eipm.n0.v02-teacher-dev-challenge-comparison.v0.1`;
- the old step-250 / step-500 metrics were readable from that same directory.

The hardened runner introduced at build head `6dbf968...` writes comparison schema v0.2 and writes `challenge_preflight.json` before evaluation. Therefore the files read by the owner were pre-existing pre-hardening artifacts, not evidence from the new hardened run.

## Root cause

The hardened runner correctly refuses to overwrite a non-empty evaluation directory. Its default output root still pointed at the historical `teacher-dev-challenge-v0.1` directory. That directory was already populated by the earlier evaluator, so the new run could not safely write there.

This is an output-namespace collision, not a model failure and not a reason to delete the old evidence.

## Fix

Build branch commit `35bc078fed4b7bab47546446870c6901e94d7519` updates the Magnolia teacher-dev sbatch wrapper to export:

`N0_V02_TEACHER_DEV_EVAL_ROOT=$HOME/rayan-compute/rayan-n0/n0-v02/teacher-dev-challenge-v0.2`

The existing `teacher-dev-challenge-v0.1` directory remains immutable. The fresh v0.2 directory is reserved for the hardened lineage/split preflight and comparison schema v0.2.

## Continuation rule

Do not interpret the old v0.1 metrics as job `575586` results. Preserve them for comparison only.

Pull the current `alice-eipm-v1-build` and rerun the one-P100 evaluation once into the fresh v0.2 namespace. Before using any behavioral result, require both:

1. `challenge_preflight.json` exists and reports PASS for lineage/split/artifact integrity;
2. `teacher_dev_challenge_comparison.json` has schema `alice.eipm.n0.v02-teacher-dev-challenge-comparison.v0.2`.

No additional gradient is authorized by this rerun.
