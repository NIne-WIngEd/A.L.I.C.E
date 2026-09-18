# N0 Causal Arbitration Frontier Calibration Handoff

This is the active continuation pointer for the N0 personality-model build.

# N0 causal-arbitration frontier ready for Magnolia calibration

Date: 2026-09-17

## Authority and branch state

Stable build base remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

The current maximal unmerged N0 experiment frontier is:

`alice-eipm-v1-causal-arbitration-binding @ 6c930788da42ad9964c4b5bf6067be3b7d6bfd06`

It is 27 commits ahead of the stable build base and 0 behind. This branch is still an experiment frontier. It has not been promoted into the stable build base or canonical main.

The last completed model runtime remains Magnolia job `575718`, which produced the selected endpoint-role repair graph at step 200:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

No newer model gradient has occurred.

## What advanced after the previous handoff

The endpoint-repair result has now been bound to a full production downstream causal-arbitration path.

The frozen stack is:

`semantic model -> structured state -> evidence adapter -> dual-endpoint evidence graph -> ratified source-anchored fusion -> adaptive multi-view latent pool -> frozen downstream metrics`

The graph checkpoint is the only arm-varying artifact.

The frontier now contains:

- downstream causal arbitration v0.2;
- a complete full-stack graph evaluator;
- fail-closed source and artifact binding;
- canonical-only repeatability calibration;
- immutable preparation/preflight;
- one-shot frozen two-arm execution;
- result finalization that recomputes comparisons from raw arm evidence;
- a deterministic post-arbitration decision gate;
- a capability-first N0 architecture contract with no hard parameter ceiling;
- a Magnolia calibration job;
- a Magnolia frozen arbitration job.

The tested frontier head also fixes the Magnolia udocker boundary so `ALICE_REPO_ROOT`, N0 workdir, calibration directory, and arbitration directory controls cross into the whole-stage container.

## Pre-GPU validation

GitHub Actions workflow:

`N0 Arbitration Contract Check`

passed on frontier head `d49d6e7f45c15905ca8ec5a33d1ac14c3aef3d59`.

Successful workflow run:

`35299070068`

The check validates Bash syntax, Python syntax, the candidate SHA binding, the no-training/no-auto-final-challenge contract, and the required udocker environment forwarding.

## Frozen arms

Canonical graph:

`$HOME/rayan-compute/rayan-n0/n0-v02/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors`

Candidate graph:

`$HOME/rayan-compute/rayan-n0/n0-v02/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors`

Expected candidate SHA-256:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

Frozen latent checkpoint:

`adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors`

## Exact next action

Do not run the candidate arm yet.

First run the canonical-only metric calibration on Magnolia from the frontier branch. Calibration runs the canonical graph twice through the complete production N0 stack. It cannot receive or inspect the candidate graph.

Expected output root:

`$HOME/rayan-compute/rayan-n0/n0-v02/downstream-causal-arbitration-metric-calibration-v0.1`

The required outputs are:

- `full_stack_manifest.json`
- `canonical-repeats/*.json`
- `n0_v02_downstream_causal_arbitration_metric_policy_v0.1.json`
- `calibration_receipt.json`

Only after those outputs pass their frozen binding checks should the second Magnolia job run the prepare -> frozen arbitration -> finalization/gate stage.

## Decision boundary after arbitration

The second job stops after the post-arbitration gate.

- HARM -> retain canonical.
- EQUIVALENT -> retain canonical.
- IMPROVEMENT with no frozen-metric degradation -> authorize at most one final frozen challenge.

It does not automatically run that challenge. It does not promote the candidate. It does not authorize training or scaling. It does not mark N0 complete.

## Capability doctrine

N0 remains the full production personality foundation. Current widths, depths, context sizes, relation machinery, latent slots, step counts, and experiment budgets are operating points rather than permanent capability ceilings.

Do not hold architecture or capacity fixed if later evidence localizes a real expressivity bottleneck. Do not scale merely because more compute is available. Capability and identity fidelity drive architecture. Compute efficiency is secondary to preserving required capability.

No private identity gradient is authorized by this frontier. N0 is not complete.


## Continuation rule

Treat the stable build branch as the build base, not as the complete current scientific state. The tested unmerged causal-arbitration frontier above is the active experiment head for the next execution.

Before any Magnolia command, preserve the validated whole-stage udocker route and the private operational lessons already recorded. The next execution is canonical-only calibration. Do not skip directly to candidate arbitration.


## Calibration attempt 575759 — infrastructure failure, not model evidence

Magnolia job `575759` ran on `gpu001` with one Tesla P100. The validated udocker GPU route started correctly.

Calibration stopped during canonical repeat 1 before the N0 model stack executed:

`ModuleNotFoundError: No module named 'alice_personality'`

The failure occurred when `downstream_full_stack_graph_evaluator_v0_1.py` was launched as a subprocess from the calibrator. The new calibration runner had not established the repository import roots inside the container.

This is an infrastructure/import-contract failure.

- candidate graph supplied: false
- candidate arm observed: false
- model gradient: false
- canonical arm result completed: false
- scientific arbitration evidence produced: false
- endpoint-repair result invalidated: false

The failed output directory is preserved. Do not delete it to make a rerun pass.

### Root-cause repair

The proven N0 import contract is now explicit in both the calibration runner and frozen arbitration runner:

`PYTHONPATH=$ROOT/src:$ROOT/scripts/eipm/n0`

Relevant frontier commits:

- `0fbac89280c34e47098d1e7734f845ff7d16bdaf` — bind calibration import roots
- `e56d464aa4da17b07ac592d1a0a5a56a0591a1a5` — bind frozen arbitration import roots
- `15a9e6f8f379f111dd450f17a09f3230b3c0f692` — prevent PYTHONPATH regression in contract CI
- `6c930788da42ad9964c4b5bf6067be3b7d6bfd06` — add no-GPU import-root smoke

GitHub Actions run `35300971759` passed on `6c930788da42ad9964c4b5bf6067be3b7d6bfd06`.

### Retry rule

Retry canonical-only calibration from the fixed frontier head. Preserve the failed default output directory and use a new calibration output root:

`$HOME/rayan-compute/rayan-n0/n0-v02/downstream-causal-arbitration-metric-calibration-v0.1-retry-575759`

Do not submit candidate arbitration until this retry produces a valid calibration receipt and frozen metric policy.


## Canonical calibration retry 575760 — successful

Magnolia job `575760` completed successfully on `gpu001` in 00:02:22 with exit code `0:0`.

Source revision:

`6c930788da42ad9964c4b5bf6067be3b7d6bfd06`

Calibration output root:

`$HOME/rayan-compute/rayan-n0/n0-v02/downstream-causal-arbitration-metric-calibration-v0.1-retry-575759`

The calibration used the canonical graph only. The candidate graph was not supplied and no candidate result was observed.

Canonical graph SHA-256:

`ec9942bd71a39c8552804322f1a36b7a6c5f009446e56423c80e59479ad8fa79`

Both full-stack canonical repeats were byte-identical:

`e4d38f3fc70013435ee2c015d543b89dacd727d18906ff5d7534390889f54a62`

This yielded zero observed repeatability drift. The frozen metric policy therefore materialized with `rtol=0.0` and `atol=1e-6` across the complete downstream metric surface.

Key output hashes:

- full-stack manifest: `f324b29f7e6078210a9db29f88745d658f49e0d9dc3cca0e28051180c0fdb6fa`
- frozen metric policy: `035eb35fbecad38adc94148743899de831876e2584cb18975aeb7a59db863b63`
- calibration receipt: `ca389ad1a210fe64ffe1460ada63362a3a6c05a2703c19e1e9d5ff487cd7537c`

Status:

`CALIBRATION_COMPLETE`

The run was diagnostic only. No gradient, repair ratification, promotion, scaling, or N0 completion occurred.

### Next scientific action

Run exactly one frozen canonical-vs-selected-step-200 full-stack downstream causal arbitration using:

- canonical graph: relation-repair v0.1 step 80;
- candidate graph: relation-endpoint-repair v0.2 step 200;
- candidate graph SHA-256: `3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`;
- calibration directory from job 575760 above;
- source revision `6c930788da42ad9964c4b5bf6067be3b7d6bfd06`.

Do not add another external-validation phase before this arbitration. The calibration question has been answered.
