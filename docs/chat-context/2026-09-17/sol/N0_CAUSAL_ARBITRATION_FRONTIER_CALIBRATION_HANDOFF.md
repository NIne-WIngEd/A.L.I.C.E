# N0 Causal Arbitration Frontier Calibration Handoff

This is the active continuation pointer for the N0 personality-model build.

# N0 causal-arbitration frontier ready for Magnolia calibration

Date: 2026-09-17

## Authority and branch state

Stable build base remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

The current maximal unmerged N0 experiment frontier is:

`alice-eipm-v1-causal-arbitration-binding @ d49d6e7f45c15905ca8ec5a33d1ac14c3aef3d59`

It is 23 commits ahead of the stable build base and 0 behind. This branch is still an experiment frontier. It has not been promoted into the stable build base or canonical main.

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
