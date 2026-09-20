# N0 QSRE T1 PASS / T2 STATIC DESIGN FRONTIER

**Date:** 2026-09-20
**Status:** durable handoff after Magnolia job 575914

## Authoritative experimental frontier

- branch: `alice-eipm-v1-qsre-t2-operator-learning`
- head: `92ea3f23109a8f0064dadaecb4f3ec25ad4678b8`
- parent T1 authorization source: `f4a9edb1915e39d893224f4503746bfd61fc6424`
- state: `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.49.json`
- T2 design: `configs/eipm/n0/n0_v02_qsre_t2_operator_learning_design_v0_1.json`
- decision: `docs/research/eipm-n0-qsre-t1-pass-t2-operator-decision-v0.1.md`
- CI: `35503534795` — SUCCESS

## T1 result

Magnolia job `575914` completed normally.

- status: `PASS_QSRE_T1_EXECUTOR_DEV_CONTRACT`
- result SHA-256: `ba6b82c10fcb13f403569ad24dbd77cf0b3657dfc3eaf9cb9d10b6690b2d32be`
- selected checkpoint: step 50, the first eligible trained checkpoint
- causal pair completion: 1.0
- control accuracy: 1.0
- family minimum success: 1.0
- all nine T1 families: 1.0
- row success: 1.0
- single-target top1: 1.0
- outside-support mass: 0.0
- outside-support invariance delta: 0.0
- plural L1: 0.015601305291056633
- TEST closed
- frozen challenge closed
- no learned operator
- no learned support
- no private identity gradient
- no automatic rerun

Interpretation: the QSRE executor/readout boundary is competent on the current governed public causal DEV workload when operator and structural support are oracle-correct and frozen production-real field states are supplied.

Do not overclaim: T1 does not establish learned natural-language operator extraction, adaptive support, end-to-end QSRE, TEST/challenge performance, private identity fidelity, production readiness, or N0 completion.

## Next causal stage

T2 is scientifically justified and is defined as:

`learned operator + oracle support + frozen proven T1 executor`

T2 static design and CPU/no-gradient mechanics are open. T2 runtime preparation, optimizer, gradient, GPU, TEST, challenge, support learning, scaling, and private identity gradient remain closed.

The first T2 study keeps focus-field binding oracle so T2 cannot silently turn into a support/entity-binding study.

## Immediate next Magnolia action

Use the post-run capture added at the frontier to bind the exact selected T1 checkpoint to the immutable T1 result.

It must emit:

`PASS_QSRE_T1_POSTRUN_CHECKPOINT_BOUND`

This is not a validation ladder. It changes a real downstream decision: exactly which T1 executor checkpoint T2 may consume.

After the receipt is returned, update the T2 design with the selected checkpoint SHA and proceed with the T2 operator mechanics/curriculum preparation. Do not rerun T1.

## Anti-loop / architecture doctrine

- no T1 hotfix or rerun
- no LR/step tuning in response to a future failure without causal localization
- one unresolved causal boundary at a time
- engineering may be parallelized, scientific causality may not
- infrastructure failures are not model evidence
- Graphify/context substrate is navigation-only; verify consequential claims in original branch/path sources
- full EIPM workload envelope remains an architecture requirement
- T1/T2/T3 are causal construction stages, not product-size or capability ceilings
- no permanent parameter/relation/role/field/edge/support/hop/context/compute ceiling
- private identity gradient remains closed

## Magnolia execution conventions

Preserve the established Magnolia workflow:

- repository: `$HOME/rayan-compute/rayan-eipm-main`
- use explicit branch fetch/pull and exact HEAD verification
- use whole-stage uDocker
- export `ALICE_N0_REPO_ROOT="$PWD"`
- export `ALICE_N0_WORKDIR="$HOME/rayan-compute/rayan-n0/n0-v02"`
- do not use host Python
- do not rely on container `$HOME`
- CPU stages use the existing wrapper and no sbatch unless the stage explicitly authorizes P100
