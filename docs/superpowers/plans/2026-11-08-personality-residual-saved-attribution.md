# Saved residual attribution implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; preserve original evidence and request fresh review before dispatch.

**Goal:** Determine what the saved residual branch learned and contributes before another training decision.

**Architecture:** A pure diagnostic helper reconstructs the final natural branch from saved native packets and checks selected AdamW transitions. An owner-isolated reader binds original artifacts and exports only aggregates. No production model changes.

**Tech stack:** Existing Python/PyTorch, native objective primitives, current Magnolia owner runtime.

**Spec:** `docs/eipm/PERSONALITY_RESIDUAL_RESULT_DIAGNOSIS_2026-11-08.md`

## Global constraints

- Frozen Gemma/N1/N3; no updates, new labels or native promotion.
- One original P2 entry before private operations; nondestructive custody thereafter.
- Full final 80 broader contexts; selected original 1->2 and 39->40 optimizer transitions per arm.
- Natural final branch derivatives only; no historical or corrected-gradient claim.
- Existing draft PR100, immutable FBM successor, no private payload in Git.

## Review focus

JSON view order, multiply authorized source copies, blocked coordinates, target
uncertainty/pair direction, and Adam step/decay semantics each receive an actual
native or actual optimizer regression. Source/config/runtime/pins must agree.

## Task 1: Pure saved-state measurement

Create `scripts/eipm/gemma_n0/conditional_residual_saved_attribution.py` and
`tests/personality/test_conditional_residual_saved_attribution.py`.

- [x] Tests fail for missing helper. Public native packets at a nonzero gate:
  restored authorized N1 source states and final readout equal actual native
  execution; loss and branch gradients equal native objective derivatives.
- [x] Implement `representation(decision)`, `replay(model, disabled, natural)`,
  `order_targets(case, decision)`, `order_loss(logits, mask, targets)` and
  `adam_transition(before, after, parameter_names)` with explicit boundaries.
- [x] Test real AdamW states: step/moment/delta reconstruction, no-grad vs zero
  gradient/decay, and refusal of corrupt moments or step metadata.
- [x] Run focused and affected residual/native tests; preserve failure logs.

## Task 2: Protected reader and dispatch

Create `magnolia_residual_saved_attribution_20261108.py/.sbatch` beside helper,
plus a versioned local launch controller and public source/verification receipts.

- [x] Implement bindings for 580417 summary/log/plan/manifest and selected snapshots; recover
  selected source cases and all 80 natural/disabled packet pairs by exact IDs.
- [x] Implement native reconstruction and objective checks before derivatives;
  retain case-level measurements privately, publish aggregate norms/effects.
- [x] Test entry guard ordering and source pins; independent review checks
  entire diagnostic and launch package. Resolve findings before submission.
- [x] Submit once under mxrayan with durable intent; observe startup once.

## Task 3: Record and continue

- [x] Add immutable FBM v29 preserving v28 prefix and process-only eligibility.
- [ ] Commit/push reviewed public artifacts to existing draft PR100; verify exact
  bytes and update continuation with job/result paths and remaining decisions.
- [ ] If diagnostic finishes during useful work, inspect it. Otherwise stop at
  that real dependency without idle polling. No next training run by guesswork.

## Execution ledger

580417 is a valid negative learning result. Original-path local verification:
79 tests/22 subtests. New affected tests:34/4; focused:7. Review-driven native
mask and recipe-format regressions preserved RED then passed. Reader581351
failed before measurement because actual retained-scope tuples also crossed
the JSON/tensor plan boundary. Its original sources and receipt are preserved.

Ruling: correct the entire metadata comparison boundary in a separate r2 reader,
with actual retained-scope construction and writer artifacts, rather than amend
the original run or infer a model defect. See the separate eight-step metadata
diagnosis. Recovery/native suite9 passed; exact-source review blockers0. Job581352
was observed RUNNING under mxrayan with owner isolation and no stderr. Measurement
success and the next learning decision remain pending. Four final governance
validators passed. FBMv29 adds trace50 and retains all49 prior rows exactly.
