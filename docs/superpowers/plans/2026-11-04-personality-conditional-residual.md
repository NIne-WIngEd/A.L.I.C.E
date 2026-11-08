# Personality conditional residual implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Existing owner
> authorization covers researched construction and owner-account compute.

**Goal:** Build and test a separate learned N1-context-to-N2 judgment route while
preserving the full personality contract and compatible trained state.

**Architecture:** An experimental N2 extension retains all original modules and
adds masked cross-attention over authorized contextual views. Its zero-initial
tanh gate adds to final candidate states before every original output head.
The exact same parent, data, corrected replay and recipe compare inactive and
learned gate arms. No production/native checkpoint promotion.

**Tech stack:** Existing PyTorch2.7.1 native model, current owner-isolated
Magnolia runtime and original source/feature admission; no new dependency.

**Spec:** ../../eipm/PERSONALITY_GRADIENT_RESULT_AND_CONDITIONAL_RESIDUAL_2026-11-04.md

## Constraints and review focus

- Frozen N0/N1/N3 and source custody; only explicitly experimental N2 changes.
- Preserve all original learned tensors, full source/query path and every head.
- Both arms use identical natural/corrected supervision; architecture access is
  the only treatment. Never silently reuse the old uncorrected control.
- Empty/masked memory cannot create evidence; candidate/authority masks persist.
- Full snapshots bind extension source and treatment mode; resume must reproduce
  optimizer/RNG/model state. Original native loaders must not admit this export.
- Use existing full evaluation, retain harms, no scalar-only success criterion.

## Task1: conditional reader and explicit experimental migration

Files: scripts/eipm/gemma_n0/conditional_judgment_residual.py;
tests/personality/test_conditional_judgment_residual.py.

- [x] Write and run missing-implementation RED tests for full zero-gate equality,
  masks/empty memory, memory permutation, magnitude-sensitive access, first gate
  and subsequent branch gradients, complete downstream heads and strict reload.
- [x] Implement residual and explicit cloned migration, preserving all original
  state keys/tensors and declaring experimental architecture identity.
- [x] Verify affected native interfaces and original source invariants.

## Task2: matched learning, complete assessment and owner dispatch

Files: separate residual learning helper/worker/sbatch, focused experiment tests,
public source/verification/dispatch receipts and FBM successor seed.

- [x] Reuse the established conditional-learning snapshot/admission machinery
  through an explicit separately versioned helper. Both arms get admitted replay;
  only residual access differs. Preserve original helpers and controllers.
- [x] Verify whole-vs-resumed actual native updates and full objective coverage.
- [x] Reuse the original worker's complete236-document/216-objective assessment,
  corrected probes, diagnostics and closing custody with explicit extension
  reconstruction. Verify exact source bindings and fresh review.
- [x] Prepare the reviewed method/evidence for publication and submit once under mxrayan. Capture job ID
  and one startup observation; stop when only the external result remains.

Submitted job580417 under mxrayan; one observation at2026-11-04T07:04:17Z
reported RUNNING on node029 with the initial isolated namespace check and
empty stderr. Full admission, updates and learned behavior remain pending.
FBMv28 adds trace49 and preserves all48 prior rows byte-for-byte.
