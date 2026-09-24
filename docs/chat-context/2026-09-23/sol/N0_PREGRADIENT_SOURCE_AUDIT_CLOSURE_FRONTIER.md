# N0 Pre-Gradient Source Audit Closure Frontier

**Date:** 2026-09-23  
**Continuity role:** authoritative handoff for the current N0 successor frontier  
**N0 source branch:** `alice-eipm-v1-n0-full-envelope-foundation-build-v1`  
**Exact N0 head:** `f78cb119b368a372c16e892646f7b7f247138eab`  
**Exact-head CI:** run `35954996471` — **green**  
**Observed exact-head static result:** **259 tests passed**, **219 static proof obligations**, **243 total proof obligations**, proof-matrix static audit PASS  
**Optimizer / gradient / FINAL:** still closed by source policy; no successor joint optimization has happened  
**N0 complete:** false

## Current decision boundary

The deep source/static audit has now moved past the earlier `ac0def...` handoff and through the full staged-training, DEV-selection, runtime-qualification and FINAL-authority source path.

At this exact head there are currently **no known unresolved STATIC_REQUIRED defects** in the proof matrix. The remaining 24 obligations are intentionally non-static: 12 CI receipts, 9 runtime blockers, 2 training blockers, and 1 sealed FINAL blocker. The remaining blockers are the empirical/runtime obligations already precommitted in source:

- P40 exact tokenizer stress;
- P40A short operator evidence/token alignment;
- P40B long-context boundary token alignment;
- P40C long semantic evidence/token alignment;
- P41 exact public-corpus runtime revalidation;
- P39PN exact full public-mixture materialization/audit;
- P42 exact 640-wide CPU no-gradient construction/forward + parameter/RSS receipt;
- P43 same-topology GPU no-gradient memory/route qualification;
- P43E actual DEV evaluator execution before stage selection;
- P44 real J1 -> J2 -> J3 joint optimization;
- P45 actual DEV generalization;
- P46 sealed FINAL-v2 evaluation after DEV-only checkpoint selection.

Do **not** reopen optimizer/gradient authority by editing the source plan after runtime receipts. Source authority remains false permanently. A separate exact-head runtime authorization receipt opens training only after all required runtime evidence passes.

## Important defects found after the previous handoff

The source audit continued after repeated green heads and found real defects each time. The important new classes are below.

### Runtime / source authority and TOCTOU

- FINAL directional endpoint reversal had been bound to semantic-program correctness rather than end-to-end reverse-traversal behavior. FINAL now gates candidate-dependent endpoint behavior.
- J2/J3 transition could use a passing DEV receipt from another checkpoint of the same stage. Stage transitions now bind the exact predecessor checkpoint receipt, system, objective state, accelerator state, source revision and selection receipt.
- Full-public-mixture audit was not bound to the exact manifest later consumed by P43/training. Manifest hash lineage is now required end to end.
- DEV and FINAL evaluators now require the exact clean candidate source checkout.
- P40/P40A/P40B/P40C auditors independently verify claimed revision against the clean checkout instead of trusting caller-supplied 40-hex values.
- The canonical CPU qualification runner rejects untracked source shadowing before any P40-P42 receipt.
- The shared source-authority helper is part of watched/compiled N0 CI authority.
- Pre-gradient runtime receipts now bind their exact producer implementations rather than only a source revision.

### Training authorization and exact-head deadlock repair

The previous source policy said optimizer/gradient/GPU-training authority was false, but no non-mutating path existed to open training after runtime qualification. Editing the plan after P40/P42/P43 would invalidate the exact-head receipts.

The correction introduced:

`scripts/eipm/n0/authorize_n0_v02_full_envelope_training_v1.py`

This authorizer:

- keeps source-file optimizer/gradient/GPU authority permanently false;
- revalidates the exact runtime receipt chain;
- revalidates the public corpus and teacher bank;
- binds the exact mixture, topology, tokenizer, semantic initialization, CPU receipt, GPU receipt, P40A/B/C receipts and static proof receipt;
- emits the only runtime authorization the trainer accepts;
- keeps FINAL closed and private-identity gradient false.

Checkpoint receipts bind this training authorization and the exact trainer implementation.

### P40 tokenizer qualification was previously too weak

The earlier “tokenizer PASS” did not execute the named P40 stress families. P40 is now a distinct exact-head runtime gate covering:

- byte-fallback / OOV behavior;
- Unicode normalization;
- held-out relation/factor fragmentation;
- long entities;
- punctuation/code/math;
- multilingual text.

The exact tokenizer auditor is source-bound and its receipt is producer-hash-bound.

### Training continuation / stage selection governance

- Same-stage continuation exists so an arbitrary runtime step budget is not a capability ceiling.
- Stage transition and same-stage resume are separate authority paths.
- The scheduler horizon can no longer silently become a zero-learning-rate stage ceiling. It decays to a precommitted nonzero floor while DEV still requires continuation.
- Checkpoint evaluation cadence is frozen before gradient. Invocation endpoints must land on that cadence, and first-passing DEV selection walks the complete linear checkpoint chain.
- J2/J3 restart optimizer/scheduler dynamics for newly activated causal owners while preserving selected predecessor model weights and the objective-balancer state.
- Checkpoint receipts now bind the exact trainer implementation; resume, DEV, selector, FINAL opening and FINAL evaluator reject trainer-implementation drift.

### DEV authority hardening

DEV receipts now bind:

- exact DEV evaluator implementation;
- DEV validation contract;
- DEV gate registry;
- proof contract;
- joint training plan;
- exact candidate checkpoint/system;
- exact runtime training authorization;
- exact training-authorizer implementation;
- exact static proof receipt.

The selector verifies this lineage before first-pass selection. FINAL opening and FINAL evaluation carry it forward.

### Static proof receipt hardening

The old proof-matrix audit could establish that required tests existed without itself proving they ran.

The exact-head static proof receipt now:

- uses the canonical proof matrix, supersession map and retrospective audit;
- runs the registered STATIC_REQUIRED pytest nodes itself;
- records JUnit collection/error/failure/skip counts;
- fails if required cases are skipped;
- requires a clean exact-source checkout both before and after execution;
- binds the authoritative N0 workflow and every CI_REQUIRED receipt name;
- binds its exact auditor implementation.

Training, DEV and FINAL re-check this lineage.

### P43 memory qualification was materially expanded

The GPU dry run no longer relies on one lexicographically “large” full-fabric row.

It independently stresses:

- max candidate cardinality;
- max field cardinality;
- max edge cardinality;
- max view cardinality;
- max reasoning depth;
- long additional-view source;
- semantic runtime axes;
- semantic factor cardinality;
- dedicated long semantic context.

It runs the cross-product of semantic and full-fabric stress cases, measures under the actual DDP replica wrapper when distributed, includes measured resident runtime overhead in the projection, and keeps projection explicitly non-authoritative for training by itself.

The Magnolia udocker wrapper and exact repository mount are now CI-watched and shell-syntax checked.

### FINAL closure lineage

FINAL-v2 now binds the complete closure authority chain:

- selected J3 checkpoint/system;
- exact trainer implementation;
- DEV evaluation and selector receipts;
- DEV evaluator implementation;
- training authorization and training-authorizer implementation;
- static proof receipt and canonical proof contract;
- FINAL opening authorization and frozen opening authorizer;
- frozen FINAL package/audit/contracts/gate registry/evaluator.

FINAL still cannot train, repair, rerun automatically, or choose checkpoints.

## Exact-head evidence

At `f78cb119b368a372c16e892646f7b7f247138eab`:

- GitHub Actions run `35954996471` completed successfully;
- 259 tests passed;
- 219 STATIC_REQUIRED obligations passed;
- 243 total proof obligations remain registered, with only empirical/runtime/training/FINAL blockers unresolved;
- full proof-matrix static audit passed;
- historical authority firewall remained intact;
- source plan still reports optimizer=false, gradient=false, gpu_training=false, n0_complete=false.

This is the strongest source/static boundary so far. Unlike earlier greens, the current path includes the actual successor trainer, training authorizer, same-stage resume, first-pass DEV selector, stage-transition lineage, P40/P42/P43 runtime authority, sealed FINAL opening and closure path.

## Necessity filter / stop condition for further static work

The owner explicitly challenged whether the deep audit was becoming over-engineered. That challenge is now part of the handoff.

Further source/static changes are justified only if they prevent one of the following:

1. a plausible false N0 PASS;
2. an accidental capability ceiling;
3. training/evaluating a different topology or artifact lineage than the one qualified;
4. DEV/FINAL goalpost movement or leakage;
5. a direct contradiction with the full-envelope successor objective.

Do **not** continue adding hashes, receipt fields, wrappers, or duplicate provenance checks merely because another lineage field could exist. At `f78cb...`, the static audit should be treated as exhausted unless a new concrete defect satisfies one of the five criteria above.

The most recent necessary addition is the canonical P39PN runtime materialization path:

- `scripts/eipm/n0/materialize_n0_v02_full_public_mixture_v1.sh`
- builds/audits the exact optimizer-facing public mixture;
- keeps FINAL frozen and unopened;
- carries exact P40/P40A/P40B/P40C/P42 evidence forward;
- materializes behavioral/runtime-view/natural lanes and the frozen FINAL-v2 package;
- emits no optimizer, gradient, training, or N0-complete authority.

This is operational closure of an already-declared runtime blocker, not a new architecture layer.

## Immediate next execution order

Do not mutate source merely to open training. If another real static defect is found, repair it first and invalidate/re-run the exact-head evidence.

Otherwise the external sequence is:

1. generate the exact static proof receipt on this head;
2. run the canonical Magnolia CPU qualification chain for P40/P40A/P40B/P40C/P42;
3. materialize/audit the exact public mixture P39PN against the unopened FINAL freeze;
4. run P43 same-topology GPU no-gradient qualification and record the safe route;
5. run the runtime training authorizer — still no optimizer inside the authorizer;
6. begin J1 only with that authorization;
7. evaluate every checkpoint on the frozen DEV cadence and select the first full-stage pass;
8. transition J1 -> J2 -> J3 only through selected predecessor receipts;
9. after J3 DEV selection, create the one-way FINAL opening authorization;
10. evaluate sealed FINAL-v2 exactly once under its precommitted rules.

If FINAL fails, preserve the failure and reopen the implicated architecture/data assumption. Do not tune on FINAL.

## Anti-hotfix / no-ceiling reminders

Still active:

- no threshold lowering;
- no automatic retry after a valid model failure;
- no FINAL-based checkpoint selection;
- no fixed parameter/width/depth/relation/factor/view/context ceiling;
- no treating runtime operating points as capability definitions;
- no reduced pilot model in place of the registered full architecture;
- no private identity gradient in N0;
- no historical PASS promoted back into successor authority;
- no source edit after exact runtime receipts merely to enable training.

## Fable transfer boundary

The transferable lesson for Fable Builder is broader than “how to build a personality model.” A shipped builder must be able to construct, qualify and link the user-specific models and runtime authority chain from a provided corpus without relying on hidden human patching.

Transfer these N0 lessons:

- source/runtime/evaluator authority must be explicit and hash-bound;
- exact empirical gates must be precommitted before results;
- one green local module is not whole-system proof;
- data lanes need ownership boundaries and leakage audits;
- operating points must not become hidden capability ceilings;
- runtime authorization should open capabilities without mutating the exact qualified source;
- checkpoint/DEV/FINAL provenance must be complete enough to prevent silent goalpost movement.

These are builder-system lessons, not N0 identity semantics.
