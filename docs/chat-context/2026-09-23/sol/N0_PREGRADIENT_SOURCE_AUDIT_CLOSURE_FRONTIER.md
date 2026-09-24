# N0 Pre-Gradient Source Audit Closure Frontier

**Date:** 2026-09-23  
**Continuity role:** authoritative handoff for the current N0 successor frontier  
**N0 source branch:** `alice-eipm-v1-n0-full-envelope-foundation-build-v1`  
**Exact N0 head:** `ab59c02d9a625b2b376d14ad0d9efd0250b305f3`  
**Exact-head CI:** run `35958513992` — **green**  
**Observed exact-head static result:** **260 tests passed**, **220 static proof obligations**, **244 total proof obligations**, proof-matrix static audit PASS  
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

At `ab59c02d9a625b2b376d14ad0d9efd0250b305f3`:

- GitHub Actions run `35958513992` completed successfully;
- 260 tests passed;
- 220 STATIC_REQUIRED obligations passed;
- 244 total proof obligations remain registered, with only empirical/runtime/training/FINAL blockers unresolved;
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

Do **not** continue adding hashes, receipt fields, wrappers, or duplicate provenance checks merely because another lineage field could exist. At `ab59c...`, the static audit should be treated as exhausted unless a new concrete defect satisfies one of the five criteria above.

The most recent necessary addition is the canonical P39PN runtime materialization path:

- `scripts/eipm/n0/materialize_n0_v02_full_public_mixture_v1.sh`
- builds/audits the exact optimizer-facing public mixture;
- keeps FINAL frozen and unopened;
- carries exact P40/P40A/P40B/P40C/P42 evidence forward;
- materializes behavioral/runtime-view/natural lanes and the frozen FINAL-v2 package;
- emits no optimizer, gradient, training, or N0-complete authority.

This is operational closure of an already-declared runtime blocker, not a new architecture layer.

### Final pre-Magnolia check: incomplete FINAL TRAIN/DEV baseline

A final necessity-filtered audit before external compute found one substantive defect at the prior `f78cb...` frontier.

The sealed FINAL-v2 independence audit checked the ordinary semantic-operator TRAIN/DEV lane, but it did **not** include the dedicated semantic-operator long-context TRAIN/DEV lane. It also computed runtime-axis extrapolation against only the base behavioral curriculum rather than every relevant full-fabric TRAIN/DEV supplement.

That could permit a false closure claim: a relation description, entity, factor combination, or runtime operating point seen in the semantic-long/runtime-view/long-context training supplements could have been treated as unseen FINAL evidence.

The fix is now complete:

- FINAL semantic independence merges ordinary semantic-operator and semantic-long TRAIN/DEV rows;
- held-out factor-combination checks include semantic-long rows;
- FINAL runtime-axis extrapolation uses behavioral + runtime-view + long-context TRAIN/DEV as the baseline;
- context-length extrapolation is evaluated against the complete full-fabric baseline using the complete FINAL package;
- the FINAL audit records hashes for every relevant TRAIN/DEV reference;
- the canonical P39PN materializer and authoritative CI workflow pass the semantic-long TRAIN/DEV artifact into the FINAL auditor;
- proof obligation `N0-P39RGA` makes this part of exact-head static authority.

Exact-head run `35958513992` passed the full strengthened package build/audit/freeze path with **260 tests**, **220 STATIC_REQUIRED obligations**, **244 total obligations**, and `PASS_N0_FULL_PROOF_MATRIX_STATIC_AUDIT`.

This defect satisfies the necessity filter because it could have produced a false claim of FINAL independence/generalization. It was not provenance-only polish. No additional source change is justified merely to keep auditing.

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


## 2026-09-24 empirical handoff continuation

**Current N0 source head:** `0441f99d7d3f7df87b999fd7fc6074a7598be5d4`  
**Current source branch:** `alice-eipm-v1-n0-full-envelope-foundation-build-v1`  
**Exact-head CI:** run `36043889153` passed at this source revision with 262 tests, shell-syntax coverage, and the N0 proof-matrix contract.  
**Current phase:** empirical pre-gradient qualification. P39PN is complete on the current source; P43 is next. Optimizer, gradient, model training, private-identity gradient, FINAL opening, and `n0_complete` remain false.

### Runtime events after the pre-Magnolia static closure

The first CPU qualification attempt, Magnolia job `576060`, exposed a real pre-gradient fixture defect rather than a model failure. In the multi-step `relation_schema` long-context row, only the first supervised relation candidate had been moved beyond the native window. Another supervised candidate remained near token 5. The repair was class-wide rather than row-specific: all supervised relation candidates are longified and span-revalidated; query/factor/step-factor evidence is checked; TRAIN+DEV regression coverage enforces the rule; and the real-token P40C auditor validates the complete supervised-candidate set. This became green at `c06aa612b1cc1a4a1f79b0c9eb62de78342ea533`.

The CPU qualification path was then performance-corrected without reducing topology or capability. At `8899239ccad264139aee9200c0b7d03ee5ae63dc`, invalid padded segment slots no longer waste semantic-backbone computation; valid segments are processed and zeros are scattered back into invalid positions. Regression coverage keeps the gradient path live. Magnolia job `576071` then passed the exact P40/P40A/P40B/P40C/P41/P42 chain with `PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1`, `combined_parameters=243693339`, and peak CPU RSS about 2533.27 MiB.

The first full-mixture materialization on that source also passed. The first subsequent P43 attempt, job `576077`, did **not** reach model/memory qualification. `torchrun --standalone` advertised `gpu001.cluster` as its TCP rendezvous host, which was not resolvable inside the Magnolia udocker network. Both ranks eventually failed with `DistNetworkError`. No P43 `result.json` was produced, so this is infrastructure/orchestration evidence, not model or memory evidence.

The canonical P43 path was repaired at source rather than bypassed in a shell hotfix. The current sbatch now uses the historically proven Magnolia whole-stage launcher:

```text
accelerate launch
--multi_gpu
--num_processes 2
--num_machines 1
--mixed_precision fp16
--dynamo_backend no
--main_process_port <per-job port>
```

A regression rejects reintroduction of `torchrun --standalone`. Because that changed the qualified source, every exact-source runtime receipt downstream of source qualification was invalidated and regenerated instead of being reused across revisions.

### Current exact-source P39PN evidence

On the current source `0441f99d7d3f7df87b999fd7fc6074a7598be5d4`, Magnolia job `576079` completed successfully and published the canonical full public mixture.

Observed current-head facts:

- `PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1`;
- source revision exactly `0441f99d7d3f7df87b999fd7fc6074a7598be5d4`;
- 8 required optimizer-facing public training lanes;
- 10 required macro families;
- FewRel natural-relation lane: 39,200 TRAIN and 5,600 DEV rows;
- governed judgment teacher bank: 1,020 rows / 51 competencies;
- `errors=[]`;
- FINAL-v2 package frozen before gradient;
- `final_results_observed=false`;
- `optimizer=false`;
- `gradient=false`;
- `gpu_training_authorized=false`;
- `training=false`;
- `final_opening=false`;
- `n0_complete=false`.

The current public-mixture root is:

```text
/homes/01/mxrayan/rayan-compute/rayan-n0/n0-v02/full-public-mixture-v1
```

The materialized manifest, audit, FINAL freeze, static proof, and CPU runtime receipt paths are all published under that root. The current P39PN pass therefore closes the exact-source public-mixture gate and leaves P43 as the immediate pre-gradient blocker.

### Magnolia operating model recovered from historical evidence

Treat these as operational invariants for the remaining N0 run:

1. Magnolia's host Git is legacy. Do not use `git -C`, `git switch`, or unproven modern Git syntax. Explicitly `cd` before Git operations.
2. The login shell is the control plane. Scientific Python execution belongs inside the qualified `rayan-n0-base` udocker/P2 environment.
3. Run a whole scientific stage in one udocker session. Do not bounce individual Python calls between host and container.
4. Host and container `$HOME` are not interchangeable. Pass Magnolia-visible absolute paths explicitly.
5. Repository cleanliness includes untracked files. Slurm stdout/stderr must remain outside the Git checkout.
6. Runtime evidence roots are immutable. Preserve failed/timed-out evidence by archiving it; never delete evidence simply to permit a retry.
7. Keep the numeric Slurm job ID returned at submission. Use `squeue` only while active and `sacct` after completion.
8. Do not use literal angle-bracket placeholders in shell assignments.
9. CPU stages use the CPU/node partition. P100 allocations are reserved for CUDA-required stages.
10. Runtime/infrastructure failure is not model evidence. Classify the failure first.
11. A second failure of the same packaging/infrastructure class triggers a root-cause audit of that class, not a hotfix ladder.
12. Do not mutate exact-qualified source merely to make orchestration convenient. Source changes require a genuine source defect and invalidate exact-source receipts.

### Immediate execution boundary

The next operation is **P43 only**: the exact 2×P100, no-gradient, DDP-wrapped J3-topology memory qualification using the repaired canonical Accelerate launcher.

P43 must still prove all of the following before training authority can open:

- exact source revision and clean checkout;
- current-head P39PN manifest/audit/FINAL-freeze lineage;
- world size 2 and DDP replica wrapping;
- all required semantic cases;
- all required full-fabric memory cases;
- all 18 semantic × full-fabric stress pairs;
- finite joint loss for every pair;
- no backward;
- no optimizer object;
- no gradients;
- no weight update;
- no model training;
- measured/projected training-memory route passes on both ranks;
- FINAL remains unopened.

If P43 passes, the next step is the non-mutating runtime training authorizer, then J1. Do not jump directly from P39PN to J1.

### N0 role remains unchanged

Recent runtime work has not changed N0's purpose. N0 is still the **full-production, identity-neutral semantic/evidential/relational/candidate-comparison foundation** for EIPM. It is not the user model, source-person identity store, assistant-self model, relationship store, memory authority, prose generator, planner, or continual-development engine. The 640-wide registered system is a qualified operating point, not a permanent parameter/width/depth/context/relation/factor/view/slot/reasoning ceiling. No reduced pilot model may substitute for this registered successor.

### Main/frontier architecture updates that affect later integration, not current N0 source

Main has advanced beyond the old `0abaed858...` boundary with the canonical personal-development architecture, memory-use calibration, adaptive retrieval/consolidation qualification, and failure-localized multi-substrate learning policy.

The relevant architectural consequences are:

- Fable/A.L.I.C.E. must keep **user/host**, **assistant-self**, and relationship state distinct; A.L.I.C.E. additionally preserves a separate source-person axis.
- Personal development must eventually be causal: authorized experience → subject-bound state → native judgment → action/outcome → governed revision → changed future judgment when relevant.
- Personal state must enter native EIPM judgment before downstream response generation; downstream LLM behavior is not sufficient proof.
- N0 remains host-neutral and should merely be able to consume semantically described personal-state views later.
- Memory influence requires calibrated `IGNORE` / `BOUND` / `CONTROL` behavior under existing authority rules; retrieval rank or repeated use cannot promote truth/identity authority.
- Failure localization should compare memory/retrieval/context-harness/model/tool/environment/evaluator substrates before deciding what to mutate.
- Usage-aware retrieval and lightweight memory-control challengers are derived/rebuildable optimization surfaces, not Claim or identity authority.
- Parametric-memory ideas are future challengers, not replacements for governed Claim/Experience memory.

These main/frontier changes **do not justify editing the active N0 source or invalidating current P39PN/P43 lineage**. They belong in later personal-learning/memory integration and Fable Builder formation logic.

### MC10D / no-hotfix lesson still governing N0

The MC10D failure mode to avoid is not merely one bad script. It is the broader pattern of repeated transport/infrastructure/validator patching until forward progress becomes a trial-error loop and narrow passes are mistaken for scientific or model evidence.

The active rule remains:

- preserve failure evidence;
- classify infrastructure vs fixture vs runtime implementation vs model capability;
- localize the failure substrate;
- repair the **failure class** only when justified;
- rerun the minimum invalidated authority chain;
- never lower thresholds, parameter-fish, silently change gates, promote a narrow PASS, or automatic-retry a valid model failure.

Job `576060` and job `576077` are now explicit negative examples of why this distinction matters.

### Graphify boundary

Graphify remains an external Sol-side context/navigation substrate, not A.L.I.C.E. runtime or truth authority. Use it to locate branch + SHA + path and then verify consequential claims against the original repository artifact. Do not merge the research/Graphify branch wholesale into N0 merely to obtain context.


## 2026-09-24 P43 job 576080 — registered replay adapter defect, class repair pending requalification

P43 job `576080` reached the real two-rank DDP joint-step runtime on `gpu001`, so the prior Magnolia rendezvous repair was effective. Both ranks then failed deterministically in the governed-judgment replay lane before any valid memory-projection receipt was produced:

`TypeError: AliceN0V02Model.forward() got an unexpected keyword argument 'group_sizes'`.

Failure classification:
- not a Magnolia networking/rendezvous failure;
- not a GPU-memory-capability failure;
- not a learned-model capability failure;
- not a gradient/optimizer/training failure;
- runtime implementation/interface defect at the registered-system replay adapter boundary.

Root cause:
`TeacherMultitaskCollator` correctly returns native semantic-model inputs plus objective-only metadata (`group_sizes`, `preferred_masks`, `principle_tags`, `ids`). `N0FullEnvelopeTrainableSystemV1.forward` incorrectly forwarded the entire teacher batch with `**dict(batch)` into `AliceN0V02Model.forward`, whose native teacher signature intentionally accepts only candidate/rationale tensors and their index map. The older standalone teacher objective path already kept these layers separate.

Why static/CPU qualification missed it:
the existing joint-step fixture used a permissive fake system, while P42 exercised the real registered system only through the full-envelope path. Thus the exact collator -> joint-step -> registered-system -> native semantic-model replay adapter contract was not executed before P43.

Class repair on active N0 branch:
- `61915721e95ea83d70c5027128b4eefd6ed76a7f`: explicit MLM/teacher native-field forwarding; objective metadata stays outside the semantic model.
- `18a36b59951b1beacda67c99e3f46e6777f00fb6`: strict registered-system replay-dispatch regression reproducing the former metadata leak.
- `91f833f8a7d0caa54797663765a5c7b9826bc7f1`: P42 CPU qualifier now executes real MLM and teacher replay adapters with objective metadata present.
- `d4c9a6e3822be104719c7067b2f301bc71d85276`: CPU handoff requires a semantic-replay-interface receipt.
- `af79bc8b79ba9401248e0e2bfd3e9e1e6588cd65`: static regression requires the cheap CPU pre-GPU replay coverage.

Current active N0 head is `af79bc8b79ba9401248e0e2bfd3e9e1e6588cd65`. It is five commits ahead of the failed `0441f99d...` runtime source and has no divergence from the active branch.

Authority consequence:
all source-bound `0441f99d...` P42/P39PN/P43 evidence is now historical only. Preserve the failed P43 root from job 576080 and the old CPU/P39PN roots; do not overwrite or delete them. The minimal legal requalification chain is exact-head static proof -> P42 CPU qualification (including the new real replay-adapter evidence) -> P39PN rematerialization/freeze -> P43. No gradient, optimizer, FINAL opening, or private-identity training is authorized.

Calibration lesson:
a final pre-GPU check must follow every real optimizer-facing lane through the production adapter boundary, not merely verify launcher, Slurm, DDP, artifact, and topology contracts. Cheap CPU qualification should execute interfaces whose failure does not intrinsically require GPU hardware. This is a direct application of the MC10D anti-hotfix rule: repair the interface failure class and move its detection earlier rather than retrying P43 with a one-off workaround.


## 2026-09-24 P42 requalification job 576083 — replay interface repair empirically closed on CPU

Magnolia job `576083` completed `0:0` on `node016` in `00:05:45` against exact source `af79bc8b79ba9401248e0e2bfd3e9e1e6588cd65`. stderr was empty.

The strengthened pre-GPU gate executed the real registered semantic replay adapters and emitted:
- `semantic_replay_interface_passed`
- `mlm_dispatch_pass=true`
- `teacher_dispatch_pass=true`
- `objective_metadata_outside_native_model=true`
- teacher objective metadata keys exactly `group_sizes`, `preferred_masks`, `principle_tags`, `ids`

Thus the job-576080 failure class is now reproduced-and-closed at the cheaper CPU qualification boundary. This is stronger evidence than the earlier static repair alone.

The full P42 runtime also passed at the repaired exact head:
`PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1`, `combined_parameters=243693339`, peak RSS `2494.34765625 MB`, 8 relations, 9 factor banks, 10 edges, 8 views, 4 reasoning steps, all required virtualized surfaces true. Static proof also passed with zero pytest failures/errors/skips and 244 obligations.

No optimizer, gradient, GPU training, private identity, FINAL opening, or N0 closure occurred.

Authority state:
- old `0441f99d...` CPU/P39PN/P43 evidence remains historical/preserved;
- `af79bc8...` P42 CPU authority is now valid;
- next minimal stage is exact-head P39PN rematerialization/freeze on `af79bc8...`;
- only after P39PN passes may P43 be retried.
