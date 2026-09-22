# N0 Full-Envelope Deep Static Audit + Personal-Development Architecture Handoff

**Date:** 2026-09-22  
**Status:** active continuity authority for the current Sol handoff  
**N0 branch:** `alice-eipm-v1-n0-full-envelope-foundation-build-v1`  
**N0 head at handoff:** `6822626db833ded0575871f2126e5770d0072293`  
Current build head: `6822626db833ded0575871f2126e5770d0072293`  
**Magnolia authorization:** **NO — deep source-level audit remains open**  
**N0 complete:** false

## Authoritative build state

```text
source_branch=alice-eipm-v1-n0-full-envelope-foundation-build-v1
current_build_head=6822626db833ded0575871f2126e5770d0072293
deep_source_audit_complete=false
exact_head_static_suite=PASS_118
proof_obligations_total=103
proof_obligations_static=89
magnolia_cpu_runtime_authorized=false
gpu_memory_dry_run_authorized=false
optimizer_authorized=false
gradient_training_authorized=false
final_validation_open_authorized=false
n0_complete=false
```

The old stable-build latent-stage pointer is historical routing context, not the current implementation frontier. For execution decisions, this handoff plus the exact unmerged full-envelope branch source takes precedence over stale stable-branch stage-state language.

## Why this handoff exists

The prior continuity branch stopped recording the most important part of the 2026-09-21 N0 rebuild. It preserved the 575986/575990 decision boundary but not the subsequent full-envelope reconstruction and deep falsification work. That made new chats incorrectly interpret a green static receipt as the next authorization boundary.

This document supersedes that inference.

A green exact-head suite is necessary. It is not sufficient. The active method is still:

> inspect original source -> try to make the architecture fail cheaply -> repair only a localized real defect -> rerun exact-head static evidence -> continue source audit -> external compute only after the static/source boundary is genuinely exhausted.

This is the same correction learned from the earlier N0 lineage: narrow PASS receipts, locally valid modules, and self-referential targets can coexist with a defective full system.

## Historical decision chain that must not be forgotten

The important current lineage is:

```text
575956 / 575957 / 575958 / 575962 / 575966
    earlier local/narrow qualification and repair lineage
            ↓
575986
    valid P2A capability failure
    frozen semantic authority did not generalize sufficiently
            ↓
575990
    zero-gradient semantic localization
    old frozen-authority direction closed
            ↓
schema-conditioned semantic/operator foundation
    step80 retained only as initialization/regression baseline
            ↓
full-envelope retrospective rebuild
    reopen weak or prematurely frozen N0 interfaces
            ↓
current branch
    alice-eipm-v1-n0-full-envelope-foundation-build-v1
```

Do not revive a superseded narrow PASS merely because a later experiment becomes inconvenient.

## Current exact-head static evidence

At `6822626db833ded0575871f2126e5770d0072293`, GitHub Actions run `35693476513` completed successfully.

Observed receipt:

- 118 tests passed;
- 103 proof obligations registered;
- 89 static obligations;
- synthetic operator curriculum audit passed;
- full-envelope behavioral-fabric static audit passed;
- natural FewRel audit passed;
- historical-authority firewall passed;
- static CPU runtime contract audit passed;
- no gradient/GPU/final authorization was granted by those receipts.

These results mean the registered static mechanics currently satisfy the precommitted tests. They do **not** mean the deep source audit is finished.

## What changed after the earlier green full-envelope head

The full-envelope rebuild continued to uncover real defects after prior green receipts. The current head includes, among other corrections:

- universal semantic-input virtualization across query, relation schema, factor schema, fields, candidates, and descriptor text;
- trainable cross-window semantic interaction rather than token-lossless stitching being misrepresented as complete semantics;
- one registered semantic-backbone + semantic-input + full-envelope successor topology for replay and full-envelope work;
- fully unavailable optional semantic banks kept inert;
- exact runtime qualification wired to the registered topology rather than detached module assembly;
- type-incompatible pre-Binder graph edges blocked before graph propagation;
- Binder null support required before graph source/target views can influence fusion;
- causal losses masked on active supervised comparisons;
- semantic-operator and full-envelope behavioral losses integrated into one governed objective family contract;
- explicit absence/sentinel support for counterfactual and intervention lanes;
- an executable behavioral curriculum and row-to-system batch compiler;
- 3-step and 4-step causal-chain coverage;
- SVD-based latent anti-collapse gradient instability removed;
- masked irrelevant-view invariance rewritten to avoid finite-forward/NaN-backward behavior;
- fail-closed non-finite gradient checks;
- loss-family EMA observation separated from differentiable microbatches;
- macro-family scaling made invariant to gradient-accumulation partitioning.

Earlier parts of the same rebuild also corrected weak survival/truncation semantics, symmetry handling, exact null support, inactive relational-path contamination, fixed/global hidden-layer mixtures, open semantic factor handling, padded-item behavior, fusion relevance routing, recoverability semantics, long-context isolation, and step-conditioned relation semantic state.

## Audit progress after the first 114-test green head

The deep audit continued after `6c4e3c02672b7a5415fb3407942d86cc9723b8d9` instead of treating that green receipt as Magnolia authorization. That immediately paid off.

### Endpoint-role causal falsification

Historical endpoint-role failures made the current source/target path a high-risk place to re-check. A new independent static intervention was added that holds evidence, direction, traversal, and learned readout preference fixed while changing only `ROLE_SOURCE` versus `ROLE_TARGET`.

The executor's structural endpoint readout and final relational probability switch to the corresponding endpoint. The exact-head proof was added as `N0-P24B`. The resulting head passed the complete static suite with 115 tests.

This closes the specific concern that source/target role semantics were only labels or support-localization proxies. It does **not** close the broader audit.

### Behavioral DEV leakage found and repaired

The behavioral curriculum's old TRAIN/DEV isolation looked stronger than it was.

The previous audit checked query-template, entity, and causal-group isolation, but DEV reused normalized TRAIN **field wording and answer-candidate wording** for the same scenario families. That could allow model-selection DEV to reward surface-template recognition instead of the intended semantic/causal transfer.

A model-free audit was added first. It failed exactly as expected at commit `f9adf24f5abbeec0002ce6695350ad62bb7d0fdd`, workflow run `35690014546`, with both:

- `TRAIN/DEV normalized field-surface overlap`;
- `TRAIN/DEV normalized candidate-surface overlap`.

The first semantic-preserving DEV rewrite removed field-surface overlap but intentionally preserved the failure evidence when two UNKNOWN/DEFER answer templates still overlapped. Commit `10336cdb09edec76dc1a415e64e10f69ada0ca0e`, run `35690063445`, failed only on the remaining candidate overlap.

Commit `7686f2d23befa3fc753b47c65cd31259ed19b3f7` repaired those remaining UNKNOWN/DEFER candidate surfaces. Its full contract run passed.

### Behavioral governance drift found and repaired

The strengthened auditor then exposed a separate contract defect:

- the builder generated `mixed_direction_composition` rows but the behavioral curriculum contract did not declare that scenario family;
- the contract did not explicitly require field-surface or candidate-surface TRAIN/DEV isolation.

The fail-closed auditor commit `9913f51eec53268b7624a2ff9db12fa307116455`, run `35690272668`, failed on exactly those three governance mismatches.

The contract was repaired at `74a7fd15b30d6af3d365be8f7aca5d7b6898252b`, and the proof requirement was strengthened at the current head:

`6822626db833ded0575871f2126e5770d0072293`

Exact-head workflow run `35690293060` is green:

- 115 tests passed;
- 100 proof obligations;
- 86 static obligations;
- behavioral field/candidate surface isolation passed;
- exact scenario-family contract coverage passed;
- all previous static gates remained green.

This sequence is important evidence for the current method: **a green suite was not enough; continuing the cheap source/data audit found real defects before Magnolia.**

### Modifier semantics bypass found and repaired

The next deep-audit target was the reliability/recency/temporal/provenance modifier path.

The runtime semantic schemas explicitly define OFF semantics such as:

- do not use reliability as an arbitration criterion;
- do not use recency as an arbitration criterion;
- do not require temporal compatibility;
- do not filter support by provenance class.

Source review found that the executor's multiplicative path gates respected those modifiers, but raw criterion values were still available through learned feature paths:

- Binder read raw edge reliability and recency directly in its support score;
- Executor fed raw reliability/recency/temporal/provenance scalars into edge-state construction even when the corresponding modifier was OFF;
- the pre-Binder evidence graph received raw explicit edge metadata;
- the evidence view received raw field reliability.

That created a real semantic bypass: the model could learn to arbitrate on a criterion that the semantic operator had said was inactive.

Independent modifier-gating tests were added first. The initial corrected test fixture exposed the direct Binder/Executor bypass, and the latest direct repair was green at `e1d9dc0b0109dce597d968527414eebcf8c78b0e`.

A stronger full-stack intervention was then added. It forced all modifiers OFF while preserving raw explicit metadata and captured what the pre-Binder graph and evidence view actually received. Commit `532e7eac79422ad417a6fb6a0c75f72b07135ac3`, workflow run `35693344066`, failed exactly on that requirement:

```text
test_full_stack_modifier_off_neutralizes_explicit_metadata_before_graph_and_evidence_view
1 failed, 117 passed
```

The causal correction is now at:

`6822626db833ded0575871f2126e5770d0072293`

The stack conditions explicit criterion metadata before pre-Binder learned paths:

- reliability/recency interpolate to neutral 0.5 when inactive;
- temporal/provenance constraint-match channels interpolate to neutral 1.0 when inactive;
- enabling the corresponding modifier restores the observed criterion signal;
- extra configured edge-metadata channels remain untouched, so this does not create a fixed metadata-cardinality ceiling.

Exact-head workflow run `35693476513` is green:

- 118 static tests passed;
- 103 proof obligations;
- 89 static obligations;
- all behavioral, FewRel, authority-firewall, CPU-contract, and training-block gates remained green.

This is another concrete example of why Magnolia remains blocked while cheap source falsification continues: the previous 115-test green head still contained a semantic control bypass.

## Current architecture intent

N0 is the public, identity-neutral semantic/judgment foundation for the EIPM.

It is responsible for capabilities such as:

- semantic language/subtext representation;
- uncertainty and evidence interpretation;
- dynamic runtime relation and factor semantics;
- structured context;
- relational execution;
- support/null-support reasoning;
- multi-view fusion;
- competitive latent allocation;
- candidate comparison and public behavioral judgment.

It is **not**:

- Elaina biography authority;
- Rayan host model;
- A.L.I.C.E. self/continuity model;
- relationship-model authority;
- Memory Formation Model;
- mission memory database;
- the whole assistant;
- a final prose generator.

There is no ratified hard parameter, width, depth, relation-count, factor-count, field-count, edge-count, view-count, slot-count, hop-count, reasoning-step, or product-context ceiling. Operating points are not capability limits.

The 136M semantic checkpoint is initialization/regression evidence, not permanent semantic authority and not a size ceiling.

## Deep source audit is still open

The prior Sol was explicitly continuing source-level falsification after green CI. The current chat must continue that work rather than jumping to Magnolia.

Areas still requiring source-level scrutiny include:

1. **Global versus step-conditioned factor semantics.** Relation semantic state, direction, and modifiers now have verified step-conditioned paths, and explicit modifier metadata bypasses are closed. Role/traversal/control ownership across multi-step programs still needs scrutiny for any whole-program shortcut that breaks valid compositions.

2. **Behavioral curriculum shortcut resistance.** TRAIN/DEV entity/query/field/candidate surface separation now exists, but scalar metadata patterns, scenario construction, answer construction, and deterministic label structure must still be audited for shortcuts that can satisfy DEV without learning the intended semantics.

3. **Causal behavioral tests versus shape tests.** A test that proves geometry, nonzero gradient, a parameter-report flag, or permutation wiring does not by itself prove the claimed behavior. Every critical claim should be traced to an independent intervention where possible.

4. **Additional runtime views.** Arbitrary extra views are supported, but future personal-state views must not turn precomputed vectors, availability, reliability, or descriptor geometry into a label leak or hidden identity axis.

5. **Fusion/latent/public-judgment causal chain.** Continue checking that relevance, decisive evidence, irrelevant evidence, recoverability, noncollapse, candidate judgment, and candidate masking are causally connected in the intended direction rather than merely jointly differentiable.

6. **Long-context semantic fidelity.** Virtualization and the trainable segment bridge remove isolated-window behavior, but source-level scrutiny must continue for boundary shifts, long candidate/schema/descriptor paths, and whether every long-text surface actually uses the same governed bridge.

7. **CPU qualifier fidelity.** Before external execution, verify from source that the CPU fixture is neither easier nor structurally different from the governed training topology and does not prove behavior by reading report booleans.

8. **Final-validation independence.** Keep final-v2 sealed. Training/DEV construction must not derive targets, templates, thresholds, or repair choices from final outcomes.

These are audit questions, not authorization for speculative fixes. A concern becomes a code change only after its causal defect is established.

## Anti-hotfix rule

The MC10D/N0 correction remains active:

- no repeated patch -> validator -> patch treadmill;
- no learning-rate/step/width/batch fishing after failures;
- no threshold lowering;
- no automatic rerun of a valid failed model experiment;
- no deleting or overwriting failure evidence;
- no treating infrastructure failure as model evidence;
- no promoting a convenient checkpoint because the intended one failed;
- no adding a mechanism merely to satisfy a static assertion.

When a valid failure or source defect is found:

1. localize it;
2. inspect the full upstream/downstream architecture;
3. choose one causal correction;
4. add the smallest independent proof that would have caught the defect;
5. rerun the complete affected suite;
6. continue the broader audit.

## Magnolia boundary

Do **not** submit `magnolia_cpu_n0_v02_full_envelope_runtime_v1.sbatch` yet merely because the current static suite is green.

The intended next external boundary remains an exact 640-wide CPU/no-gradient full-envelope construction/forward + parameter/memory receipt, but only **after the unfinished source audit closes without another statically discoverable defect**.

If the source audit finds a real defect, fix and requalify first.

After source-audit closure, the order is:

```text
exact-head static/source qualification
        ↓
full 640-wide CPU/no-gradient runtime + memory/parameter receipt
        ↓
same-topology GPU no-gradient memory dry run
        ↓
only then consider first authorized joint gradient stage
```

No optimizer, gradient training, FINAL opening, or N0 completion is authorized by this handoff.

## Personal-development architecture finding accepted on 2026-09-22

A separate Astra architecture review found a real broader-system gap. The finding is accepted, but it must not be confused with N0's role.

The missing destination loop is:

```text
experience / observation
    -> subject-bound user / source / relationship / assistant-self state
    -> native personal judgment
    -> action / response
    -> observed outcome
    -> governed state revision
    -> changed future judgment
```

Current memory/event/projection infrastructure does not by itself demonstrate this loop. The released conversation runtime's fixed constitutional prompt and replaceable downstream model also do not prove learned personal judgment.

Identity structure:

- A.L.I.C.E.: Rayan host + Mehejabin Elaina source-person foundation + A.L.I.C.E. current self;
- general Fable: user/host + Fable current self.

A.L.I.C.E. must not become an idealized Rayan. A Fable user is not required to declare an ideal/true self; such declarations are evidence, not identity authority.

Continuing user learning and assistant self-development are required destination capabilities.

### Repository corrections made

The accepted architecture correction is now canonical on `main` at:

`d77b990b76de37984c4fd1a523aeb628d5b84afa`

It reached main through protected PR #90 after the repository's policy, product-family, capability-barrier, phase-boundary, Phase 0, Phase 1, and dedicated personal-development checks passed.

Canonical corrections include:

- `docs/research/PERSONAL_DEVELOPMENT_AUDIT_2026-09-22.md`;
- product-neutral `assistant_self` projection subject while retaining `alice_self` compatibility;
- policy requiring user/host + assistant-self + relationship development;
- explicit experience -> state -> judgment -> outcome -> revision causal contract;
- explicit rule that prompts, storage schemas, or generic model behavior are insufficient proof;
- regression tests for subject separation and the architecture contract.

The Fable builder branch was also updated at `fable-builder-model@467fb30689500f30b0bf23d04c214362d63ff10d`:

- `docs/fable-builder/README.md` now treats post-activation personal development/reflection as required rather than optional;
- `docs/fable-builder/traces/FBM_TRACE_20260922_PERSONAL_DEVELOPMENT_ARCHITECTURE.jsonl` records the transferable lessons.

These architecture changes do **not** claim the learned personal-development loop is implemented.

The canonical main correction adds the product-neutral `assistant_self` projection subject, the causal personal-development policy, a canonical Fable personal-development architecture document, and regression tests. The learned loop itself remains future implementation work; documentation/storage tests are explicitly insufficient to claim it.

## Current next action

Continue the deep source-level N0 audit at exact head `6822626db833ded0575871f2126e5770d0072293`.

Do not ask the owner for a Magnolia run until that audit has either:

- found and repaired the remaining statically discoverable defects and reached a new exact-head green receipt; or
- produced a reasoned source-level conclusion that the remaining unknowns are genuinely empirical and cannot be resolved cheaply before runtime.

Graphify is navigation only. Original branch-qualified source remains authority.
