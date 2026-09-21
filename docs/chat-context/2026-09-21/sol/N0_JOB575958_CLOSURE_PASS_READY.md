# N0 job 575958 closure-pass — exact-head verified and ready for one Magnolia launch

**Date:** 2026-09-21  
**Status:** closure package implemented, exact-head static/CPU qualification passed, Graphify exact-source topology verified, Fable failure/success seeds recorded, one owner Magnolia launch may proceed  
**Closure branch:** `alice-eipm-v1-n0-closure-pass`  
**Qualified exact HEAD:** `fefc2e4ec54ddde113165b1a3103a1add9be34e8`  
**Closure qualification:** GitHub Actions `35561705280` — SUCCESS  
**Graphify exact-source verification:** workflow `35561824623`, attempt 2 — SUCCESS  
**Graphify source:** `alice-eipm-v1-n0-closure-pass@fefc2e4ec54ddde113165b1a3103a1add9be34e8`  
**Fable branch after catch-up:** `fable-builder-model@0994838c62c57912b1808243397c08e1b3f368a0`

## Why the prior final-v3 package is superseded for the next run

Magnolia job `575958` was a valid learned-model failure, not infrastructure noise.

The job completed the governed Production P2-v3 budget on the exact qualified source and returned:

`FAIL_QSRE_PRODUCTION_P2_OPERATOR_V3`

with no selected checkpoint and no P3 authorization.

The failure trajectory showed:
- TRAIN relation/factor objectives fitting strongly;
- DEV relation-program exactness remaining low;
- Production open-schema transfer degrading toward zero;
- traversal DEV accuracy remaining near a stable sub-gate plateau despite near-zero TRAIN traversal loss.

This invalidated the claim that the v3 package had solved the semantic induction boundary.

The response is not another local P2 patch. The closure package changes the missing capability boundary before another Production operator run.

## Causal architecture diagnosis retained from 575958

### 1. Implicit six-relation ontology

Production v3 removed an explicit learned relation-ID table but still allowed the shared semantic projection/normalization geometry to optimize directly around the six core Production relation meanings.

The Production open-schema meanings were correctly excluded from gradient, but the matcher itself could still specialize to the six observed meanings.

Dynamic candidate cardinality therefore did not prove dynamic semantic generalization.

### 2. Ordered-evidence mechanism was lost during branch recomposition

The preserved `alice-eipm-v1-qsre-production-p2-ordered-evidence-v3` head contains the independently motivated query-token coverage / remaining-evidence mechanism derived after job 575957 localized later-step relation identity/order failure.

The later final-v3 recomposition correctly kept continuous relation hypotheses but omitted the explicit remaining-evidence state.

Both properties are compatible and are now present together:
- continuous semantic relation hypotheses before structural binding;
- differentiable ordered query-evidence consumption across recurrent relation steps.

### 3. Factor surfaces were still opaque small class heads

Role, traversal, direction, control and modifiers still used fixed learned class heads in the failed package.

Job 575958 showed that these heads could fit TRAIN while remaining below DEV gates.

Closure pass replaces those opaque semantic authorities with runtime factor descriptions scored by the same reusable semantic matcher.

### 4. Previous static tests did not test the missing capability

The prior qualification proved mechanics such as:
- dynamic relation cardinality;
- candidate permutation equivariance;
- continuous relation distributions;
- structural-only exact sparsity.

It did not prove that a matcher trained on one collection of relation meanings could transfer to genuinely unseen relation meanings after optimization.

Closure pass adds that proof before Production P2.

## Closure architecture

### P2S — broad public schema-generalization stage

New matcher:

`src/alice_personality/n0/qsre_schema_matcher.py`

`QSRESchemaMatcher` has:
- zero relation-identity parameters;
- zero candidate-count-dependent parameters;
- zero reasoning-step-dependent parameters;
- shared query/schema semantic projection;
- candidate-conditioned query evidence;
- symmetric query-to-schema and schema-to-query late interaction;
- runtime-variable schema cardinality.

P2S trains only on public identity-neutral auxiliary relation meanings, the six core Production meanings, semantic factor meanings and paired paraphrases.

It must pass a separate unseen auxiliary relation gate before Production P2 is authorized.

The actual Production/final holdout meanings are explicitly forbidden from P2S TRAIN and its auxiliary holdout:

- `CONFLICTS_WITH`
- `EXEMPLIFIES`
- `PREREQUISITE_FOR`
- `PART_OF`
- `ENABLES`
- `PREVENTS`

The config explicitly keeps:
- Production open-schema descriptions out of gradient;
- final-only descriptions out of gradient;
- auxiliary holdout descriptions out of gradient;
- private identity data out of gradient.

### Production P2 — frozen semantic matcher plus ordered program learning

After P2S passes, the schema matcher is loaded into:

`src/alice_personality/n0/qsre_production_operator_v3.py`

and frozen.

Production P2 may train recurrent ordering/event/applicability/continuous-execution state, but it cannot rotate the semantic metric around the six core relation meanings.

The operator now contains explicit differentiable query-token coverage:

`remaining = 1 - query_coverage`

Candidate-conditioned relation evidence updates coverage after each recurrent step.

Coverage is:
- token-position based;
- shared across semantic layers;
- not a left-to-right pointer;
- recoverable through a nonzero floor;
- relation-grounded rather than generic attention-only.

Relation hypotheses remain continuous softmax distributions through the operator.

No exact-zero relation sparsity is introduced before Binder v2.

### Semantic factor schemas

The closure operator does not contain fixed learned:

- `role_head`
- `traversal_head`
- `direction_head`
- `modifier_head`
- `control_head`

Role/traversal/direction/control/modifier meanings are supplied as semantic schema descriptions and scored by the same frozen matcher.

Stable factor indices remain executor/runtime interfaces only.

### Binder / executor boundary remains

`QSREProductionBinderV2` remains the structural evidence-selection boundary.

It still owns:
- exact sparse edge support;
- adaptive variable support;
- runtime type constraints;
- no fixed top-k;
- no operator-continuous-state support authority.

Thus the intended causal order is:

`frozen semantic matcher -> ordered continuous operator -> exact sparse structural binding -> proven executor -> structural readout -> N0 fabric`

## Proven components reused rather than rewritten

No 575958 evidence invalidated:
- the corrected P1 executor;
- the STOP-tail semantic fix;
- unchanged P1 step-50 checkpoint;
- P1 checkpoint SHA-256 `91f2c78dcc35967064189af4a7110cdee83e5652fb037d3d88f58500ac3aaab9`;
- Binder v2 structural sparse boundary;
- full 640 semantic width;
- original Production P2/P3/P4 eligibility gates;
- original Production LR/batch/step settings;
- original frozen native N0 validation.

No P1 retraining is authorized.

## Exact-source verification after qualification

The closure workflow `35561705280` passed on exact HEAD:

`fefc2e4ec54ddde113165b1a3103a1add9be34e8`

It proved:
- Python/shell/config compilation;
- closure matcher mechanics;
- job-575958 failure authority binding;
- real Production/final holdout isolation from P2S;
- reusable matcher contract with no relation-ID/candidate-count/hop parameter axes;
- frozen matcher in Production P2;
- restored ordered query-evidence coverage;
- absence of fixed factor class heads;
- unchanged Production compute settings and gates;
- continuous operator -> structurally sparse Binder v2 boundary;
- one-run/frozen-final lineage.

Graphify was then rebuilt from the exact closure HEAD using Graphify v0.9.63.

Graphify workflow `35561824623` first assembled the exact graph successfully but failed only at publication because `fable-builder-model` moved while the catalog freshness check was running. This was a context-catalog race, not an A.L.I.C.E. architecture or source failure.

Attempt 2 succeeded after the Fable head stabilized.

Published `LAST_GRAPH_BUILD.json` records:

- source branch: `alice-eipm-v1-n0-closure-pass`
- source commit: `fefc2e4ec54ddde113165b1a3103a1add9be34e8`
- extraction: `exact-source-worktree`
- authority: navigation-only.

The refreshed maximal unmerged experiment heads are:
1. `alice-eipm-v1-n0-closure-pass@fefc2e4...`
2. `alice-eipm-v1-qsre-production-p2-ordered-evidence-v3@342bdbb...`
3. `alice-eipm-v1-qsre-production-p2-failure-localization@c9b1d35...`

All three were inspected during the 575958 review.

Graphify navigation was then checked against original source at the exact closure HEAD. Direct source verification confirmed:

- matcher reports zero relation-identity/candidate-count/reasoning-step parameter axes;
- candidate-conditioned query/schema evidence exists;
- Production operator loads and freezes the pretrained matcher;
- ordered `query_coverage` is present and fed back as `query_remaining`;
- relation distributions remain continuous softmax;
- fixed factor class heads are absent;
- factor-schema cache is configured in Production P2;
- its SHA is forwarded through P3, P4 and final validation;
- runner passes the same factor-schema cache to P2/P3/P4/final;
- Production open-schema descriptions remain out of gradient;
- the original frozen final validation is reused;
- final closure still requires `PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE`, `n0_complete=true`, and `n1_authorized=true`.

No source-level bypass of the frozen matcher, reintroduction of relation/factor identity authority, loss of ordered coverage, holdout-description gradient leak, or factor-schema-lineage gap was found in the qualified exact-head path.

## Fable continuity

The Fable branch now includes:

`docs/fable-builder/traces/FBM_TRACE_20260921_N0_JOB575958_CLOSURE_PASS.jsonl`

It records:
- 575958 as a valid schema-generalization/package failure;
- the branch-recomposition loss of ordered evidence;
- the implicit-ontology lesson;
- fixed factor-head overfitting;
- the closure-pass architecture and anti-hotfix rule.

This is builder-process evidence only. It does not promote model success.

## Single authorized next action

Use:

`scripts/eipm/n0/magnolia_p100_n0_v02_qsre_n0_closure_pass_v1.sbatch`

New immutable output root:

`$ALICE_N0_WORKDIR/qsre-n0-closure-pass-v1`

Required sequence:

`P2S -> Production P2 -> P3 -> zero-gradient P4 -> original frozen native N0 self-validation`

Each stage fail-closes the next.

No automatic rerun.
No LR search.
No step search.
No width search.
No batch search.
No threshold change.
No TEST opening.
No private identity gradient.

If P2S fails, stop before Production P2.

If Production P2/P3/P4/final fails, preserve the evidence and stop for causal localization.

N0 closes only if the final native result is exactly:

`PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE`

with:

`n0_complete=true`

and:

`n1_authorized=true`.
