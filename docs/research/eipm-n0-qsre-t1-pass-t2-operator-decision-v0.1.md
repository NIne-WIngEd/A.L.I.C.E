# QSRE T1 PASS → T2 Learned-Operator Decision v0.1

**Date:** 2026-09-20  
**Status:** T1 causal question passed on DEV; T2 static design/CPU mechanics justified; T2 gradient remains closed  
**T1 Magnolia job:** `575914`  
**T1 source revision:** `f4a9edb1915e39d893224f4503746bfd61fc6424`

## 1. Result

The governed one-shot T1 experiment completed normally and stopped at the first eligible trained checkpoint.

- result status: `PASS_QSRE_T1_EXECUTOR_DEV_CONTRACT`
- selected checkpoint: step `50`
- result SHA-256: `ba6b82c10fcb13f403569ad24dbd77cf0b3657dfc3eaf9cb9d10b6690b2d32be`
- prepared cache: `03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564`
- curriculum: `155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063`
- `causal_pair_completion = 1.0`
- `control_accuracy = 1.0`
- `family_min_success = 1.0`
- every one of the nine T1 workload families = `1.0`
- `row_success_accuracy = 1.0`
- `single_target_top1_accuracy = 1.0`
- `mean_target_support_mass = 0.9997953772544861`
- `outside_support_mass_max = 0.0`
- `outside_support_invariance_max_delta = 0.0`
- `plural_l1 = 0.015601305291056633`

TEST remained closed. The frozen challenge remained closed. The run did not learn operator extraction or support selection and did not use private identity data.

## 2. Scientific interpretation

T1 answers one causal question cleanly:

> When relation/operator semantics and structural support are correct, the QSRE relational executor and structural readout can learn the current public causal execution workload over frozen production-real field representations.

This is important because the preceding architecture failures repeatedly mixed query semantics, binding, execution, and global parent competition. T1 isolates execution and now removes that boundary as the current primary uncertainty.

This does **not** establish learned natural-language operator extraction, adaptive support discovery, end-to-end QSRE, TEST/challenge performance, private identity fidelity, production readiness, or N0 completion.

No T1 rerun is justified. The first eligible checkpoint already satisfied every predeclared DEV gate. More steps, LR changes, a second seed, or a larger T1 model would add compute without answering a new causal question.

## 3. Why T2 is now justified

The ratified QSRE sequence defines T2 as:

> learned operator + oracle support + proven frozen executor.

That is now the next unresolved boundary. T2 asks whether frozen natural-language semantic states can recover the relation sequence, requested argument role, operation/control semantics, and continuous operator state required by the executor that T1 already proved.

T2 must not learn structural support at the same time. Otherwise a failure would again become ambiguous between language/operator semantics and evidence binding.

## 4. T2 architecture boundary

The first T2 study must:

- freeze the selected T1 executor/readout checkpoint;
- keep structural support oracle;
- keep focus-field binding oracle initially so T2 does not silently become a support/entity-binding study;
- freeze the semantic backbone and all existing parent modules;
- learn only the query-operator encoder when gradient is later authorized;
- consume the semantic hidden-state stack rather than only one final pooled vector;
- make relation and argument role first-class and separately probeable;
- preserve ordered relation sequences;
- expose token/layer diagnostics for relation, role, operation, and applicability;
- keep a continuous residual operator state and uncertainty/plurality alongside current anchor predictions;
- never use parent global field logits or return to additive scalar-residual execution.

The earlier relation-layer audit can provide soft initialization priors. It cannot become a hard layer mask or a permanent ontology.

## 5. Full-EIPM requirement

T2 is not being designed around a six-relation benchmark as the product target.

The active QSRE workload envelope remains authoritative: the full N0/EIPM foundation must support language pragmatics, provenance, relational/causal reasoning, temporal change, relationship-sensitive interpretation, values/tradeoffs, candidate comparison, emotional interpretation, uncertainty/plurality, heterogeneous context, long-context sparse relevance, and fast/deliberate runtime paths.

Current relation count, role count, path length, support cardinality, context shape, model width, parameter count, and compute route are operating points. None is a permanent personality-model ceiling.

## 6. Anti-MC10 / anti-hotfix rule

This T1 PASS does not start a validation ladder.

The only missing evidence needed before T2 runtime preparation is durable binding of the exact selected step-50 checkpoint to the immutable T1 result. That binding changes a real decision: which executor T2 is allowed to consume.

After that:

- no T1 rerun;
- no T1 hotfix;
- no TEST opening;
- no support learning during T2;
- no automatic T2 GPU authorization.

Engineering for T2 may be prepared in parallel, but scientific causality remains sequential.

If T2 later fails, localize relation/role/operation/control semantics and downstream causal-pair behavior first. Do not respond with LR tuning, extra steps, support unfreezing, or executor redesign without a new causal diagnosis.

## 7. Immediate next action

Run the CPU/no-gradient T1 post-run capture once. It verifies the immutable result SHA, reads the checkpoint SHA already recorded inside `result.json`, independently hashes the selected checkpoint file, and emits one receipt.

Only that exact checkpoint may become the frozen executor parent for T2.
