# N0 job 575966 — frozen semantic authority v3

**Date:** 2026-09-21  
**Source run:** Magnolia job `575966`  
**Failed source revision:** `1c94065cd30abb9764260e239b2227b27d098040`  
**Failed stage:** semantic-v2 P2S  
**Observed result:** `FAIL_QSRE_CLOSURE_SCHEMA_MATCHER` / exit 41  
**Production P2 authorization:** false  
**Repair branch:** `alice-eipm-v1-n0-frozen-semantic-authority-v3`

## Classification

Job 575966 is valid model evidence.

The run reached the P100 on the exact semantic-v2 source revision. P1 lineage, job-575958 Production failure lineage, job-575962 P2S-v1 lineage and the original frozen final-validation lineage all passed before P2S-v2 started.

P2S-v2 used the complete precommitted 1000-step budget and failed closed. Production P2 never ran.

There is no infrastructure failure to repair and no permission to extend the run.

## What changed relative to 575962

Semantic-v2 was useful even though it failed.

The v1 P2S run at job 575962 had weak performance everywhere. Semantic-v2 repaired several real alignment defects:
- auxiliary schemas used the Production runtime relation-description grammar;
- relation keys were removed from semantic text;
- factor train phrases became gradient-bearing;
- auxiliary paired-view consistency became gradient-bearing;
- the matcher gained an anchored bidirectional interaction path.

Those changes substantially improved fitting of the six Production core relations.

By step 400, Production core single-relation top-1 reached 1.0 and remained there.

Yet held-out semantic generalization did not follow.

The best selected-by-precommitted-score observation was step 150:
- auxiliary unseen relation top-1: `0.6666666666666666`;
- auxiliary seen relation top-1: `0.71875`;
- Production core single-relation top-1: `0.9477272727272728`;
- held-out factor macro accuracy: `0.7166666666666667`.

Later:
- Production core reached `1.0`;
- relation/factor training losses approached zero;
- auxiliary held-out relation accuracy remained roughly `0.56-0.67`;
- held-out factor macro accuracy remained roughly `0.61-0.73`.

This is a different failure signature from 575958 and 575962.

## Causal interpretation

The learned P2S matcher can fit its public training substrate but does not acquire the required broad runtime-schema semantics.

The failure is no longer explained by:
- an explicit relation-ID table;
- mismatched Production schema formatting;
- missing factor train phrases;
- insufficient Production-core fit.

The remaining boundary is the decision to train another semantic authority head on a small P2S relation/factor substrate.

The P2S-v2 matcher had about 1.23M trainable parameters, while the auxiliary relation substrate contained only 16 train relation meanings with a small paraphrase bank. The trajectory shows that additional optimization makes the training surface easier without improving the held-out semantic surface.

Therefore v3 does **not** train P2S again.

No learning-rate change, step extension, width change, batch change, threshold reduction or additional matcher head is authorized.

## Missed authority already present in N0

The ratified `alice-n0-semantic-v0.2` model is not just a raw hidden-state generator.

Its public semantic training already contains three useful surfaces:

1. **joint prompt/candidate preference scoring**  
   The teacher collator tokenizes a prompt and candidate as one sequence pair. The model then scores that joint representation with the trained preference scorer.

2. **semantic projection**  
   The semantic projection is explicitly trained under the public semantic/rationale contrastive objective.

3. **contextual token states**  
   Token states remain available for fine-grained local evidence and ordered coverage.

P2S-v1/v2 largely ignored the first two trained surfaces and tried to learn a new semantic metric from cached raw hidden states.

That is the architectural boundary changed by v3.

## Frozen semantic authority v3

P2A-v3 is a **zero-gradient qualification stage**.

It evaluates one precommitted frozen ensemble derived from the already-ratified semantic checkpoint.

There is:
- no optimizer;
- no new semantic matcher training;
- no checkpoint selection;
- no relation/factor class head;
- no relation-ID parameter;
- no factor-ID parameter.

The three authority components are:

### Joint prompt/candidate preference score

For each query and runtime schema candidate, the frozen semantic model receives the query as the prompt and the schema meaning as the paired candidate.

This reuses the same joint prompt/candidate pathway that the public semantic model was trained to rank.

### Shared semantic-projection cosine

Query text and schema text are encoded through the same fixed semantic-representation prompt and the frozen trained semantic projection.

Their normalized semantic projections are compared by cosine similarity.

### Parameter-free token evidence

The existing hidden-state tensors are reduced with one deterministic layer mixture.

Normalized query and schema tokens use symmetric max-style local matching.

This component also exposes candidate-conditioned query evidence and remaining-support mass for the recurrent ordered program.

It has zero parameters.

## Fixed component fusion

The raw joint, semantic-projection and token-evidence scores are normalized across **only the runtime candidates active in the current call**.

Each component receives candidate-axis z-normalization.

The final semantic authority is their equal-weight mean.

No component weight is learned.

Production TRAIN caches contain only the six core relation candidates. Therefore open-schema relation descriptions cannot affect TRAIN normalization or gradients.

Production DEV contains the full runtime schema.

Final validation uses its own full frozen runtime schema in a separate zero-gradient authority cache.

## Ordered operator boundary

The trainable Production operator no longer owns semantic relation identity.

For every recurrent step:

`frozen authority + log(remaining evidence support) + bounded recurrent order residual`

produces the relation logits.

The authority establishes semantic identity.

Remaining support lets consumed evidence suppress already-used semantic candidates.

The recurrent state handles ordering/program context but is bounded and cannot become a replacement relation ontology.

Relation hypotheses remain continuous.

Exact zero sparsity still begins only in Binder v2 structural support.

## Factor semantics

Role, traversal, direction, control and modifier identity also come from the frozen authority.

The operator has no fixed factor class heads.

The factor schema hidden states remain available only for parameter-free token evidence and continuous factor summaries.

Applicability, event termination and continuous execution state remain trainable because they are runtime operator behavior rather than semantic label identity.

## P2A causal gate

P2A-v3 uses the same semantic generalization eligibility values already committed for P2S:
- auxiliary seen relation top-1 >= `0.95`;
- auxiliary unseen relation top-1 >= `0.85`;
- Production core single-relation top-1 >= `0.95`;
- held-out factor macro accuracy >= `0.90`.

The data semantics and holdouts are unchanged.

If P2A fails, the evidence localizes the problem below the operator: the ratified semantic base itself does not currently supply enough zero-shot schema semantics for N0 closure.

Production P2 must remain closed in that case.

This prevents another learned-head repair loop.

## Research precedent

External work is used as precedent, not authority.

- **RE-Matching** (ACL 2023, DOI `10.18653/v1/2023.acl-long.369`) supports fine-grained semantic matching for zero-shot relation extraction.
- **AlignRE** (Findings ACL 2024, DOI `10.18653/v1/2024.findings-acl.174`) identifies encoding/semantic alignment between instances and relation prototypes as a key zero-shot boundary.
- **Fusion Makes Perfection / EMMA** (NAACL 2024 Short) reports that separately encoded query/description representations can be insufficient and combines coarse retrieval with fine-grained jointly encoded instance-description scoring.
- **GLiREL** (NAACL 2025, DOI `10.18653/v1/2025.naacl-long.418`) provides further precedent for shared interaction over runtime relation labels rather than a fixed relation classifier.

The A.L.I.C.E.-specific transfer is narrow: reuse the semantic model's already-trained joint ranking and semantic projection surfaces, combine them with parameter-free token evidence, and stop training another small closed semantic head.

## Preserved architecture

Job 575966 produced no evidence against:
- corrected P1;
- exact P1 step-50 checkpoint;
- ordered query-evidence coverage;
- continuous relation hypotheses;
- Binder v2;
- Production P2/P3/P4 compute settings and eligibility thresholds;
- the original frozen native final validation;
- private-identity separation.

Those remain unchanged.

## New evidence root

Preserve all previous roots.

The new immutable root is:

`$ALICE_N0_WORKDIR/qsre-n0-frozen-semantic-authority-v3`

The runner first proves the exact 575966 P2S-v2 failure, including its step-150 best metrics and full 1000-step completion.

Then:

`P2A frozen authority -> authority caches -> Production P2 -> P3 -> P4 -> original frozen native final validation`.

Every arrow is fail-closed.

## Closure condition

N0 is still incomplete.

The only N0-complete state remains:

`PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE`

with:

`n0_complete=true`

and:

`n1_authorized=true`.
