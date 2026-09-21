# N0 job 575962 — semantic matcher v2 causal repair

**Date:** 2026-09-21  
**Source run:** Magnolia job `575962`  
**Failed source revision:** `fefc2e4ec54ddde113165b1a3103a1add9be34e8`  
**Failed stage:** P2S closure schema matcher  
**Observed result:** `FAIL_QSRE_CLOSURE_SCHEMA_MATCHER` / exit 41  
**Production P2 authorization:** false  
**Repair branch:** `alice-eipm-v1-n0-closure-semantic-v2`

## Classification

Job 575962 is valid model evidence, not an infrastructure failure.

The P100 runtime, exact Git revision, P1 lineage, job-575958 failure lineage and original frozen final-validation lineage all passed before P2S. P2S then consumed its full precommitted 1000-step budget and fail-closed exactly as designed.

No Production P2, P3, P4 or final-validation result was produced from the failed matcher.

The P2S gate therefore did its job: it localized the next scientific boundary before a more expensive downstream run.

## What the P2S trajectory says

The best observed P2S checkpoint was step 850:

- auxiliary unseen-holdout relation top-1: `0.5416666666666666`
- auxiliary seen-relation top-1: `0.5625`
- Production core single-relation top-1: `0.7090909090909091`
- held-out factor macro accuracy: `0.6666666666666667`

None approached the precommitted eligibility boundary.

The trajectory also plateaued rather than moving toward the gates. Production core single-relation accuracy peaked near 0.72 earlier in training. Held-out factor macro accuracy peaked near 0.725. Additional steps did not resolve the capability deficit.

Most importantly, **seen and unseen auxiliary relation accuracy were similarly sub-gate**. That is different from job 575958. It weakens the hypothesis that P2S merely learned another closed seen-relation ontology. The failure is broader: the semantic matcher and its encoding/training alignment are not expressive enough for the required runtime schema matching problem.

This run does **not** authorize:
- more P2S steps;
- another learning-rate choice;
- a threshold reduction;
- a wider blind search;
- an identical rerun.

## Source inspection after the failure

The failed P2S implementation exposed three defects at one causal boundary.

### 1. The semantic matcher was still too shallow

The v1 matcher projected frozen contextual token states with one shared linear metric and scored candidates mainly through cosine token similarity, tokenwise max interaction and weighted pooling.

That removes relation-ID parameters, but it is not the same as learning compositional semantic compatibility.

Direction, argument-role reversals, negation, dependency language and closely related relation descriptions can require candidate-conditioned composition across multiple tokens. A single projected token metric can preserve lexical similarity while still confusing the actual relation.

The repair therefore keeps relation identity out of parameters but adds a shared **bidirectional query/schema pair refinement**.

### 2. P2S auxiliary schemas did not use the Production runtime encoding grammar

The failed P2S auxiliary schema input used:

`Schema key: <opaque key>. Meaning: <description>`

Actual Production runtime relation schemas use:

`Relation meaning: ... Source argument: ... Target argument: ... The relation is directional/symmetric.`

So P2S was asked to prove zero-shot semantic transfer using a different encoding schema from the one the frozen matcher would later consume in Production.

This is a representational alignment defect, not a reason to tune the optimizer.

Semantic-v2 calls the exact Production `relation_schema_text(...)` formatter for auxiliary relation schemas. Relation keys remain metadata and are not inserted into semantic text.

### 3. Factor training and factor holdout evaluation used different semantic tasks

The v1 meta-config already contained explicit factor `train_phrases` and held-out factor paraphrases.

However, the v1 trainer never used the factor `train_phrases` in gradient. Factor matching was trained from full Production queries and then evaluated on standalone semantic factor instructions.

That is a direct train/evaluation domain mismatch.

Semantic-v2 makes the factor train phrase bank gradient-bearing while retaining Production factor-query supervision. Their aggregate weight is averaged so the precommitted factor-loss magnitude is not simply doubled.

Held-out factor phrases remain evaluation-only.

## Semantic matcher v2 architecture

`QSRESchemaMatcher` remains one shared runtime-schema matcher.

It still has:
- zero relation-identity parameters;
- zero factor-identity parameters;
- zero candidate-count parameter axis;
- zero reasoning-step parameter axis;
- runtime-variable schema cardinality;
- candidate-conditioned query evidence;
- continuous semantic scores.

The v2 matcher adds:

### Frozen semantic anchor + learned residual

A fixed relation-independent projection preserves the frozen N0 semantic geometry.

A learned projection is residual to that anchor rather than being allowed to freely rotate the entire semantic space around P2S training classes.

This directly addresses the risk that a trainable matcher can destroy useful zero-shot geometry even without an explicit relation-ID table.

### Shared semantic-layer mixture

The 17 frozen semantic hidden states are combined through one globally shared layer mixture.

The mixture is not indexed by relation, factor, candidate count or hop.

### Bidirectional pair refinement

For every runtime candidate:

1. query tokens softly align to schema tokens;
2. schema tokens softly align back to query tokens;
3. shared token-pair features compose:
   - left semantic state;
   - right semantic state;
   - elementwise interaction;
   - absolute difference;
4. candidate-conditioned query evidence and schema evidence are refined;
5. pair summaries produce a bounded learned residual on top of the direct semantic anchor score.

This provides cross-candidate semantic interaction without turning runtime relations into learned class IDs.

The recurrent Production operator still receives the historical `[B,C,L,T]` query-evidence surface. Ordered query coverage remains intact.

The operator still keeps relation hypotheses continuous until Binder v2.

## Training-objective repair

The P2S v2 run keeps the same:
- seed;
- batch size;
- maximum steps;
- evaluation cadence;
- learning rate;
- weight decay;
- gradient clip;
- eligibility thresholds.

This is deliberately **not** a hyperparameter search.

The objective changes only where job 575962 localized the deficit:

- auxiliary relation schemas use Production runtime description grammar;
- opaque relation keys are removed from semantic inputs;
- auxiliary relation paired-view consistency joins the existing core paired-view consistency;
- factor train phrases become gradient-bearing;
- Production factor-query supervision remains;
- held-out factor phrases remain gradient-free;
- auxiliary unseen relation descriptions remain gradient-free;
- Production open-schema meanings remain forbidden;
- final-only relation meanings remain forbidden;
- private identity remains absent.

The v2 matcher initialization seed is also applied before matcher construction, repairing a reproducibility defect in the v1 trainer.

## Research precedent used as design evidence

The architecture is still A.L.I.C.E.-specific. External papers are precedent, not authority.

Relevant observations:

- **RE-Matching: A Fine-Grained Semantic Matching Method for Zero-Shot Relation Extraction** (ACL 2023, DOI `10.18653/v1/2023.acl-long.369`) motivates fine-grained semantic matching rather than only coarse class prototypes.
- **AlignRE: An Encoding and Semantic Alignment Approach for Zero-Shot Relation Extraction** (Findings ACL 2024, DOI `10.18653/v1/2024.findings-acl.174`) explicitly identifies an encoding-schema gap between instances and prototypes as a zero-shot failure source and aligns their encoding.
- **GLiREL: Generalist Model for Zero-Shot Relation Extraction** (NAACL 2025, DOI `10.18653/v1/2025.naacl-long.418`) uses shared processing and cross-interaction between relation-label and entity-pair representations while supporting labels that were unseen during training.
- Dense-retrieval-style ZSRE work also points toward hard semantic alignment and contrastive objectives rather than fixed relation classification heads.

We do not import their task assumptions wholesale. The useful transfer is the architecture principle: **runtime semantic schemas need shared candidate-conditioned interaction and aligned encoding, not a hidden closed class ontology**.

## Preserved downstream architecture

Job 575962 produced no evidence against:
- the corrected P1 executor;
- the exact P1 step-50 checkpoint;
- ordered query-evidence coverage;
- continuous relation hypotheses;
- semantic factor schemas as the Production interface;
- Binder v2 exact sparse structural support;
- the original Production P2/P3/P4 gates and compute plan;
- the original frozen native final validation.

Therefore semantic-v2 changes the P2S semantic matching boundary only.

If P2S v2 passes, its matcher is frozen before Production P2.

If P2S v2 fails, Production P2 remains closed and the evidence is preserved. No automatic repair chain is authorized.

## Evidence roots

Preserve the failed v1 root:

`$ALICE_N0_WORKDIR/qsre-n0-closure-pass-v1`

The semantic-v2 runner must verify that its P2S result is the exact valid 575962 failure before it can create new evidence.

New immutable evidence root:

`$ALICE_N0_WORKDIR/qsre-n0-closure-semantic-v2`

No overwrite or automatic rerun is allowed.

## Closure condition

N0 is still incomplete.

The only valid closure remains the unchanged original frozen native result:

`PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE`

with:

`n0_complete=true`

and:

`n1_authorized=true`.
