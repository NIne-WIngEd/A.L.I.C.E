# N0 QSRE post-reality-audit interpretation v0.57

**Date:** 2026-09-20  
**Trigger:** Magnolia CPU reality audit job 575950  
**Decision:** do **not** reopen the semantic backbone from the v0.56 decision alone.

## What 575950 established cleanly

Job 575950 completed successfully on a CPU node with empty stderr.

It established:

- actual Bash/uDocker runtime transfer is healthy;
- superseded recovery scripts fail closed;
- the T2 -> frozen T1 evaluator is total over 480 deliberately varied provisional operator combinations;
- the old synthetic T2 language distribution is easier for shallow lexical models than the realistic audit distribution;
- mean-pooled frozen semantic states have weak transfer for several operator labels.

These are valid results.

## Why the automatic v0.56 semantic-reopen conclusion is not ratified

The v0.56 semantic-sufficiency probe did **not** test the representation class consumed by T2.

The probe:

- mean-pooled every hidden layer;
- represented each query with one pooled vector per layer;
- used nearest-centroid classification.

The T2 operator encoder instead consumes:

- every token state;
- from all 17 hidden-state outputs;
- with learned latent queries;
- token-level cross-attention;
- learned layer embeddings and refinement.

The ratified N0 semantic model also explicitly states that mean pooling is a compatibility readout and is **not** the representational ceiling.

This mismatch matters because prior no-gradient evidence already showed the exact failure mode:

- job 575801: token late interaction role accuracy 0.6875 vs 0.5625 mean pool;
- job 575804: causes and supports directional signals appeared in intermediate token layers even when later/pool views lost them;
- that evidence was the reason relation-conditioned multi-layer token access was introduced.

Therefore a weak mean-pooled centroid probe cannot support the stronger statement:

> frozen semantic states do not retain enough transferable operator signal.

It supports only:

> simple pooled readouts do not expose enough transferable operator signal.

That is not the same claim.

## Additional evidence from the interrupted T2 run

Job 575933 never produced a valid T2 PASS/FAIL because the evaluator crashed on a provisional PATH_FOLLOW prediction.

Before that crash, step 25 had already moved materially from initialization:

- control accuracy: 0.8854;
- relational role accuracy: 0.7773;
- relational operation accuracy: 0.6406;
- relation-sequence exact accuracy: 0.2014;
- full operator exact accuracy: 0.0972.

This is not a success result. It is evidence that the frozen token stack is learnable by the T2 interface and contradicts treating the semantic representation as already proven absent.

## Data finding

The v0.56 shallow audit remains useful.

For T2 v0.1 query-only unigram NB on DEV:

- control: 0.7847;
- operation: 0.7014;
- role: 0.7083;
- full signature: 0.2639.

On the realistic 56-row audit set, full signature fell to 0.0714.

So the next causal experiment should not reuse the old generated instruction language.

## Architecture decision

Keep:

- QSRE T1 executor/readout;
- frozen semantic checkpoint for one more causal stage;
- T2 multi-layer token operator family;
- oracle structural support and frozen T1 executor.

Replace:

- the T2 v0.1 synthetic query curriculum.

The next experiment is T2 v0.2:

- realistic entity-rich public language;
- cue collisions;
- disjoint train/dev names and phrasing;
- exact same T1 structural distribution;
- exact same T2 architecture and optimizer;
- no LR/step/model-size tuning;
- one preparation job and one dependent P100 run;
- one newly locked operator-only challenge may open only after a perfect governed DEV pass.

If the realistic T2 v0.2 run is a valid model failure, do not patch it. That failure will reopen the semantic/operator boundary with much stronger evidence than the v0.56 pooled probe.

## Anti-loop

This correction does not erase job 575950. Its runtime, fuzz, and pooled-representation evidence remain valid.

It only rejects the over-broad inference made from a representation-mismatched probe.
