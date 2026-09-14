# N0 step-1000 evidence and teacher-probe boundary — 2026-09-14

## Durable state

Job `575548` completed the first corrected continuation of the public N0 semantic backbone from canonical step 200 to canonical step 1000.

The run used the corrected global scheduler horizon instead of repeating the defective 200-step cosine cycle. Step 1000 records 16,384,000 total tokens seen and remains a public-only lineage with `private_identity_gradient=false`.

Same-held-out-set evaluation improved from step 200 to step 1000:

- mean masked-token NLL: `8.014384633844548` -> `7.957240052656694`
- masked-token perplexity: `3024.147866767514` -> `2856.179160206486`
- held-out visible tokens per evaluation: 3,316
- masked tokens per evaluation: 495

This is a real directional improvement, but the dev split is small because the original immutable materialized corpus assigned only 0.1% of documents to dev and 0.1% to test.

## Evaluation rule

Do not retroactively enlarge the held-out split by reclassifying existing training rows. Step-1000 weights have already been allowed to see the training split, so doing that would create false held-out evidence.

Instead, preserve the true dev document set and reduce dynamic-mask variance by evaluating the same held-out documents across multiple deterministic masking seeds. Keep test untouched for milestone use.

## Teaching boundary

Do not blindly continue MLM just because step 1000 exists. The next action is a cheap teacher diagnostic that answers two questions before another long run:

1. Is the MLM improvement stable across multiple independent masking patterns on the true dev documents?
2. Does the current backbone already encode enough of the 43 public semantic/pragmatic/social/epistemic/ranking/alignment competencies for a frozen ranking head to generalize from the governed Sol train examples to the governed Sol dev examples?

A frozen-backbone probe is diagnostic only. It trains the generic ranking head while preserving the step-1000 backbone exactly. It is not a substitute for semantic fine-tuning and must not be described as personality learning.

The Sol curriculum currently consists of the original 60-row seed plus the 26-row coverage shard. Together they provide train and dev representation across all 43 registered N0 competencies. Future Sol teaching examples should be driven by observed per-competency failures rather than bulk synthetic generation without evidence.

## Compute discipline

The teacher diagnostic is the only justified immediate accelerator work. It should use one P100 because the backbone is frozen and the dataset is small. No two-GPU continuation should be launched until the diagnostic result determines whether additional public MLM, targeted public semantic teaching, or both are warranted.

Magnolia remains an accelerator, not the project workflow. Git/local reasoning and curriculum construction continue independently of the cluster.

No private E0/E-INF/A-SYN gradient is authorized.
