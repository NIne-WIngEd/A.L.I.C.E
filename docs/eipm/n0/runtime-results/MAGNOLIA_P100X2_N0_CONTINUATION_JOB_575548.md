# Magnolia N0 continuation — job 575548

**Date:** 2026-09-14  
**Status:** completed successfully  
**Role:** first evidence-driven continuation of the durable public N0 semantic backbone after the step-200 scheduler defect was repaired

## Scheduler/runtime result

- job: `575548`
- node: `gpu001`
- route: 2× Tesla P100-PCIE-12GB
- parent checkpoint: `step-00000200`
- target checkpoint: `step-00001000`
- sequence length: 512
- micro batch size: 1 per process
- gradient accumulation: 16
- mixed precision: fp16
- global scheduler horizon: 10,000 optimizer steps
- internal Accelerate scheduler horizon: 20,000 scheduler steps for 2 processes with split_batches=false
- save interval: 200 steps
- private identity data/gradient: false

The fail-fast teaching-code preflight passed before model execution.

## Scheduler repair confirmation

The resumed run explicitly recorded:

- `legacy_scheduler_rebase=true`
- `scheduler_total_steps_global=10000`
- `scheduler_total_steps_internal=20000`
- resume step: 200

The learning rate no longer repeated the defective 200-step cosine cycle. It remained near the intended early-training region and decayed slowly from approximately `2.9973e-4` at step 210 to `2.9292e-4` at step 1000.

## Durable step-1000 state

The run completed at:

- checkpoint step: 1000
- tokens seen total: 16,384,000
- resume parent: canonical `step-00000200`
- private identity gradient: false

The training loss remained noisy, as expected for dynamic span masking and a still-early random-initialized 352M-parameter model, but useful low points appeared throughout the segment. Examples include approximately 5.99 at step 400, 6.08 at step 800, and 6.16 at step 1000. Training-loss noise is not used by itself as the promotion criterion.

## Same-held-out-set before/after evidence

The deterministic dev MLM evaluator used the same document split and mask seed before and after the segment.

At step 200:

- batches: 2
- visible tokens: 3,316
- masked tokens: 495
- mean masked-token NLL: `8.014384633844548`
- masked-token perplexity: `3024.147866767514`

At step 1000:

- batches: 2
- visible tokens: 3,316
- masked tokens: 495
- mean masked-token NLL: `7.957240052656694`
- masked-token perplexity: `2856.179160206486`

This is a real directional improvement on the same held-out documents: NLL decreased by about 0.0571 and masked-token perplexity decreased by about 5.6%.

## Evaluation limitation discovered

The current corpus split policy reserves only 0.1% of documents for dev and 0.1% for test. On the bounded 5,933-row corpus, the dev set produces only two 512-token batches. The evaluation is genuinely held out, but it is too small to serve as a high-confidence sole decision signal.

Do **not** enlarge the dev set retroactively by relabeling training documents: the existing step-1000 model has already been allowed to train on those documents, so doing that would create false held-out claims. Instead, preserve the existing true held-out documents and reduce masking variance by evaluating them under multiple deterministic masking seeds. Keep the test split untouched for milestone evaluation.

## Next teaching decision

Do not launch another long MLM continuation solely because step 1000 completed. First run a cheap teacher diagnostic at step 1000:

1. evaluate the true dev documents under multiple deterministic mask seeds;
2. train a frozen-backbone semantic ranking probe on the governed Sol curriculum so the probe tests what the backbone already represents rather than rewriting it;
3. evaluate aggregate and per-competency dev performance;
4. use those two signals to decide whether the next action is more public MLM learning, targeted public semantic teaching, or both.

This diagnostic is public N0 work only. It does not authorize any private E0/E-INF/A-SYN gradient.
