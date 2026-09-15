# N0 v0.2 First Native Multitask Tranche — PASS

**Date:** 2026-09-14  
**Magnolia job:** `575583`  
**Node:** `gpu001`  
**Runtime:** 25m52s  
**Accelerators:** 2 × Tesla P100-PCIE-12GB  
**Private identity data:** false  
**Private identity gradient:** false

## What ran

The first permanent native `alice-n0-semantic-v0.2` weight update completed from random initialization. The 136,594,435-parameter v0.2 model used the governed 48k tokenizer, the 21-source public v0.2.1 corpus, and the 1,020-row / 51-competency public teacher bank.

The bounded first tranche ran 500 optimizer steps at sequence length 512 with:

- span MLM weight 0.70;
- candidate preference weight 0.10;
- principle/rationale alignment weight 0.10;
- semantic contrastive weight 0.10;
- 8 MLM microbatches per optimizer step per prepared DDP schedule;
- teacher batch 2 per process;
- fp16;
- learning rate 3e-4 after 100-step warmup;
- deterministic global public-corpus row ordering from `sha256(seed:text_sha256)` to avoid source-order bias in a bounded tranche.

Permanent checkpoints were written at steps 250 and 500.

## Fixed core readiness

Random baseline:

- top-1: 0.359375;
- separation: 0.359375;
- full invariance: 0.325581;
- mean margin: -0.041153.

Step 250:

- top-1: 0.9765625;
- separation: 0.9765625;
- full invariance: 0.976744;
- mean margin: 4.513219.

Step 500:

- top-1: 0.9765625;
- separation: 0.9765625;
- full invariance: 0.976744;
- mean margin: 15.460137.

The only surviving fixed-core failure at both step 250 and step 500 was `N0V02-FIX-ALIGN-01`. Its mean margin worsened from approximately -0.330 at step 250 to -0.697 at step 500 while the other 42 competencies remained correct. This is a real targeted repair signal, not a reason to restart the model.

## Fixed voice readiness

Random baseline:

- top-1: 0.50;
- full invariance: 0.375;
- mean margin: 0.019201.

Step 250:

- top-1: 1.00;
- separation: 1.00;
- full invariance: 1.00;
- mean margin: 4.375031.

Step 500:

- top-1: 1.00;
- separation: 1.00;
- full invariance: 1.00;
- mean margin: 13.764771.

All eight public voice-expression competencies passed every fixed order/paraphrase variant by step 250 and remained passing at step 500.

## Held-out public MLM

Random baseline:

- mean masked-token NLL: 10.826638;
- masked-token perplexity: 50,344.18.

Step 250:

- mean NLL: 7.069161;
- perplexity: 1,175.16.

Step 500:

- mean NLL: 6.660788;
- perplexity: 781.17.

The semantic language objective continued to improve materially from step 250 to step 500 even though the current small fixed ranker suite had already saturated.

## Interpretation

This tranche proves that the v0.2 architecture, objective wiring, tokenizer/corpus lineage, DDP path, and rationale-bearing teacher supervision can produce rapid real learning on Magnolia. It does **not** establish N0 readiness for private N1 identity gradients.

The current fixed readiness suite is too small to justify promotion: 43 core base cases plus 8 voice base cases, even after deterministic candidate-order/paraphrase expansion, can saturate quickly. The step-250 → step-500 pattern also shows increasing score magnitude without accuracy gain. Candidate-preference loss frequently approached zero late in the run. This is evidence to strengthen generalization and calibration tests before spending another training segment on the same supervision mix.

The correct next step is therefore **challenge evaluation before more gradient updates**:

1. preserve both step-250 and step-500 checkpoints;
2. treat step-500 as the current semantic candidate because MLM continued to improve, but do not declare it champion yet;
3. expand evaluation with genuinely new held-out cases, cross-competency cases, ambiguity/tie cases, and harder negative candidates;
4. test both step-250 and step-500 on that challenge suite;
5. generate targeted repair only from observed failures, with special attention to `ALIGN-01`;
6. continue from the better checkpoint rather than restarting;
7. add score-calibration pressure before allowing preference margins to grow without behavioral gain.

## Runtime warnings

Transformers reported the MLM prediction-head tensors as `UNEXPECTED` when `ModernBertModel` loaded the MLM checkpoint for ranker evaluation. This is expected for the deliberate task change from `ModernBertForMaskedLM` to backbone-only `ModernBertModel`; the ranker backbone and scorer state loaded successfully and evaluation completed.

The old Magnolia kernel warning (`3.10.0` below the recommended distributed-runtime kernel) also appeared. The run completed without a hang, so this is not a new blocker.

## Local-path bug fixed upstream

The initial submission failed because `$HOME` inside the udocker session resolved to `/root`, causing the script to search `/root/rayan-compute/...`. The owner patched the script locally and job 575583 then completed. The build branch now uses the repository's mounted compute root (`$(dirname "$ROOT")/rayan-n0/n0-v02`) so future container executions do not depend on container `$HOME`.
