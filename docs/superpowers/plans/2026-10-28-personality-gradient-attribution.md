# Personality trajectory gradient attribution implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Execute and review the bounded tasks below; preserve completed scientific sources.

**Goal:** Distinguish weak conditional teaching, paired-context cancellation and competing native objectives in the completed negative learning trajectory before choosing another model change.

**Architecture:** Read the intervention's forty original pre-update states and forty corresponding post-update states. Reproduce each of its 216 cases at the original state, natural plus replay, through the existing model and admitted targets. Differentiate the actual objectives without optimizer steps and compare gradients with the saved parameter change.

**Tech stack:** Existing owner-isolated Magnolia Python3.11.16/PyTorch2.7.1 CPU runtime, original scientific revision333fcd0c, complete closed features and reviewed teaching. Windows public mechanics use the existing personality environment.

**Spec:** `docs/eipm/PERSONALITY_CONDITIONAL_LEARNING_DIAGNOSIS_2026-10-28.md` and verified saved audit580080.

## Constraints and interpretation

- Keep Gemma/N1/N3 and the full native architecture unchanged. No new labels, no optimizer steps, no candidate checkpoint or qualification claim.
- All216 N2 cases, original forty groups and their 1/(2*group_size) coefficients remain present. Include every native loss term exactly once.
- Three disjoint objective components: preferenceCE+candidate-pairBCE; concept-pointer loss; every remaining native term. Apply the actual recipe weights before differentiation.
- Preserve per-case/per-view preference gradients before group aggregation, especially each of the forty broader context pairs. Correction-minus-natural and pair cancellation are separate measurements.
- Report complete parameter coverage and nonoverlapping architectural partitions: conditional projections; candidate reader; scalar head; all other N2 parameters. Do not rename scalar head output into an independently trained preference module.
- Compare with actual saved Δtheta. All original updates were clipped, and AdamW has momentum and coordinate scaling. A cosine is local sensitivity, not proof that gradient surgery repairs the personality.
- Reproduce original natural/replay field losses, coverage and accumulated total gradient norm before interpreting each group. Report numerical residuals explicitly under PyTorch float32 comparison tolerances; do not silently relax after failure.
- First protected operation is the established namespace guard, called once. Full source/custody admission happens at acquisition and close; use nondestructive live checks between groups. Preserve exact public log copy permissions and immutable snapshot manifest/tensor pins.
- Save completed measurement groups, source pins and cursor so interruption does not lose the work. Incomplete coverage cannot yield a completed diagnostic claim.

## Task1: actual objective-gradient analysis

Files: `scripts/eipm/gemma_n0/conditional_gradient_attribution.py`; `tests/personality/test_conditional_gradient_attribution.py`.

- [x] Add discriminating RED tests using actual native objective/teaching fixtures and a small analytic cancellation control. Verify complete weighted decomposition, actual-update versus negative-gradient distinction, unknown/zero objectives, frozen parameters and no accumulated `.grad` or state mutation.
- [x] Implement per-view component gradients with `torch.autograd.grad`, complete native objective values and architectural parameter partitions. Keep actual-total gradients independent for decomposition checks.
- [x] Aggregate original endpoint coefficients; retain individual/pair/correction-specific summaries and actual-update directional derivatives. Do not invent independent owner/outcome labels.
- [x] Run affected native teaching/conditional regressions and fresh review. Final54 tests and8 subtests passed. Review exposed missing combined natural/replay endpoint cancellation; analytic and actual native regressions verified its correction.

## Task2: exact trajectory reader and execution

Files: `scripts/eipm/gemma_n0/conditional_learning_readonly_context.py`, `scripts/eipm/gemma_n0/magnolia_conditional_gradient_attribution_20261028.py`, matching `.sbatch`, public receipts and FBM seed.

- [x] Implement reuse of the verified source, complete feature and batch-admission logic of579741 in a separate read-only context with explicit closing custody. Original worker is unchanged; no runtime monkey-patches skip its optimizer.
- [x] Implement reading original experiment plan, progress manifests and tensor bodies by externally published pins. Initial group uses original9f3de2; later groups use prior intervention snapshot. Check exact N1/N3/objective/final state and original event prefix. Actual full private execution remains pending.
- [x] Implement per-group saved pre-state/RNG loading, original full case replay, native loss/correction reproduction, gradient measurement and post-state Δtheta binding. Persist measured groups without updating either model. Private details stay in owner custody. Saved groups have public pins for later explicit recovery; automatic resume is not claimed.
- [ ] Verify public mechanics, source closure and fresh review before one owner-account dispatch. Preserve failures. Record exact job/source identity, FBM process and explicit resume conditions, then stop if only the external result remains.

## Research and decision rules

[IntCEM](https://arxiv.org/html/2309.16928) motivates intervention-aware task teaching but does not guarantee improvement here. [PCGrad](https://arxiv.org/html/2001.06782v4) requires task-gradient evidence; this diagnostic measures the relevant condition without adopting its optimizer. PyTorch's [autograd.grad](https://docs.pytorch.org/docs/2.7/generated/torch.autograd.grad.html), [AdamW](https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html), and [clipping](https://docs.pytorch.org/docs/2.7/generated/torch.nn.utils.clip_grad_norm_.html) define the measured mechanism. Context7 documentation supplements these pinned-runtime references; it does not authorize a dependency upgrade.

Substantial per-context signals that cancel support a grouping/objective question. Opposed task/remainder signals with actual updates locally increasing task loss support competition at those updates. Tiny correction-specific task signals before aggregation support weak learning through the corrected channel. Preserve alternative explanations and use a targeted intervention only after the measurement supports it. No result automatically selects an architecture or training recipe.
