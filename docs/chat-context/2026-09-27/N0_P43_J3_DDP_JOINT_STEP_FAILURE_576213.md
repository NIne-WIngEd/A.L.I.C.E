# N0 P43 J3 two-rank backward finding: job 576213

## Observation and authority

Owner-supplied Magnolia `sacct` reports job `576213` on `gpu001` **FAILED `1:0`** after 1m46s. The pinned diagnostic's stdout binds scientific source `a19f8e8893422702c138182f239064385addf91c`, two ranks, the full J3 objective and `optimizer=false`, `weight_update=false`, `P43_AUTHORITY=false`. Both ranks printed `GRADIENT_CASE_START` for the **first** case `max_runtime_axes__max_candidate_cardinality`. Neither printed `GRADIENT_CASE_DONE`. Both rank tracebacks point to `accelerator.backward(loss)` and `RuntimeError: Expected to mark a variable ready only once`; PyTorch names `stack.public_judgment_probe.score.2.weight`, parameter index 385. The elastic `ChildFailedError` is a consequence of that error. The kernel-version warning does not explain the specific reducer exception. The result is an **incomplete gradient diagnostic**, not a measured OOM, an 85% capacity PASS/FAIL, or a training authorization.

The published probe checks the runtime checkpoint mode before entering the stress loop. Both ranks reached `GRADIENT_CASE_START`. The owner subsequently supplied the preserved per-rank JSONL receipts, detailed below. Do not infer backward peak, parameter nonmutation hash-after, or complete pair coverage from the interrupted Slurm allocation. Preserve the evidence root `~/rayan-compute/rayan-n0/n0-v02/full-envelope-j3-gradient-memory-a19f8e88-v1/` and original P43/forward roots.

## Preserved rank receipts and measurement limit

Owner-provided SHA-256: `rank-0.jsonl` `a8d462003f3d8f4abe6f1da1d721cb7409e6e5b7ffcc457b02103c02df7f4ab8`; `rank-1.jsonl` `1698e5153064e8959e118983cc91a3a7c3134daa84155c68f067355c3c1bf7d5`. These digests and observations are transcribed from the terminal; the remote files were not independently read here. No `diagnostic.json` was printed because the experiment did not complete.

| Receipt field | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| GPU, total bytes | Tesla P100-PCIE-12GB; 12,778,733,568 | Tesla P100-PCIE-12GB; 12,778,733,568 |
| 85% capacity bytes | 10,861,923,532 | 10,861,923,532 |
| Trainable parameters | 242,697,624 | 242,697,624 |
| Checkpoint modules / `use_reentrant` | 17 / all `false` | 17 / all `false` |
| Versions | Torch 2.7.1+cu118; Accelerate 1.15.0; Transformers 5.17.0 | same |
| Loaded allocator bytes / reserved bytes | 1,973,318,656 / 2,053,111,808 | same |
| Interrupted backward peak allocated / reserved bytes | 5,531,400,704 / 6,257,901,568 | same |
| Failed case | `max_runtime_axes__max_candidate_cardinality` | same |
| Error | `RuntimeError`, parameter 385 marked ready twice | same |

The receipt says `oom=false` on each rank; neither records a `case_completed` or `rank_completed` event. `failure_memory` is a partial snapshot at the reducer exception, **not a completed backward peak or optimizer-step peak**. Both loaded receipts bind the exact public mixture `791f342287d124770a193c175413f3df5d7043a963d1d3d655127baf5c38c8be`, original trainer hash `a75a35c4ded5c16939e499f6ecf18f2bcc173428888764e8508fd449d5f2c6db`, and source `a19f8e8893422702c138182f239064385addf91c`. No optimizer was created, so AdamW states, gradient accumulation and checkpoint/resume are still unmeasured.

## Systemic explanation to test

The registered joint step calls the same DDP-wrapped `system` repeatedly: MLM, teacher, semantic operator, primary full fabric, three counterfactual full-fabric views, and natural relation. It combines the losses **outside** these DDP `forward` calls and then runs one backward. The pinned trainer uses `DistributedDataParallelKwargs(find_unused_parameters=True)` and this exact joint step. P43 `576210` exercised those calls only inside `torch.inference_mode()`; it could not expose reducer readiness during backward. PyTorch's upstream issue [#60844](https://github.com/pytorch/pytorch/issues/60844) describes a multitask, shared-encoder, multiple-DDP-forwards/one-backward failure with `find_unused_parameters=True`. This is a strong mechanism match, **not proof** that changing one flag or enabling `static_graph` is sound for N0's dynamic stage and row geometry. The named public-judgment parameter is used in multiple full-fabric views. The probe and trainer must be examined together; 40–80 GB GPUs would still encounter this correctness defect under the unchanged trainer.

## Full-route correction program

1. Preserve both inspected per-rank JSONL receipts and hashes. They show the same reducer exception on both ranks, with nonreentrant checkpointing verified. Retain the absence of completed backward/optimizer memory evidence as a first-class uncertainty.
2. Specify a **single DDP boundary around one intact joint J3 step**. An outer module may own the existing registered system and the exact `FullEnvelopeJointTrainingObjectiveV1`; its `forward` calls the unchanged all-lane `execute_full_envelope_joint_step` through the *unwrapped inner module*. DDP then sees one outer forward and one backward per microbatch while shared weights can participate in all eight inner task calls. Keep the ten precommitted macro families, teacher batch 2, replay 512, four full-fabric views, all 18 extrema and the 85% memory policy. Do not split into independent optimizer steps or detach cross-view gradients.
3. Treat the outer module as a **versioned training-route change**, not a one-line hotfix. Audit `build_optimizer`'s backbone/interface parameter groups, staged `apply_stage_trainability`, objective EMA and loss balancer, Accelerate preparation and gradient accumulation, `no_sync`, model/objective checkpoint keys, exact same-stage resume and J1→J2→J3 predecessor loading. The currently registered P43 and training authorizer bind the prior DDP layout, so a successor needs new source-bound receipts; the old a19 P43 FAIL remains historical evidence.
4. Before another full GPU job, prove on reproducible small public/synthetic CPU fixtures that the wrapped and unwrapped full steps have matching forward values and per-parameter gradients for all active families, including counterfactual paths, unused/frozen parameters and variable schema/cardinality. Test two Gloo DDP ranks across changing J1/J2/J3 shapes and repeated accumulation; verify optimizer grouping and checkpoint/resume equivalence. Static text assertions alone will miss this failure.
5. Only after those checks, run a **fresh two-rank full-model 18-case gradient-memory diagnostic** and measure forward/backward peaks under the real registered topology. A legal whole-route optimizer-state/step and durable resume check must follow before a new P43-equivalent authority and training decision. No paid provider purchase, P43 pass, gradient training, or FINAL has been justified by job `576213`.

Local reference: source `full_envelope_joint_step_v1.py` calls the wrapped system eight times; `train_n0_v02_full_envelope_joint_v1.py` configures `find_unused_parameters=True` and calls that joint step before `accelerator.backward(loss)`. Upstream DDP reference: [PyTorch issue #60844](https://github.com/pytorch/pytorch/issues/60844) and [DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). The outer-module route is a **design hypothesis**, pending equivalence and live distributed validation.
