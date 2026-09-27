# N0 P43 full-J3 forward attribution: Magnolia 576211

## Evidence and scope

The owner ran the versioned `n0_p43_j3_forward_attribution_v1.py` on `gpu001` against a clean `a19f8e8893422702c138182f239064385addf91c` checkout. Slurm `576211` completed `0:0` in 3m15s. Its `forward_trace.json` SHA-256 is `4accf8d82c0aa15ea59d2f2d4ae99240c72e4951ad15fd60233ac5cd86acf87e`; `diagnostic.json` SHA-256 is `6779c513c67ea782234bdf2c2338e93c395d00280bc70621545aaca69a9d3410`. The pasted trace was reconstructed byte for byte to the reported hash, and its internal diagnostic hash matches. Both artifacts remain under `~/rayan-compute/rayan-n0/n0-v02/full-envelope-j3-forward-attribution-a19f8e88-v1/` on Magnolia. The trace also binds mixture manifest SHA-256 `791f342287d124770a193c175413f3df5d7043a963d1d3d655127baf5c38c8be` and original qualifier SHA-256 `a4c33e247fcb05b97cf7ea3dbd632fb8eb6e4444ea72d971a0d9e243d732a7ef`.

One CUDA process performed the diagnostic even though the Slurm request allocated two GPUs. The original diagnostic passed all 18 semantic × fabric cases with finite joint losses and all public lanes. The trace contains 144 system calls: 18 each of MLM, teacher, semantic operator and natural relation, plus 72 full-fabric forwards (primary plus three counterfactuals in every case). It is a no-gradient, no-optimizer, no-training-authority observation; there is no measured backward, DDP training, optimizer step, checkpoint/resume, or P43 PASS. Host MaxRSS `1357476K` is not GPU VRAM.

## Measured pattern on the single P100

| Task | Calls | Highest allocated CUDA bytes | Peak GiB | Case of highest peak |
| --- | ---: | ---: | ---: | --- |
| Full envelope | 72 | 3,926,628,864 | 3.657 | `long_context_semantic__long_additional_view_source` |
| Semantic operator | 18 | 2,696,761,856 | 2.512 | same |
| Natural relation | 18 | 2,055,293,952 | 1.914 | same |
| MLM | 18 | 1,729,153,536 | 1.610 | same |
| Teacher | 18 | 1,548,701,696 | 1.442 | same |

The full-envelope maximum is the seventh call (third counterfactual) of the longest additional-view case. It starts with ~1.381 GiB allocated and peaks ~2.276 GiB above that per-call baseline. The maximum CUDA allocator reservation is 6,557,794,304 bytes (6.107 GiB); it is cached allocator reservation, not simultaneous live tensor allocation. Other full-fabric cases peak near 1.39–1.45 GiB allocated. The `long_additional_view_source` fabric case drives ~3.656–3.657 GiB full-envelope peaks in *each* semantic pairing. These per-call values must not be summed across calls: inference mode releases outputs/temporary activations differently from an intact differentiable J3 graph.

## Decision and next experiment

The original two-rank P43 projection remains **17,754,461,984 bytes/rank**, above the unchanged 85% P100 and observed Kaggle T4 limits. That number assumes four times a no-gradient peak delta for backward activations and is not a measured optimizer-training peak. The production trainer *already* enables backbone gradient checkpointing, which the original inference-only qualifier cannot characterize. The new trace localizes a large long-view forward transient but does not prove full-fabric rematerialization will recover the required memory or that forward allocation predicts retained backward activations. A full-system differentiable measurement is the next discriminating test, not a reduction of sequence length, views, families, gate or model geometry.

Specify a separate, versioned **two-rank full J3 gradient-memory diagnostic**, not the original P43 receipt. It must load the exact source-bound model, teacher/mixture and 18 stress pairs with the trainer's existing checkpoint mode, real DDP and fp16; run the actual single intact joint objective through backward without weight update, with bounded fail-closed OOM recording. Measure allocated, reserved and driver memory after load, forward, backward and gradient lifetime per rank and record parameter/gradient counts, finite and nonzero gradients across every active family. If legally and technically possible under the unchanged training authorization, account separately for AdamW state and optimizer-step temporaries; otherwise label optimizer peak **unmeasured**. A no-step gradient probe cannot replace the existing P43 authorization or justify training. Any memory-efficient alternative must prove equivalent full-joint gradients, the unchanged 85% cap across the whole route, two-rank checkpoint/resume and fresh authority receipts before training. Preserve the failed P43 root and this forward root.

Kaggle's `30.00h` remaining GPU quota and stored `NvidiaL4` kernel metadata do not reverse the verified two-T4 allocation on that run. Actual L4×2 entitlement is still unknown. Paid high-memory compute remains an option if a complete low-memory route fails or proves costlier to engineer; no paid resource has been launched.
