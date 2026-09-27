# N0 P43 memory route reassessment — 2026-09-27

## Verdict and evidence boundary

The existing two-rank replicated-DDP P43 **fails its precommitted 85% memory projection** on Magnolia P100×2. The hardware-only Kaggle request stored `NvidiaL4` but its only observed execution allocated Tesla T4×2, which also fails that *unchanged P100-derived estimate*. Neither observation measures backward or optimizer peak; neither proves the full 243,693,339-parameter model cannot be trained on those devices using an independently qualified memory-efficient implementation. Do not change the 85% limit, full J3 objective, required 18 stress pairs, public lanes, checkpoint lineage, or FINAL isolation to declare a PASS.

Exact scientific source: `a19f8e8893422702c138182f239064385addf91c`; P42 `576206` PASS; P39PN `576207` PASS; complete single-rank fp16 diagnostic `576209` PASS without P43 authority; P43 `576210` finite full J3 on both DDP ranks, memory projection FAIL, no gradients, no optimizer and no weights. P39PN manifest SHA-256 `791f342287d124770a193c175413f3df5d7043a963d1d3d655127baf5c38c8be`. Preserve failed roots.

## What 17.75 GB actually means

The P43 qualifier runs `torch.inference_mode()` with fp16 autocast over 18 semantic × full-fabric pairings. From each rank's P100 receipt:

| Term | Bytes | GiB | Origin |
| --- | ---: | ---: | --- |
| Pre-forward resident baseline | 1,973,318,656 | 1.838 | Measured, including 974,773,356 model parameter bytes |
| Gradients and Adam moments | 2,912,371,488 | 2.712 | 242,697,624 trainable parameters × (4 + 8) bytes |
| Backward activation allowance | 11,795,030,016 | 10.985 | 2,948,757,504 measured no-gradient allocation delta × frozen 4× multiplier |
| Allocator safety | 1,073,741,824 | 1.000 | Precommitted margin |
| **Projected** | **17,754,461,984** | **16.535** | Sum, **not** measured backward peak |

The measured no-gradient peak **allocated** was 4,922,076,160 bytes; peak **reserved** was 7,564,427,264 bytes. The projection uses allocated memory, not a synchronized training peak. PyTorch distinguishes allocated/reserved and offers snapshot tracing; CUDA library allocations may be outside its allocator. Backbone gradient checkpointing is enabled in the *trainer* but absent from the P43 no-gradient forward; a no-gradient run cannot establish its benefit to a retained autograd graph. This can make the frozen 4× proxy conservative or optimistic for an actual optimizer step. Retain it as the present gate; replace only with a versioned, validated training-memory evidence contract, never by simply changing the multiplier.

| Actual device | Physical bytes per rank | 85% budget bytes | Reduction in current projection required | Status |
| --- | ---: | ---: | ---: | --- |
| Magnolia P100 | 12,778,733,568 | 10,861,923,532.8 | 6,892,538,451.2 | Current P43 fails; fit after new design unproven |
| Kaggle observed T4 | 15,636,037,632 | 13,290,631,987.2 | 4,463,829,996.8 | Hardware-only probe fails old projection; fit after new design unproven |
| Two actual L4 | ~24 GB specified by NVIDIA per GPU | Provider-specific | Old P100 projection suggests headroom | Account has **not** yielded L4×2; must remeasure |

At unchanged other terms, a 2-rank T4 route needs effective activation reserve below `2.4862 ×` the observed no-gradient delta (versus current 4×); P100 needs below `1.6626 ×`. These are arithmetic targets, not measured savings, and neither is a proposed gate relaxation. Sharding half of 4-byte gradients plus 8-byte Adam moments saves at most ~1.456 GB per rank before communication/all-gather overhead; **sharding state alone does not close either gap under the frozen proxy**. The full model has four differentiable full-fabric views (primary, decisive ablation, irrelevant removal, permutation) plus semantic, natural relation, MLM, and teacher branches in a single J3 graph; separating branches into independent optimizer steps can change gradients, joint loss interactions and DEV interpretation.

## Highest-value routes, in order

1. **Do the account-specific Kaggle entitlement check without N0 payload.** Official CLI documents `NvidiaL4` and `NvidiaL4X1` but warns some GPU choices are competition/admin restricted. Existing private probe requested `NvidiaL4` in the CLI and metadata; pull confirms stored metadata, while SHA-verified runtime says Tesla T4×2. Metadata is a *request*, not proof of allocation. Determine from account UI/support or an independent live allocation whether this account can actually obtain **two** L4s; single-L4 mode is not the current 2-rank DDP route. Do not repush the old identical probe and infer entitlement from another stored request.
2. **Design one full-workload lower-memory challenger if free GPUs matter.** First profile individual J3 branches and retained tensors using PyTorch allocator snapshots and phase peaks on a controlled preflight. Keep input row, all ten families, optimizer objective, and geometry fixed. Candidate work: activation rematerialization beyond the already checkpointed backbone, streaming/recomputation of independent forward subgraphs with proof of equal joint gradients, optimizer state sharding or CPU offload, and FSDP. Quantify the effect of DDP graph traversal, four behavioral views, optimizer `foreach` temporaries, synchronization and exact-state checkpoint/resume. These are engineering hypotheses, **not** a qualified alternative today.
3. **Create a versioned, source-bound full-system challenger before gradients.** Compare forward loss and gradients with the original implementation on a safe fixture, test all 18 stress pairs and both ranks on actual target hardware, validate numerical/optimizer/DEV equivalence where expected, record allocated/reserved and driver memory across a complete representative forward/backward/step under an appropriate authorization procedure, and verify state save/readback and cross-session resume. Preserve the old a19 P42/P39PN/P43 receipts; changed code or topology needs its own P42/P39PN bindings and memory/training authority. No silent FSDP substitution under a DDP receipt.
4. **Use paid 2× high-VRAM DDP only if** actual two-L4 access fails or the versioned memory challenger cannot prove a full, durable route within a reasonable effort budget. The preexisting 2×48GB or 2×40GB options reduce engineering and operational risk; check live prices, runtime, storage and account access before an authorized launch. Multi-run checkpointing solves session duration, never a per-step peak.

## New read-only Magnolia inventory: transfer is tractable but one path is bound

The owner's exact `a19` read-only inventory ran successfully inside `magnolia_udocker_exec.sh` after the login shell's `python3: command not found` response; script SHA-256 `c37845f1208b56bc829bfc681c89b8542b594dd2770b0bbde46558d42c8c71ea`. It confirms 21 corpus shards totaling 123,677,368 bytes; production checkpoint 669,270,116 bytes, SHA-256 `6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43`; tokenizer 3,402,578 bytes; seven optimizer-facing lane files including FewRel 67,814,760 bytes; teacher runtime registry SHA-256 `4c06e08cf7ca217fe803ec16a63da2d7381b1d400172af8ffd9f90de120f6de9`; audit SHA-256 `e08c7606d27011b59e3f5803702f1bb0d3254c465be30120725597fbcaa4f123`; no FINAL files staged, no private identity data, training unauthorized. Seven teacher shard pairs resolve relative to repo; v0.5 pair uses absolute Magnolia `/homes/01/mxrayan/.../teacher-bank-v0.5/` paths. A portable environment must bind that exact path or create a versioned registry with fresh public-mixture/authorization audit: silently rewriting the registry breaks existing bound hashes. Source and data need not move merely to study a free-GPU challenger.

## Sources and unresolved questions

Primary documentation: [PyTorch checkpoint memory tradeoff](https://pytorch.org/blog/activation-checkpointing-techniques/), [PyTorch CUDA memory traces](https://docs.pytorch.org/docs/stable/torch_cuda_memory), [PyTorch FSDP](https://docs.pytorch.org/docs/main/fsdp.html), [PyTorch distributed checkpoints](https://docs.pytorch.org/docs/main/distributed.checkpoint.html), [Kaggle accelerator CLI and restrictions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [NVIDIA L4 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf). No verified Kaggle L4 entitlement, live backward peak, validated lower-memory whole-model route, or paid allocation is yet recorded.
