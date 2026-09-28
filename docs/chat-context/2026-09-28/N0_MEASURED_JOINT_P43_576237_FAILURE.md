# N0 measured joint P43 job 576237: capacity failure and retained evidence

Status: **FAIL, no GPU qualification or training authorization**. This is an
assessment of the owner's pasted Slurm stdout/stderr, not an independent read
of Magnolia's retained JSONL receipts. Scientific source remains
`5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2`.

## Exact-head predecessor chain

The owner reported CPU job `576234` COMPLETED `0:0`. Its six printed receipt
statuses include static P42, full CPU runtime, tokenizer stress, operator
evidence alignment, semantic operator long context and long-context token
boundary alignment. Every displayed `source_revision` matches the pinned
source. Mixture job `576235` COMPLETED `0:0`. Its public manifest status is
`MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT`, audit status is
`PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1`, and FINAL remains frozen
before gradient. The printed audit reports eight training lanes, ten macro
families, zero FINAL training rows, no private identity data and no training
authorization. These jobs supplied the fresh input chain for the bounded GPU
attempt. The handoff GPU preflight checked source and input hashes before
submitting the job; its own stdout has not been supplied separately.

## What the one bounded GPU run established

Slurm job `576237` ran on `gpu001` for 30m29s, then FAILED `1:0`. The wrapper
printed the exact source, executed the old two-rank, 18-pair no-gradient
diagnostic, and preserved its `FAIL_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1`
projection. That predecessor recorded 12,778,733,568 bytes per P100 and a
projected 17,754,461,984 training bytes per rank. It had no backward or
optimizer authority.

The measured successor entered the complete joint two-rank J3 route.
Stdout records both ranks passing the first five pairs:
`max_runtime_axes` paired with candidate, field, edge, view and reasoning
stress cases. Both ranks then entered
`max_runtime_axes__long_additional_view_source`. Stderr names the first
optimizer step of that pair: `measured full step exceeds 85% per-rank capacity:
max_runtime_axes__long_additional_view_source/0`. The source raises this only
after the registered eight-microbatch optimizer step returned, checked that
the FP16 scaler did not skip it, checked that AdamW state exists, and compared
before/after state digests. The qualifier writes a failure row containing the
actual measured memory before raising. The traceback is a policy capacity
failure, not evidence of a CUDA OOM or a new dtype failure. The kernel/NCCL
messages do not supersede the explicit exception.

The owner subsequently pasted the last `rank-0.jsonl` and `rank-1.jsonl`
failure rows. Both report the identical actual step measurement:

| Per-rank quantity | Bytes | GiB |
| --- | ---: | ---: |
| P100 total capacity | 12,778,733,568 | 11.901 |
| 85% policy limit (floor) | 10,861,923,532 | 10.116 |
| Peak PyTorch allocation | 9,350,800,384 | 8.709 |
| Peak PyTorch reservation | 10,498,342,912 | 9.777 |
| Maximum sampled whole-device use | 10,886,447,104 | 10.139 |
| Safety margin | 1,073,741,824 | 1.000 |
| Conservative measured peak | 11,960,188,928 | 11.139 |
| Over the 85% policy limit | 1,098,265,396 | 1.023 |

The conservative peak is the larger of allocator allocation, reservation and
sampled whole-device use plus the frozen margin. The last term dominates. The
sampled device use alone was already 24,523,572 bytes above the 85% limit;
the 1 GiB margin is an explicit part of the original capacity policy, not an
optional extra to drop. Both rows report finite loss, all eight microbatches,
finite nonzero gradients, nonzero named public judgment gradient, and changed
weight/AdamW state. The stage samples show device use growing at backward
microbatch 1 and staying at 10,886,447,104 bytes through optimizer. The
allocator's higher instantaneous allocation peak is not localized by those
post-stage snapshots. No CUDA OOM occurred.

The same measured demand would require at least 14,070,810,504 total device
bytes to pass an 85% arithmetic check **if all memory behavior stayed
identical**. This is only a lower bound for the one failing step. Different
hardware, the other 12 pairs, a later AdamW step, and checkpoint transport
can change the peak. It cannot certify a T4, L4, 16 GB or 24 GB offer. Even
the five reported PASS pairs do not establish all 18 pairs or the
checkpoint/save/load/resumed-step requirement. The source keeps
`measured/result.json` marked `INCOMPLETE_N0_MEASURED_JOINT_ROUTE_NOT_AUTHORITY`
on this early failure; a successful measured GPU receipt and joint-route
receipt are not reported. The older projected FAIL remains intact.

## Next decision, no new compute

Inspect the preserved root at
`$HOME/rayan-compute/rayan-n0/n0-v02/measured-joint-5e29f7f-v1`.
Preserve both rank files and their source/input hashes; do not reset or rename
the root. Compare faithful complete-step memory routes and verified two-GPU
high-memory offers against the **unmeasured** remainder, not just the 1.023
GiB gap in the first failing pair. A larger device may solve this observed
capacity failure but still requires all 18 pairs, second steps and
checkpoint/resume on the actual purchased runtime. No second Magnolia/Kaggle attempt,
reduced family/shape, weaker threshold, fabricated PASS, production training,
checkpoint selection or FINAL opening follows from this result. Any source
change invalidates this exact-head predecessor chain for later authorization.

FBM may retain this as an observed public **procedure failure** with source,
jobs, hardware, executed stage and evidence gaps. It is not entity or
personality gold. `training_eligible=false`, `evaluation_eligible=false`.

## Paid fallback, still unallocated

The measured lower bound makes a larger-memory replicated DDP host a simpler
candidate than a new sharding/offload implementation. Lambda's current public
listing describes one two-GPU A6000 instance with 48 GB per card, 28 vCPUs,
200 GiB RAM and 1 TiB SSD at $1.09 per GPU-hour ($2.18/hour for the pair,
before taxes). Its two-A100 PCIe instance lists 40 GB per card at $1.99 per
GPU-hour. These are catalogue prices and shapes, not a checked account offer
or availability. [Lambda instance shapes](https://docs.lambda.ai/public-cloud/on-demand/)
and [pricing](https://lambda.ai/pricing). A different provider's advertised
GPU cannot substitute for an observed two-device allocation.

Before any allocation, inventory the exact public source/data/checkpoint
closure and the absolute teacher shard path, then prepare a digest-pinned
container or compatible locked runtime with the current trainer. Confirm the
specific offer's two unsliced GPUs, driver, Python/Torch/CUDA/Accelerate
versions, host RAM/disk, persistence and billing. On the rented host, first
verify hashes, rank mapping and NCCL, then run one bounded original full-route
qualification; stop on mismatch or failure. The P100 receipt does not promise
that 18 pairs, checkpoint/resume or numerical behavior will pass there. No
instance is selected, purchased or started by this note.
