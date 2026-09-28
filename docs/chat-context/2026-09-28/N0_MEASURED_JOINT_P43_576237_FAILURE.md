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

The pasted stdout and stderr do **not** include the per-rank `rank-0.jsonl`
and `rank-1.jsonl` measurement objects. Thus the exact peak, overage,
allocator-versus-device-used contribution, and timing within the step remain
unverified here. Even the five reported PASS pairs do not establish all 18
pairs or the checkpoint/save/load/resumed-step requirement. The source keeps
`measured/result.json` marked `INCOMPLETE_N0_MEASURED_JOINT_ROUTE_NOT_AUTHORITY`
on this early failure; a successful measured GPU receipt and joint-route
receipt are not reported. The older projected FAIL remains intact.

## Next decision, no new compute

Inspect the preserved root at
`$HOME/rayan-compute/rayan-n0/n0-v02/measured-joint-5e29f7f-v1`.
Read `measured/rank-0.jsonl` and `measured/rank-1.jsonl` for each step's
`max_allocated_bytes`, `max_reserved_bytes`, `max_sampled_device_used_bytes`,
`conservative_measured_bytes`, capacity fraction, stage samples and named
gradient flags. Confirm the failure row on each rank and check that the
root's incomplete result and old projection are preserved. Use the actual
overage and peak stage to compare faithful complete-step memory routes and
verified two-GPU high-memory offers. No second Magnolia/Kaggle attempt,
reduced family/shape, weaker threshold, fabricated PASS, production training,
checkpoint selection or FINAL opening follows from this result. Any source
change invalidates this exact-head predecessor chain for later authorization.

FBM may retain this as an observed public **procedure failure** with source,
jobs, hardware, executed stage and evidence gaps. It is not entity or
personality gold. `training_eligible=false`, `evaluation_eligible=false`.
