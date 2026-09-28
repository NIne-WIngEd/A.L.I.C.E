# N0 measured joint P43 v2: source and execution gate

Status (2026-09-28): the versioned source implementation is published on
`alice-eipm-v1-n0-full-envelope-joint-ddp-v2` at
`5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2`. The [exact-head CPU workflow](https://github.com/NIne-WIngEd/A.L.I.C.E/actions/runs/36368390448)
passed **125 tests, one CUDA-only skip, zero failures**. There has been **no new full-model GPU
run, no measured P43 PASS, no production training, and no paid allocation**.
The original `a19f8e88` failed P43 projection, Magnolia job 576213 failed
backward, and the Kaggle request that physically allocated T4×2 remain
independent evidence. Do not relabel their receipts or copy them to this head.

## What the source now does

`qualify_n0_v02_complete_joint_route_v2.py` runs the registered J3 system and
the *same* `execute_registered_optimizer_step` called by the production
trainer. The outer DDP forward owns the intact ten-family joint objective.
Each rank exercises all 18 semantic × fabric stress pairs, resetting full
model, objective, scheduler and AdamW state before each pair. Each pair has
two real AdamW steps with eight microbatches apiece; the first step allocates
optimizer state and the next tests the resident state. The qualifier checks
finite whole loss and named gradients, including the public judgment probe.
For the long-context/additional-view pair, it saves model, objective,
optimizer, scaler and scheduler state, loads it again, and compares the next
step to an uninterrupted control. It measures save, load and resumed-step
device capacity too. Strict J1/J2/J3 state loading proves key-space transfer;
it does **not** claim actual DEV-based predecessor selection.

Each rank's allocated and reserved peaks and sampled physical device usage
are conservatively combined with a 1 GiB device margin, and every first,
later and resumed step, save and load must fit within the original **85% per
rank**. The versioned P43 receipt and separate whole-route receipt are
written only after both ranks finish every case. Rank JSONL and an initial
`INCOMPLETE` marker preserve partial failures. Diagnostic weight updates and
the checkpoint under the fresh evidence root have no production authority.

The previous no-gradient P43 is rerun on the final source and public mixture
with `--preserve-projection-failure`; a failed projection is recorded as a
FAIL, even if its full no-gradient forward covers all 18 pairs. The measured
successor accepts it as historical diagnostic data only if the no-gradient
forward itself fits below 85%. The authorizer and trainer require both the
actual measured step receipt and whole-route receipt; status strings without
per-rank/per-pair step arithmetic and exact source/input hashes cannot pass.

## Gate order before one GPU attempt

1. Finish the exact-head CPU CI and review any failure at the full-route
   level. No branch update may silently reuse an older source-bound receipt.
2. Materialize and audit a **new** full public mixture at the final checked-out
   source head. Refresh the P42/static and CPU qualification, tokenizer and
   operator-evidence token audits and long-context boundary receipts required
   by `verify_pre_gradient_runtime`. Verify the teacher bank, all seven public
   lane inputs, the original semantic checkpoint and public corpus hashes.
   Old `a19f8e88` P39PN/P42 evidence cannot stand in for this closure.
3. Check that the allocated host exposes two distinct, compatible GPUs and
   that its driver, Torch CUDA build, Accelerate, Transformers, tokenizer and
   input paths match the reviewed runtime. The rank receipts record installed
   versions, device properties, driver query and source/data hashes. The
   `n0_two_gpu_runtime_inventory_v1.py` tool is data-free and can check the
   physical allocation; it does not authorize a model step.
4. Only then run the precommitted **one** full-route wrapper
   `scripts/eipm/n0/run_n0_v02_measured_joint_route_v2.sh` in a fresh evidence
   root; the Magnolia `magnolia_p100x2_n0_v02_measured_joint_memory_v2.sbatch`
   is an optional wrapper for the *already qualified* P100 allocation. It is
   not a request to retry infrastructure until something passes. Stop after
   a failure, preserve both ranks' files and fix the scientific route before
   any further attempt.
5. A full measured PASS plus the rest of the exact-head receipts can be
   checked by the training authorizer. This is a later decision: a CPU green,
   hardware name, projected memory figure or partially completed GPU run
   cannot authorize gradient training, DEV selection, FINAL or N0 completion.

No provider is certified here. A paid provider can change driver, GPU
architecture, wheel ABI, data mounts and checkpoint storage. Inspect an
**actual** two-GPU offer and pin compatible artifacts before expenditure;
the first full GPU proof on that host can still fail. The absolute Magnolia
path embedded in one teacher registry pair must be mounted or replaced
through a separately versioned data and mixture audit. This document
commits no paid resources and starts no GPU job.
