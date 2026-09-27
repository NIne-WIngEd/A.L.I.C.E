# N0 P43 full-model memory experiment protocol — 2026-09-27

## State, authority and outcome

Scientific checkout remains clean at `a19f8e8893422702c138182f239064385addf91c`. P42 and P39PN are PASS; source-bound P43 on Magnolia P100×2 is projection FAIL. A single-Kaggle-run `NvidiaL4` request executed on T4×2; Kaggle CLI metadata is a request, not an allocation receipt. No live two-L4 entitlement, backward peak, optimizer memory, sharding success or training authorization has been measured. Do not repush the same L4 probe as a pretend entitlement test.

The first versioned artifact is `n0_p43_j3_forward_attribution_v1.py` with `n0_p43_j3_forward_attribution_v1.sbatch`. The script invokes the **original** exact-source qualifier in `--diagnostic-only` mode, traces every model forward of all 18 semantic × full-fabric pairings, and demands exactly four full-fabric views per pair. It reports allocated and reserved CUDA bytes immediately before/after each real task forward and its per-call peak. This is one rank's complete all-lane J3 **forward attribution only**, with no gradient, optimizer or checkpoint evaluation. It neither measures retained backward tensors nor grants P43 or training authority. The same full objective remains intact. The Slurm allocation asks for two P100 GPUs to match Magnolia's previously admitted 12 CPU/96 GB shape; the diagnostic explicitly runs one CUDA process. Original failed roots are preserved.

Syntax and shell parsing were checked locally; actual execution, PyTorch graph hooks, device allocation and numeric results **remain unverified** until Magnolia returns a receipt. Script SHA-256 `b2d3af77466b3d861e1dc25e6e8a3701fe3e07e4f16e1a1a6dfd19fcc865e8b2`; sbatch SHA-256 `26d5a32d2a7c2cdf83515d0912f68dcd479c816b4a52b449b231c7d351ad532b`. GitHub fetched bytes exactly matched the checked local files. An execution failure is a new observation to diagnose, not permission to weaken the objective or mark the gate PASS.

## Kaggle account evidence

The cloud workspace is **not logged into Kaggle** and has no local Kaggle CLI credentials. The user's existing Windows CLI had authenticated for the previous probe. The official Kaggle CLI `kaggle quota` can give remaining GPU/TPU hours but does **not certify an accelerator shape**. `kernels status` gives run status, and `kernels pull -m` gives requested/stored shape. The owner account's Notebook accelerator picker, Kaggle support/account-specific statement, or a *new, justified* hardware-only allocation with actual device count/name/memory are needed for a two-L4 claim. A single `NvidiaL4X1` allocation cannot qualify the existing two-rank DDP trainer. Do not upload the model/corpus to Kaggle until the device and transport path qualify.

Read-only Windows checks on the already authenticated machine: `kaggle --version`, `kaggle quota` (only if the installed version lists it), `kaggle kernels status mkrayanyan/rayan-n0-l4-p43-dcdd890fade8`, and inspect the existing `$HOME\Downloads\RAYAN_N0_L4_P43_dcdd890fade8\output\probe_result.json` and `remote-metadata\kernel-metadata.json`. Never print tokens, credentials or private notebook inputs. Existing SHA-verified probe already shows 2×T4, no training; repeating reads will not prove L4 entitlement.

## Engineering experiments after forward attribution

**Baseline / instrumentation.** Attribute each of the original system calls: MLM, teacher, dedicated semantic, four full-fabric views, natural relation. Include transient peak and retained end-of-call allocation; no-gradient figures cannot stand in for backward peak. Identify biggest four-view components before changing the implementation. Preserve source, all eight public training lanes, ten macro families, teacher batch 2, microbatch 1, replay length 512, 18 stress pairings, FP16 and 85% cap.

**Activation candidate.** Selectively rematerialize full-fabric subgraphs where the trace and later legal backward evidence indicate retained activations dominate. Keep `FullEnvelopeJointTrainingObjectiveV1` and its complete one-step graph; do not turn four views or macro families into independent optimizer steps. With Torch DDP `find_unused_parameters=True` and repeated module calls, verify `use_reentrant=False` explicitly and check the *installed* Transformers checkpoint behavior rather than assuming a particular version. PyTorch documents stronger DDP restrictions for reentrant checkpointing. Compare exact unsplit loss and per-parameter gradients on a reproducible safe CPU/CUDA fixture, including dropout/RNG, EMA state, all weights, unused/frozen params, and stress extrema.

**State candidate.** FSDP/ZeRO optimizer-state sharding or CPU offload can preserve the full model, but the exact configured DDP receipt does not cover it. Two-rank sharding of gradients and Adam state has an optimistic savings bound of ~1,456,185,744 bytes per rank before communication/all-gathers, leaving ~3.008 GB of the T4 gap and ~5.436 GB of the P100 gap under the old formula. Offloading Adam moments alone has an optimistic ~1,941,580,992-byte bound, before transfer/allocator costs. Do not add these hypothetical savings as if independent. Require distributed loading, optimizer step, full checkpoint and sharded state persistence, correct same-stage resume and J1→J2→J3 DEV-selection semantics.

**Acceptance.** After implementing an isolated versioned source branch, get fresh exact-head P42 and P39PN bindings, prove full-joint gradient equivalence, then run a newly specified two-rank **whole-route** memory qualification on actual GPUs across all 18 cases. A legally authorized representative optimizer step must record allocated/reserved and driver memory, stability across accumulation and 25-step state readback where the authority contract permits; stop on numerical failures or OOM and preserve roots. The old a19 P43 failure remains attached. Its status cannot be flipped by editing `backward_activation_multiplier_over_no_grad_peak_delta`, `projected_training_fraction_max`, cutting public lanes, or using a one-rank profiling result.

**Fallback.** If two L4s cannot actually be allocated and a complete T4/P100 route does not pass under the full obligations, choose a verified two-GPU high-memory provider with live cost and durable checkpoint storage; obtain owner approval before launching billed compute. Multiple time-sliced sessions solve quota/duration only after a full step fits.

Primary documentation: [Kaggle CLI accelerator and restrictions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md); [PyTorch DDP checkpoint restrictions](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html); [PyTorch checkpoint modes](https://docs.pytorch.org/docs/stable/checkpoint.html); [FSDP](https://docs.pytorch.org/docs/main/fsdp.html); [CUDA memory snapshots](https://docs.pytorch.org/docs/stable/torch_cuda_memory).

## Guarded Magnolia submission (no gradient)

Run on the Magnolia login shell with the already audited `a19f8e88` mixture and teacher bank. The script creates a **fresh diagnostic root**; previous roots remain untouched. `--nodelist=gpu001` targets the observed P100×2 node so a K80 allocation cannot be mistaken for the intended test.

```bash
(
  set -euo pipefail
  cd "$HOME/rayan-compute/rayan-eipm-main"
  SHA=a19f8e8893422702c138182f239064385addf91c
  WORK="$HOME/rayan-compute/rayan-n0/n0-v02"
  MIXTURE="$WORK/full-public-mixture-a19f8e88-v1"
  TEACHER_REGISTRY="$WORK/teacher-bank-v0.5/n0_v02_teacher_bank_v0.5.runtime.json"
  TEACHER_AUDIT="$WORK/teacher-bank-v0.5/teacher-bank-v0.5-audit.json"
  TRACE="$WORK/full-envelope-j3-forward-attribution-a19f8e88-v1"
  LOGDIR="$HOME/rayan-compute/rayan-n0/slurm/j3-forward-attribution-a19f8e88-v1"
  PROFILE="$WORK/n0_p43_j3_forward_attribution_v1.py"
  BATCH="$WORK/n0_p43_j3_forward_attribution_v1.sbatch"
  test "$(git rev-parse HEAD)" = "$SHA"
  test -z "$(git status --porcelain)"
  test -f "$MIXTURE/full_public_mixture_manifest.json"
  test -f "$TEACHER_REGISTRY"
  test -f "$TEACHER_AUDIT"
  test ! -e "$TRACE"
  git fetch origin alice-context
  git show 4cd9f9860a43f923dd8d272fccfd8e62fc9aad29:docs/chat-context/2026-09-27/n0_p43_j3_forward_attribution_v1.py > "$PROFILE"
  git show 4cd9f9860a43f923dd8d272fccfd8e62fc9aad29:docs/chat-context/2026-09-27/n0_p43_j3_forward_attribution_v1.sbatch > "$BATCH"
  printf '%s  %s\n' b2d3af77466b3d861e1dc25e6e8a3701fe3e07e4f16e1a1a6dfd19fcc865e8b2 "$PROFILE" |
    sha256sum -c -
  printf '%s  %s\n' 26d5a32d2a7c2cdf83515d0912f68dcd479c816b4a52b449b231c7d351ad532b "$BATCH" |
    sha256sum -c -
  mkdir -p "$LOGDIR"
  export ALICE_N0_WORKDIR="$WORK"
  export ALICE_N0_FULL_MIXTURE_ROOT="$MIXTURE"
  export ALICE_N0_TEACHER_REGISTRY="$TEACHER_REGISTRY"
  export ALICE_N0_TEACHER_AUDIT="$TEACHER_AUDIT"
  export ALICE_N0_FORWARD_TRACE_SCRIPT="$PROFILE"
  export ALICE_N0_FORWARD_TRACE_ROOT="$TRACE"
  JOBID="$(sbatch --parsable --nodelist=gpu001 \
    --output="$LOGDIR/forward-%j.out" \
    --error="$LOGDIR/forward-%j.err" \
    --export=ALL "$BATCH")"
  JOBID="${JOBID%%;*}"
  echo "JOBID=$JOBID"
  echo "TRACE=$TRACE"
  echo "LOGDIR=$LOGDIR"
)
```

After the job leaves the queue, inspect `sacct -j "$JOBID" --format=JobID,State,ExitCode,Elapsed,NodeList`, the `forward-$JOBID.{out,err}` logs, `$TRACE/diagnostic.json`, and `$TRACE/forward_trace.json`; preserve the entire root whether it passes or fails. An SBATCH submission or successful syntax check does not confer P43 or gradient authority. Actual numeric findings must be inspected before selecting a rematerialization site.
