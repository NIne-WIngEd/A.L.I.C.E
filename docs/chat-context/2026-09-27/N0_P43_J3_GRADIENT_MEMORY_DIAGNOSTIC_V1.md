# N0 P43 two-rank full-J3 gradient-memory diagnostic — 2026-09-27

## Question and authority

Job 576211 isolated a 3.657 GiB live-allocation forward peak in the long additional-view case. The old P43 16.535 GiB/rank training figure is a *projection* from inference mode. The production trainer already enables backbone gradient checkpointing, which inference mode cannot measure. This versioned experiment asks whether the **actual, intact J3 gradient graph** reaches backward within the unchanged 85% per-device limit on Magnolia P100×2, and whether the production DDP/checkpoint mode can perform the repeated forwards in one joint objective.

The diagnostic invokes the original, pinned `execute_full_envelope_joint_step` across the full 3 × 6 stress matrix. Every step includes MLM, teacher, semantic operator, four full-fabric views and natural relation with all ten J3 macro families. It uses the trainer's `Accelerator(mixed_precision="fp16")`, `DistributedDataParallelKwargs(find_unused_parameters=True)`, `accelerator.prepare(system)`, `accelerator.backward(loss)`, and *its unmodified* `enable_precommitted_gradient_checkpointing`. It checks runtime checkpoint mode before the first stress case. PyTorch documents nonreentrant checkpointing as supported with this DDP setting. If runtime mode is different or cannot be verified, the diagnostic fails before executing any J3 backward; do not adjust checkpoint arguments in this probe.

It creates **no optimizer**, runs **no optimizer step**, saves **no model weights**, and does **not authorize training or FINAL**. It hashes all system parameters after DDP setup and again after all cases to check that no weight was updated. Forward and backward CUDA allocated/reserved peaks are recorded separately per case and per rank. Driver use is sampled before, after forward, and after backward; it is not a true driver high-water mark. The 85% check uses the greater of PyTorch peak reservation and sampled driver use. The initial eight-microbatch accumulation, AdamW moment allocation, optimizer-step temporaries, DEV selection, 25-step checkpoint readback and resume are **not measured**. Even a complete, within-budget diagnostic is **not P43 PASS**.

The original P43 FAIL root, job 576211 forward root and N0 scientific commit remain untouched. A runtime DDP error, overflow or OOM is a finding about the complete route. Preserve rank progress and inspect it before deciding on a versioned architecture or hardware route; never turn this into a one-view or one-family success.

## Published artifacts

- Scientific N0 source: `a19f8e8893422702c138182f239064385addf91c`.
- Diagnostic: `docs/chat-context/2026-09-27/n0_p43_j3_gradient_memory_diagnostic_v1.py`, SHA-256 `4cb68fd590225e7c240e2e065f310bd07a36ab27de1e9356697e1dcb83ae39e2`.
- Slurm launcher: `docs/chat-context/2026-09-27/n0_p43_j3_gradient_memory_diagnostic_v1.sbatch`, SHA-256 `ae6f74e2da5cd4aa95dfacbc6e75f851bcf3183bb5cfe95e13678eb1df1e99e6`.
- Pin the committed artifact source with `git show d6d2ba1036b96bc970da1e426fa99f05f613a5d9:<path>`; this commit does not change the N0 scientific branch.

Local verification: Python compilation, CLI argument parse, shell syntax, literal script/launcher SHA binding, original qualifier and trainer hashes, static checks for the required full J3/backward path and absence of `optimizer.step()`. **The distributed CUDA path has not run.** In particular, the installed Magnolia Transformers checkpoint mode and the repeated-forward DDP backward remain empirical questions.

## Guarded Magnolia command

Run in the existing Magnolia login Bash shell. It creates one new evidence root and log directory. If any target already exists, stop and inspect it before choosing a new versioned root.

```bash
(
  set -euo pipefail
  cd "$HOME/rayan-compute/rayan-eipm-main"
  SHA=a19f8e8893422702c138182f239064385addf91c
  SOURCE=d6d2ba1036b96bc970da1e426fa99f05f613a5d9
  WORK="$HOME/rayan-compute/rayan-n0/n0-v02"
  MIXTURE="$WORK/full-public-mixture-a19f8e88-v1"
  TEACHER_REGISTRY="$WORK/teacher-bank-v0.5/n0_v02_teacher_bank_v0.5.runtime.json"
  TEACHER_AUDIT="$WORK/teacher-bank-v0.5/teacher-bank-v0.5-audit.json"
  OUT="$WORK/full-envelope-j3-gradient-memory-a19f8e88-v1"
  LOGDIR="$HOME/rayan-compute/rayan-n0/slurm/j3-gradient-memory-a19f8e88-v1"
  PROFILE="$WORK/n0_p43_j3_gradient_memory_diagnostic_v1.py"
  BATCH="$WORK/n0_p43_j3_gradient_memory_diagnostic_v1.sbatch"

  test "$(git rev-parse HEAD)" = "$SHA"
  test -z "$(git status --porcelain)"
  test -f "$MIXTURE/full_public_mixture_manifest.json"
  test -f "$MIXTURE/full_public_mixture_audit.json"
  test -f "$TEACHER_REGISTRY"
  test -f "$TEACHER_AUDIT"
  test ! -e "$OUT"
  test ! -e "$PROFILE"
  test ! -e "$BATCH"

  git fetch origin alice-context
  git show "$SOURCE:docs/chat-context/2026-09-27/n0_p43_j3_gradient_memory_diagnostic_v1.py" > "$PROFILE"
  git show "$SOURCE:docs/chat-context/2026-09-27/n0_p43_j3_gradient_memory_diagnostic_v1.sbatch" > "$BATCH"
  printf '%s  %s\n' 4cb68fd590225e7c240e2e065f310bd07a36ab27de1e9356697e1dcb83ae39e2 "$PROFILE" | sha256sum -c -
  printf '%s  %s\n' ae6f74e2da5cd4aa95dfacbc6e75f851bcf3183bb5cfe95e13678eb1df1e99e6 "$BATCH" | sha256sum -c -
  bash -n "$BATCH"
  export ALICE_N0_REPO_ROOT="$PWD"
  RAYAN_UDOCKER_NVIDIA=0 bash scripts/eipm/n0/magnolia_udocker_exec.sh -lc 'python -m py_compile "$1"' _ "$PROFILE"

  mkdir -p "$LOGDIR"
  export ALICE_N0_WORKDIR="$WORK"
  export ALICE_N0_FULL_MIXTURE_ROOT="$MIXTURE"
  export ALICE_N0_TEACHER_REGISTRY="$TEACHER_REGISTRY"
  export ALICE_N0_TEACHER_AUDIT="$TEACHER_AUDIT"
  export ALICE_N0_GRADIENT_DIAGNOSTIC_SCRIPT="$PROFILE"
  export ALICE_N0_GRADIENT_DIAGNOSTIC_ROOT="$OUT"
  JOBID="$(sbatch --parsable --nodelist=gpu001 \
    --output="$LOGDIR/gradient-%j.out" \
    --error="$LOGDIR/gradient-%j.err" \
    --export=ALL "$BATCH")"
  JOBID="${JOBID%%;*}"
  echo "JOBID=$JOBID"
  echo "OUT=$OUT"
  echo "LOGDIR=$LOGDIR"
)
```

Readout after submission (set `JOBID` to the printed number):

```bash
JOBID=REPLACE_WITH_SUBMITTED_JOB_ID
OUT="$HOME/rayan-compute/rayan-n0/n0-v02/full-envelope-j3-gradient-memory-a19f8e88-v1"
LOGDIR="$HOME/rayan-compute/rayan-n0/slurm/j3-gradient-memory-a19f8e88-v1"
sacct -j "$JOBID" --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS,NodeList
for file in "$LOGDIR/gradient-$JOBID.out" "$LOGDIR/gradient-$JOBID.err"; do
  if test -f "$file"; then
    printf '\n===== %s =====\n' "$file"
    cat "$file"
  fi
done
if test -d "$OUT"; then
  ls -lh "$OUT"
  for file in "$OUT"/rank-*.jsonl "$OUT"/diagnostic.json; do
    if test -f "$file"; then
      sha256sum "$file"
      cat "$file"
    fi
  done
fi
```

If rank logs show failure before case 18, treat it as incomplete. Preserve the root and Slurm logs. A `COMPLETE_TWO_RANK_GRADIENT_DIAGNOSTIC_NOT_P43_AUTHORITY` marker requires two ranks and all 18 cases but still leaves optimizer-time peak and training qualification unmeasured.
