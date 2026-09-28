# N0 5e29 exact-head CPU → mixture → one measured GPU handoff

Status (2026-09-28): **prepared but not run on Magnolia**. This ChatGPT
workspace has no mounted Magnolia corpus or `sbatch`, and `magnolia01` does
not resolve here. The pinned scientific source is
`alice-eipm-v1-n0-full-envelope-joint-ddp-v2` at
`5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2`.
[Exact-head CPU CI](https://github.com/NIne-WIngEd/A.L.I.C.E/actions/runs/36368390448)
passed 125 tests and skipped its single CUDA-only case. That workflow does
not generate Magnolia's source-bound P42 or P39PN receipts.

The two accompanying files are an operator handoff, not part of the scientific
source: `n0_5e29_exact_head_handoff_v1.sh` and
`n0_5e29_exact_head_handoff_check_v1.py`. They invoke the existing canonical
CPU proof runner, public-mixture materializer and full-route GPU wrapper.
The checker compares the exact source, producer hashes, CPU receipt family,
public-mixture manifest/audit and lane hashes before submitting later phases.
Each phase gets a fresh output root and a durable submission record. The
GPU phase submits **at most one** 10-hour, two-P100 Magnolia job; it does not
turn a no-gradient projection FAIL into PASS or authorize production training.

## On Magnolia: pin source and obtain the handoff

Run from the login node. Stop if your source worktree is dirty. These
commands use `cd` and standard Git subcommands; Magnolia's older Git does
not support the global `git -C` option.

```bash
set -euo pipefail
REPO="$HOME/rayan-compute/rayan-eipm-main"
WORK="$HOME/rayan-compute/rayan-n0/n0-v02"
SHA=5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2
cd "$REPO"
test -z "$(git status --porcelain)"
git fetch origin alice-eipm-v1-n0-full-envelope-joint-ddp-v2 alice-context
git checkout --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain)"

git show origin/alice-context:docs/chat-context/2026-09-28/n0_5e29_exact_head_handoff_v1.sh \
  > "$WORK/n0_5e29_exact_head_handoff_v1.sh"
git show origin/alice-context:docs/chat-context/2026-09-28/n0_5e29_exact_head_handoff_check_v1.py \
  > "$WORK/n0_5e29_exact_head_handoff_check_v1.py"
printf '%s  %s\n' df7db4340c843c0ff891b9e19cb57aea3943dac67d39fd745bf44e46ec9c6817 \
  "$WORK/n0_5e29_exact_head_handoff_v1.sh" | sha256sum -c -
printf '%s  %s\n' e715af3266babb31594ee0343de94167544340ea59733e2cb730b5b2cc55a1cc \
  "$WORK/n0_5e29_exact_head_handoff_check_v1.py" | sha256sum -c -
bash -n "$WORK/n0_5e29_exact_head_handoff_v1.sh"
```

The handoff defaults to the existing v0.2 tokenizer, corpus, semantic
checkpoint, FewRel and teacher-bank v0.5 paths. It creates new isolated roots:
`full-envelope-cpu-runtime-5e29f7f-v1`,
`full-public-mixture-5e29f7f-v1`, and
`measured-joint-5e29f7f-v1` under `$WORK`. Occupied roots are preserved and
stop submission. The old `a19f8e88` and other failed evidence are untouched.

## Phase 1: fresh P42/static/CPU receipts

```bash
bash "$WORK/n0_5e29_exact_head_handoff_v1.sh" cpu
```

This submits the existing 6-hour CPU Slurm runner, which executes the
registered static proof suite and builds tokenizer, operator evidence,
long-context and full-envelope CPU receipts on the pinned head. Record the
printed job ID. Wait for `sacct` to show `COMPLETED 0:0`, then read its stdout
and stderr before advancing. The next phase revalidates each receipt and
producer hash. Do not rename or copy older CPU roots to these paths.

## Phase 2: fresh public mixture and sealed FINAL package

```bash
bash "$WORK/n0_5e29_exact_head_handoff_v1.sh" mixture
```

This submits one 6-hour CPU materialization job. It uses the original
`materialize_n0_v02_full_public_mixture_v1.sh` within udocker and passes
FewRel, teacher and exact-head CPU roots explicitly. It publishes the public
mixture only after its behavioral, view, natural relation and sealed FINAL
package audits pass. Inspect `sacct`, stdout and stderr after completion.
This materialization freezes the FINAL package without opening or evaluating
FINAL and does not train the model.

## Phase 3: one bounded two-rank J3 proof

```bash
bash "$WORK/n0_5e29_exact_head_handoff_v1.sh" gpu
```

The GPU submission is refused unless the fresh CPU and mixture receipts,
their canonical producer hashes, teacher/input hashes, all six materialized
stress-lane row hashes and source revision match this head. It also refuses
an occupied GPU evidence root, an earlier submission record, or another
queued/running job with the same name. The existing `.sbatch` wrapper asks
for `gpu001` with two P100s and a 10-hour limit. The N0 runner first records
the original no-gradient P43 outcome. It then attempts the intact full J3
two-rank, 18-pair, eight-microbatch, first/later AdamW and checkpoint/resume
qualification at the original 85% per-rank threshold. A failure remains a
failure. Do not submit a second shape or platform automatically.

After any phase, use the `job_id` in
`$WORK/n0-exact-head-5e29f7f-<phase>-submission.txt`:

```bash
PHASE=cpu  # change to mixture or gpu for that phase
JOBID="$(awk -F= '$1=="job_id" { print $2 }' \
  "$WORK/n0-exact-head-5e29f7f-${PHASE}-submission.txt")"
test -n "$JOBID"
sacct -j "$JOBID" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,NodeList
cat "$HOME/rayan-compute/rayan-n0/slurm/exact-head-5e29f7f-${PHASE}"/*"${JOBID}"*.out
cat "$HOME/rayan-compute/rayan-n0/slurm/exact-head-5e29f7f-${PHASE}"/*"${JOBID}"*.err
```

For a GPU result, also inspect the preserved root's
`no_gradient_projection.json`, `measured/result.json`,
`measured/rank-0.jsonl`, `measured/rank-1.jsonl`,
`measured/measured_gpu_memory_result.json` and
`measured/complete_joint_route_result.json` where present. A CPU PASS or
partial GPU trace does **not** authorize training. The training authorizer
still separately requires its full exact-head receipt chain, DEV stage
selection and unopened FINAL boundary.
