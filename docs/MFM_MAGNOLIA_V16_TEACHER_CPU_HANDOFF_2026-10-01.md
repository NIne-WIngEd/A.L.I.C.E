# Magnolia full teacher-corpus CPU handoff

`scripts/mfm/magnolia_v16_teacher_cpu_preflight.sbatch` is separate from the
21-case public diagnostic. It requires an owner-authorized
`mfm-v16-owner-teacher-corpus-v1` manifest with authenticated source rights,
exact target provenance and distinct train/diagnostic-development lineage.
The [issued synthetic-only manifest](MFM_V16_CHAT_DIRECTED_SYNTHETIC_CORPUS_2026-10-01.md)
admits 16 fictional train and nine fictional diagnostic development cases
under the owner's chat direction. Its manifest SHA-256 is
`4e59403ee2205d5219eed09a1dfd712c87b4bbae09dc0435a2275aac18ca7573`.
This is a bounded teacher processor run, not a full capability corpus or
independent gold. There is **no teacher CPU receipt yet**. Do not use the
earlier unissued candidates, AMI/ICSI teacher candidates or old v1.5 mixture.

Before submission, record the following values in owner custody independently
of the job. Export them on the Magnolia login node. All paths must be absolute
and inside `$HOME/rayan-compute`; keep the corpus outside the Git checkout.

| Environment variable | Required value |
| --- | --- |
| `MFM_REPO_ROOT`, `MFM_REPO_COMMIT` | Clean v1.6 checkout and exact 40-digit commit |
| `MFM_FOUNDATION_ROOT`, `MFM_FOUNDATION_COMMIT` | Clean pinned Gemma foundation checkout and exact commit |
| `MFM_TEACHER_MANIFEST`, `MFM_TEACHER_MANIFEST_SHA256` | Frozen teacher manifest and externally recorded SHA-256 |
| `MFM_OWNER_AUTH_REF` | Exact owner authorization reference in every case |
| `MFM_PREPARED_BASE_DIR`, `MFM_PREPARED_BASE_RECEIPT` | Existing local MFM role clone and sealed receipt |
| `MFM_PREPARED_BASE_RECEIPT_FILE_SHA256` | Externally recorded SHA-256 of the receipt file bytes |
| `MFM_PREFLIGHT_RECEIPT` | Fresh, nonexistent absolute output path; its parent exists |

The two exact local Pillow and TorchVision wheels used in job 576510 must
remain under `$HOME/rayan-compute/mfm/wheels`. The job rehashes them and
installs them into a private job-specific directory. The `rayan-n0-base` P2
container must already be in the local udocker store. Nothing is pulled from
the network. The job pins Torch 2.7.1+cu118, Transformers 5.17.0, Pillow
11.3.0 and TorchVision 0.22.1+cu118 before verifying the 23.9 GB role clone.
Media cases additionally require working ffmpeg/ffprobe and decoder support;
the earlier text-only diagnostic did not establish that.

Run the separate [no-data namespace probe](MFM_MAGNOLIA_PRIVATE_CPU_NAMESPACE_PROBE_2026-10-01.md)
first. Once it passes, set the required variables above from the frozen record
and submit the teacher job, with logs outside Git:

Jobs 576516 and 576517 did not pass. The first reached udocker as UID 0; the
second found that util-linux 2.23.2 lacks the same-user mapping option. Job
576551 then passed the no-data P2 network check on node005 at MFM `6bddc31`:
the P2 process reported only `lo` in a namespace distinct from the host.
This clears the measured network capability gate for that node. The teacher
script uses the same helper and repeats the P2 check on its assigned node
before opening private input. The synthetic-only manifest is now issued, but
must be transferred, hashed and pinned on Magnolia along with all the other
inputs above. No teacher corpus CPU job has been submitted.

```bash
MFM_LOGDIR="$HOME/rayan-compute/mfm/slurm"
mkdir -p "$MFM_LOGDIR"
JOBID="$(sbatch --parsable --export=ALL \
  --output="$MFM_LOGDIR/rayan-mfm-v16-teacher-cpu-%j.out" \
  --error="$MFM_LOGDIR/rayan-mfm-v16-teacher-cpu-%j.err" \
  "$MFM_REPO_ROOT/scripts/mfm/magnolia_v16_teacher_cpu_preflight.sbatch")"
JOBID="${JOBID%%;*}"
echo "JOBID=$JOBID"
```

The Slurm defaults are four CPUs, 32 GB and two hours on `node`. Override
`--mem` and `--time` at submission only after sizing the actual frozen corpus;
those defaults are not a training fit estimate. Then inspect accounting and
the receipt under owner custody:

```bash
sacct -j "$JOBID" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,NodeList
cat "$MFM_LOGDIR/rayan-mfm-v16-teacher-cpu-${JOBID}.out"
cat "$MFM_LOGDIR/rayan-mfm-v16-teacher-cpu-${JOBID}.err"
```

The script checks a fresh Linux network namespace, then verifies the actual
P2 container sees only `lo` before opening the manifest. If namespace
creation fails, it stops. A failure after corpus access never triggers a
fallback. Within isolation it verifies the exact clean checkouts, corpus SHA,
role receipt SHA and pinned wheels; runs `data-preflight --teacher-fit`; then
runs `processor-preflight --teacher-fit` over **every** admitted train and
development case. FINAL is not opened by the training lane. The v1.6 trainer
rehashes the entire prepared base and writes a new sealed CPU receipt. It
rejects exceeded source or target budgets without truncation. Optional
`MFM_MAX_SOURCE_TOKENS`, `MFM_MAX_TARGET_TOKENS`, `MFM_SPECIALIST_WIDTH`,
`MFM_SPECIALIST_LAYERS` and `MFM_SPECIALIST_HEADS` may be exported before
submission; absent values use the current trainer defaults and enter the
receipt. If budgets change, run a new CPU preflight before any GPU probe.

`COMPLETED 0:0` plus a validated new receipt proves only the processor/data
contract on those exact corpus and code bytes. It does not prove training,
independent FINAL performance, product capability or GPU fit. Do not buy GPU
time from this script alone.
