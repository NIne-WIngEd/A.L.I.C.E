#!/usr/bin/env bash
# Run on Magnolia login node, one phase at a time after inspecting each receipt.
# This handoff is outside the exact-source N0 repository and never trains N0.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != cpu && "$MODE" != mixture && "$MODE" != gpu ]]; then
  echo "usage: bash $0 cpu|mixture|gpu" >&2
  exit 64
fi

SHA=5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2
REPO="${ALICE_N0_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORK="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
FEWREL="${ALICE_N0_FEWREL_ROOT:-$HOME/rayan-compute/rayan-n0/sources/FewRel}"
CPU_ROOT="${ALICE_N0_CPU_RUNTIME_ROOT:-$WORK/full-envelope-cpu-runtime-5e29f7f-v1}"
MIXTURE="${ALICE_N0_FULL_MIXTURE_ROOT:-$WORK/full-public-mixture-5e29f7f-v1}"
EVIDENCE="${ALICE_N0_MEASURED_JOINT_ROOT:-$WORK/measured-joint-5e29f7f-v1}"
TEACHER="${ALICE_N0_TEACHER_REGISTRY:-$WORK/teacher-bank-v0.5/n0_v02_teacher_bank_v0.5.runtime.json}"
TEACHER_AUDIT="${ALICE_N0_TEACHER_AUDIT:-$WORK/teacher-bank-v0.5/teacher-bank-v0.5-audit.json}"
CHECK="$WORK/n0_5e29_exact_head_handoff_check_v1.py"
LOGDIR="$HOME/rayan-compute/rayan-n0/slurm/exact-head-5e29f7f-$MODE"
RECORD="$WORK/n0-exact-head-5e29f7f-$MODE-submission.txt"

cd "$REPO"
if [[ "$(git rev-parse HEAD)" != "$SHA" || -n "$(git status --porcelain)" ]]; then
  echo "STOP: expected fully clean source at $SHA" >&2
  exit 90
fi
if [[ ! -f "$CHECK" ]]; then
  echo "STOP: missing versioned preflight checker: $CHECK" >&2
  exit 91
fi
if [[ -e "$RECORD" ]]; then
  echo "STOP: submission already attempted; inspect record before planning another attempt: $RECORD" >&2
  exit 92
fi

export ALICE_N0_REPO_ROOT="$REPO"
export ALICE_N0_WORKDIR="$WORK"
export ALICE_N0_EXPECTED_REVISION="$SHA"
export ALICE_N0_CPU_RUNTIME_ROOT="$CPU_ROOT"
export ALICE_N0_FULL_MIXTURE_ROOT="$MIXTURE"
export ALICE_N0_MEASURED_JOINT_ROOT="$EVIDENCE"
export ALICE_N0_TEACHER_REGISTRY="$TEACHER"
export ALICE_N0_TEACHER_AUDIT="$TEACHER_AUDIT"
export ALICE_N0_FEWREL_ROOT="$FEWREL"

# The checker runs in the existing CPU container; the mounted WORK contains
# this script. FewRel is passed explicitly because udocker's allow-list does
# not forward ALICE_N0_FEWREL_ROOT.
RAYAN_UDOCKER_NVIDIA=0 bash "$REPO/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  -lc 'set -euo pipefail; export ALICE_N0_FEWREL_ROOT="$1"; python "$2" "$3"' \
  _ "$FEWREL" "$CHECK" "$MODE"

mkdir -p "$LOGDIR"
if [[ "$MODE" == cpu ]]; then
  JOB_NAME=rayan-n0-full-cpu
  JOB_SCRIPT="$REPO/scripts/eipm/n0/magnolia_cpu_n0_v02_full_envelope_runtime_v1.sbatch"
  OPTIONS=(--output="$LOGDIR/cpu-%j.out" --error="$LOGDIR/cpu-%j.err")
elif [[ "$MODE" == mixture ]]; then
  JOB_NAME=rayan-n0-p39pn-5e29
  JOB_SCRIPT="$WORK/n0_exact_head_p39pn_5e29f7f_v1.sbatch"
  if [[ -e "$JOB_SCRIPT" ]]; then
    echo "STOP: preserve existing CPU materialization job script: $JOB_SCRIPT" >&2
    exit 93
  fi
  cat > "$JOB_SCRIPT" <<'BATCH'
#!/usr/bin/env bash
set -euo pipefail
export RAYAN_UDOCKER_NVIDIA=0
bash "$ALICE_N0_REPO_ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc '
  set -euo pipefail
  export ALICE_N0_REPO_ROOT="$1"
  export ALICE_N0_WORKDIR="$2"
  export ALICE_N0_EXPECTED_REVISION="$3"
  export ALICE_N0_FEWREL_ROOT="$4"
  export ALICE_N0_TEACHER_REGISTRY="$5"
  export ALICE_N0_TEACHER_AUDIT="$6"
  export ALICE_N0_CPU_RUNTIME_ROOT="$7"
  export ALICE_N0_FULL_MIXTURE_ROOT="$8"
  bash "$1/scripts/eipm/n0/materialize_n0_v02_full_public_mixture_v1.sh"
' _ "$ALICE_N0_REPO_ROOT" "$ALICE_N0_WORKDIR" \
  "$ALICE_N0_EXPECTED_REVISION" "$ALICE_N0_FEWREL_ROOT" \
  "$ALICE_N0_TEACHER_REGISTRY" "$ALICE_N0_TEACHER_AUDIT" \
  "$ALICE_N0_CPU_RUNTIME_ROOT" "$ALICE_N0_FULL_MIXTURE_ROOT"
BATCH
  chmod 700 "$JOB_SCRIPT"
  OPTIONS=(--partition=node --nodes=1 --ntasks=1 --cpus-per-task=12
    --mem=64G --time=06:00:00
    --output="$LOGDIR/mixture-%j.out" --error="$LOGDIR/mixture-%j.err")
else
  JOB_NAME=rayan-n0-joint-v2
  JOB_SCRIPT="$REPO/scripts/eipm/n0/magnolia_p100x2_n0_v02_measured_joint_memory_v2.sbatch"
  OPTIONS=(--output="$LOGDIR/gpu-%j.out" --error="$LOGDIR/gpu-%j.err")
  JOB_LIST="$(squeue -u "$USER" -h -o '%j')" || {
    echo "STOP: scheduler queue could not be inspected" >&2
    exit 94
  }
  if awk '$0=="rayan-n0-joint-v2" { found=1 } END { exit !found }' <<< "$JOB_LIST"; then
    echo "STOP: another full-route GPU qualification is already queued or running" >&2
    exit 96
  fi
fi

# A durable attempt record blocks accidental resubmission even when a queued
# job has finished before the owner inspects its result.
(set -C; printf 'source=%s\nmode=%s\nscript=%s\nstatus=submission_attempted\n' \
  "$SHA" "$MODE" "$JOB_SCRIPT" > "$RECORD")
if JOBID="$(sbatch --parsable --job-name="$JOB_NAME" "${OPTIONS[@]}" \
  --export=ALL "$JOB_SCRIPT")"; then
  JOBID="${JOBID%%;*}"
  printf 'job_id=%s\n' "$JOBID" >> "$RECORD"
  printf 'JOBID=%s\nMODE=%s\nSOURCE=%s\nLOGDIR=%s\nRECORD=%s\n' \
    "$JOBID" "$MODE" "$SHA" "$LOGDIR" "$RECORD"
else
  printf 'status=submission_rejected\n' >> "$RECORD"
  echo "STOP: Slurm rejected submission; inspect preserved record" >&2
  exit 95
fi
