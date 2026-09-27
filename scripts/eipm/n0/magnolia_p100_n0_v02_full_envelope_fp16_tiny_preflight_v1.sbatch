#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-fp16-tiny
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?expected source revision required}"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: fp16 tiny preflight source drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "STOP: fp16 tiny preflight requires a fully clean checkout" >&2
  exit 91
fi

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAYAN_UDOCKER_NVIDIA=1

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc '
  set -euo pipefail
  python -c "import torch; assert torch.cuda.is_available()"
  python -m pytest -q \
    tests/eipm/test_n0_full_envelope_trainable_system_v1.py::test_full_trainable_system_autocast_additional_text_views_keep_gradients \
    tests/eipm/test_n0_full_envelope_trainable_system_v1.py::test_full_trainable_system_cuda_fp16_additional_text_views_no_gradient
'
