#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ALICE_N0_REPO_ROOT="$ROOT"
export RAYAN_UDOCKER_NVIDIA=0

cd "$ROOT"

echo "===== N0 V0.2 CPU-ONLY PREFLIGHT ====="
date -Is
echo "repo_root=$ROOT"
echo "gpu_requested=false"
echo "private_identity_gradient=false"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  "export PYTHONPATH='$ROOT/src'; python '$ROOT/scripts/eipm/n0/preflight_n0_v02.py' --repo-root '$ROOT'"

echo "===== N0 V0.2 CPU-ONLY PREFLIGHT COMPLETE ====="
date -Is
