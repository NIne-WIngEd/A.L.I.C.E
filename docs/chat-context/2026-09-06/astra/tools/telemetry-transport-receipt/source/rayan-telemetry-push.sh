#!/bin/bash
set -euo pipefail
umask 077
[ "$#" -eq 7 ] || { echo "usage: RUN_ID STAGE STATUS JOB_ID LOG MANIFEST RECEIPT" >&2; exit 64; }
RUN_ID="$1"; STAGE="$2"; STATUS="$3"; JOB_ID="$4"; LOG="$5"; MANIFEST="$6"; RECEIPT="$7"
case "$RUN_ID" in *[!A-Za-z0-9._-]*|"") exit 65;; esac
case "$STAGE" in *[!A-Za-z0-9._-]*|"") exit 66;; esac
case "$STATUS" in COMPLETED|FAILED|CANCELLED|RUNNING) ;; *) exit 67;; esac
case "$JOB_ID" in *[!0-9]*|"") exit 68;; esac
for f in "$LOG" "$MANIFEST" "$RECEIPT"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 69; }; done
ROOT="$HOME/rayan-compute"; GH="$ROOT/runtime/github"; LEDGER="$ROOT/telemetry/ledger"; SROOT="$ROOT/telemetry/spool"; LOCK="$ROOT/telemetry/push.lock"
KEY="$GH/rayan_compute_ledger_ed25519"; KNOWN="$GH/known_hosts"; SPOOL="$SROOT/$RUN_ID"
export GIT_SSH="$ROOT/bin/rayan-github-ssh"
export GIT_SSH_VARIANT=ssh
mkdir -p "$SROOT"; chmod 700 "$SROOT"; rm -rf "$SPOOL"; mkdir -p "$SPOOL"; chmod 700 "$SPOOL"
cp "$MANIFEST" "$SPOOL/manifest.json"; cp "$RECEIPT" "$SPOOL/receipt.json"
BYTES="$(wc -c < "$LOG" | tr -d ' ')"
if [ "$BYTES" -le 45000000 ]; then cp "$LOG" "$SPOOL/application.log"; else mkdir -p "$SPOOL/log-parts"; split -b 45000000 -d -a 4 "$LOG" "$SPOOL/log-parts/application.log.part."; (cd "$SPOOL/log-parts" && sha256sum application.log.part.* > SHA256SUMS); fi
printf '{\n  "schema": "rayan.compute.telemetry.v1",\n  "run_id": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "provider": "magnolia",\n  "slurm_job_id": "%s",\n  "log_bytes": %s,\n  "published_at": "%s"\n}\n' "$RUN_ID" "$STAGE" "$STATUS" "$JOB_ID" "$BYTES" "$(date -Is)" > "$SPOOL/telemetry.json"
if grep -R -E -q 'GOCSPX-|github_pat_|ghp_|-----BEGIN [A-Z ]*PRIVATE KEY-----' "$SPOOL"; then echo "TELEMETRY_SECRET_SCAN_FAILED=true" >&2; exit 70; fi
(cd "$SPOOL" && find . -type f ! -name bundle.sha256 -print0 | sort -z | xargs -0 sha256sum > bundle.sha256)
acq=false; for _ in $(seq 1 120); do if mkdir "$LOCK" 2>/dev/null; then acq=true; break; fi; sleep 1; done; [ "$acq" = true ] || exit 71
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
[ -d "$LEDGER/.git" ] || exit 72
ok=false
for attempt in 1 2 3 4 5; do
  cd "$LEDGER"; git fetch origin main >/dev/null 2>&1 || { sleep 2; continue; }; git checkout -B main origin/main >/dev/null 2>&1
  DEST="runs/$RUN_ID"
  if [ -f "$DEST/bundle.sha256" ]; then if cmp -s "$DEST/bundle.sha256" "$SPOOL/bundle.sha256"; then echo "telemetry_idempotent=true"; ok=true; break; else echo "RUN_ID_COLLISION=true" >&2; exit 73; fi; fi
  mkdir -p "$DEST"; cp -R "$SPOOL/." "$DEST/"; git add "$DEST"
  if git diff --cached --quiet; then ok=true; break; fi
  git commit -m "telemetry: $RUN_ID $STATUS" >/dev/null
  if git push origin main >/dev/null 2>&1; then ok=true; break; fi
  git reset --hard HEAD~1 >/dev/null 2>&1 || true; sleep 2
done
[ "$ok" = true ] || exit 74
cd "$LEDGER"; git fetch origin main >/dev/null 2>&1; git show "origin/main:runs/$RUN_ID/bundle.sha256" >/dev/null 2>&1 || exit 75
echo "RAYAN_GITHUB_TELEMETRY_PUSHED=true"; echo "run_id=$RUN_ID"; echo "status=$STATUS"
