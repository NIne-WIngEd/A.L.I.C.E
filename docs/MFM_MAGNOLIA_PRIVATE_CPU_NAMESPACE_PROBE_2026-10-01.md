# Magnolia MFM private CPU namespace capability probe

`scripts/mfm/magnolia_private_network_namespace_probe.sbatch` is a separate,
five-minute `node` partition job. It checks whether the existing `rayan-n0-base`
P2 udocker container can start inside a new loopback-only Linux network
namespace. It reads no MFM corpus, model weight or private source and does not
run a processor. A successful result establishes only this runtime capability;
it is not a data admission or training receipt.

After this script is present in the Magnolia MFM checkout, submit it from the
login node. Keep logs under the owner-controlled compute root:

```bash
MFM_REPO_ROOT="$HOME/rayan-compute/mfm/alice-mfm-v16-412ae32a"
MFM_LOGDIR="$HOME/rayan-compute/mfm/slurm"
mkdir -p "$MFM_LOGDIR"
JOBID="$(sbatch --parsable \
  --output="$MFM_LOGDIR/rayan-mfm-netns-probe-%j.out" \
  --error="$MFM_LOGDIR/rayan-mfm-netns-probe-%j.err" \
  "$MFM_REPO_ROOT/scripts/mfm/magnolia_private_network_namespace_probe.sbatch")"
JOBID="${JOBID%%;*}"
echo "JOBID=$JOBID"
```

When Slurm finishes, read accounting and the small output:

```bash
sacct -j "$JOBID" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,NodeList
cat "$MFM_LOGDIR/rayan-mfm-netns-probe-${JOBID}.out"
cat "$MFM_LOGDIR/rayan-mfm-netns-probe-${JOBID}.err"
```

`COMPLETED 0:0` counts only if `p2_interfaces=lo` appears and `p2_netns`
differs from `host_netns`. The job tries rootless user-plus-network `unshare`
and then direct network `unshare`; neither route uses private evidence. If
both fail, exit 3 deliberately blocks the private teacher CPU pass on this
runtime. Do not rerun the public 21-case job as a substitute. Keep the
trainer's loopback-only guard in place. Test an administrator-approved isolated
runtime separately if Magnolia denies namespace creation.

Job 576516 failed with exit 3 in two seconds on node005. The first route,
`--map-root-user`, reached udocker with UID 0 and udocker refused to run; the
direct `--net` route returned `Operation not permitted`. It opened no private
corpus and does **not** prove that user-plus-network namespaces are denied.
The revised probe checks Magnolia's `unshare` version and support for
`--map-current-user`, maps the caller's UID/GID to the same nonzero values,
and verifies the P2 network namespace and interfaces. Some older `unshare`
versions lack that flag. If it is absent or the P2 check fails, the job still
exits 3; inspect its log before trying another runtime. Neither the failed
job nor the revised code authorizes the private CPU pass. Use the revised
teacher CPU script only after this no-data probe passes on Magnolia.

After a passing capability probe, a separate private-corpus job must still
pin its own branch head, source rights and target provenance, complete v1.6
manifest and prepared-base receipt. It must enter the same tested namespace
*before* `data-preflight` or `processor-preflight` opens any source bytes. The
public diagnostic `.sbatch` hardcodes a different 21-case corpus and is not
the template for a full-corpus receipt.
