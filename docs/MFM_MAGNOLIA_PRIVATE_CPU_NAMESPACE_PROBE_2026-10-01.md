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
JOBID="$(sbatch --parsable --export=ALL,MFM_REPO_ROOT="$MFM_REPO_ROOT" \
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
differs from `host_netns`. The job uses a small host Python 3.11 helper to
create user and network namespaces together, maps the owner's UID/GID to the
same nonzero values before exec, and closes inherited file descriptors above
standard streams. The actual P2 process must separately pass the loopback
check. Any failure deliberately blocks the private teacher CPU pass. Do not
rerun the public 21-case job as a substitute. Keep the trainer's loopback-only
guard in place.

Jobs 576516 and 576517 both failed with exit 3 on node005 without opening a
private corpus. The former mapped the owner to UID 0, so udocker refused; the
latter found util-linux 2.23.2, which lacks `--map-current-user`, and direct
`--net` returned `Operation not permitted`. The new helper calls the Linux
`unshare` syscall directly and writes a single same-UID map, `setgroups=deny`,
and a same-GID map in the process before it executes bash.

The owner-reported **job 576551** ran that helper on node005 at MFM commit
`6bddc31e724edee6485ad9306148dd0a390b8858`. Slurm reported
`COMPLETED 0:0` in five seconds with empty stderr. The host network namespace
was `net:[4026531956]`. The helper kept UID/GID `1905/100` and entered
`net:[4026532988]` with only `lo`. The actual P2 process reported that same
new namespace and `p2_interfaces=lo`. This is a successful **no-data P2
network capability check on node005**. It is based on pasted accounting and
logs; the original Magnolia files were not independently fetched. It does
not establish complete privacy, source rights, corpus admission, a processor
receipt or learned formation behavior. The teacher job repeats the P2 check
on its assigned node before opening any private corpus input.

After a passing capability probe, a separate private-corpus job must still
pin its own branch head, source rights and target provenance, complete v1.6
manifest and prepared-base receipt. It must enter the same tested namespace
*before* `data-preflight` or `processor-preflight` opens any source bytes. The
public diagnostic `.sbatch` hardcodes a different 21-case corpus and is not
the template for a full-corpus receipt.
