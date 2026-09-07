# Read-only Magnolia provider and a2 revision observation

This diagnostic implements the next observation specified in Sol's master
handoff. It is independent of the Qwen execution package. It does not import or
run a workload, stage a package on Magnolia, submit a job, download a model,
change an ALICE state file or publish telemetry.

The controller uses the recorded native Windows SSH identity and sends a fixed
LF script through SSH stdin. It reuses the known Magnolia Python/shared-library
bootstrap. All scheduler commands are read-only. `sbatch` appears only in a
command-path lookup; it is never invoked.

The remote report records eleven bounded commands: time, hostname, username,
scheduler binary paths, two client versions, controller ping, the user-wide
active queue, exact historical source accounting, exact v103 a2 accounting,
and the failing source-job queue query for comparison. Each result retains
argv, timestamps, exit code and both streams. Base64 preserves exact bytes;
text fields are readable views. Truncation or incomplete reads are explicit.
A failed comparison never prevents collection of the other observations.

Both local and remote inventories are read before and after observation.
The local side includes both controller states, a2 controller diagnostics,
revision archives/intents/acknowledgements and hashes of the R13/R18 receipts.
The remote side includes a2 lifecycle and revision files, task markers and two
small package manifests. Unknown revision directories are listed; only the
specified filenames are read. Links, unreadable files, missing files and read
limits remain distinct. Raw model weights, unrelated Vault data, credentials,
private candidates and MC8 are outside the read set. CA/HTTPS checks are not
repeated because the prior trace already captured them successfully.

Only the launcher's new diagnostic directory in Downloads is written. The
result ZIP is returned to the owner. Raw diagnostic output is not automatically
published to GitHub. Existing receipts are observed; no authority is inferred
from their presence, a successful command, an empty queue or this diagnostic's
exit code.

Exit 0 means collection completed. It can include real scheduler errors under
`COLLECTED_PROVIDER_ERRORS`; that is useful evidence, not provider approval.
Exit 74 means transport, inventory, output limits or changing state left an
incomplete observation; return the ZIP anyway. Exit 76 is a local launcher or
package-verification stop; return its transcript. Do not rerun Qwen v103 or
change revision intent to clear a stop.

The nine offline tests use explicitly synthetic scheduler results. They check
raw byte retention, independent query results, SSH failure/timeout/parse errors,
state preservation and partially recorded revisions. They do not establish
Magnolia's actual error or qualify a new execution adapter.
