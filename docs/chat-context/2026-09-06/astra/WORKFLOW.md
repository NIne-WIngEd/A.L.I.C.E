# Working agreement and recovered lessons

Current override: follow [START_HERE](START_HERE.md) and the
[actual v104 result review](QWEN_V104_RESULT_REVIEW.json). A2 is closed after
initial telemetry failure. Collect the installed helper source next; earlier
ready/run instructions below are history.

Rayan explicitly switched ongoing ALICE work to Astra on 6 September 2026. The earlier Sol-default preference is superseded. Work in substantial coherent batches: recover evidence, make one bounded decision, implement it, verify the relevant failure modes, record the result, and leave one executable next step. Do not trade knowledge retention or final cognitive semantics for speed.

## Locations and authority

- Canonical release: `NIne-WIngEd/A.L.I.C.E`, `main`, frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.
- Continuity: `alice-context`, `docs/chat-context/README.md`; older source index: `2026-08-31/SOURCE_INDEX.md`.
- Operational history: `alice-mc10b-live`, `alice-mc10c-live`, `alice-telemetry`; compute receipts: `NIne-WIngEd/Rayan-Compute-Ledger`.
- User machine: `C:\A.L.I.C.E-main`; data/artifacts: `C:\ALICE_Vault`; Downloads contains versioned packages and launchers.
- Remote archive: `rayan_gdrive:Rayan-Compute`; Magnolia working root: `/homes/01/mxrayan/rayan-compute`.
- A telemetry heartbeat proves liveness, not useful scientific progress. Publish completed-stage summaries, hashes, failures and actual receipt locations. Never include credentials or private records.
- Keep context work in an isolated checkout. Do not switch, reset or clean the user's working main checkout. Never force-push context. Reconcile an advanced remote before publishing.

## Package and PowerShell workflow

Use a versioned ZIP and a small PowerShell launcher: SHA-256 verification, fresh extraction, compile gate, meaningful offline selftest, then one bounded action with a deterministic exit and receipt. Preserve this workflow instead of introducing unfamiliar single-file packaging. Resolve the user's Anaconda Python first, then the active Conda environment, then real python.exe applications. Python 3.13 compatibility matters; the user's Qualcomm laptop has no CUDA.

Use script files and argument arrays. Long inline PowerShell-to-SSH command chains have corrupted `$?`, awk `$1`, and newline escaping. Do not embed large scripts into interpolated shell strings. Preserve actual newlines in JSON and Markdown; do not publish literal backslash-n paragraph separators. Native Python exit codes must be captured explicitly rather than depending on PowerShell's treatment of stderr. Do not repurpose HOME or other system variables.

Only remove a disposable run folder created for that exact package. Keep receipts, raw result rows and source artifacts separate and immutable. Validate output before replacing canonical state. A safe recovery reads old outputs; it does not silently overwrite qualified bindings while discovering whether they are valid.

## Compute allocation

Prefer local or Magnolia CPU for extraction, validation, scoring, data preparation and analysis. Use Magnolia for supported jobs whenever it avoids spending scarce Kaggle GPU time. Verify the requested hardware, model footprint and scheduler authorization before placement.

Recovered Magnolia context describes 20-core/128-GB CPU nodes and an earlier 48-GiB, 20-core exclusive request. The authorized P100 allocation exposed about 12 GiB VRAM and did not fit the frozen roughly 30B judge artifact. An A100 path was outside the authorized QOS. These are historical constraints, not a live hardware inventory. Do not change quantization, profiles, CPU/GPU policy or scheduler permissions silently to force a fit. SSH credentials stay on the user's machine.

Before any future remote submission, reconcile exact existing lifecycle IDs and preserve terminal output. An old GPU-hours estimate is not current quota. Do not launch a new identity just because a controller stopped. CPU preflight must resolve transport/runtime/protocol problems before a GPU job. Download results before deleting ephemeral runtime resources.

## What the recent failure chain taught us

| Observed failure | Durable correction |
|---|---|
| Transport, helper identity, mounts, runtime/TLS and disk failures | Separate environment readiness from model behavior; exercise captured protocol transitions offline. |
| Receipt serialization and worker-lineage mismatches | Hash the exact serialized bytes and exact rendered worker; preserve parent and effective-contract lineage. |
| GLM Q04 thinking exhausted 6,144 / 8,192 / 12,288 budgets with no final answer | Stop blind budget escalation; the ratified v186 profile changed to thinking_off before the fresh semantic result. |
| v186 GLM then failed semantic gates | Inspect all existing result rows. The later failure is not proof of the earlier empty-content problem recurring. |
| Repeated rubric amendments after viewing the same public 16 tasks | Label that suite calibration; use prospective independently authored evidence for generalization. Keep MC8 sealed. |
| Validator trusts embedded gold and collapses task IDs into a dictionary | Diagnostic recomputation must use the frozen external task gold and reject duplicate/missing IDs. Do not infer that tampering caused the actual model failure. |
| Recovery replaces canonical qualification directory before checking its contents | New diagnostic reads downloaded output only and writes to its own run directory. |
| Qwen v100 passed 35 native Windows tests but SSH mkdir used Windows-formatted remote paths; SCP then failed | Use explicit POSIX path values for Magnolia addresses while keeping local Windows paths native. Test client and server conventions independently, including result download. |

Preserve the exact 63 replacements, one deferred item, 287-candidate pool, Gemma v182 rows and existing qualified bindings. The new audit is a continuation, not a restart. Neither version numbers nor green structural validators demonstrate learned identity or memory cognition.

## Approved Qwen implementation checkpoint

Owner activation of the Qwen fallback is recorded in `fallback/OWNER_APPROVAL.json`; no renewed permission is required for that exact public run. `QWEN_RELEASE.json` records the checked package and launcher identities. The implementation carries forward the exact pinned Ollama archive/binary and Python shared-library fix. It uses a durable Slurm submission intent, no automatic requeue, per-task request/stream checkpoints, a fixed token ceiling, independent score reconstruction and terminal telemetry recovery. Resume means attaching or collecting from the same run; it never means silently repeating model judgments.

A source build, a test pass, and a real calibration pass are different evidence states. Future chats must inspect `qwen-public/LATEST_QWEN_PUBLIC.json` and the actual returned result ZIP before advancing any eligibility gate. The package's approved six-hour CPU envelope is not a new GPU quota estimate or scheduler authorization.

## Windows transport repair checkpoint

`QWEN_V100_WINDOWS_TRANSFER_FINDINGS.md` records the reproduced defect and `QWEN_TRANSPORT_REPAIR_V101.json` identifies the historical v101 launcher. The original v100 workload ZIP and stable run ID remain frozen. The repair wraps that exact ZIP and adjusts only local remote-address metadata after authority validation. Never replace its original workload identity with the transport wrapper's ZIP hash in controller state. No manual state cleanup, new run or approval is required.

The supplied native Windows log passed all 35 original tests, then stopped before submission. The repair release passed 11 focused tests on Linux. One actual bash reproduction is intentionally skipped by the next Windows selftest; the other 10 run before live transport. Keep these observations separate from actual Magnolia success. Record a client repair receipt in both the fresh launcher folder and the persistent run archive. Return the transcript and actual result ZIP for verification after the next run.

## Recovered v1.7.6 CA and immutable-ledger requirements

The v101 result confirmed two regressions. Job 575089 stopped before any inference because the new worker omitted the documented CA setup. The ledger helper accepted the first heartbeat and rejected changed bytes under the same run ID with exit 73. These are infrastructure failures; do not interpret zero attempted tasks as Qwen semantic accuracy.

Carry the earlier explicit CA policy into every external HTTPS client and dependent process. Preserve certificate and hostname verification. Test the endpoint and freeze the model manifest before scheduler submission. Each changed telemetry payload must use its own immutable snapshot identity. Retry saved bytes under the same snapshot ID only. Never overwrite an old ledger run to clear a collision.

V102 uses one named source-bound infrastructure successor after exact zero-inference source evidence and live terminal reconciliation. The old job/state remain intact. Original probe/task request hashes are checked independently. Full worker-lifecycle tests now model the actual immutable publisher. The source terminal failure was appended through the GitHub connector under a separate snapshot; the original ZIP remains an unchanged historical record.

## V102 producer/consumer boundary lesson — 7 September 2026

Rayan explicitly instructed: do not make blind hotfixes; trace the failure and
the larger purpose before choosing a change. Work in evidence-led batches.
Astra's v102 source verifier incorrectly used the exported ZIP layout as the
live run layout. Its 48 passing native Windows tests recreated the same mistake.
The real collector reproduction is recorded in `QWEN_V102_ROOT_CAUSE_AND_DIRECTION.md`.
The actual read-only trace is now received in `QWEN_HOST_TRACE_REVIEW.json`.
Use the v103 release as the next step; earlier launchers above are historical.
The captured a2 stop was before source HTTPS and new submission.

For every evidence member, distinguish producer, physical origin, exported name,
expected bytes and authority. Exercise real producer-to-consumer transitions.
Fixtures must model package storage, live runs and exported bundles separately.
Check scheduler exit status before interpreting empty output. Preserve failed
pre-submission descriptors; a changed package must have explicit revision lineage.
Do not erase state, remove a hash check or copy exported metadata into the source
run to satisfy an incorrect locator. Test counts do not replace these proofs.

The bounded judge task serves eligible synthetic candidate evaluation. Measure
progress toward learned identity, AMFM, continuing missions and memory correctness
with the experiment ledger; package releases are operational work, not evidence
of those capabilities. Preserve existing approval and CPU/Kaggle preferences.


## Confirmed origins and recoverable revision — 7 September 2026

The host trace confirms intact package/source bytes and the missing exported
name in the live run. Keep one package-projection rule for live verification
and result collection. The expected manifest must come from its original
package; a copied export cannot establish that origin. Check the full original
package and ZIP, not only the convenient manifest file.

An unsubmitted descriptor still carries identity. V103 requires the exact
observed local state and preserves it before remote mutation. The remote
revision uses the original submission lock, checks for jobs and execution
markers, preserves original descriptor/failure bytes, writes intent first and
records completion after the active reference changes. Retry only the exact
recorded transition. A new descriptor without that history is a stop. Test the
actual old launcher rejecting revised state and lost acknowledgements causing
attachment, not duplicate submission.

The trace's HTTPS success is login-node HEAD evidence only. Preserve certificate
verification, pinned runtime, full model digest resolution and the compute-node
throughput gate. The six native diagnostic tests and sixty offline v103 tests
have different scope. No native v103 or scientific pass has yet been received.
Do not rerun the original a1 launcher to clear its historical telemetry state;
its terminal ledger recovery has a separate immutable receipt.


## V103 provider review — 7 September 2026

Astra used different scheduler query shapes in the real diagnostic and production
guard. Tests accepted the unobserved shape through programmed successful replies.
Provider tests must consume raw actual command observations. The same read-only
functions and arguments must serve preflight and production state interpretation.
Do not treat a nearby successful query as proof of a different query.

Preserve every scheduler command receipt before applying assertions. Store argv,
exit code, raw stdout/stderr and observation time. A high-level SSH stop and a
stderr byte count cannot support the next diagnosis. A completed source job may
be absent from the active controller; exact accounting remains a separate authority.
Do not ignore failed queue queries to compensate for choosing the wrong query.

Reconcile local intent before changing release identity. V103 writes its immutable
local intent before remote revision. Another package hash under that revision ID
conflicts even when the remote revision never completes. Archive and interpret
both sides of the transaction before selecting a successor or recovery action.
Never delete intent to make a new launcher run.

Consolidate provider behavior instead of creating another science package for
scheduler diagnostics. Existing science remains frozen. New packages must name
the exact changed provider/workload contract and its actual evidence. Keep prior
packages as historical records; avoid making old wrappers live dependencies.

Current agent routing is Astra primary while available and Sol fallback for
capacity or repeated failure. Resume from the newest shared observed checkpoint.
Before handoff, update both narrative and machine-readable state so neither
continues to recommend a launcher invalidated by the latest terminal.


## Read-only provider observation and roadmap — 7 September 2026

The provider trace uses the exact v103 user queue, source accounting, a2 accounting
and historical-job comparison arguments. Preserve raw bytes before interpreting
them. Scheduler failures are collected independently. Keep local and remote
inventories before and after the observation; concurrent changes leave an explicit
incomplete result. SSH timeout/nonzero/malformed responses still retain available
raw transport and local evidence in a result ZIP. No trace exit grants execution.

The nine offline tests use synthetic results, including error cases. They verify
the collector's behavior; the returned real provider receipt is still required.
Never claim that a later trace recovers a lost earlier stderr message.

Follow PHASE2_REPLACEMENT_ROADMAP.md. Stage G acceptance, G+ recollection and the
H/I/J authority stages remain distinct. E05 follows accepted G. Translate the
accepted deployment-profile scope amendment into machine policy and acceptance
evidence before Stage G closure. Final replacement requires accepted Stage J.


## Confirmed provider boundary and resource discipline — 7 September 2026

The real source-job query exits 1 with Invalid job id; user-wide queue and exact
accounting succeed. Reuse one shared scheduler boundary across reconciliation,
lookup and monitoring. Preserve raw receipts before interpreting every result.
Retain the unfinished local intent and bind its explicit supersession; another
remote revision or local acknowledgement requires review, not deletion.

Keep fault investigations bounded. Reuse existing evidence and tests. Avoid
repeating the full audit or regenerating roadmap documents for one provider
failure. Make one necessary implementation/verification/context batch. A passing
local test suite never substitutes for the live provider or scientific result.


## V104 actual telemetry boundary — 7 September 2026

V104 reached submission and passed login-node source/TLS preflight. The first
immutable snapshot publication then returned 74 before runtime preparation.
The wrapper retained selected diagnostic substrings but discarded the helper's
raw output. Apply the existing raw-command-receipt lesson to telemetry too.
Inspect the installed helper before attributing the exit or changing behavior.
Synthetic publisher tests cannot establish that installed helper's compatibility.

The connector recovered the exact terminal snapshot. Keep that recovery separate
from the original failed host receipts and local controller state. Do not replay
closed inference, reset state or bypass the initial telemetry gate to clear a
publication failure. Manifest resolution, runtime preparation, a throughput probe
and task completion are distinct observations.

For this missing source file, a direct native-key SCP copy is sufficient; a new
diagnostic package and another full selftest pass add no value. Resume a bounded
repair only after the actual helper establishes which evidence or change is needed.
