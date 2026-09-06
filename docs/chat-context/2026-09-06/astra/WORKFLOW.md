# Working agreement and recovered lessons

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

`QWEN_V100_WINDOWS_TRANSFER_FINDINGS.md` records the reproduced defect and `QWEN_TRANSPORT_REPAIR_V101.json` identifies the active launcher. The original v100 workload ZIP and stable run ID remain frozen. The repair wraps that exact ZIP and adjusts only local remote-address metadata after authority validation. Never replace its original workload identity with the transport wrapper's ZIP hash in controller state. No manual state cleanup, new run or approval is required.

The supplied native Windows log passed all 35 original tests, then stopped before submission. The repair release passed 11 focused tests on Linux. One actual bash reproduction is intentionally skipped by the next Windows selftest; the other 10 run before live transport. Keep these observations separate from actual Magnolia success. Record a client repair receipt in both the fresh launcher folder and the persistent run archive. Return the transcript and actual result ZIP for verification after the next run.

## Recovered v1.7.6 CA and immutable-ledger requirements

The v101 result confirmed two regressions. Job 575089 stopped before any inference because the new worker omitted the documented CA setup. The ledger helper accepted the first heartbeat and rejected changed bytes under the same run ID with exit 73. These are infrastructure failures; do not interpret zero attempted tasks as Qwen semantic accuracy.

Carry the earlier explicit CA policy into every external HTTPS client and dependent process. Preserve certificate and hostname verification. Test the endpoint and freeze the model manifest before scheduler submission. Each changed telemetry payload must use its own immutable snapshot identity. Retry saved bytes under the same snapshot ID only. Never overwrite an old ledger run to clear a collision.

V102 uses one named source-bound infrastructure successor after exact zero-inference source evidence and live terminal reconciliation. The old job/state remain intact. Original probe/task request hashes are checked independently. Full worker-lifecycle tests now model the actual immutable publisher. The source terminal failure was appended through the GitHub connector under a separate snapshot; the original ZIP remains an unchanged historical record.
